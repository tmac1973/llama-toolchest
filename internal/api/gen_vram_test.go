package api

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/tmac1973/llama-toolchest/internal/generation"
	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/monitor"
)

// vramServer is instanceServer on a 32,624 MiB card with 1,300 MiB used by
// the desktop: the asmodean R9700.
func vramServer(t *testing.T) *Server {
	t.Helper()
	s := instanceServer(t)
	s.testHardware = &models.Hardware{GPUs: []models.GPUSpec{{Index: 0, Name: "R9700", VRAMTotalMiB: 32624, OtherUsedMiB: 1300}}}
	return s
}

func addInstance(t *testing.T, s *Server, inst generation.Instance) {
	t.Helper()
	if inst.ModelSetID == "" {
		inst.ModelSetID = "ming-image"
	}
	if inst.Name == "" {
		inst.Name = inst.ID
	}
	if err := s.gen.AddInstance(inst); err != nil {
		t.Fatal(err)
	}
}

func TestSuggestReservation(t *testing.T) {
	// The October measurements: Qwen-Image 10,283 MiB, ACE-Step 7,239 MiB.
	for peak, want := range map[int]int{10283: 11776, 7239: 8192, 512: 1024, 465: 512} {
		if got := suggestReservation(peak); got != want {
			t.Errorf("suggestReservation(%d) = %d, want %d", peak, got, want)
		}
	}
}

// Enabled instances' reservations reach the planner; disabled ones and
// everything while generation is off do not.
func TestHardwareCarriesReservations(t *testing.T) {
	s := vramServer(t)
	addInstance(t, s, generation.Instance{ID: "a", Port: 8081, Enabled: true, ReservedMiB: 11776})
	addInstance(t, s, generation.Instance{ID: "b", Port: 8082, Enabled: true, ReservedMiB: 8192})
	addInstance(t, s, generation.Instance{ID: "c", Port: 8083, Enabled: false, ReservedMiB: 4096})

	if got := s.hardware().GPUs[0].ReservedMiB; got != 11776+8192 {
		t.Errorf("ReservedMiB = %d, want the two enabled instances", got)
	}
	s.cfg.GenerationEnabled = false
	if got := s.hardware().GPUs[0].ReservedMiB; got != 0 {
		t.Errorf("ReservedMiB with generation off = %d, want 0", got)
	}
	if s.testHardware.GPUs[0].ReservedMiB != 0 {
		t.Error("hardware() changed the test hardware it was given")
	}
}

func TestGenFitCheck(t *testing.T) {
	s := vramServer(t)
	addInstance(t, s, generation.Instance{ID: "qwen", Port: 8081, Enabled: true, ReservedMiB: 11776})
	ace := generation.Instance{ID: "ace", Name: "ace", ModelSetID: "ming-image", Port: 8082}

	if err := s.genFitCheck(ace, 8192); err != nil {
		t.Errorf("8 GiB next to 11.5 GiB on a 32 GB card: %v", err)
	}
	err := s.genFitCheck(ace, 20000)
	if err == nil {
		t.Fatal("20,000 MiB next to 11.5 GiB was accepted")
	}
	for _, want := range []string{"GPU 0", "ace needs 19.5 GiB", "qwen 11.5 GiB", "other programs 1.3 GiB"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("message lacks %q: %v", want, err)
		}
	}
	// The instance's own current reservation does not count against it.
	qwen, _ := s.gen.Instance("qwen")
	if err := s.genFitCheck(qwen, 11776); err != nil {
		t.Errorf("an instance re-checked against itself: %v", err)
	}
}

func TestStartNeedsAReservation(t *testing.T) {
	s := vramServer(t)
	addInstance(t, s, generation.Instance{ID: "m", Port: 8081, Enabled: true})
	err := s.startGenInstance("m")
	if err == nil || !strings.Contains(err.Error(), "Measure") {
		t.Errorf("start without a reservation: %v, want a pointer to Measure", err)
	}
}

func TestEnableAndReservationAreFitChecked(t *testing.T) {
	s := vramServer(t)
	addInstance(t, s, generation.Instance{ID: "big", Port: 8081, Enabled: true, ReservedMiB: 20480})
	addInstance(t, s, generation.Instance{ID: "ming", Port: 8082, Enabled: false, ReservedMiB: 19456})

	rec := httptest.NewRecorder()
	s.handleGenInstanceAction(rec, instanceReq("POST", "/", "ming", "enable", nil))
	if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), "not enough VRAM") {
		t.Errorf("enabling a set that does not fit: %d %s", rec.Code, rec.Body.String())
	}
	if inst, _ := s.gen.Instance("ming"); inst.Enabled {
		t.Error("enabled despite not fitting")
	}

	rec = httptest.NewRecorder()
	s.handleGenSetReservation(rec, instanceReq("PUT", "/", "big", "", url.Values{"reserved_gib": {"30"}}))
	if rec.Code != http.StatusConflict {
		t.Errorf("a 30 GiB reservation: %d, want 409", rec.Code)
	}
	rec = httptest.NewRecorder()
	s.handleGenSetReservation(rec, instanceReq("PUT", "/", "big", "", url.Values{"reserved_gib": {"12.5"}}))
	if rec.Code != http.StatusOK {
		t.Fatalf("a 12.5 GiB reservation: %d %s", rec.Code, rec.Body.String())
	}
	if inst, _ := s.gen.Instance("big"); inst.ReservedMiB != 12800 {
		t.Errorf("ReservedMiB = %d, want 12800", inst.ReservedMiB)
	}
	// A disabled instance's reservation may be set to anything; the fit is
	// checked when it is enabled.
	rec = httptest.NewRecorder()
	s.handleGenSetReservation(rec, instanceReq("PUT", "/", "ming", "", url.Values{"reserved_gib": {"25"}}))
	if rec.Code != http.StatusOK {
		t.Errorf("reservation of a disabled instance: %d", rec.Code)
	}
}

// An on-demand LLM load that would crowd out the reservations is refused
// with a 507-mapped error naming them.
func TestLLMLoadRespectsReservations(t *testing.T) {
	s := vramServer(t)
	s.cfg.ModelsMax = 1
	addInstance(t, s, generation.Instance{ID: "qwen", Port: 8081, Enabled: true, ReservedMiB: 11776})
	addInstance(t, s, generation.Instance{ID: "ace", Port: 8082, Enabled: true, ReservedMiB: 8192})

	add := func(id string, gib float64) *models.Model {
		m := &models.Model{ID: id, ModelID: "org/" + id, Filename: id + ".gguf", SizeBytes: int64(gib / 1.1 * (1 << 30))}
		if err := s.registry.Add(m); err != nil {
			t.Fatal(err)
		}
		return m
	}
	small := add("small", 5.5) // about 5.7 GiB with overhead: fits in the 8.6 GiB left
	big := add("big", 11.5)    // about 11.7 GiB: does not

	if err := s.checkLLMFitsReservations(small); err != nil {
		t.Errorf("small model: %v", err)
	}
	err := s.checkLLMFitsReservations(big)
	var fitErr *genLLMFitError
	if !errors.As(err, &fitErr) || !strings.Contains(err.Error(), "19.5 GiB reserved for generation backends on GPU 0") {
		t.Errorf("big model: %v", err)
	}
	if !strings.Contains(err.Error(), "the LLM needs about 11.") {
		t.Errorf("big model message does not say what it needs: %v", err)
	}

	// A benchmark cell is checked with its own settings, alone.
	cfg, _ := s.registry.GetConfig("big")
	if err := s.checkConfigFitsReservations(big, cfg); !errors.As(err, &fitErr) {
		t.Errorf("benchmark cell of the big model: %v, want refused", err)
	}
	cfg, _ = s.registry.GetConfig("small")
	if err := s.checkConfigFitsReservations(small, cfg); err != nil {
		t.Errorf("benchmark cell of the small model: %v", err)
	}

	// The Models page's Load button is refused the same way, with 507,
	// before the router is started.
	rec := httptest.NewRecorder()
	req := httptest.NewRequest("PUT", "/api/models/big/activate", nil)
	rc := chi.NewRouteContext()
	rc.URLParams.Add("id", "big")
	s.handleActivateModel(rec, req.WithContext(context.WithValue(req.Context(), chi.RouteCtxKey, rc)))
	if rec.Code != http.StatusInsufficientStorage || !strings.Contains(rec.Body.String(), "reserved for generation backends") {
		t.Errorf("Load button for the big model: %d %q, want 507 naming the reservations", rec.Code, rec.Body.String())
	}

	s.cfg.GenerationEnabled = false
	if err := s.checkLLMFitsReservations(big); err != nil {
		t.Errorf("with generation off: %v", err)
	}
}

// A load that replaces the loaded model, and a benchmark cell, are
// planned on the cards as they were with nothing of ours loaded: what the
// loaded model holds beyond its buffer report reads as other programs'
// memory until it is unloaded. A load beside it is planned on the cards
// as they are.
func TestReplacingLoadsUseIdleRoom(t *testing.T) {
	s := vramServer(t)
	// 2 GiB of the loaded model's memory counted as other programs'.
	s.testHardware.GPUs[0].OtherUsedMiB = 1300 + 2048
	addInstance(t, s, generation.Instance{ID: "qwen", Port: 8081, Enabled: true, ReservedMiB: 11776})
	addInstance(t, s, generation.Instance{ID: "ace", Port: 8082, Enabled: true, ReservedMiB: 8192})
	m := &models.Model{ID: "mid", ModelID: "org/mid", Filename: "mid.gguf", SizeBytes: 7 << 30}
	if err := s.registry.Add(m); err != nil {
		t.Fatal(err)
	}
	cfg, _ := s.registry.GetConfig("mid")
	var fitErr *genLLMFitError

	// No reading with nothing loaded yet: the cards as they are.
	if err := s.checkConfigFitsReservations(m, cfg); !errors.As(err, &fitErr) {
		t.Errorf("benchmark cell without an idle reading: %v, want refused", err)
	}

	s.otherVRAM.idleOther = []int{1300}
	if err := s.checkConfigFitsReservations(m, cfg); err != nil {
		t.Errorf("benchmark cell: %v", err)
	}
	s.cfg.ModelsMax = 1
	if err := s.checkLLMFitsReservations(m); err != nil {
		t.Errorf("load replacing the loaded model: %v", err)
	}
	s.cfg.ModelsMax = 2
	if err := s.checkLLMFitsReservations(m); !errors.As(err, &fitErr) {
		t.Errorf("load beside the loaded model: %v, want refused", err)
	}

	if got := s.idleHardware().GPUs[0]; got.OtherUsedMiB != 1300 || got.ReservedMiB != 11776+8192 {
		t.Errorf("idleHardware GPU 0 = other %d, reserved %d; want 1300 and both reservations", got.OtherUsedMiB, got.ReservedMiB)
	}
}

func TestWithoutGenUse(t *testing.T) {
	gpus := []monitor.GPUInfo{{Index: 0}, {Index: 1}, {Index: 2}}
	gen := map[int]*genVRAM{1: {Using: 158}, 2: {Using: 9000}}
	got := withoutGenUse([]int{1300, 500, 4000}, gpus, gen)
	if want := []int{1300, 342, 0}; !slices.Equal(got, want) {
		t.Errorf("withoutGenUse = %v, want %v", got, want)
	}
}

func TestMeasureRefusals(t *testing.T) {
	s := vramServer(t)
	addInstance(t, s, generation.Instance{ID: "m", Port: 8081, Enabled: true})
	inst, _ := s.gen.Instance("m")
	s.genMgr.SetMeasuring("m", true)
	if err := s.canMeasure(inst); err == nil {
		t.Error("measuring twice was allowed")
	}
	s.genMgr.SetMeasuring("m", false)
	if err := s.canMeasure(inst); err != nil {
		t.Errorf("an idle GPU: %v", err)
	}
}

func TestGPUMapShowsReservations(t *testing.T) {
	s := vramServer(t)
	addInstance(t, s, generation.Instance{ID: "qwen", Name: "qwen", Port: 8081, Enabled: true, ReservedMiB: 11776})
	addInstance(t, s, generation.Instance{ID: "ace", Name: "ace", Port: 8082, Enabled: true, ReservedMiB: 8192})
	// One LLM is already on the map, so the reservations take the
	// palette's second and third colors.
	allocs := s.genReservationAllocations(1)
	if len(allocs) != 2 || allocs[0].ModelName != "qwen (reserved)" || allocs[0].PerGPUGB != 11.5 || !allocs[0].Reserved {
		t.Fatalf("allocations = %+v", allocs)
	}
	if allocs[0].Color != models.AllocationColor(1) || allocs[1].Color != models.AllocationColor(2) {
		t.Errorf("colors = %s, %s; want the palette's 2nd and 3rd, each its own", allocs[0].Color, allocs[1].Color)
	}
}

// After a test run the card's reading lags behind the backend's exit; the
// wait returns once it is back near the reading from before the run, and
// gives up after its limit.
func TestWaitGPUReleased(t *testing.T) {
	readings := []int{11300, 11300, 1400, 1300}
	i := 0
	used := func() int {
		v := readings[min(i, len(readings)-1)]
		i++
		return v
	}
	if !waitGPUReleased(used, 1200, time.Second, time.Millisecond) {
		t.Error("reading came back within the slack, but the wait gave up")
	}
	if i != 3 {
		t.Errorf("read %d times, want 3 (stop at the first released reading)", i)
	}
	if waitGPUReleased(func() int { return 9000 }, 1200, 20*time.Millisecond, time.Millisecond) {
		t.Error("reading never came back, but the wait reported success")
	}
}
