package api

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/generation"
	"github.com/tmac1973/llama-toolchest/internal/models"
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
	s.cfg.GenerationEnabled = false
	if err := s.checkLLMFitsReservations(big); err != nil {
		t.Errorf("with generation off: %v", err)
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
	allocs := s.genReservationAllocations()
	if len(allocs) != 1 || allocs[0].ModelName != "qwen (reserved)" || allocs[0].PerGPUGB != 11.5 {
		t.Errorf("allocations = %+v", allocs)
	}
}
