package api

import (
	"context"
	"errors"
	"github.com/tmac1973/llama-toolchest/internal/config"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/generation"
	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/monitor"
)

// addRoomModel registers an enabled model of sizeGiB on disk.
func addRoomModel(t *testing.T, s *Server, id string, sizeGiB int64) *models.Model {
	t.Helper()
	m := &models.Model{
		ID: id, ModelID: "org/" + id + "-GGUF", Quant: "Q4_K_M",
		Filename: id + ".gguf", FilePath: filepath.Join(t.TempDir(), id+".gguf"),
		SizeBytes: sizeGiB << 30,
	}
	if err := s.registry.Add(m); err != nil {
		t.Fatal(err)
	}
	if err := s.registry.SetConfig(m.ID, &models.ModelConfig{Enabled: true}); err != nil {
		t.Fatal(err)
	}
	return m
}

// tryEvict never waits: it fails at once for a model with a request in
// progress, and an unload it starts holds new requests back until done.
func TestLLMUseTryEvict(t *testing.T) {
	var u llmUse
	ctx := context.Background()
	end, err := u.begin(ctx, "qwen")
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := u.tryEvict("qwen"); ok {
		t.Fatal("tryEvict took a model with a request in progress")
	}
	end()

	done, ok := u.tryEvict("qwen")
	if !ok {
		t.Fatal("tryEvict refused an idle model")
	}
	if _, ok := u.tryEvict("qwen"); ok {
		t.Error("a second tryEvict took a model already being unloaded")
	}
	if !u.busy("qwen") {
		t.Error("a model being unloaded is not busy")
	}
	began := make(chan struct{})
	go func() {
		end, _ := u.begin(ctx, "qwen")
		end()
		close(began)
	}()
	select {
	case <-began:
		t.Fatal("a request began while the model was being unloaded")
	case <-time.After(30 * time.Millisecond):
	}
	done()
	<-began
}

// The idle list has the loaded models no request uses, least recently
// used first, each with its planned memory; a busy model, one still
// loading, an unknown one and except are left out.
func TestIdleLLMsByLRU(t *testing.T) {
	s, f, _ := startFakeRouter(t)
	s.testHardware = &models.Hardware{GPUs: []models.GPUSpec{{Index: 0, Name: "R9700", VRAMTotalMiB: 32624, OtherUsedMiB: 1300}}}
	for _, id := range []string{"chat", "embed", "rerank", "busy", "loading", "except"} {
		addRoomModel(t, s, id, 2)
	}
	f.reset(map[string]string{
		"chat": "loaded", "embed": "loaded", "rerank": "loaded", "busy": "loaded",
		"loading": "loading", "except": "loaded", "stranger": "loaded",
	}, nil)

	s.llmUse.touch("rerank")
	time.Sleep(2 * time.Millisecond)
	s.llmUse.touch("chat")
	end, err := s.llmUse.begin(context.Background(), "busy")
	if err != nil {
		t.Fatal(err)
	}
	defer end()

	got := s.idleLLMsByLRU("except")
	var ids []string
	for _, l := range got {
		ids = append(ids, l.id)
		if l.mib[0] <= 0 {
			t.Errorf("%s has no planned memory on GPU 0: %v", l.id, l.mib)
		}
	}
	// embed was never used, so it sorts first.
	want := []string{"embed", "rerank", "chat"}
	if len(ids) != len(want) {
		t.Fatalf("idle = %v, want %v", ids, want)
	}
	for i := range want {
		if ids[i] != want[i] {
			t.Fatalf("idle = %v, want %v", ids, want)
		}
	}

	// loaded and loading models of ours count; stranger is not ours.
	if n := s.loadedLLMCount(); n != 6 {
		t.Errorf("loadedLLMCount = %d, want 6", n)
	}
}

// roomServer is startFakeRouter in toolchest memory mode on the asmodean
// R9700 (budget 28,818 MiB for models). Planned sizes on it: 8 GiB on
// disk → 9,216 MiB, 12 → 13,721, 16 → 18,227, 18 → 20,480.
func roomServer(t *testing.T) (*Server, *fakeRouter) {
	t.Helper()
	s, f, _ := startFakeRouter(t)
	s.testHardware = &models.Hardware{GPUs: []models.GPUSpec{{Index: 0, Name: "R9700", VRAMTotalMiB: 32624, OtherUsedMiB: 1300}}}
	s.cfg.VRAMMarginMiB = ptr(oldTestMarginMiB)
	s.memoryMode.Store(int32(memoryToolchest))
	return s, f
}

// loaded sets the router's list to ids loaded, used in that order.
func loaded(t *testing.T, s *Server, f *fakeRouter, ids ...string) {
	t.Helper()
	st := map[string]string{}
	for _, id := range ids {
		st[s.registry.RouterName(id)] = "loaded"
		s.llmUse.touch(id)
		time.Sleep(2 * time.Millisecond)
	}
	f.reset(st, nil)
}

func loadWithRoom(t *testing.T, s *Server, m *models.Model) error {
	t.Helper()
	release, err := s.loadLLMWithRoom(context.Background(), m, s.registry.RouterName(m.ID))
	release()
	return err
}

// A model that fits beside the loaded ones loads with nothing unloaded.
func TestLoadWithRoomFitsBeside(t *testing.T) {
	s, f := roomServer(t)
	addRoomModel(t, s, "chat", 16)
	embed := addRoomModel(t, s, "embed", 1)
	loaded(t, s, f, "chat")
	if err := loadWithRoom(t, s, embed); err != nil {
		t.Fatal(err)
	}
	if u := f.unloadCalls(); len(u) != 0 {
		t.Errorf("unloaded %v for a model that fits", u)
	}
	if l := f.loadCalls(); !slices.Equal(l, []string{s.registry.RouterName("embed")}) {
		t.Errorf("loads = %v", l)
	}
}

// With models loaded, hardware counts what they hold beyond their buffer
// report as other programs' memory; their plans count it as well. The
// room is taken from the reading with nothing of ours loaded, so a model
// that fits by plan loads beside them with nothing unloaded.
func TestLoadWithRoomCountsLoadedOnce(t *testing.T) {
	s, f := roomServer(t)
	s.testHardware.GPUs[0].OtherUsedMiB = 1300 + 2048
	s.otherVRAM.idleOther = []int{1300}
	addRoomModel(t, s, "chat", 16)
	embed := addRoomModel(t, s, "embed", 8)
	loaded(t, s, f, "chat")
	// 18,227 + 9,216 MiB: under the 28,818 MiB budget, over the 26,900
	// the 2 GiB counted twice would leave.
	if blocked, err := s.neverFits(embed); err != nil || blocked {
		t.Errorf("neverFits: blocked %v, err %v", blocked, err)
	}
	if err := loadWithRoom(t, s, embed); err != nil {
		t.Fatal(err)
	}
	if u := f.unloadCalls(); len(u) != 0 {
		t.Errorf("unloaded %v for a model that fits by plan", u)
	}
}

// When it does not fit, the least recently used idle model goes, with
// its reason noted; a busy model stays even when it is older.
func TestLoadWithRoomUnloadsLRU(t *testing.T) {
	s, f := roomServer(t)
	addRoomModel(t, s, "a", 12)
	addRoomModel(t, s, "b", 8)
	c := addRoomModel(t, s, "c", 8)

	loaded(t, s, f, "a", "b")
	if err := loadWithRoom(t, s, c); err != nil {
		t.Fatal(err)
	}
	if u := f.unloadCalls(); !slices.Equal(u, []string{s.registry.RouterName("a")}) {
		t.Fatalf("unloads = %v, want only a", u)
	}
	if got := s.takeUnloadReason(s.registry.RouterName("a"), time.Now()); got != "room for "+c.PublicName() {
		t.Errorf("reason = %q", got)
	}

	loaded(t, s, f, "a", "b")
	end, err := s.llmUse.begin(context.Background(), "a")
	if err != nil {
		t.Fatal(err)
	}
	defer end()
	if err := loadWithRoom(t, s, c); err != nil {
		t.Fatal(err)
	}
	if u := f.unloadCalls(); !slices.Equal(u, []string{s.registry.RouterName("b")}) {
		t.Fatalf("unloads with a busy = %v, want b", u)
	}
}

// With only a busy model in the way, nothing is unloaded and the load is
// sent: the plan is an estimate, so llama-server decides (issue #561).
func TestLoadWithRoomTriesWhenAllBusy(t *testing.T) {
	s, f := roomServer(t)
	addRoomModel(t, s, "a", 18)
	c := addRoomModel(t, s, "c", 12)
	loaded(t, s, f, "a")
	end, _ := s.llmUse.begin(context.Background(), "a")
	defer end()
	if err := loadWithRoom(t, s, c); err != nil {
		t.Fatal(err)
	}
	if len(f.unloadCalls()) != 0 {
		t.Errorf("unloads %v; want none", f.unloadCalls())
	}
	if l := f.loadCalls(); !slices.Equal(l, []string{s.registry.RouterName("c")}) {
		t.Errorf("loads = %v, want c", l)
	}
}

// A model whose plan alone is over the card's budget, with no
// reservation, is still loaded once the idle models on the card are
// gone: issue #561, a 24 GB card refused a model it ran on 2.50.
func TestLoadWithRoomOverPlanAlone(t *testing.T) {
	s, f := roomServer(t)
	addRoomModel(t, s, "a", 8)
	huge := addRoomModel(t, s, "huge", 26)
	loaded(t, s, f, "a")
	if err := loadWithRoom(t, s, huge); err != nil {
		t.Fatal(err)
	}
	if u := f.unloadCalls(); !slices.Equal(u, []string{s.registry.RouterName("a")}) {
		t.Errorf("unloads = %v, want a", u)
	}
	if l := f.loadCalls(); !slices.Equal(l, []string{s.registry.RouterName("huge")}) {
		t.Errorf("loads = %v, want huge", l)
	}
}

// Max Loaded is a limit toolchest keeps: at the limit the least recently
// used idle model goes, even with room to spare.
func TestLoadWithRoomMaxLoaded(t *testing.T) {
	s, f := roomServer(t)
	s.cfg.ModelsMax = 2
	addRoomModel(t, s, "a", 1)
	addRoomModel(t, s, "b", 1)
	c := addRoomModel(t, s, "c", 1)
	loaded(t, s, f, "a", "b")
	if err := loadWithRoom(t, s, c); err != nil {
		t.Fatal(err)
	}
	if u := f.unloadCalls(); !slices.Equal(u, []string{s.registry.RouterName("a")}) {
		t.Errorf("unloads = %v, want a", u)
	}
}

// An unload that fails stops the load with the error.
func TestLoadWithRoomUnloadFails(t *testing.T) {
	s, f := roomServer(t)
	addRoomModel(t, s, "a", 18)
	c := addRoomModel(t, s, "c", 12)
	loaded(t, s, f, "a")
	f.mu.Lock()
	f.unloadFails = true
	f.mu.Unlock()
	err := loadWithRoom(t, s, c)
	if err == nil || !strings.Contains(err.Error(), "make room for "+c.PublicName()) {
		t.Fatalf("err = %v, want the wrapped unload error", err)
	}
	if len(f.loadCalls()) != 0 {
		t.Error("loaded after a failed unload")
	}
}

// The real free check unloads one more model when the card shows less
// free than the plan says, and counts a model still loading in full.
func TestLoadWithRoomRealFreeCheck(t *testing.T) {
	s, f := roomServer(t)
	a := addRoomModel(t, s, "a", 12)
	c := addRoomModel(t, s, "c", 8)
	loaded(t, s, f, "a")
	// The card shows a's plan used, plus 7 GiB more than the plan knows.
	s.testGPUReadings = func() []monitor.GPUInfo {
		used := 1300 + 7*1024
		f.mu.Lock()
		if f.status[s.registry.RouterName("a")] == "loaded" {
			used += 13721
		}
		f.mu.Unlock()
		return []monitor.GPUInfo{{Index: 0, VRAMTotalMB: 32624, VRAMUsedMB: used}}
	}
	if err := loadWithRoom(t, s, c); err != nil {
		t.Fatal(err)
	}
	if u := f.unloadCalls(); !slices.Equal(u, []string{s.registry.RouterName("a")}) {
		t.Fatalf("unloads = %v, want a", u)
	}

	// a still loading counts with its whole plan.
	f.reset(map[string]string{s.registry.RouterName("a"): "loading"}, nil)
	s.testGPUReadings = func() []monitor.GPUInfo {
		return []monitor.GPUInfo{{Index: 0, VRAMTotalMB: 32624, VRAMUsedMB: 1300}}
	}
	room := models.RoomOf(s.idleHardware())
	if got := s.loadingLLMMiB(room, c.ID)[0]; got != 13721 {
		t.Fatalf("loading = %d, want a's plan", got)
	}
	if short := s.realFreeShortGPUs(c); len(short) != 0 {
		t.Fatalf("short = %+v with room for both", short)
	}
	big := addRoomModel(t, s, "big", 16)
	short := s.realFreeShortGPUs(big)
	if len(short) != 1 || short[0].needed != 18227+13721+oldTestMarginMiB {
		t.Errorf("short = %+v, want big + loading a + the margin", short)
	}
	_ = a
}

// When the card shows less free than the plan needs and nothing idle is
// left to unload, the load is still sent: llama-server decides.
func TestLoadWithRoomRealFreeShortStillLoads(t *testing.T) {
	s, f := roomServer(t)
	c := addRoomModel(t, s, "c", 16)
	s.testGPUReadings = func() []monitor.GPUInfo {
		return []monitor.GPUInfo{{Index: 0, VRAMTotalMB: 32624, VRAMUsedMB: 20000}}
	}
	if short := s.realFreeShortGPUs(c); len(short) != 1 {
		t.Fatalf("short = %+v, want GPU 0", short)
	}
	if err := loadWithRoom(t, s, c); err != nil {
		t.Fatal(err)
	}
	if l := f.loadCalls(); !slices.Equal(l, []string{s.registry.RouterName("c")}) {
		t.Errorf("loads = %v, want c", l)
	}
}

// Two loads at once run one after the other: the second sees the first
// as loading, so it unloads a to fit.
func TestLoadWithRoomOneAtATime(t *testing.T) {
	s, f := roomServer(t)
	addRoomModel(t, s, "a", 12)
	c := addRoomModel(t, s, "c", 8)
	d := addRoomModel(t, s, "d", 8)
	loaded(t, s, f, "a")
	errs := make(chan error, 2)
	for _, m := range []*models.Model{c, d} {
		go func() { errs <- loadWithRoom(t, s, m) }()
	}
	for range 2 {
		if err := <-errs; err != nil {
			t.Fatal(err)
		}
	}
	if u := f.unloadCalls(); !slices.Equal(u, []string{s.registry.RouterName("a")}) {
		t.Errorf("unloads = %v, want a once", u)
	}
	if l := f.loadCalls(); len(l) != 2 {
		t.Errorf("loads = %v, want c and d", l)
	}
}

// In toolchest mode Max Loaded 1 is not a replace for the reservation
// check: the router replaces nothing on its own.
func TestToolchestModeIsNotReplacing(t *testing.T) {
	s, _ := roomServer(t)
	withGen(t, s)
	s.cfg.ModelsMax = 1
	addInstance(t, s, generation.Instance{ID: "img", Port: 8081, Enabled: true, ReservedMiB: 4096})
	c := addRoomModel(t, s, "c", 1)
	if _, _, replacing := s.llmReservationNeed(c); replacing {
		t.Error("replacing in toolchest mode")
	}
	s.memoryMode.Store(int32(memoryCount))
	if _, _, replacing := s.llmReservationNeed(c); !replacing {
		t.Error("not replacing with Max Loaded 1 in count mode")
	}
}

// withGen turns generation backends on for s, with the image model set
// instanceServer has.
func withGen(t *testing.T, s *Server) {
	t.Helper()
	s.cfg.GenerationEnabled = true
	store, err := generation.Load(s.cfg.DataDir)
	if err != nil {
		t.Fatal(err)
	}
	s.gen = store
	if err := store.AddModelSet(generation.ModelSet{ID: "ming-image", Name: "Ming-Image", Kind: generation.KindSD, Complete: true,
		Components: []generation.Component{{Slot: "diffusion_model", Path: "/m/dit.gguf"}}}); err != nil {
		t.Fatal(err)
	}
}

// runningBackend starts a fake backend reserving reserveMiB on GPU 0.
func runningBackend(t *testing.T, s *Server, reserveMiB int) {
	t.Helper()
	runFakeInstance(t, s, "ming", "ming-image", 0, newFakeBackend(t, 0))
	inst, _ := s.gen.Instance("ming")
	inst.ReservedMiB = reserveMiB
	if err := s.gen.UpdateInstance(inst); err != nil {
		t.Fatal(err)
	}
}

// With a running backend in the way, idle LLMs go first; the backend is
// stopped (eviction on) only when they are not enough.
func TestLoadWithRoomLLMsBeforeBackends(t *testing.T) {
	s, f := roomServer(t)
	withGen(t, s)
	s.cfg.GenEvict, s.cfg.Eviction = true, config.EvictionToolchest
	runningBackend(t, s, 10000) // budget 18,818 MiB beside it
	addRoomModel(t, s, "a", 8)
	c := addRoomModel(t, s, "c", 12)
	loaded(t, s, f, "a")
	if err := loadWithRoom(t, s, c); err != nil {
		t.Fatal(err)
	}
	if u := f.unloadCalls(); !slices.Equal(u, []string{s.registry.RouterName("a")}) {
		t.Fatalf("unloads = %v, want a", u)
	}
	if !s.genMgr.IsActive("ming") {
		t.Fatal("the backend was stopped though unloading a was enough")
	}

	big := addRoomModel(t, s, "big", 18)
	loaded(t, s, f, "a")
	if err := loadWithRoom(t, s, big); err != nil {
		t.Fatal(err)
	}
	if u := f.unloadCalls(); !slices.Equal(u, []string{s.registry.RouterName("a")}) {
		t.Errorf("unloads = %v, want a", u)
	}
	if s.genMgr.IsActive("ming") {
		t.Error("the backend still runs though the idle LLMs were not enough")
	}
}

// With eviction off, the same load is refused after the idle LLMs are
// unloaded, naming the reservation.
func TestLoadWithRoomEvictOffRefuses(t *testing.T) {
	s, f := roomServer(t)
	withGen(t, s)
	runningBackend(t, s, 10000)
	addRoomModel(t, s, "a", 8)
	big := addRoomModel(t, s, "big", 18)
	loaded(t, s, f, "a")
	err := loadWithRoom(t, s, big)
	var fitErr *genLLMFitError
	if !errors.As(err, &fitErr) || !strings.Contains(err.Error(), "reserved for generation backends") {
		t.Fatalf("err = %v, want the reservation refusal", err)
	}
	if !s.genMgr.IsActive("ming") {
		t.Error("the backend was stopped with eviction off")
	}
	if len(f.loadCalls()) != 0 {
		t.Error("the model was loaded")
	}
	// The reservation alone leaves no room, so a stays loaded.
	if u := f.unloadCalls(); len(u) != 0 {
		t.Errorf("unloaded %v for a load that could never fit", u)
	}
	if strings.Contains(err.Error(), "unload a model first") {
		t.Errorf("err = %v, suggests an unload with nothing left to unload", err)
	}
}

// A load that cannot fit beside a busy model, even with every idle model
// gone, unloads nothing: the idle one stays (the asmodean field test,
// step 5, unloaded the reranker for nothing). The load is still sent.
func TestLoadWithRoomBusyBlocksUnloadsNothing(t *testing.T) {
	s, f := roomServer(t)
	addRoomModel(t, s, "a", 12)
	addRoomModel(t, s, "b", 8)
	big := addRoomModel(t, s, "big", 18)
	loaded(t, s, f, "b", "a")
	end, err := s.llmUse.begin(context.Background(), "a")
	if err != nil {
		t.Fatal(err)
	}
	defer end()
	if err := loadWithRoom(t, s, big); err != nil {
		t.Fatal(err)
	}
	if u := f.unloadCalls(); len(u) != 0 {
		t.Errorf("unloaded %v for a load that cannot fit by plan", u)
	}
	if l := f.loadCalls(); !slices.Equal(l, []string{s.registry.RouterName("big")}) {
		t.Errorf("loads = %v, want big", l)
	}
}

// A backend kept running counts with eviction on too: it is never
// stopped, so a load that needs its room is refused with nothing
// unloaded.
func TestLoadWithRoomKeptRunningBackend(t *testing.T) {
	s, f := roomServer(t)
	withGen(t, s)
	s.cfg.GenEvict, s.cfg.Eviction = true, config.EvictionToolchest
	runningBackend(t, s, 10000)
	inst, _ := s.gen.Instance("ming")
	inst.KeepRunning = true
	if err := s.gen.UpdateInstance(inst); err != nil {
		t.Fatal(err)
	}
	addRoomModel(t, s, "a", 8)
	big := addRoomModel(t, s, "big", 18)
	loaded(t, s, f, "a")
	err := loadWithRoom(t, s, big)
	if err == nil || !strings.Contains(err.Error(), "kept running") {
		t.Fatalf("err = %v, want the refusal naming the kept-running backend", err)
	}
	if u := f.unloadCalls(); len(u) != 0 {
		t.Errorf("unloaded %v for a load that could never fit", u)
	}
	if !s.genMgr.IsActive("ming") {
		t.Error("a kept-running backend was stopped")
	}
}

// With eviction on and the backend stopped, no reservation counts, but a
// load is still checked: idle LLMs are unloaded before the load is sent.
func TestLoadWithRoomNoReservation(t *testing.T) {
	s, f := roomServer(t)
	withGen(t, s)
	s.cfg.GenEvict, s.cfg.Eviction = true, config.EvictionToolchest
	addInstance(t, s, generation.Instance{ID: "ming", Port: 8081, Enabled: true, ReservedMiB: 10000})
	addRoomModel(t, s, "a", 12)
	addRoomModel(t, s, "b", 8)
	c := addRoomModel(t, s, "c", 12)
	loaded(t, s, f, "a", "b")
	if s.genReservationsApply() {
		t.Fatal("a stopped backend's reservation counts with eviction on")
	}
	if err := loadWithRoom(t, s, c); err != nil {
		t.Fatal(err)
	}
	if u := f.unloadCalls(); len(u) == 0 {
		t.Fatal("loaded past the card's budget with nothing unloaded")
	}
}

// A request whose model is unloaded again each time between its load and
// its count loads it again through the room check, and in toolchest mode
// gives up after loadAttempts instead of letting the router load it with
// no check. Outside toolchest mode the last attempt goes on, as before.
func TestLoadAndCountRequestRetriesRunOut(t *testing.T) {
	for _, mode := range []memoryMode{memoryToolchest, memoryCount} {
		s, f := roomServer(t)
		s.memoryMode.Store(int32(mode))
		c := addRoomModel(t, s, "c", 1)
		name := s.registry.RouterName(c.ID)
		// A load finishes at the next list; once the request is counted,
		// the model is gone, as if another load took its room.
		f.reset(map[string]string{name: "unloaded"}, func(f *fakeRouter, _ int) {
			switch {
			case f.status[name] == "loading":
				defer func() { f.status[name] = "loaded" }()
			case f.status[name] == "loaded" && s.llmUse.busy(c.ID):
				f.status[name] = "unloaded"
			}
		})
		end, err := s.loadAndCountRequest(context.Background(), []byte(`{"model":"c"}`))
		if mode == memoryToolchest {
			if err == nil || !strings.Contains(err.Error(), "unloaded 4 times") {
				t.Errorf("toolchest mode: err = %v, want the retry refusal", err)
			}
			if s.llmUse.busy(c.ID) {
				t.Error("toolchest mode: the request is still counted")
			}
		} else if err != nil {
			t.Errorf("count mode: err = %v, want the request to go on", err)
		}
		end()
		if n := len(f.loadCalls()); n != loadAttempts {
			t.Errorf("mode %d: %d loads, want %d", mode, n, loadAttempts)
		}
	}
}
