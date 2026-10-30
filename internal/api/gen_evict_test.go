package api

import (
	"context"
	"errors"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/generation"
	"github.com/tmac1973/llama-toolchest/internal/models"
)

// An unload waits for the model's requests in progress, and a request
// that arrives during the unload waits for it to finish.
func TestLLMUseOrdersUnloadsAndRequests(t *testing.T) {
	var u llmUse
	ctx := context.Background()
	end, err := u.begin(ctx, "qwen")
	if err != nil {
		t.Fatal(err)
	}

	short, cancel := context.WithTimeout(ctx, 30*time.Millisecond)
	defer cancel()
	if _, err := u.evict(short, "qwen"); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("evict with a request in progress = %v, want it to wait", err)
	}
	if done, err := u.evict(ctx, "gemma"); err != nil {
		t.Fatalf("another model's unload waited: %v", err)
	} else {
		done()
	}

	evicted := make(chan func())
	go func() {
		done, _ := u.evict(ctx, "qwen")
		evicted <- done
	}()
	end()
	done := <-evicted

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

// evictServer is vramServer with one LLM too big to fit beside a running
// backend's 20,000 MiB reservation on its only GPU.
func evictServer(t *testing.T) (*Server, *models.Model) {
	t.Helper()
	s := vramServer(t)
	s.cfg.ModelsMax = 1
	runFakeInstance(t, s, "ming", "ming-image", 0, newFakeBackend(t, 0))
	inst, _ := s.gen.Instance("ming")
	inst.ReservedMiB = 20000
	if err := s.gen.UpdateInstance(inst); err != nil {
		t.Fatal(err)
	}
	big := &models.Model{ID: "big", ModelID: "org/big", Filename: "big.gguf", SizeBytes: 11 << 30}
	if err := s.registry.Add(big); err != nil {
		t.Fatal(err)
	}
	return s, big
}

// Without eviction, a load that does not fit beside a running backend is
// refused and the backend keeps running.
func TestMakeRoomForLLMOffRefuses(t *testing.T) {
	s, big := evictServer(t)
	release, err := s.makeRoomForLLM(context.Background(), big)
	release()
	var fitErr *genLLMFitError
	if !errors.As(err, &fitErr) {
		t.Fatalf("err = %v, want the reservation refusal", err)
	}
	if !s.genMgr.IsActive("ming") {
		t.Error("the backend was stopped with eviction off")
	}
}

// With eviction on, the load stops the backend in its way and holds the
// GPU's job queue until release, so a job cannot start it again first.
func TestMakeRoomForLLMStopsTheBackend(t *testing.T) {
	s, big := evictServer(t)
	s.cfg.GenEvict = true
	release, err := s.makeRoomForLLM(context.Background(), big)
	if err != nil {
		t.Fatal(err)
	}
	if s.genMgr.IsActive("ming") {
		t.Error("the backend is still running")
	}
	ticket, err := s.genQueue.Submit(0, "ming", generation.KindSD)
	if err != nil {
		t.Fatal(err)
	}
	defer ticket.Done()
	waited := make(chan error, 1)
	go func() { waited <- ticket.Wait(context.Background()) }()
	select {
	case <-waited:
		t.Fatal("a job got the GPU before the load released it")
	case <-time.After(30 * time.Millisecond):
	}
	release()
	if err := <-waited; err != nil {
		t.Fatal(err)
	}
	ticket.Done()

	// Stopped, with no job of its own on the GPU, the backend's
	// reservation no longer counts against a load.
	if err := s.checkLLMFitsReservations(big); err != nil {
		t.Errorf("with the backend stopped: %v", err)
	}
	s.cfg.GenEvict = false
	if err := s.checkLLMFitsReservations(big); err == nil {
		t.Error("with eviction off, a stopped backend's reservation still counts")
	}
}

// A stopped backend can be picked for a job, with eviction on or off (a
// running one first).
func TestPickStoppedBackends(t *testing.T) {
	s := vramServer(t)
	addInstance(t, s, generation.Instance{ID: "ming", Port: 8081, Enabled: true, ReservedMiB: 4096})
	for _, evict := range []bool{false, true} {
		s.cfg.GenEvict = evict
		inst, pe := s.pickInstance("ming", generation.KindSD)
		if pe != nil || inst.ID != "ming" {
			t.Fatalf("eviction %v: pick = %+v, %v; want the stopped backend", evict, inst, pe)
		}
	}

	runFakeInstance(t, s, "ming-2", "ming-image", 0, newFakeBackend(t, 0))
	if inst, _ := s.pickInstance("ming-image", generation.KindSD); inst.ID != "ming-2" {
		t.Errorf("picked %s, want the running backend first", inst.ID)
	}
}

// A job for a stopped backend first stops the other backends on its GPU
// when they are what is in the way.
func TestMakeRoomForGenStopsOtherBackends(t *testing.T) {
	s := vramServer(t)
	s.cfg.GenEvict = true
	runFakeInstance(t, s, "other", "ming-image", 0, newFakeBackend(t, 0))
	other, _ := s.gen.Instance("other")
	other.ReservedMiB = 20000
	s.gen.UpdateInstance(other)
	addInstance(t, s, generation.Instance{ID: "ming", Port: 8089, Enabled: true, ReservedMiB: 12000})
	inst, _ := s.gen.Instance("ming")

	if err := s.genFitCheck(inst, inst.ReservedMiB); err == nil {
		t.Fatal("the test backends fit side by side")
	}
	if err := s.makeRoomForGen(context.Background(), inst); err != nil {
		t.Fatal(err)
	}
	if s.genMgr.IsActive("other") {
		t.Error("the other backend is still running")
	}
}

// With eviction off, a job for a stopped backend starts it without
// stopping anything, and is refused when it does not fit.
func TestGenReadyWithEvictionOffStopsNothing(t *testing.T) {
	s := vramServer(t)
	runFakeInstance(t, s, "other", "ming-image", 0, newFakeBackend(t, 0))
	other, _ := s.gen.Instance("other")
	other.ReservedMiB = 20000
	s.gen.UpdateInstance(other)
	addInstance(t, s, generation.Instance{ID: "ming", Port: 8089, Enabled: true, ReservedMiB: 12000})
	inst, _ := s.gen.Instance("ming")

	err := s.genReady(context.Background(), inst)
	if err == nil || s.genFitCheck(inst, inst.ReservedMiB) == nil {
		t.Fatalf("genReady = %v; want the fit check's refusal", err)
	}
	if !s.genMgr.IsActive("other") {
		t.Error("the other backend was stopped with eviction off")
	}
}

// The wait for freed memory ends at the target, or once the reading
// stops falling (an idle backend held less than its reservation), and
// gives up when no reading comes.
func TestGPUFreed(t *testing.T) {
	feed := func(vals ...int) <-chan int {
		ch := make(chan int, len(vals))
		for _, v := range vals {
			ch <- v
		}
		return ch
	}
	if !gpuFreed(feed(20000, 9000), 9000, time.Second) {
		t.Error("reaching the target did not end the wait")
	}
	if !gpuFreed(feed(20000, 18000, 17900), 9000, time.Second) {
		t.Error("a reading that stopped falling did not end the wait")
	}
	if gpuFreed(feed(20000, 15000), 9000, 30*time.Millisecond) {
		t.Error("a reading still falling ended the wait")
	}
}

// With eviction on, the planner's view marks the reservations a loading
// model may take back: every enabled backend's, but not one kept running.
func TestHardwareMarksEvictableReservations(t *testing.T) {
	s := vramServer(t)
	addInstance(t, s, generation.Instance{ID: "ming", Port: 8081, Enabled: true, ReservedMiB: 8000})
	addInstance(t, s, generation.Instance{ID: "ace", Port: 8082, Enabled: true, ReservedMiB: 6000, KeepRunning: true})

	g := s.hardware().GPUs[0]
	if g.ReservedMiB != 14000 || g.EvictableMiB != 0 {
		t.Errorf("eviction off: reserved %d, evictable %d; want 14000 and 0", g.ReservedMiB, g.EvictableMiB)
	}
	s.cfg.GenEvict = true
	g = s.hardware().GPUs[0]
	if g.ReservedMiB != 14000 || g.EvictableMiB != 8000 {
		t.Errorf("eviction on: reserved %d, evictable %d; want 14000 and 8000 (ace is kept running)", g.ReservedMiB, g.EvictableMiB)
	}
	if room := models.RoomOf(s.idleHardware()); room.Whole == nil || room.Whole[0] <= room.Shared[0] {
		t.Errorf("room = %+v, want more room on the whole card", room)
	}
}

// A job starting a stopped backend has the GPU before the backend runs.
// An LLM load arriving then (a chat waiting for the model the job just
// unloaded) waits for the job instead of taking the room it needs.
func TestMakeRoomForLLMWaitsForAStartingBackend(t *testing.T) {
	s, big := evictServer(t)
	s.cfg.GenEvict = true
	if err := s.genMgr.Stop("ming"); err != nil {
		t.Fatal(err)
	}
	if err := s.checkLLMFitsReservations(big); err != nil {
		t.Fatalf("with the backend stopped and no job: %v", err)
	}

	ticket, err := s.genQueue.Submit(0, "ming", generation.KindSD)
	if err != nil {
		t.Fatal(err)
	}
	if err := ticket.Wait(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := s.checkLLMFitsReservations(big); err == nil {
		t.Fatal("a backend whose job has the GPU did not count against the load")
	}

	loaded := make(chan error, 1)
	go func() {
		release, err := s.makeRoomForLLM(context.Background(), big)
		release()
		loaded <- err
	}()
	select {
	case err := <-loaded:
		t.Fatalf("the load went ahead while the job had the GPU: %v", err)
	case <-time.After(50 * time.Millisecond):
	}
	// The job ends without its backend running (it failed to start):
	// there is room again, with nothing to stop.
	ticket.Done()
	select {
	case err := <-loaded:
		if err != nil {
			t.Errorf("load after the job ended: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("the load did not go ahead after the job ended")
	}
}

// While a load waits for a job on one of its GPUs, a job arriving for
// another of them waits behind the load instead of starting there and
// leaving the load short again (seen on compute2: the 27B's load waited
// for flux on GPU 1, ace's job took GPU 2, and the load failed).
func TestMakeRoomForLLMHoldsAllItsGPUs(t *testing.T) {
	s, _ := evictServer(t)
	s.cfg.GenEvict = true
	// Short beside ming on GPU 0 even when split over two cards.
	big := &models.Model{ID: "huge", ModelID: "org/huge", Filename: "huge.gguf", SizeBytes: 30 << 30}
	if err := s.registry.Add(big); err != nil {
		t.Fatal(err)
	}
	s.testHardware.GPUs = append(s.testHardware.GPUs, models.GPUSpec{Index: 1, Name: "R9700", VRAMTotalMiB: 32624, OtherUsedMiB: 1300})
	cfg, _ := s.registry.GetConfig(big.ID)
	c := *cfg
	c.GPUAssign, c.TensorSplit = "all", "1,1"
	if err := s.registry.SetConfig(big.ID, &c); err != nil {
		t.Fatal(err)
	}
	addInstance(t, s, generation.Instance{ID: "ace", Port: 8083, GPU: 1, Enabled: true, ReservedMiB: 30000, ModelSetID: "ming-image"})

	if hw, need, _ := s.llmReservationNeed(big); need[1] == 0 || !slices.Contains(s.reservationShortGPUs(hw, need), 0) {
		t.Fatalf("need = %v; want the model on both GPUs and short on GPU 0", need)
	}
	// ming's job has GPU 0; the load is short there and waits.
	ticket, _ := s.genQueue.Submit(0, "ming", generation.KindSD)
	if err := ticket.Wait(context.Background()); err != nil {
		t.Fatal(err)
	}
	loaded := make(chan error, 1)
	go func() {
		release, err := s.makeRoomForLLM(context.Background(), big)
		release()
		loaded <- err
	}()
	time.Sleep(100 * time.Millisecond) // the load is waiting in Hold by now

	// A job for GPU 1 arrives meanwhile: it must not start.
	other, _ := s.genQueue.Submit(1, "ace", generation.KindAce)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()
	if err := other.Wait(ctx); err == nil {
		t.Fatal("a job started on GPU 1 while the load was waiting for it")
	}
	other, _ = s.genQueue.Submit(1, "ace", generation.KindAce)

	ticket.Done()
	select {
	case err := <-loaded:
		if err != nil {
			t.Errorf("load after the job ended: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("the load did not go ahead")
	}
	wctx, wcancel := context.WithTimeout(context.Background(), time.Second)
	defer wcancel()
	if err := other.Wait(wctx); err != nil {
		t.Errorf("the waiting job did not start after the load: %v", err)
	}
	other.Done()
}

// The Start button makes room the way a job does when the backend does
// not fit: with eviction on it stops what is in the way and goes on to
// start (here it stops launching the test's missing binary, past every room check), and
// gives the GPU's turn back. With eviction off it is refused and stops
// nothing.
func TestStartGenFromUIMakesRoom(t *testing.T) {
	s := vramServer(t)
	runFakeInstance(t, s, "other", "ming-image", 0, newFakeBackend(t, 0))
	other, _ := s.gen.Instance("other")
	other.ReservedMiB = 20000
	s.gen.UpdateInstance(other)
	addInstance(t, s, generation.Instance{ID: "ming", Port: 8089, Enabled: true, ReservedMiB: 12000})
	inst, _ := s.gen.Instance("ming")

	err := s.startGenFromUI(context.Background(), inst)
	if err == nil || !strings.Contains(err.Error(), "not enough VRAM") {
		t.Fatalf("eviction off: err = %v, want the fit refusal", err)
	}
	if !s.genMgr.IsActive("other") {
		t.Fatal("eviction off: the other backend was stopped")
	}

	s.cfg.GenEvict = true
	err = s.startGenFromUI(context.Background(), inst)
	t.Logf("eviction on: %v", err)
	if err == nil || strings.Contains(err.Error(), "not enough VRAM") {
		t.Errorf("eviction on: err = %v, want it past the room check (only the test's missing binary left)", err)
	}
	if s.genMgr.IsActive("other") {
		t.Error("eviction on: the backend in the way is still running")
	}
	if snap := s.genQueue.Snapshot(0); snap.Running != nil || len(snap.Waiting) != 0 {
		t.Errorf("the GPU's turn was not given back: %+v", snap)
	}
}

// With eviction on, a measurement takes the GPU the way a job does: it
// stops the other backends there instead of being refused, and holds the
// GPU's job queue until release. A kept-running backend still refuses
// it. With eviction off nothing is stopped.
func TestMakeRoomForMeasure(t *testing.T) {
	s := vramServer(t)
	runFakeInstance(t, s, "other", "ming-image", 0, newFakeBackend(t, 0))
	addInstance(t, s, generation.Instance{ID: "m", Port: 8089, Enabled: true})
	inst, _ := s.gen.Instance("m")

	if err := s.canMeasure(inst); err == nil || !strings.Contains(err.Error(), "stop other first") {
		t.Errorf("eviction off: canMeasure = %v, want stop other first", err)
	}
	release, err := s.makeRoomForMeasure(context.Background(), inst)
	release()
	if err != nil || !s.genMgr.IsActive("other") {
		t.Fatalf("eviction off: err %v, other running %v; want nothing stopped", err, s.genMgr.IsActive("other"))
	}

	s.cfg.GenEvict = true
	other, _ := s.gen.Instance("other")
	other.KeepRunning = true
	s.gen.UpdateInstance(other)
	if err := s.canMeasure(inst); err == nil {
		t.Error("eviction on: measuring beside a kept-running backend was allowed")
	}
	other.KeepRunning = false
	s.gen.UpdateInstance(other)
	if err := s.canMeasure(inst); err != nil {
		t.Errorf("eviction on: canMeasure = %v", err)
	}
	release, err = s.makeRoomForMeasure(context.Background(), inst)
	if err != nil {
		t.Fatal(err)
	}
	if s.genMgr.IsActive("other") {
		t.Error("eviction on: the other backend is still running")
	}
	ticket, _ := s.genQueue.Submit(0, "other", generation.KindSD)
	defer ticket.Done()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()
	if err := ticket.Wait(ctx); err == nil {
		t.Fatal("a job got the GPU during the measurement")
	}
	release()
	if snap := s.genQueue.Snapshot(0); snap.Running != nil || len(snap.Waiting) != 0 {
		t.Errorf("the GPU was not given back: %+v", snap)
	}
}

// With eviction on, an LLM load waits for a measurement on its GPU even
// when it fits beside what runs there now: the backend may grow into the
// room.
func TestMakeRoomForLLMWaitsForAMeasurement(t *testing.T) {
	s, big := evictServer(t)
	s.cfg.GenEvict = true
	if err := s.genMgr.Stop("ming"); err != nil {
		t.Fatal(err)
	}
	if err := s.checkLLMFitsReservations(big); err != nil {
		t.Fatalf("with the backend stopped: %v", err)
	}
	s.genMgr.SetMeasuring("ming", true)
	inst, _ := s.gen.Instance("ming")
	hold, err := s.makeRoomForMeasure(context.Background(), inst)
	if err != nil {
		t.Fatal(err)
	}
	loaded := make(chan error, 1)
	go func() {
		release, err := s.makeRoomForLLM(context.Background(), big)
		release()
		loaded <- err
	}()
	select {
	case err := <-loaded:
		t.Fatalf("the load went ahead during the measurement: %v", err)
	case <-time.After(50 * time.Millisecond):
	}
	s.genMgr.SetMeasuring("ming", false)
	hold()
	select {
	case err := <-loaded:
		if err != nil {
			t.Errorf("load after the measurement: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("the load did not go ahead after the measurement")
	}
}
