package api

import (
	"context"
	"errors"
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

	// Stopped, the backend's reservation no longer counts against a load.
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
