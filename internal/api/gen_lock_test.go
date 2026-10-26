package api

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/go-chi/chi/v5"
	"github.com/tmac1973/llama-toolchest/internal/generation"
	"github.com/tmac1973/llama-toolchest/internal/models"
)

// setReservation sets an instance's reservation and keep-running mark.
func setReservation(t *testing.T, s *Server, id string, mib int, keep bool) {
	t.Helper()
	inst, err := s.gen.Instance(id)
	if err != nil {
		t.Fatal(err)
	}
	inst.ReservedMiB, inst.KeepRunning = mib, keep
	if err := s.gen.UpdateInstance(inst); err != nil {
		t.Fatal(err)
	}
}

// A locked instance's reservation counts with eviction on even while it
// is stopped, as every reservation does with eviction off.
func TestLockedReservationCountsWhileStopped(t *testing.T) {
	s := vramServer(t)
	s.cfg.ModelsMax = 1
	s.cfg.GenEvict = true
	big := &models.Model{ID: "big", ModelID: "org/big", Filename: "big.gguf", SizeBytes: 11 << 30}
	if err := s.registry.Add(big); err != nil {
		t.Fatal(err)
	}
	addInstance(t, s, generation.Instance{ID: "ming", Port: 8081, Enabled: true, ReservedMiB: 20000})
	if err := s.checkLLMFitsReservations(big); err != nil {
		t.Fatalf("unlocked and stopped, the reservation still counts: %v", err)
	}
	setReservation(t, s, "ming", 20000, true)
	var fitErr *genLLMFitError
	if err := s.checkLLMFitsReservations(big); !errors.As(err, &fitErr) {
		t.Errorf("locked and stopped: err = %v, want the reservation refusal", err)
	}
}

// With eviction on, a load refuses rather than stop a locked backend.
func TestMakeRoomForLLMKeepsLockedBackend(t *testing.T) {
	s, big := evictServer(t)
	s.cfg.GenEvict = true
	setReservation(t, s, "ming", 20000, true)
	release, err := s.makeRoomForLLM(context.Background(), big)
	release()
	var fitErr *genLLMFitError
	if !errors.As(err, &fitErr) {
		t.Fatalf("err = %v, want the reservation refusal", err)
	}
	if !s.genMgr.IsActive("ming") {
		t.Error("the locked backend was stopped")
	}
}

// Making room for a backend stops only as many others as it needs,
// largest first, and never a locked one.
func TestMakeRoomForGenStopsOnlyWhatItNeeds(t *testing.T) {
	s := vramServer(t)
	s.cfg.GenEvict = true
	for id, mib := range map[string]int{"small": 6000, "large": 9000, "locked": 10000} {
		runFakeInstance(t, s, id, "ming-image", 0, newFakeBackend(t, 0))
		setReservation(t, s, id, mib, id == "locked")
	}
	addInstance(t, s, generation.Instance{ID: "new", Port: 8089, Enabled: true, ReservedMiB: 12000})
	inst, _ := s.gen.Instance("new")
	if err := s.genFitCheck(inst, inst.ReservedMiB); err == nil {
		t.Fatal("the test backends fit side by side")
	}

	if err := s.makeRoomForGen(context.Background(), inst); err != nil {
		t.Fatal(err)
	}
	for id, wantRunning := range map[string]bool{"large": false, "small": true, "locked": true} {
		if got := s.genMgr.IsActive(id); got != wantRunning {
			t.Errorf("%s running = %v, want %v", id, got, wantRunning)
		}
	}
}

// Locking a stopped instance is refused when its reservation does not
// fit now, since it would then count; unlocking always works.
func TestLockRefusedWhenItDoesNotFit(t *testing.T) {
	s := vramServer(t)
	s.cfg.GenEvict = true
	runFakeInstance(t, s, "running", "ming-image", 0, newFakeBackend(t, 0))
	setReservation(t, s, "running", 20000, false)
	addInstance(t, s, generation.Instance{ID: "ming", Port: 8089, Enabled: true, ReservedMiB: 12000})

	act := func(action string) int {
		rc := chi.NewRouteContext()
		rc.URLParams.Add("id", "ming")
		rc.URLParams.Add("action", action)
		r := httptest.NewRequest("POST", "/", nil)
		r = r.WithContext(context.WithValue(r.Context(), chi.RouteCtxKey, rc))
		rec := httptest.NewRecorder()
		s.handleGenInstanceAction(rec, r)
		return rec.Code
	}
	if code := act("keep-running"); code != http.StatusBadRequest {
		t.Errorf("lock that does not fit = %d, want 400", code)
	}
	if inst, _ := s.gen.Instance("ming"); inst.KeepRunning {
		t.Error("locked although it did not fit")
	}

	setReservation(t, s, "running", 4096, false)
	if code := act("keep-running"); code != http.StatusOK {
		t.Errorf("lock that fits = %d, want 200", code)
	}
	if inst, _ := s.gen.Instance("ming"); !inst.KeepRunning {
		t.Error("not locked")
	}
	if code := act("allow-stop"); code != http.StatusOK {
		t.Errorf("unlock = %d, want 200", code)
	}
	if inst, _ := s.gen.Instance("ming"); inst.KeepRunning {
		t.Error("still locked")
	}
}
