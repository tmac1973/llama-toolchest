package api

import (
	"errors"
	"github.com/tmac1973/llama-toolchest/internal/config"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/models"
)

// A benchmark cell that asks to make room (Autoconfigure's test load)
// stops the backend in its way rather than being refused for it.
func TestMakeRoomForConfigStopsTheBackend(t *testing.T) {
	s, big := evictServer(t)
	s.cfg.GenEvict, s.cfg.Eviction = true, config.EvictionToolchest
	cfg := &models.ModelConfig{}
	var fitErr *genLLMFitError
	if err := s.checkConfigFitsReservations(big, cfg); !errors.As(err, &fitErr) {
		t.Fatalf("before: err = %v, want the reservation refusal", err)
	}
	if err := s.makeRoomForConfig(big, cfg, []int{0}, "test"); err != nil {
		t.Fatal(err)
	}
	if s.genMgr.IsActive("ming") {
		t.Error("the backend in the way is still running")
	}
}

// A backend kept running is never stopped, so the cell is still refused.
func TestMakeRoomForConfigKeepsALockedBackend(t *testing.T) {
	s, big := evictServer(t)
	s.cfg.GenEvict, s.cfg.Eviction = true, config.EvictionToolchest
	inst, _ := s.gen.Instance("ming")
	inst.KeepRunning = true
	if err := s.gen.UpdateInstance(inst); err != nil {
		t.Fatal(err)
	}
	var fitErr *genLLMFitError
	if err := s.makeRoomForConfig(big, &models.ModelConfig{}, []int{0}, "test"); !errors.As(err, &fitErr) {
		t.Fatalf("err = %v, want the reservation refusal", err)
	}
	if !s.genMgr.IsActive("ming") {
		t.Error("a backend kept running was stopped")
	}
}
