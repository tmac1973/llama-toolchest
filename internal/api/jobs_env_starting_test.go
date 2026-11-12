package api

import (
	"context"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/process"
)

// startStartingRouter starts the fake router and checks it is still starting.
func startStartingRouter(t *testing.T, s *Server, proc *process.Manager) {
	t.Helper()
	if err := proc.Start(process.RouterConfig{
		BinaryPath: filepath.Join(s.builderDataDirForTest(t), "llama-server"),
		Host:       "127.0.0.1", Port: s.cfg.LlamaPort, ModelsMax: 1,
	}); err != nil {
		t.Fatalf("start: %v", err)
	}
	if st := proc.GetStatus().State; st != process.StateStarting {
		t.Fatalf("state = %q, want %q", st, process.StateStarting)
	}
}

// Seen on compute2: an autotune run started right after a container
// restart failed all 12 cells at once with "router already starting".
// A job that finds the router still starting waits for it, then takes
// it over.
func TestJobWaitsForAStartingRouter(t *testing.T) {
	var ready atomic.Bool
	s, proc := fakeRouterServer(t, ready.Load)
	startStartingRouter(t, s, proc)
	env := newJobEnv(s)

	done := make(chan error, 1)
	go func() { done <- env.restartRouter(context.Background(), "a benchmark cell") }()

	select {
	case err := <-done:
		t.Fatalf("restartRouter returned while the router was still starting: %v", err)
	case <-time.After(500 * time.Millisecond):
	}
	ready.Store(true)

	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("restartRouter: %v", err)
		}
	case <-time.After(30 * time.Second):
		t.Fatal("restartRouter did not return after the router came up")
	}
	if !proc.IsRunning() {
		t.Error("the router is not running after the job took it over")
	}
}

// A router that never comes up is not waited for forever: the job says
// what to do instead.
func TestJobGivesUpOnARouterThatStaysStarting(t *testing.T) {
	old := routerStartingWait
	routerStartingWait = 100 * time.Millisecond
	t.Cleanup(func() { routerStartingWait = old })

	s, proc := fakeRouterServer(t, func() bool { return false })
	startStartingRouter(t, s, proc)
	env := newJobEnv(s)

	err := env.restartRouter(context.Background(), "a benchmark cell")
	if err == nil || !strings.Contains(err.Error(), "still starting") {
		t.Fatalf("err = %v, want one saying the server is still starting", err)
	}
	if strings.Contains(err.Error(), "already starting") {
		t.Errorf("err = %v: the job tried to start a second router", err)
	}
}
