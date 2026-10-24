package process

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"
	"time"
)

// fakeProgram writes a shell script to a temp dir and returns its path.
func fakeProgram(t *testing.T, body string) string {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("shell-script fakes need a unix shell")
	}
	path := filepath.Join(t.TempDir(), "fake")
	if err := os.WriteFile(path, []byte("#!/bin/sh\n"+body+"\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	return path
}

// healthServer answers /health with 200.
func healthServer(t *testing.T) string {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	t.Cleanup(srv.Close)
	return srv.URL + "/health"
}

func testConfig(t *testing.T, body string) SupervisorConfig {
	return SupervisorConfig{
		Name:           "fake",
		BinaryPath:     fakeProgram(t, body),
		Env:            os.Environ(),
		HealthURL:      healthServer(t),
		HealthInterval: 20 * time.Millisecond,
		HealthTimeout:  5 * time.Second,
		StopTimeout:    5 * time.Second,
	}
}

// waitFor polls cond until it is true or the deadline passes.
func waitFor(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s", what)
}

func TestSupervisorStartRunStop(t *testing.T) {
	s := NewSupervisor("Fake", 50)
	logs := s.Subscribe()
	defer s.Unsubscribe(logs)

	if err := s.Start(testConfig(t, "echo hello from fake\nexec sleep 30")); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := s.WaitRunning(ctx); err != nil {
		t.Fatalf("WaitRunning: %v", err)
	}
	if s.PID() == 0 {
		t.Error("PID() = 0 while running")
	}

	waitFor(t, "the fake's output line", func() bool {
		select {
		case line := <-logs:
			return line == "hello from fake"
		default:
			return false
		}
	})

	if err := s.Stop(); err != nil {
		t.Fatalf("Stop: %v", err)
	}
	if st := s.GetStatus(); st.State != StateStopped {
		t.Errorf("state after Stop = %q, want %q", st.State, StateStopped)
	}
	if s.PID() != 0 {
		t.Error("PID() != 0 after Stop")
	}
	if !slices.Contains(s.RecentLogs(), "==> Fake stopped") {
		t.Errorf("logs %q lack the stop line", s.RecentLogs())
	}
}

// A program that ignores SIGTERM is killed once StopTimeout passes.
func TestSupervisorKillsAfterStopTimeout(t *testing.T) {
	s := NewSupervisor("Fake", 50)
	cfg := testConfig(t, "trap '' TERM\necho ready\nwhile true; do sleep 0.1; done")
	cfg.StopTimeout = 300 * time.Millisecond
	if err := s.Start(cfg); err != nil {
		t.Fatal(err)
	}
	waitFor(t, "the fake to start", s.IsRunning)

	start := time.Now()
	if err := s.Stop(); err != nil {
		t.Fatalf("Stop: %v", err)
	}
	if took := time.Since(start); took < cfg.StopTimeout {
		t.Errorf("Stop returned after %v, before StopTimeout %v: SIGTERM was not ignored", took, cfg.StopTimeout)
	}
	if st := s.GetStatus(); st.State != StateStopped {
		t.Errorf("state after Stop = %q, want %q", st.State, StateStopped)
	}
}

// A program that exits before it is ready leaves the state "failed", with
// its last output still readable.
func TestSupervisorEarlyExitFails(t *testing.T) {
	s := NewSupervisor("Fake", 50)
	cfg := testConfig(t, "echo cannot open model file\nexit 3")
	cfg.HealthURL = "http://127.0.0.1:1/health" // never answers
	if err := s.Start(cfg); err != nil {
		t.Fatal(err)
	}

	err := s.WaitRunning(context.Background())
	if err == nil || !strings.Contains(err.Error(), "fake failed to start") {
		t.Fatalf("WaitRunning: %v, want a failure", err)
	}
	if st := s.GetStatus(); st.State != StateFailed || !strings.Contains(st.Error, "exit status 3") {
		t.Errorf("status = %+v, want failed with exit status 3", st)
	}
	waitFor(t, "the fake's output in RecentLogs", func() bool {
		return slices.Contains(s.RecentLogs(), "cannot open model file")
	})
}

func TestSupervisorStartTwiceRefused(t *testing.T) {
	s := NewSupervisor("Fake", 50)
	cfg := testConfig(t, "exec sleep 30")
	if err := s.Start(cfg); err != nil {
		t.Fatal(err)
	}
	defer s.Stop()

	if err := s.Start(cfg); err == nil || !strings.Contains(err.Error(), "fake already") {
		t.Fatalf("second Start: %v, want an 'already' error", err)
	}
}

func TestSupervisorStopWhenNotRunning(t *testing.T) {
	err := NewManager().Stop()
	if err == nil || err.Error() != "router not running" {
		t.Fatalf("Stop on a fresh Manager: %v, want %q", err, "router not running")
	}
}

func TestBuildEnvAddsLibraryPathAndCUDAOrder(t *testing.T) {
	env := BuildEnv("/opt/builds/b1", []string{"HIP_VISIBLE_DEVICES=1"})
	if n, last := countCUDAOrder(env); n != 1 || last == "" {
		t.Errorf("CUDA_DEVICE_ORDER entries = %d (%q), want exactly one", n, last)
	}
	if !slices.Contains(env, "HIP_VISIBLE_DEVICES=1") && os.Getenv("HIP_VISIBLE_DEVICES") == "" {
		t.Errorf("extra HIP_VISIBLE_DEVICES=1 missing from %v", env)
	}
	found := false
	for _, kv := range env {
		if strings.Contains(kv, "/opt/builds/b1") {
			found = true
		}
	}
	if !found {
		t.Errorf("binary dir missing from the library path in %v", env)
	}
}
