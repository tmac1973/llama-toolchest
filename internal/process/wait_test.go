package process

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"
)

func (m *Manager) setState(state, reason string) {
	m.mu.Lock()
	m.status.State, m.status.Error = state, reason
	m.mu.Unlock()
}

// Activate used to stop waiting on a passing /health ping and then check
// IsRunning, which only flips on pollHealth's first 2s tick, so a fast
// start was reported as a failure. WaitRunning waits for the flip itself.
func TestWaitRunningWaitsForStateToFlip(t *testing.T) {
	m := NewManager()
	m.setState(StateStarting, "")
	go func() {
		time.Sleep(700 * time.Millisecond)
		m.setState(StateRunning, "")
	}()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := m.WaitRunning(ctx); err != nil {
		t.Fatalf("WaitRunning: %v, want nil once the router is running", err)
	}
}

func TestWaitRunningStopsOnFailure(t *testing.T) {
	m := NewManager()
	m.setState(StateFailed, "exit status 1")

	err := m.WaitRunning(context.Background())
	if err == nil || !strings.Contains(err.Error(), "exit status 1") {
		t.Fatalf("WaitRunning: %v, want the failure reason", err)
	}
}

func TestWaitRunningHonoursContext(t *testing.T) {
	m := NewManager()
	m.setState(StateStarting, "")

	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	if err := m.WaitRunning(ctx); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("WaitRunning: %v, want context.DeadlineExceeded", err)
	}
}
