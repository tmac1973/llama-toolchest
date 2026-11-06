package process

import (
	"context"
	"testing"
)

// An instance that never ran reads as stopped, with no logs and nothing
// to stop, wait for or forget.
func TestInstancesNeverRun(t *testing.T) {
	m := NewInstances(10)
	if st := m.Status("x"); st.State != StateStopped {
		t.Errorf("Status = %q, want stopped", st.State)
	}
	if m.IsActive("x") || m.IsRunning("x") || m.PID("x") != 0 {
		t.Error("a never-run instance reads as active, running or with a PID")
	}
	if m.Subscribe("x") != nil || m.RecentLogs("x") != nil {
		t.Error("a never-run instance has logs")
	}
	if err := m.Stop("x"); err != nil {
		t.Errorf("Stop: %v", err)
	}
	if err := m.WaitRunning(context.Background(), "x"); err != nil {
		t.Errorf("WaitRunning: %v", err)
	}
	m.Forget("x")
	m.StopAll()
}
