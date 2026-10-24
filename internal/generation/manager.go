package generation

import (
	"context"
	"sort"
	"sync"

	"github.com/tmac1973/llama-toolchest/internal/process"
)

// instanceLogHistory is how many output lines each instance keeps for new
// log viewers.
const instanceLogHistory = 500

// Manager runs generation backend instances, one supervised process each.
// It knows nothing about the store or builds; callers hand it a ready
// launch config (see LaunchSpec).
type Manager struct {
	mu   sync.Mutex
	sups map[string]*process.Supervisor
	// measuring holds the instances in a measured test run: the API sends
	// them no requests, and nothing else may start on their GPU.
	measuring map[string]bool
}

// SetMeasuring marks an instance as in, or out of, a measured test run.
func (m *Manager) SetMeasuring(id string, on bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.measuring == nil {
		m.measuring = map[string]bool{}
	}
	if on {
		m.measuring[id] = true
	} else {
		delete(m.measuring, id)
	}
}

// IsMeasuring reports whether an instance is in a measured test run.
func (m *Manager) IsMeasuring(id string) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.measuring[id]
}

// NewManager returns a manager with nothing running.
func NewManager() *Manager {
	return &Manager{sups: map[string]*process.Supervisor{}}
}

// supervisor returns the instance's supervisor, creating it on first use.
// It is kept after the process stops, so its last log lines stay readable.
func (m *Manager) supervisor(id string, kind BackendKind) *process.Supervisor {
	m.mu.Lock()
	defer m.mu.Unlock()
	s, ok := m.sups[id]
	if !ok {
		label := "sd-server"
		if kind == KindAce {
			label = "ace-server"
		}
		s = process.NewSupervisor(label+" "+id, instanceLogHistory)
		m.sups[id] = s
	}
	return s
}

func (m *Manager) existing(id string) *process.Supervisor {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.sups[id]
}

// Start runs an instance with the given launch config.
func (m *Manager) Start(id string, kind BackendKind, cfg process.SupervisorConfig) error {
	return m.supervisor(id, kind).Start(cfg)
}

// Stop stops an instance. Stopping one that is not running is not an
// error.
func (m *Manager) Stop(id string) error {
	s := m.existing(id)
	if s == nil || !m.active(s) {
		return nil
	}
	return s.Stop()
}

// Status returns an instance's process status; stopped when it never ran.
func (m *Manager) Status(id string) process.Status {
	if s := m.existing(id); s != nil {
		return s.GetStatus()
	}
	return process.Status{State: process.StateStopped}
}

// IsRunning reports whether the instance has passed its health check.
func (m *Manager) IsRunning(id string) bool {
	s := m.existing(id)
	return s != nil && s.IsRunning()
}

// IsActive reports whether the instance is starting or running.
func (m *Manager) IsActive(id string) bool {
	s := m.existing(id)
	return s != nil && m.active(s)
}

func (m *Manager) active(s *process.Supervisor) bool {
	st := s.GetStatus().State
	return st == process.StateStarting || st == process.StateRunning
}

// PID returns the instance's process ID, or 0 when it is not running.
func (m *Manager) PID(id string) int {
	if s := m.existing(id); s != nil {
		return s.PID()
	}
	return 0
}

// WaitRunning waits until the instance passes its health check, stops, or
// ctx ends.
func (m *Manager) WaitRunning(ctx context.Context, id string) error {
	if s := m.existing(id); s != nil {
		return s.WaitRunning(ctx)
	}
	return nil
}

// Subscribe returns the instance's log lines, starting with its recent
// history; nil when the instance never ran.
func (m *Manager) Subscribe(id string) chan string {
	if s := m.existing(id); s != nil {
		return s.Subscribe()
	}
	return nil
}

// Unsubscribe removes a log subscriber.
func (m *Manager) Unsubscribe(id string, ch chan string) {
	if s := m.existing(id); s != nil {
		s.Unsubscribe(ch)
	}
}

// Forget drops a stopped instance's supervisor, after the instance is
// deleted.
func (m *Manager) Forget(id string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if s, ok := m.sups[id]; ok && !m.active(s) {
		delete(m.sups, id)
	}
}

// StopAll stops every running instance, for shutdown or when the
// generation backends are switched off.
func (m *Manager) StopAll() {
	m.mu.Lock()
	ids := make([]string, 0, len(m.sups))
	for id := range m.sups {
		ids = append(ids, id)
	}
	m.mu.Unlock()
	sort.Strings(ids)
	var wg sync.WaitGroup
	for _, id := range ids {
		wg.Add(1)
		go func() {
			defer wg.Done()
			m.Stop(id)
		}()
	}
	wg.Wait()
}
