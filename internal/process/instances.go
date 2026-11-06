package process

import (
	"context"
	"sort"
	"sync"
)

// Instances runs a set of named instances, one Supervisor each: the
// generation backends and the fork llama-servers. It knows nothing about
// their stores or builds; callers hand it a ready launch config.
type Instances struct {
	history int // log lines each supervisor keeps for new viewers

	mu   sync.Mutex
	sups map[string]*Supervisor
}

// NewInstances returns a set with nothing running, whose supervisors keep
// history log lines.
func NewInstances(history int) *Instances {
	return &Instances{history: history, sups: map[string]*Supervisor{}}
}

// supervisor returns the instance's supervisor, creating it on first use
// with label as its log name. It is kept after the process stops, so its
// last log lines stay readable.
func (m *Instances) supervisor(id, label string) *Supervisor {
	m.mu.Lock()
	defer m.mu.Unlock()
	s, ok := m.sups[id]
	if !ok {
		s = NewSupervisor(label, m.history)
		m.sups[id] = s
	}
	return s
}

func (m *Instances) existing(id string) *Supervisor {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.sups[id]
}

// Start runs an instance with the given launch config. label names the
// process in its log lines; it is set the first time the instance starts.
func (m *Instances) Start(id, label string, cfg SupervisorConfig) error {
	return m.supervisor(id, label).Start(cfg)
}

// Stop stops an instance. Stopping one that is not running is not an
// error.
func (m *Instances) Stop(id string) error {
	s := m.existing(id)
	if s == nil || !active(s) {
		return nil
	}
	return s.Stop()
}

// Status returns an instance's process status; stopped when it never ran.
func (m *Instances) Status(id string) Status {
	if s := m.existing(id); s != nil {
		return s.GetStatus()
	}
	return Status{State: StateStopped}
}

// IsRunning reports whether the instance has passed its health check.
func (m *Instances) IsRunning(id string) bool {
	s := m.existing(id)
	return s != nil && s.IsRunning()
}

// IsActive reports whether the instance is starting or running.
func (m *Instances) IsActive(id string) bool {
	s := m.existing(id)
	return s != nil && active(s)
}

func active(s *Supervisor) bool {
	st := s.GetStatus().State
	return st == StateStarting || st == StateRunning
}

// PID returns the instance's process ID, or 0 when it is not running.
func (m *Instances) PID(id string) int {
	if s := m.existing(id); s != nil {
		return s.PID()
	}
	return 0
}

// WaitRunning waits until the instance passes its health check, stops, or
// ctx ends.
func (m *Instances) WaitRunning(ctx context.Context, id string) error {
	if s := m.existing(id); s != nil {
		return s.WaitRunning(ctx)
	}
	return nil
}

// Subscribe returns the instance's log lines, starting with its recent
// history; nil when the instance never ran.
func (m *Instances) Subscribe(id string) chan string {
	if s := m.existing(id); s != nil {
		return s.Subscribe()
	}
	return nil
}

// Unsubscribe removes a log subscriber.
func (m *Instances) Unsubscribe(id string, ch chan string) {
	if s := m.existing(id); s != nil {
		s.Unsubscribe(ch)
	}
}

// RecentLogs returns the instance's buffered log lines, oldest first:
// what it printed since it last started. Nil when it never ran.
func (m *Instances) RecentLogs(id string) []string {
	if s := m.existing(id); s != nil {
		return s.RecentLogs()
	}
	return nil
}

// Forget drops a stopped instance's supervisor, after the instance is
// deleted.
func (m *Instances) Forget(id string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if s, ok := m.sups[id]; ok && !active(s) {
		delete(m.sups, id)
	}
}

// StopAll stops every running instance, for shutdown or when the feature
// is switched off.
func (m *Instances) StopAll() {
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
