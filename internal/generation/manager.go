package generation

import (
	"context"
	"sync"

	"github.com/tmac1973/llama-toolchest/internal/process"
)

// instanceLogHistory is how many output lines each instance keeps for new
// log viewers.
const instanceLogHistory = 500

// Manager runs generation backend instances, one supervised process each.
// It knows nothing about the store or builds; callers hand it a ready
// launch config (see LaunchSpec). Stop, Status, the logs and the rest come
// from process.Instances.
type Manager struct {
	*process.Instances

	mu sync.Mutex
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
	return &Manager{Instances: process.NewInstances(instanceLogHistory)}
}

// Start runs an instance with the given launch config.
func (m *Manager) Start(id string, kind BackendKind, cfg process.SupervisorConfig) error {
	label := "sd-server"
	if kind == KindAce {
		label = "ace-server"
	}
	return m.Instances.Start(id, label+" "+id, cfg)
}

// WaitRunning waits until the instance passes its health check, stops, or
// ctx ends. When the backend failed, the error ends with the last error
// line of its log, so the user sees why without opening the log.
func (m *Manager) WaitRunning(ctx context.Context, id string) error {
	err := m.Instances.WaitRunning(ctx, id)
	if err != nil && m.Status(id).State == process.StateFailed {
		return withLastLogLine(err, m.RecentLogs(id))
	}
	return err
}
