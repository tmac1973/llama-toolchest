package forks

import (
	"time"

	"github.com/tmac1973/llama-toolchest/internal/process"
)

// instanceLogHistory is how many output lines each instance keeps for new
// log viewers.
const instanceLogHistory = 500

// StartTimeout is how long a fork instance may take to answer its health
// check. Avalon reads an 89 GiB model and fills its expert cache before
// /health answers 200; with less RAM than the model that read comes from
// disk.
const StartTimeout = 20 * time.Minute

// Manager runs fork instances, one supervised llama-server each. It knows
// nothing about the store or builds; callers hand it a ready launch
// config. Stop, Status, the logs and the rest come from
// process.Instances.
type Manager struct {
	*process.Instances
}

// NewManager returns a manager with nothing running.
func NewManager() *Manager {
	return &Manager{process.NewInstances(instanceLogHistory)}
}

// Start runs an instance with the given launch config.
func (m *Manager) Start(id string, cfg process.SupervisorConfig) error {
	return m.Instances.Start(id, "fork llama-server "+id, cfg)
}
