package process

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"path/filepath"
	"strings"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/routerclient"
)

// ModelStatus represents the state of a model in the router.
type ModelStatus = routerclient.ModelStatus

// Process states.
const (
	StateStopped  = "stopped"
	StateStarting = "starting"
	StateRunning  = "running"
	StateFailed   = "failed"
)

// Status represents the state of the router process itself.
type Status struct {
	State     string    `json:"state"` // StateStopped, StateStarting, StateRunning, StateFailed
	PID       int       `json:"pid,omitempty"`
	Uptime    string    `json:"uptime,omitempty"`
	StartedAt time.Time `json:"started_at,omitempty"`
	Error     string    `json:"error,omitempty"`
	HealthOK  bool      `json:"health_ok"`
}

// RouterConfig defines how to start the llama-server router.
type RouterConfig struct {
	BinaryPath string
	ModelsDir  string
	PresetPath string
	ModelsMax  int
	Host       string
	Port       int
	// ExtraEnv holds KEY=VALUE pairs appended to the child environment,
	// for curated performance-related variables set in Settings.
	ExtraEnv []string
}

const logHistorySize = 500

// Manager manages a single llama-server router process. The process
// lifecycle (start, stop, health, logs) comes from the embedded
// Supervisor; Manager adds the router's arguments and model control.
type Manager struct {
	*Supervisor

	// routerURL is guarded by Supervisor.mu. It is kept after the router
	// stops, so model calls fail at the HTTP level rather than here.
	routerURL string
}

func NewManager() *Manager {
	return &Manager{Supervisor: NewSupervisor("Router", logHistorySize)}
}

// Start launches the llama-server in router mode.
func (m *Manager) Start(cfg RouterConfig) error {
	if cfg.Host == "" {
		cfg.Host = "0.0.0.0"
	}
	if cfg.Port == 0 {
		cfg.Port = 8080
	}

	// Build command args — no --model flag means router mode
	args := []string{
		"--host", cfg.Host,
		"--port", fmt.Sprintf("%d", cfg.Port),
	}
	if cfg.ModelsDir != "" {
		args = append(args, "--models-dir", cfg.ModelsDir)
	}
	if cfg.PresetPath != "" {
		args = append(args, "--models-preset", cfg.PresetPath)
	}
	if cfg.ModelsMax >= 0 {
		args = append(args, "--models-max", fmt.Sprintf("%d", cfg.ModelsMax))
	}

	url := fmt.Sprintf("http://localhost:%d", cfg.Port)
	err := m.Supervisor.Start(SupervisorConfig{
		Name:       "llama-server router",
		BinaryPath: cfg.BinaryPath,
		Args:       args,
		// Tell the child process where to find co-located shared libraries.
		Env:       BuildEnv(filepath.Dir(cfg.BinaryPath), cfg.ExtraEnv),
		HealthURL: url + "/health",
	})
	if err != nil {
		return err
	}

	m.mu.Lock()
	m.routerURL = url
	m.mu.Unlock()
	return nil
}

// LoadModel tells the router to load a model by name. A model that is
// already loaded counts as success.
func (m *Manager) LoadModel(name string) error {
	url, err := m.routerBase()
	if err != nil {
		return err
	}
	if err := routerclient.Load(context.Background(), url, name); err != nil {
		return fmt.Errorf("load model %q: %w", name, err)
	}
	return nil
}

// UnloadModel tells the router to unload a model by name.
func (m *Manager) UnloadModel(name string) error {
	url, err := m.routerBase()
	if err != nil {
		return err
	}
	if err := routerclient.Unload(context.Background(), url, name); err != nil {
		return fmt.Errorf("unload model %q: %w", name, err)
	}
	return nil
}

// ListModels queries the router for all known models and their status.
func (m *Manager) ListModels() ([]ModelStatus, error) {
	url, err := m.routerBase()
	if err != nil {
		return nil, err
	}
	return routerclient.List(context.Background(), url)
}

// routerBase returns the running router's base URL.
func (m *Manager) routerBase() (string, error) {
	m.mu.Lock()
	url := m.routerURL
	m.mu.Unlock()
	if url == "" {
		return "", errors.New("router not running")
	}
	return url, nil
}

// applyExtraEnv appends configured KEY=VALUE pairs, skipping any whose
// key is already present in the inherited environment. Matching
// pinCUDADeviceOrder's rule: a value the operator exported around the
// process wins over one set in the UI, so a systemd drop-in or container
// env stays authoritative.
func applyExtraEnv(env []string, extra []string) []string {
	if len(extra) == 0 {
		return env
	}
	existing := make(map[string]bool, len(env))
	for _, kv := range env {
		if i := strings.IndexByte(kv, '='); i > 0 {
			existing[kv[:i]] = true
		}
	}
	for _, kv := range extra {
		i := strings.IndexByte(kv, '=')
		if i <= 0 {
			continue
		}
		if existing[kv[:i]] {
			// Inherited values win, but silently ignoring a setting the
			// user made in Settings looks like the setting doesn't work.
			// The value is not logged: it can be a token.
			slog.Warn("runtime environment variable ignored: already set in the service environment",
				"name", kv[:i])
			continue
		}
		env = append(env, kv)
	}
	return env
}

// pinCUDADeviceOrder forces llama-server's CUDA backend to enumerate GPUs in
// PCI bus order, matching nvidia-smi (whose indices drive the web UI's "GPU N"
// labels and the --main-gpu / --tensor-split values we generate). Without this
// the CUDA runtime defaults to CUDA_DEVICE_ORDER=FASTEST_FIRST — a perf-based
// ordering that diverges from nvidia-smi on heterogeneous multi-GPU boxes, so
// "GPU 0" in the UI could map to a different physical card in llama-server
// (issue #68). A user-provided CUDA_DEVICE_ORDER is left untouched.
func pinCUDADeviceOrder(env []string) []string {
	for _, kv := range env {
		if strings.HasPrefix(kv, "CUDA_DEVICE_ORDER=") {
			return env
		}
	}
	return append(env, "CUDA_DEVICE_ORDER=PCI_BUS_ID")
}
