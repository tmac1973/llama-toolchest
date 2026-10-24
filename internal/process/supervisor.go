package process

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/exec"
	"strings"
	"sync"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/broadcast"
	"github.com/tmac1973/llama-toolchest/internal/libpath"
)

// Default timings for a supervised process.
const (
	defaultHealthInterval = 2 * time.Second
	defaultHealthTimeout  = 5 * time.Minute
	defaultStopTimeout    = 30 * time.Second
)

// SupervisorConfig describes one program run by a Supervisor.
type SupervisorConfig struct {
	// Name identifies the program in the service log, e.g.
	// "llama-server router" or "sd-server:ming-image".
	Name       string
	BinaryPath string
	Args       []string
	// Env is the complete child environment; see BuildEnv.
	Env     []string
	WorkDir string
	// HealthURL is polled until it answers 200; the process is "running"
	// from then on.
	HealthURL      string
	HealthInterval time.Duration
	HealthTimeout  time.Duration
	StopTimeout    time.Duration
}

// Supervisor starts one child process, streams its output to log
// subscribers, polls its health endpoint until it is ready, and stops it
// gracefully.
type Supervisor struct {
	// label starts the "==> ..." lines added to the program's log and,
	// in lower case, its error messages, e.g. "Router".
	label string

	mu        sync.Mutex
	cmd       *exec.Cmd
	cancel    context.CancelFunc
	status    Status
	config    *SupervisorConfig
	healthURL string
	done      chan struct{}

	logs *broadcast.Broadcaster[string]
}

// NewSupervisor returns a stopped supervisor that keeps the last
// logHistory output lines for new log subscribers. label names the program
// in its log lines and error messages, e.g. "Router".
func NewSupervisor(label string, logHistory int) *Supervisor {
	return &Supervisor{
		label:  label,
		status: Status{State: StateStopped},
		logs:   broadcast.New[string](logHistory, 256),
	}
}

// BuildEnv returns the environment for a child whose binary and shared
// libraries live in binDir: the inherited environment, the library path
// pointed at binDir, CUDA devices in PCI bus order, and the extra
// KEY=VALUE pairs (inherited values win).
func BuildEnv(binDir string, extra []string) []string {
	return applyExtraEnv(pinCUDADeviceOrder(libpath.Prepend(os.Environ(), binDir)), extra)
}

// noun is the lower-case name used in error messages, e.g. "router".
func (s *Supervisor) noun() string {
	return strings.ToLower(s.label)
}

func (c SupervisorConfig) withDefaults() SupervisorConfig {
	if c.HealthInterval <= 0 {
		c.HealthInterval = defaultHealthInterval
	}
	if c.HealthTimeout <= 0 {
		c.HealthTimeout = defaultHealthTimeout
	}
	if c.StopTimeout <= 0 {
		c.StopTimeout = defaultStopTimeout
	}
	return c
}

// Start launches the program described by cfg.
func (s *Supervisor) Start(cfg SupervisorConfig) error {
	cfg = cfg.withDefaults()

	s.mu.Lock()
	defer s.mu.Unlock()

	if s.status.State == StateRunning || s.status.State == StateStarting {
		return fmt.Errorf("%s already %s", s.noun(), s.status.State)
	}

	s.logs.ClearHistory()

	ctx, cancel := context.WithCancel(context.Background())
	cmd := exec.CommandContext(ctx, cfg.BinaryPath, cfg.Args...)
	cmd.Env = cfg.Env
	cmd.Dir = cfg.WorkDir

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		cancel()
		return fmt.Errorf("stdout pipe: %w", err)
	}
	cmd.Stderr = cmd.Stdout

	if err := cmd.Start(); err != nil {
		cancel()
		return fmt.Errorf("start: %w", err)
	}

	done := make(chan struct{})
	s.cmd = cmd
	s.cancel = cancel
	s.config = &cfg
	s.done = done
	s.healthURL = cfg.HealthURL
	s.status = Status{
		State:     StateStarting,
		PID:       cmd.Process.Pid,
		StartedAt: time.Now(),
	}

	go func() {
		scanner := bufio.NewScanner(stdout)
		scanner.Buffer(make([]byte, 0, 64*1024), 256*1024)
		for scanner.Scan() {
			s.broadcast(scanner.Text())
		}
	}()

	go s.monitorProcess(cmd, done)
	go s.pollHealth(cfg)

	slog.Info(cfg.Name+" started", "pid", cmd.Process.Pid)
	return nil
}

// Stop asks the program to exit (SIGTERM), waits up to StopTimeout, then
// kills it.
func (s *Supervisor) Stop() error {
	s.mu.Lock()
	if s.cmd == nil || s.cmd.Process == nil {
		s.mu.Unlock()
		return fmt.Errorf("%s not running", s.noun())
	}
	cmd := s.cmd
	cancel := s.cancel
	done := s.done
	cfg := *s.config
	s.mu.Unlock()

	if err := terminate(cmd.Process); err != nil {
		slog.Debug("terminate failed", "error", err)
	}

	select {
	case <-done:
	case <-time.After(cfg.StopTimeout):
		cmd.Process.Kill()
		<-done
	}

	cancel()

	s.mu.Lock()
	s.status = Status{State: StateStopped}
	s.cmd = nil
	s.cancel = nil
	s.config = nil
	s.mu.Unlock()

	s.broadcast("==> " + s.label + " stopped")
	slog.Info(cfg.Name + " stopped")
	return nil
}

// Restart stops then starts with the current config.
func (s *Supervisor) Restart() error {
	s.mu.Lock()
	cfg := s.config
	s.mu.Unlock()

	if cfg == nil {
		return errors.New("no config to restart with")
	}
	cfgCopy := *cfg

	if err := s.Stop(); err != nil {
		slog.Debug("stop during restart", "error", err)
	}

	time.Sleep(500 * time.Millisecond)
	return s.Start(cfgCopy)
}

// GetStatus returns the process status.
func (s *Supervisor) GetStatus() Status {
	s.mu.Lock()
	defer s.mu.Unlock()

	st := s.status
	if st.State == StateRunning && !st.StartedAt.IsZero() {
		st.Uptime = time.Since(st.StartedAt).Truncate(time.Second).String()
	}
	return st
}

// IsRunning returns true once the process has passed its health check.
func (s *Supervisor) IsRunning() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.status.State == StateRunning
}

// PID returns the process ID while the process is starting or running,
// and 0 otherwise.
func (s *Supervisor) PID() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.status.State != StateStarting && s.status.State != StateRunning {
		return 0
	}
	return s.status.PID
}

// WaitRunning blocks until the process has passed its first health check
// (IsRunning turns true), the process stops or fails, or ctx ends. This is
// the one meaning of "the process is up" that callers should wait on: a
// passing health ping alone can come before the state flips.
func (s *Supervisor) WaitRunning(ctx context.Context) error {
	for {
		s.mu.Lock()
		state, reason := s.status.State, s.status.Error
		s.mu.Unlock()
		noun := s.noun()

		switch state {
		case StateRunning:
			return nil
		case StateStopped:
			return fmt.Errorf("%s stopped before it was ready", noun)
		case StateFailed:
			if reason == "" {
				reason = "unknown error"
			}
			return fmt.Errorf("%s failed to start: %s", noun, reason)
		}

		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(500 * time.Millisecond):
		}
	}
}

// Subscribe returns a channel that receives log lines, starting with the
// buffered history.
func (s *Supervisor) Subscribe() chan string {
	return s.logs.Subscribe()
}

// Unsubscribe removes a log subscriber.
func (s *Supervisor) Unsubscribe(ch chan string) {
	s.logs.Unsubscribe(ch)
}

// RecentLogs returns the buffered log lines, oldest first. The buffer is
// emptied every time the process starts, so after a restart it holds only
// what the new process has printed.
func (s *Supervisor) RecentLogs() []string {
	return s.logs.History()
}

// ClearLogs discards the buffered log history so it won't be replayed to new subscribers.
func (s *Supervisor) ClearLogs() {
	s.logs.ClearHistory()
}

// CheckHealth pings the health endpoint of the most recently started
// process.
func (s *Supervisor) CheckHealth() bool {
	s.mu.Lock()
	url := s.healthURL
	s.mu.Unlock()

	if url == "" {
		return false
	}

	client := &http.Client{Timeout: 2 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		return false
	}
	resp.Body.Close()

	healthy := resp.StatusCode == http.StatusOK

	s.mu.Lock()
	s.status.HealthOK = healthy
	s.mu.Unlock()

	return healthy
}

func (s *Supervisor) broadcast(line string) {
	s.logs.Broadcast(line)
}

func (s *Supervisor) monitorProcess(cmd *exec.Cmd, done chan struct{}) {
	err := cmd.Wait()
	close(done)

	s.mu.Lock()
	defer s.mu.Unlock()

	if s.cmd != cmd {
		return
	}

	if err != nil {
		s.status.State = StateFailed
		s.status.Error = err.Error()
		s.broadcast(fmt.Sprintf("==> %s exited with error: %v", s.label, err))
	} else {
		s.status.State = StateStopped
		s.broadcast("==> " + s.label + " exited normally")
	}
	s.status.HealthOK = false
}

func (s *Supervisor) pollHealth(cfg SupervisorConfig) {
	client := &http.Client{Timeout: 2 * time.Second}
	deadline := time.Now().Add(cfg.HealthTimeout)

	for time.Now().Before(deadline) {
		time.Sleep(cfg.HealthInterval)

		s.mu.Lock()
		if s.status.State != StateStarting && s.status.State != StateRunning {
			s.mu.Unlock()
			return
		}
		s.mu.Unlock()

		resp, err := client.Get(cfg.HealthURL)
		if err != nil {
			continue
		}
		resp.Body.Close()

		if resp.StatusCode == http.StatusOK {
			s.mu.Lock()
			s.status.State = StateRunning
			s.status.HealthOK = true
			s.mu.Unlock()
			s.broadcast("==> " + s.label + " health check passed — ready")
			return
		}
	}

	s.mu.Lock()
	if s.status.State == StateStarting {
		s.status.State = StateFailed
		s.status.Error = "health check timeout"
	}
	s.mu.Unlock()
}
