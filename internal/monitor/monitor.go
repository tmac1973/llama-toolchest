package monitor

import (
	"time"

	"github.com/tmac1973/llama-toolchest/internal/broadcast"
)

// Metrics holds a snapshot of system resource usage.
type Metrics struct {
	Timestamp time.Time  `json:"timestamp"`
	GPU       []GPUInfo  `json:"gpu,omitempty"`
	CPU       CPUInfo    `json:"cpu"`
	Memory    MemoryInfo `json:"memory"`
}

// GPUInfo holds per-GPU metrics.
type GPUInfo struct {
	Index       int     `json:"index"`
	Name        string  `json:"name"`
	UtilPercent int     `json:"util_percent"` // 0-100
	VRAMUsedMB  int     `json:"vram_used_mb"`
	VRAMTotalMB int     `json:"vram_total_mb"`
	TempC       int     `json:"temp_c"`
	PowerW      float64 `json:"power_w,omitempty"`
	// HasFan says the card reports a fan. FanPercent is its duty, 0-100,
	// and FanRPM its speed, 0 when not reported (NVIDIA reports no RPM).
	// Both are 0 on a card whose fans stop when it is cool; without
	// HasFan that would look the same as a card with no fan.
	HasFan     bool   `json:"has_fan,omitempty"`
	FanPercent int    `json:"fan_percent,omitempty"`
	FanRPM     int    `json:"fan_rpm,omitempty"`
	Arch       string `json:"arch,omitempty"`    // gfx target (ROCm only)
	IsIGPU     bool   `json:"is_igpu,omitempty"` // integrated GPU (APU)
}

// CPUInfo holds CPU usage metrics.
type CPUInfo struct {
	UsagePercent float64 `json:"usage_percent"` // 0-100
	Cores        int     `json:"cores"`
}

// MemoryInfo holds system memory metrics.
type MemoryInfo struct {
	UsedMB  int `json:"used_mb"`
	TotalMB int `json:"total_mb"`
}

// GPUBackend provides GPU-specific metric collection.
type GPUBackend interface {
	Name() string
	Collect() ([]GPUInfo, error)
}

// Monitor polls system metrics at a regular interval.
type Monitor struct {
	gpu      GPUBackend
	interval time.Duration

	// metrics holds the latest snapshot (history of one) and fans each new
	// one out to subscribers without blocking the poller.
	metrics *broadcast.Broadcaster[Metrics]

	stop chan struct{}
}

// New creates a Monitor that polls at the given interval.
// It auto-detects the GPU backend.
func New(interval time.Duration) *Monitor {
	return &Monitor{
		gpu:      detectGPUBackend(),
		interval: interval,
		metrics:  broadcast.New[Metrics](1, 4),
		stop:     make(chan struct{}),
	}
}

// Start begins polling in the background.
func (m *Monitor) Start() {
	// Collect once immediately
	m.collect()

	go func() {
		ticker := time.NewTicker(m.interval)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				m.collect()
			case <-m.stop:
				return
			}
		}
	}()
}

// Stop halts the polling loop.
func (m *Monitor) Stop() {
	close(m.stop)
}

// Current returns the latest metrics snapshot, or a zero one before the
// first poll.
func (m *Monitor) Current() Metrics {
	cur, _ := m.metrics.Last()
	return cur
}

// Subscribe returns a channel that receives metrics updates, starting
// with the latest snapshot so a new viewer does not wait for the next
// poll.
func (m *Monitor) Subscribe() chan Metrics {
	return m.metrics.Subscribe()
}

// Unsubscribe removes a subscription channel.
func (m *Monitor) Unsubscribe(ch chan Metrics) {
	m.metrics.Unsubscribe(ch)
}

func (m *Monitor) collect() {
	metrics := Metrics{
		Timestamp: time.Now(),
		CPU:       collectCPU(),
		Memory:    collectMemory(),
	}

	if m.gpu != nil {
		if gpus, err := m.gpu.Collect(); err == nil {
			metrics.GPU = gpus
		}
	}

	// A slow subscriber misses a snapshot rather than stalling the poller.
	m.metrics.Broadcast(metrics)
}

func detectGPUBackend() GPUBackend {
	if b := newNVIDIA(); b != nil {
		return b
	}
	if b := newROCm(); b != nil {
		return b
	}
	return nil
}
