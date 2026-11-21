package monitor

import (
	"fmt"
	"os/exec"
	"runtime"

	"golang.org/x/sys/unix"
)

type appleBackend struct{}

// newApple returns the Metal backend on an Apple Silicon Mac. An Intel Mac
// has no unified memory, and llama.cpp does not use Metal there.
func newApple() GPUBackend {
	if runtime.GOARCH != "arm64" {
		return nil
	}
	if _, err := readAppleGPU(); err != nil {
		return nil
	}
	return &appleBackend{}
}

func (a *appleBackend) Name() string { return "metal" }

func (a *appleBackend) Collect() ([]GPUInfo, error) {
	s, err := readAppleGPU()
	if err != nil {
		return nil, err
	}
	mem, err := unix.SysctlUint64("hw.memsize")
	if err != nil {
		return nil, fmt.Errorf("sysctl hw.memsize: %w", err)
	}
	// Not set on most machines; the read fails on older macOS.
	limit, _ := unix.SysctlUint32("iogpu.wired_limit_mb")
	total := appleWiredLimitMiB(int(mem/1024/1024), int(limit))
	return []GPUInfo{appleGPUInfo(s, total)}, nil
}

func readAppleGPU() (appleGPUStats, error) {
	out, err := exec.Command("ioreg", "-r", "-d", "1", "-c", "IOAccelerator").Output()
	if err != nil {
		return appleGPUStats{}, fmt.Errorf("ioreg: %w", err)
	}
	s, ok := parseIOAccelerator(string(out))
	if !ok {
		return s, fmt.Errorf("ioreg lists no GPU")
	}
	return s, nil
}
