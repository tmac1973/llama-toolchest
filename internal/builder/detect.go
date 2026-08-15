package builder

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"
)

// FindROCmTool locates a ROCm binary. PATH is tried first, then
// $ROCM_PATH/bin, then /opt/rocm/bin. Fedora and Debian symlink the
// ROCm tools into /usr/bin, but Arch-family distros install everything
// under /opt/rocm without touching PATH — a bare exec of "rocminfo"
// fails there even with ROCm fully installed. Returns "" if the tool
// isn't found anywhere.
func FindROCmTool(name string) string {
	if p, err := exec.LookPath(name); err == nil {
		return p
	}
	var dirs []string
	if rp := os.Getenv("ROCM_PATH"); rp != "" {
		dirs = append(dirs, filepath.Join(rp, "bin"))
	}
	dirs = append(dirs, "/opt/rocm/bin")
	for _, d := range dirs {
		p := filepath.Join(d, name)
		if st, err := os.Stat(p); err == nil && !st.IsDir() {
			return p
		}
	}
	return ""
}

// RunningInContainer reports whether the current process is running inside a
// Docker- or Podman-style container. Docker creates /.dockerenv at the
// container root; Podman/CRI-O create /run/.containerenv. Either marker is
// sufficient; we don't fall back to /proc/1/cgroup parsing because cgroup v2
// no longer reliably names the runtime.
func RunningInContainer() bool {
	for _, p := range []string{"/.dockerenv", "/run/.containerenv"} {
		if _, err := os.Stat(p); err == nil {
			return true
		}
	}
	return false
}

// apuArchs are gfx IDs belonging to integrated GPUs (APUs). Used to keep
// iGPU targets out of the default GPU_TARGETS: on a box with a discrete
// GPU, the iGPU is a tiny memory carve-out nobody wants kernels for.
// APU-only boxes (a Strix Halo, gfx1151) keep their targets — see
// rocmGPUTargets.
var apuArchs = map[string]bool{
	"gfx902": true, "gfx909": true, "gfx90c": true, // Raven/Renoir/Cezanne
	"gfx1013": true, "gfx1033": true, // Van Gogh
	"gfx1035": true, "gfx1036": true, "gfx1037": true, // Rembrandt/Raphael/Mendocino
	"gfx1103": true,                                                    // Phoenix/Hawk Point
	"gfx1150": true, "gfx1151": true, "gfx1152": true, "gfx1153": true, // Strix/Krackan
}

// IsIGPUArch reports whether a gfx target belongs to an integrated GPU.
func IsIGPUArch(gfx string) bool { return apuArchs[gfx] }

// Backend represents a detected GPU compute backend.
type Backend struct {
	Name      string   `json:"name"` // "rocm", "cuda", "vulkan", "metal", "cpu"
	Available bool     `json:"available"`
	GPUs      []string `json:"gpus"` // e.g. ["gfx1201", "gfx1201"]
	Info      string   `json:"info"` // human-readable summary
}

var (
	detectMu       sync.Mutex
	detectCache    []Backend
	detectCachedAt time.Time
)

// DetectBackends probes the system for available GPU compute backends.
//
// Results are cached briefly: probing execs rocminfo, nvidia-smi, and
// vulkaninfo (the latter alone costs ~100ms), and callers like the
// build-option list and profile lookup run on every page render. Ten
// seconds is long enough to collapse a render into one probe and short
// enough that a freshly installed SDK appears on the next refresh.
func DetectBackends() []Backend {
	detectMu.Lock()
	defer detectMu.Unlock()
	if detectCache != nil && time.Since(detectCachedAt) < 10*time.Second {
		return detectCache
	}
	detectCache = []Backend{
		detectROCm(),
		detectCUDA(),
		detectVulkan(),
		detectMetal(),
		{Name: "cpu", Available: true, Info: "CPU fallback (always available)"},
	}
	detectCachedAt = time.Now()
	return detectCache
}

func detectROCm() Backend {
	b := Backend{Name: "rocm"}

	rocminfo := FindROCmTool("rocminfo")
	if rocminfo == "" {
		b.Info = "rocminfo not found (checked PATH, $ROCM_PATH/bin, /opt/rocm/bin)"
		return b
	}
	out, err := exec.Command(rocminfo).Output()
	if err != nil {
		b.Info = "rocminfo failed"
		return b
	}

	// Parse GPU agent names from rocminfo output.
	// Only match short gfx IDs (e.g. "gfx1100"), skip triple-format like "amdgcn-amd-amdhsa--gfx1100".
	for _, line := range strings.Split(string(out), "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "Name:") {
			continue
		}
		name := strings.TrimSpace(strings.TrimPrefix(line, "Name:"))
		if strings.HasPrefix(name, "gfx") && !strings.Contains(name, "-") {
			b.GPUs = append(b.GPUs, name)
		}
	}

	if len(b.GPUs) > 0 {
		b.Available = true
		b.Info = strings.Join(b.GPUs, ", ")
	} else {
		b.Info = "rocminfo found but no GPU agents detected"
	}
	return b
}

func detectCUDA() Backend {
	b := Backend{Name: "cuda"}

	out, err := exec.Command("nvidia-smi",
		"--query-gpu=name",
		"--format=csv,noheader,nounits").Output()
	if err != nil {
		b.Info = "nvidia-smi not found or failed"
		return b
	}

	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		name := strings.TrimSpace(line)
		if name != "" {
			b.GPUs = append(b.GPUs, name)
		}
	}

	if len(b.GPUs) > 0 {
		b.Available = true
		b.Info = strings.Join(b.GPUs, ", ")
	} else {
		b.Info = "nvidia-smi found but no GPUs detected"
	}
	return b
}

func detectVulkan() Backend {
	b := Backend{Name: "vulkan"}

	// Vulkan in containers requires careful host driver/ICD passthrough that
	// we don't manage; restrict the profile to host-mode installs.
	if RunningInContainer() {
		b.Info = "vulkan builds are only supported in host mode"
		return b
	}

	if _, err := exec.LookPath("vulkaninfo"); err != nil {
		b.Info = "vulkaninfo not found"
		return b
	}

	out, err := exec.Command("vulkaninfo", "--summary").Output()
	if err != nil {
		b.Info = "vulkaninfo failed to run"
		return b
	}

	// Parse "deviceName" lines from --summary; one per physical device.
	for _, line := range strings.Split(string(out), "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "deviceName") {
			continue
		}
		// Format: "deviceName        = AMD Radeon RX 7900 XTX"
		if idx := strings.Index(line, "="); idx >= 0 {
			name := strings.TrimSpace(line[idx+1:])
			if name != "" && !strings.Contains(strings.ToLower(name), "llvmpipe") {
				b.GPUs = append(b.GPUs, name)
			}
		}
	}

	if len(b.GPUs) > 0 {
		b.Available = true
		b.Info = strings.Join(b.GPUs, ", ")
	} else {
		// vulkaninfo ran but only reported software (llvmpipe) or nothing useful.
		// Mark unavailable so users don't pick it expecting GPU acceleration.
		b.Info = "vulkaninfo found but no hardware Vulkan devices"
	}
	return b
}

func detectMetal() Backend {
	b := Backend{Name: "metal"}
	if runtime.GOOS != "darwin" {
		b.Info = "Metal is macOS-only"
		return b
	}
	b.Available = true
	b.Info = "Apple Metal (always available on macOS)"
	return b
}
