//go:build linux

package monitor

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
)

type rocmBackend struct{}

func newROCm() GPUBackend {
	// Check for /dev/kfd (ROCm kernel driver)
	if _, err := os.Stat("/dev/kfd"); err != nil {
		return nil
	}
	return &rocmBackend{}
}

func (r *rocmBackend) Name() string { return "rocm" }

func (r *rocmBackend) Collect() ([]GPUInfo, error) {
	// Try rocm-smi first
	if gpus, err := r.collectROCmSMI(); err == nil && len(gpus) > 0 {
		return gpus, nil
	}
	// Fallback to sysfs
	return r.collectSysfs()
}

func (r *rocmBackend) collectROCmSMI() ([]GPUInfo, error) {
	out, err := exec.Command("rocm-smi",
		"--showuse", "--showmemuse", "--showtemp", "--showpower",
		"--csv").Output()
	if err != nil {
		return nil, fmt.Errorf("rocm-smi: %w", err)
	}

	lines := strings.Split(strings.TrimSpace(string(out)), "\n")
	if len(lines) < 2 {
		return nil, fmt.Errorf("rocm-smi: unexpected output")
	}

	// Parse CSV header to find column indices
	header := strings.Split(lines[0], ",")
	colIdx := make(map[string]int)
	for i, h := range header {
		colIdx[strings.TrimSpace(h)] = i
	}

	var gpus []GPUInfo
	for _, line := range lines[1:] {
		fields := strings.Split(line, ",")
		if len(fields) < 2 {
			continue
		}

		gpu := GPUInfo{}
		if i, ok := colIdx["device"]; ok && i < len(fields) {
			// rocm-smi reports the device as "card0", "card1", etc.
			// (or sometimes "GPU 0", "GPU 1"). Plain Atoi silently
			// fails and leaves Index=0 for every GPU, which made
			// every GPU show up as GPU0 in the sidebar.
			s := strings.TrimSpace(fields[i])
			s = strings.TrimPrefix(s, "card")
			s = strings.TrimPrefix(s, "GPU ")
			s = strings.TrimPrefix(s, "GPU")
			gpu.Index, _ = strconv.Atoi(s)
		}
		if i, ok := colIdx["GPU use (%)"]; ok && i < len(fields) {
			gpu.UtilPercent, _ = strconv.Atoi(strings.TrimSpace(fields[i]))
		}
		if i, ok := colIdx["GPU memory use (%)"]; ok && i < len(fields) {
			// rocm-smi reports percentage, we need to convert if we have total
			_ = fields[i] // we'll get absolute values from sysfs if needed
		}
		if i, ok := colIdx["Temperature (Sensor edge) (C)"]; ok && i < len(fields) {
			f, _ := strconv.ParseFloat(strings.TrimSpace(fields[i]), 64)
			gpu.TempC = int(f)
		}
		if i, ok := colIdx["Average Graphics Package Power (W)"]; ok && i < len(fields) {
			gpu.PowerW, _ = strconv.ParseFloat(strings.TrimSpace(fields[i]), 64)
		}

		// Get VRAM info from sysfs (more reliable than rocm-smi CSV)
		vramUsed, vramTotal := readVRAMSysfs(gpu.Index)
		gpu.VRAMUsedMB = vramUsed
		gpu.VRAMTotalMB = vramTotal

		// Get GPU name from sysfs
		gpu.Name = readGPUNameSysfs(gpu.Index)

		gpus = append(gpus, gpu)
	}
	return gpus, nil
}

// listAMDGPUDirs returns the sysfs device directories of AMD GPUs
// (vendor 0x1002) in card order. Shared by collectSysfs and readVRAMSysfs
// so the card enumeration lives in one place.
func listAMDGPUDirs() []string {
	cards, _ := filepath.Glob("/sys/class/drm/card[0-9]*/device/vendor")
	var dirs []string
	for _, vendorFile := range cards {
		vendor, _ := os.ReadFile(vendorFile)
		if strings.TrimSpace(string(vendor)) != "0x1002" {
			continue
		}
		dirs = append(dirs, filepath.Dir(vendorFile))
	}
	return dirs
}

func (r *rocmBackend) collectSysfs() ([]GPUInfo, error) {
	var gpus []GPUInfo

	for idx, deviceDir := range listAMDGPUDirs() {
		gpu := GPUInfo{Index: idx}

		// GPU utilization
		if data, err := os.ReadFile(filepath.Join(deviceDir, "gpu_busy_percent")); err == nil {
			gpu.UtilPercent, _ = strconv.Atoi(strings.TrimSpace(string(data)))
		}

		// VRAM
		gpu.VRAMUsedMB, gpu.VRAMTotalMB = readVRAMFromDir(deviceDir)

		// Temperature from hwmon
		hwmonDirs, _ := filepath.Glob(filepath.Join(deviceDir, "hwmon", "hwmon*"))
		for _, hwmon := range hwmonDirs {
			if data, err := os.ReadFile(filepath.Join(hwmon, "temp1_input")); err == nil {
				millideg, _ := strconv.Atoi(strings.TrimSpace(string(data)))
				gpu.TempC = millideg / 1000
				break
			}
		}

		// Power from hwmon
		for _, hwmon := range hwmonDirs {
			if data, err := os.ReadFile(filepath.Join(hwmon, "power1_average")); err == nil {
				microwatts, _ := strconv.Atoi(strings.TrimSpace(string(data)))
				gpu.PowerW = float64(microwatts) / 1_000_000
				break
			}
		}

		gpu.Name = readGPUNameSysfs(idx)
		gpus = append(gpus, gpu)
	}

	if len(gpus) == 0 {
		return nil, fmt.Errorf("no AMD GPUs found in sysfs")
	}
	return gpus, nil
}

// readVRAMSysfs reads VRAM usage for the Nth AMD GPU. Used by the rocm-smi
// path, which only knows the GPU index.
func readVRAMSysfs(gpuIdx int) (usedMB, totalMB int) {
	dirs := listAMDGPUDirs()
	if gpuIdx < 0 || gpuIdx >= len(dirs) {
		return 0, 0
	}
	return readVRAMFromDir(dirs[gpuIdx])
}

// readVRAMFromDir reads /sys/class/drm/card*/device/mem_info_vram_* from an
// already-resolved device directory.
func readVRAMFromDir(deviceDir string) (usedMB, totalMB int) {
	if data, err := os.ReadFile(filepath.Join(deviceDir, "mem_info_vram_used")); err == nil {
		bytes, _ := strconv.ParseInt(strings.TrimSpace(string(data)), 10, 64)
		usedMB = int(bytes / (1024 * 1024))
	}
	if data, err := os.ReadFile(filepath.Join(deviceDir, "mem_info_vram_total")); err == nil {
		bytes, _ := strconv.ParseInt(strings.TrimSpace(string(data)), 10, 64)
		totalMB = int(bytes / (1024 * 1024))
	}
	return
}

func readGPUNameSysfs(gpuIdx int) string {
	if out, err := exec.Command("rocminfo").Output(); err == nil {
		names := parseROCmGPUNames(string(out))
		if gpuIdx >= 0 && gpuIdx < len(names) {
			return names[gpuIdx]
		}
	}
	return fmt.Sprintf("AMD GPU %d", gpuIdx)
}

// parseROCmGPUNames extracts the marketing names of GPU agents from rocminfo
// output, in agent order. rocminfo lists every HSA agent — including the host
// CPU — under "Agent N" blocks, each carrying a "Device Type:" (CPU or GPU)
// and a "Marketing Name:". Only GPU agents are returned.
//
// The previous approach blacklisted marketing names starting with "AMD Ryzen"
// or "AMD EPYC" to skip the CPU agent. That let any other CPU leak through and
// get labelled as GPU 0 — e.g. an Intel Xeon host showed up as "GPU 0 Intel(R)
// Xeon(R) W-2225 CPU" (issue #68). Keying off "Device Type: GPU" is robust to
// the CPU vendor.
//
// When a GPU agent reports no marketing name, its "Name:" (e.g. "gfx1100") is
// used as a fallback so the entry is never blank.
func parseROCmGPUNames(out string) []string {
	type agent struct {
		name      string
		marketing string
		isGPU     bool
	}
	var agents []agent
	cur := -1
	for _, line := range strings.Split(out, "\n") {
		line = strings.TrimSpace(line)
		switch {
		case strings.HasPrefix(line, "Agent "):
			agents = append(agents, agent{})
			cur = len(agents) - 1
		case cur < 0:
			// Header lines before the first agent block.
			continue
		case strings.HasPrefix(line, "Marketing Name:"):
			agents[cur].marketing = strings.TrimSpace(strings.TrimPrefix(line, "Marketing Name:"))
		case strings.HasPrefix(line, "Name:"):
			// Record only the first "Name:" of a block (the agent name);
			// later Name fields belong to nested pool/cache entries.
			if agents[cur].name == "" {
				agents[cur].name = strings.TrimSpace(strings.TrimPrefix(line, "Name:"))
			}
		case strings.HasPrefix(line, "Device Type:"):
			if strings.Contains(line, "GPU") {
				agents[cur].isGPU = true
			}
		}
	}

	var names []string
	for _, a := range agents {
		if !a.isGPU {
			continue
		}
		if a.marketing != "" {
			names = append(names, a.marketing)
		} else {
			names = append(names, a.name)
		}
	}
	return names
}
