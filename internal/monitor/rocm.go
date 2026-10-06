//go:build linux

package monitor

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"

	"github.com/tmac1973/llama-toolchest/internal/builder"
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
	smi := builder.FindROCmTool("rocm-smi")
	if smi == "" {
		return nil, fmt.Errorf("rocm-smi: not found")
	}
	out, err := exec.Command(smi,
		"--showbus", "--showuse", "--showmemuse", "--showtemp", "--showpower",
		"--showfan", "--csv").Output()
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

	// rocm-smi's "device" column ("card0", "card1", ...) is NOT a
	// usable identity: depending on version and machine it is either
	// the DRM card number (driver probe order, shifted by BMC/iGPU
	// display devices) or rocm-smi's own row number (sorted by PCI bus
	// address — the reverse of KFD order on some boards). Interpreting
	// it produced swapped metrics on one machine and duplicate GPU
	// indices on another. The PCI bus address is the only unambiguous
	// key both sides share, so rows are matched to KFD positions by
	// it; without the column the whole collection is rejected and the
	// sysfs fallback (consistent by construction) takes over.
	busCol, ok := colIdx["PCI Bus"]
	if !ok {
		return nil, fmt.Errorf("rocm-smi: no PCI Bus column")
	}

	// KFD-ordered device dirs: position N belongs to the GPU
	// llama-server addresses as ROCm<N>.
	dirs := listAMDGPUDirs()
	byBDF := kfdIndexByBDF(dirs)
	seen := make(map[int]bool)

	var gpus []GPUInfo
	for _, line := range lines[1:] {
		fields := strings.Split(line, ",")
		if len(fields) < 2 {
			continue
		}

		gpu := GPUInfo{}
		if busCol >= len(fields) {
			return nil, fmt.Errorf("rocm-smi: row without PCI Bus field")
		}
		bdf := strings.ToLower(strings.TrimSpace(fields[busCol]))
		idx, ok := byBDF[bdf]
		if !ok || seen[idx] {
			// A device KFD doesn't know, or two rows claiming one
			// GPU: the mapping is unreliable — let sysfs take over
			// rather than render wrong numbers.
			return nil, fmt.Errorf("rocm-smi: device %s not uniquely in KFD topology", bdf)
		}
		seen[idx] = true
		gpu.Index = idx
		if i, ok := colIdx["GPU use (%)"]; ok && i < len(fields) {
			gpu.UtilPercent, _ = strconv.Atoi(strings.TrimSpace(fields[i]))
		}
		if i, ok := colIdx["Temperature (Sensor edge) (C)"]; ok && i < len(fields) {
			f, _ := strconv.ParseFloat(strings.TrimSpace(fields[i]), 64)
			gpu.TempC = int(f)
		}
		if i, ok := colIdx["Average Graphics Package Power (W)"]; ok && i < len(fields) {
			gpu.PowerW, _ = strconv.ParseFloat(strings.TrimSpace(fields[i]), 64)
		}
		// The fan columns' names vary across ROCm versions.
		for _, col := range []string{"Fan speed (%)", "Fan Speed (%)"} {
			if i, ok := colIdx[col]; ok && i < len(fields) {
				if f, err := strconv.ParseFloat(strings.TrimSpace(fields[i]), 64); err == nil {
					gpu.FanPercent, gpu.HasFan = int(f), true
				}
				break
			}
		}
		for _, col := range []string{"Fan RPM", "Fan speed (RPM)"} {
			if i, ok := colIdx[col]; ok && i < len(fields) {
				if f, err := strconv.ParseFloat(strings.TrimSpace(fields[i]), 64); err == nil {
					gpu.FanRPM, gpu.HasFan = int(f), true
				}
				break
			}
		}

		// Get VRAM info from sysfs (more reliable than rocm-smi CSV).
		// gpu.Index is a KFD position now, so it indexes dirs directly.
		if gpu.Index >= 0 && gpu.Index < len(dirs) {
			gpu.VRAMUsedMB, gpu.VRAMTotalMB = readVRAMFromDir(dirs[gpu.Index])
			// rocm-smi leaves the fan columns out when the fans have
			// stopped, so the card's own fan files fill in.
			if !gpu.HasFan {
				gpu.FanPercent, gpu.FanRPM, gpu.HasFan = readFanFromDir(dirs[gpu.Index])
			}
		}

		// Get GPU name from sysfs
		gpu.Name = readGPUNameSysfs(gpu.Index)
		tagROCmGPU(&gpu)

		gpus = append(gpus, gpu)
	}
	// Rows arrive in rocm-smi's PCI-address order; present them in
	// GPU-index order so the sidebar reads ROCm0, ROCm1, ...
	sort.Slice(gpus, func(i, j int) bool { return gpus[i].Index < gpus[j].Index })
	return gpus, nil
}

// listAMDGPUDirs returns the sysfs device directories of AMD GPUs in
// KFD topology order — the order rocminfo and llama-server's HIP
// runtime enumerate in. (rocm-smi does NOT share it: its rows are
// sorted by PCI bus address, which is why collectROCmSMI matches rows
// by that address instead of by position or label.) Shared by
// collectSysfs and collectROCmSMI so the enumeration lives in one
// place.
//
// DRM card numbers follow driver probe order instead, which on desktop
// APU boxes puts the iGPU at card0 while KFD lists the discrete GPU
// first. Indexing sysfs by card number therefore attributed the iGPU's
// 2GB carve-out to the discrete card (and vice versa) on such boxes —
// and would have mismatched the per-model --device indices llama-server
// resolves. KFD ordering keeps every index consumer aligned; the card
// glob remains as a fallback for kernels without KFD topology.
func listAMDGPUDirs() []string {
	if dirs := listAMDGPUDirsKFD(); len(dirs) > 0 {
		return dirs
	}
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

// listAMDGPUDirsKFD enumerates GPU nodes from /sys/class/kfd, mapping
// each node's drm_render_minor to its device directory. CPU agents
// carry gfx_target_version 0 and are skipped.
func listAMDGPUDirsKFD() []string {
	nodes, _ := filepath.Glob("/sys/class/kfd/kfd/topology/nodes/*/properties")
	// Glob order is lexical ("10" before "2"); sort by numeric node id.
	sort.Slice(nodes, func(i, j int) bool {
		return kfdNodeID(nodes[i]) < kfdNodeID(nodes[j])
	})
	var dirs []string
	for _, propsPath := range nodes {
		data, err := os.ReadFile(propsPath)
		if err != nil {
			continue
		}
		gfx, minor := 0, -1
		for _, line := range strings.Split(string(data), "\n") {
			fields := strings.Fields(line)
			if len(fields) != 2 {
				continue
			}
			switch fields[0] {
			case "gfx_target_version":
				gfx, _ = strconv.Atoi(fields[1])
			case "drm_render_minor":
				minor, _ = strconv.Atoi(fields[1])
			}
		}
		if gfx == 0 || minor <= 0 {
			continue
		}
		dir := fmt.Sprintf("/sys/class/drm/renderD%d/device", minor)
		if vendor, err := os.ReadFile(filepath.Join(dir, "vendor")); err == nil &&
			strings.TrimSpace(string(vendor)) == "0x1002" {
			dirs = append(dirs, dir)
		}
	}
	return dirs
}

// kfdNodeID extracts the numeric node id from a topology properties path.
func kfdNodeID(propsPath string) int {
	id, _ := strconv.Atoi(filepath.Base(filepath.Dir(propsPath)))
	return id
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

		gpu.FanPercent, gpu.FanRPM, gpu.HasFan = readFanFromDir(deviceDir)

		gpu.Name = readGPUNameSysfs(idx)
		tagROCmGPU(&gpu)
		gpus = append(gpus, gpu)
	}

	if len(gpus) == 0 {
		return nil, fmt.Errorf("no AMD GPUs found in sysfs")
	}
	return gpus, nil
}

// readFanFromDir reads the fan of the card at deviceDir from the first
// of its hwmon directories that has one.
func readFanFromDir(deviceDir string) (percent, rpm int, ok bool) {
	hwmonDirs, _ := filepath.Glob(filepath.Join(deviceDir, "hwmon", "hwmon*"))
	for _, hwmon := range hwmonDirs {
		if percent, rpm, ok = readFanHwmon(hwmon); ok {
			return percent, rpm, true
		}
	}
	return 0, 0, false
}

// readFanHwmon reads a card's fan from its hwmon directory: the duty
// from pwm1 (0 to pwm1_max, 255 when that file is missing) as a
// percentage, and the speed in RPM from fan1_input. ok is false when the
// directory has neither.
func readFanHwmon(hwmon string) (percent, rpm int, ok bool) {
	readInt := func(name string) (int, bool) {
		data, err := os.ReadFile(filepath.Join(hwmon, name))
		if err != nil {
			return 0, false
		}
		v, err := strconv.Atoi(strings.TrimSpace(string(data)))
		return v, err == nil
	}
	if pwm, have := readInt("pwm1"); have {
		max := 255
		if m, have := readInt("pwm1_max"); have && m > 0 {
			max = m
		}
		percent = (pwm*100 + max/2) / max
		ok = true
	}
	if r, have := readInt("fan1_input"); have {
		rpm = r
		ok = true
	}
	return percent, rpm, ok
}

// kfdIndexByBDF maps each device's PCI bus address (lowercase, e.g.
// "0000:c7:00.0") to its position in the KFD-ordered dir list. The
// address is the final component of the resolved sysfs device path,
// and it is the same string rocm-smi's "PCI Bus" column reports —
// the one identity shared by both enumerations.
func kfdIndexByBDF(dirs []string) map[string]int {
	m := make(map[string]int, len(dirs))
	for i, d := range dirs {
		resolved, err := filepath.EvalSymlinks(d)
		if err != nil {
			continue
		}
		m[strings.ToLower(filepath.Base(resolved))] = i
	}
	return m
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
	names, _ := rocmAgents()
	if gpuIdx >= 0 && gpuIdx < len(names) {
		return names[gpuIdx]
	}
	return fmt.Sprintf("AMD GPU %d", gpuIdx)
}

// rocmGPUNames is the display name of each GPU agent: its marketing name,
// or its gfx id when it reports none, so the entry is never blank.
func rocmGPUNames(agents []builder.ROCmAgent) []string {
	names := make([]string, 0, len(agents))
	for _, a := range agents {
		if a.MarketingName != "" {
			names = append(names, a.MarketingName)
		} else {
			names = append(names, a.Name)
		}
	}
	return names
}

// rocmGPUArchs is the gfx target of each GPU agent, in the same order as
// rocmGPUNames.
func rocmGPUArchs(agents []builder.ROCmAgent) []string {
	archs := make([]string, 0, len(agents))
	for _, a := range agents {
		archs = append(archs, a.Name)
	}
	return archs
}

// rocmAgentInfo caches the rocminfo agent scan: names and gfx archs of
// the GPU agents, in agent order. The hardware doesn't change while the
// process runs, and the previous per-poll rocminfo exec was pure waste.
// Card index → agent index relies on both following the same KFD/PCI
// enumeration order, the same assumption readGPUNameSysfs has always
// made for names.
var (
	rocmAgentsOnce sync.Once
	rocmAgentNames []string
	rocmAgentArchs []string
)

func rocmAgents() (names, archs []string) {
	rocmAgentsOnce.Do(func() {
		rocminfo := builder.FindROCmTool("rocminfo")
		if rocminfo == "" {
			return
		}
		out, err := exec.Command(rocminfo).Output()
		if err != nil {
			return
		}
		agents := builder.ParseROCmGPUAgents(string(out))
		rocmAgentNames = rocmGPUNames(agents)
		rocmAgentArchs = rocmGPUArchs(agents)
	})
	return rocmAgentNames, rocmAgentArchs
}

// tagROCmGPU fills the arch-derived fields for a GPU by index: its gfx
// target and whether it's an integrated GPU (used by the model-config
// assignment guard rails).
func tagROCmGPU(gpu *GPUInfo) {
	_, archs := rocmAgents()
	if gpu.Index >= 0 && gpu.Index < len(archs) {
		gpu.Arch = archs[gpu.Index]
		gpu.IsIGPU = builder.IsIGPUArch(gpu.Arch)
	}
}
