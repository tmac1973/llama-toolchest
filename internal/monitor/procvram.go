package monitor

import (
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// kfdProcRoot is where the AMD kernel driver lists each GPU process's
// memory; a var so tests can point it at a fixture.
var kfdProcRoot = "/sys/class/kfd/kfd/proc"

// nvidiaComputeApps runs nvidia-smi's per-process memory query; a var so
// tests can replace it.
var nvidiaComputeApps = func() ([]byte, error) {
	cmd := exec.Command("nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits")
	timer := time.AfterFunc(5*time.Second, func() {
		if cmd.Process != nil {
			cmd.Process.Kill()
		}
	})
	defer timer.Stop()
	return cmd.Output()
}

// ProcessVRAMMiB returns the GPU memory one process holds, in MiB, summed
// over its GPUs, and whether it could be read.
//
// AMD: the kernel driver's /sys/class/kfd/kfd/proc/<pid>/vram_<gpu_id>
// files, readable without rocm-smi or root. NVIDIA: nvidia-smi's compute
// apps list. A process missing from both reports ok=false, never 0: inside
// a container's PID namespace the tools list host PIDs, and Vulkan
// backends show up in neither, so "not listed" means "not known".
func ProcessVRAMMiB(pid int) (mib int, ok bool) {
	if pid <= 0 {
		return 0, false
	}
	if mib, ok := kfdProcessVRAM(pid); ok {
		return mib, true
	}
	return nvidiaProcessVRAM(pid)
}

func kfdProcessVRAM(pid int) (int, bool) {
	files, _ := filepath.Glob(filepath.Join(kfdProcRoot, strconv.Itoa(pid), "vram_*"))
	if len(files) == 0 {
		return 0, false
	}
	var bytes int64
	for _, f := range files {
		raw, err := os.ReadFile(f)
		if err != nil {
			return 0, false
		}
		n, err := strconv.ParseInt(strings.TrimSpace(string(raw)), 10, 64)
		if err != nil {
			return 0, false
		}
		bytes += n
	}
	return int(bytes >> 20), true
}

func nvidiaProcessVRAM(pid int) (int, bool) {
	out, err := nvidiaComputeApps()
	if err != nil {
		return 0, false
	}
	return parseComputeApps(out, pid)
}

// parseComputeApps sums the used_memory (MiB) of pid's lines in
// "pid, used_memory" CSV; a process on several GPUs has a line per GPU.
func parseComputeApps(out []byte, pid int) (int, bool) {
	total, found := 0, false
	for _, line := range strings.Split(string(out), "\n") {
		fields := strings.Split(line, ",")
		if len(fields) < 2 {
			continue
		}
		p, err := strconv.Atoi(strings.TrimSpace(fields[0]))
		if err != nil || p != pid {
			continue
		}
		mib, err := strconv.Atoi(strings.TrimSpace(fields[1]))
		if err != nil {
			continue
		}
		total += mib
		found = true
	}
	return total, found
}
