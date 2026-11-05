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
// files, readable without rocm-smi or root. Those list host PIDs, so
// inside a container they do not match; there the amdgpu driver's
// per-client figures in /proc/<pid>/fdinfo are read instead, which use
// the PIDs the container sees. NVIDIA: nvidia-smi's compute apps list. A
// process missing from all of them reports ok=false, never 0: Vulkan
// backends on NVIDIA show up in none, so "not listed" means "not known".
func ProcessVRAMMiB(pid int) (mib int, ok bool) {
	if pid <= 0 {
		return 0, false
	}
	if mib, ok := kfdProcessVRAM(pid); ok {
		return mib, true
	}
	if mib, ok := drmFdinfoVRAM(pid); ok {
		return mib, true
	}
	return nvidiaProcessVRAM(pid)
}

// drmFdinfoVRAM sums the VRAM the amdgpu driver reports for pid's DRM
// clients (drm-memory-vram, or drm-resident-vram on kernels that name it
// so), each client once: several open files can share one client. ok is
// false when pid has no amdgpu client. On asmodean (ROCm 10, kernel 7.2)
// it read 10,017 MiB where the kfd file of the same process read 10,159.
func drmFdinfoVRAM(pid int) (int, bool) {
	files, _ := filepath.Glob(filepath.Join(procRoot, strconv.Itoa(pid), "fdinfo", "*"))
	seen := map[string]bool{}
	var kib int64
	found := false
	for _, f := range files {
		raw, err := os.ReadFile(f)
		if err != nil {
			continue
		}
		fields := map[string]string{}
		for _, line := range strings.Split(string(raw), "\n") {
			k, v, ok := strings.Cut(line, ":")
			if ok {
				fields[strings.TrimSpace(k)] = strings.TrimSpace(v)
			}
		}
		id := fields["drm-client-id"]
		if fields["drm-driver"] != "amdgpu" || id == "" || seen[id] {
			continue
		}
		v, ok := fields["drm-memory-vram"]
		if !ok {
			v, ok = fields["drm-resident-vram"]
		}
		if !ok {
			continue
		}
		n, ok := parseKiB(v)
		if !ok {
			continue
		}
		seen[id] = true
		kib += n
		found = true
	}
	return int(kib >> 10), found
}

// parseKiB reads a DRM fdinfo memory figure, "52 KiB" or "10 MiB", as KiB.
// A bare number is bytes.
func parseKiB(v string) (int64, bool) {
	num, unit, _ := strings.Cut(v, " ")
	n, err := strconv.ParseInt(num, 10, 64)
	if err != nil {
		return 0, false
	}
	switch unit {
	case "KiB":
		return n, true
	case "MiB":
		return n << 10, true
	case "GiB":
		return n << 20, true
	case "":
		return n >> 10, true
	}
	return 0, false
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
