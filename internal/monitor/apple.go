package monitor

import (
	"regexp"
	"strconv"
)

// Apple Silicon has no memory of its own for the GPU: the CPU and the GPU
// share the machine's RAM (unified memory). macOS lets the GPU wire only
// part of it, and Metal's recommendedMaxWorkingSetSize, which llama.cpp
// reports as the device's memory, is that limit. The monitor shows it as
// the GPU's memory so the fit planner, the load checks and eviction work
// with the same figure llama.cpp does.

// appleWiredLimitMiB is the part of memMiB macOS lets the GPU use, in
// MiB. limitMiB is the iogpu.wired_limit_mb sysctl, which an admin sets
// to change the limit; 0 means the macOS default. Without it the default
// is about two thirds of the RAM up to 16 GiB and about three quarters
// above (Metal reports 10922 MiB on 16 GiB, 18186 MiB on 24 GiB and 49152
// MiB on 64 GiB). The small rounding down keeps the figure at or under
// what Metal reports.
func appleWiredLimitMiB(memMiB, limitMiB int) int {
	if limitMiB > 0 {
		return min(limitMiB, memMiB)
	}
	if memMiB <= 16*1024 {
		return memMiB * 2 / 3
	}
	return memMiB * 74 / 100
}

// appleGPUStats is what ioreg's IOAccelerator entry tells about the GPU.
type appleGPUStats struct {
	Model       string
	Cores       int
	UsedBytes   int64 // "Alloc system memory": memory the GPU holds
	UtilPercent int   // "Device Utilization %"
}

var (
	ioregModel = regexp.MustCompile(`"model" = "([^"]+)"`)
	ioregCores = regexp.MustCompile(`"gpu-core-count" = (\d+)`)
	ioregAlloc = regexp.MustCompile(`"Alloc system memory"=(\d+)`)
	ioregUtil  = regexp.MustCompile(`"Device Utilization %"=(\d+)`)
)

// parseIOAccelerator reads the output of
// "ioreg -r -d 1 -c IOAccelerator". ok is false when it names no GPU.
func parseIOAccelerator(out string) (s appleGPUStats, ok bool) {
	m := ioregModel.FindStringSubmatch(out)
	if m == nil {
		return s, false
	}
	s.Model = m[1]
	if m := ioregCores.FindStringSubmatch(out); m != nil {
		s.Cores, _ = strconv.Atoi(m[1])
	}
	if m := ioregAlloc.FindStringSubmatch(out); m != nil {
		s.UsedBytes, _ = strconv.ParseInt(m[1], 10, 64)
	}
	if m := ioregUtil.FindStringSubmatch(out); m != nil {
		s.UtilPercent, _ = strconv.Atoi(m[1])
	}
	return s, true
}

// appleGPUInfo is the GPU as the monitor lists it. It is not marked as an
// integrated GPU: on these machines it is the only GPU and llama.cpp puts
// whole models on it, so the planner treats it as a card of its own.
func appleGPUInfo(s appleGPUStats, totalMiB int) GPUInfo {
	name := s.Model
	if s.Cores > 0 {
		name += " (" + strconv.Itoa(s.Cores) + "-core GPU)"
	}
	return GPUInfo{
		Index:        0,
		Name:         name,
		UtilPercent:  s.UtilPercent,
		VRAMUsedMiB:  min(totalMiB, int(s.UsedBytes/1024/1024)),
		VRAMTotalMiB: totalMiB,
	}
}
