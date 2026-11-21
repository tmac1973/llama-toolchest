package monitor

import "golang.org/x/sys/unix"

// perfCores is the number of performance cores of an Apple Silicon CPU,
// which macOS lists as performance level 0. It is 0 when the CPU has one
// kind of core (hw.nperflevels is 1) or macOS does not say.
func perfCores() int {
	levels, err := unix.SysctlUint32("hw.nperflevels")
	if err != nil || levels < 2 {
		return 0
	}
	n, err := unix.SysctlUint32("hw.perflevel0.physicalcpu")
	if err != nil {
		return 0
	}
	return int(n)
}
