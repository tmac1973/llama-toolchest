package monitor

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// procRoot is the proc file system; a var so tests can point it at a
// fixture.
var procRoot = "/proc"

// ProcessPeakRAMMiB returns the most system RAM one process has held so
// far (VmHWM in /proc/<pid>/status), in MiB, and whether it could be read.
func ProcessPeakRAMMiB(pid int) (mib int, ok bool) {
	if pid <= 0 {
		return 0, false
	}
	data, err := os.ReadFile(filepath.Join(procRoot, strconv.Itoa(pid), "status"))
	if err != nil {
		return 0, false
	}
	for _, line := range strings.Split(string(data), "\n") {
		rest, found := strings.CutPrefix(line, "VmHWM:")
		if !found {
			continue
		}
		fields := strings.Fields(rest)
		if len(fields) == 0 {
			return 0, false
		}
		kb, err := strconv.Atoi(fields[0])
		if err != nil {
			return 0, false
		}
		return kb / 1024, true
	}
	return 0, false
}
