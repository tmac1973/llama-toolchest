package monitor

import (
	"os"
	"path/filepath"
	"testing"
)

func TestProcessPeakRAMMiB(t *testing.T) {
	old := procRoot
	procRoot = t.TempDir()
	t.Cleanup(func() { procRoot = old })
	os.MkdirAll(filepath.Join(procRoot, "42"), 0o755)
	os.WriteFile(filepath.Join(procRoot, "42", "status"), []byte("Name:\tsd-server\nVmPeak:\t 99999999 kB\nVmHWM:\t  6553600 kB\nVmRSS:\t 1024 kB\n"), 0o644)
	if mib, ok := ProcessPeakRAMMiB(42); !ok || mib != 6400 {
		t.Errorf("ProcessPeakRAMMiB(42) = %d, %v; want 6400, true", mib, ok)
	}
	for _, pid := range []int{0, 43} {
		if _, ok := ProcessPeakRAMMiB(pid); ok {
			t.Errorf("ProcessPeakRAMMiB(%d) read something", pid)
		}
	}
}
