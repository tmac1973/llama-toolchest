package monitor

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

// The figure a running sd-server reported on asmodean (R9700) on
// 2026-10-25: 9,693,564,928 bytes in vram_63146.
func TestProcessVRAMFromKFD(t *testing.T) {
	root := t.TempDir()
	dir := filepath.Join(root, "4242")
	os.MkdirAll(dir, 0o755)
	os.WriteFile(filepath.Join(dir, "vram_63146"), []byte("9693564928\n"), 0o444)
	os.WriteFile(filepath.Join(dir, "sdma_63146"), []byte("0\n"), 0o444)

	old := kfdProcRoot
	kfdProcRoot = root
	t.Cleanup(func() { kfdProcRoot = old })

	if mib, ok := ProcessVRAMMiB(4242); !ok || mib != 9244 {
		t.Errorf("ProcessVRAMMiB = %d, %v; want 9244, true", mib, ok)
	}
}

func TestProcessVRAMFromNvidiaSMI(t *testing.T) {
	oldRoot, oldSMI := kfdProcRoot, nvidiaComputeApps
	kfdProcRoot = t.TempDir()
	nvidiaComputeApps = func() ([]byte, error) {
		return []byte("1234, 8000\n4242, 6100\n4242, 512\n"), nil
	}
	t.Cleanup(func() { kfdProcRoot, nvidiaComputeApps = oldRoot, oldSMI })

	if mib, ok := ProcessVRAMMiB(4242); !ok || mib != 6612 {
		t.Errorf("ProcessVRAMMiB = %d, %v; want 6612 summed over both GPUs", mib, ok)
	}
	// Not listed (another PID namespace, or a Vulkan backend): unknown,
	// never zero.
	if mib, ok := ProcessVRAMMiB(999); ok {
		t.Errorf("unlisted PID = %d, true; want ok=false", mib)
	}
}

func TestProcessVRAMNoTools(t *testing.T) {
	oldRoot, oldSMI := kfdProcRoot, nvidiaComputeApps
	kfdProcRoot = t.TempDir()
	nvidiaComputeApps = func() ([]byte, error) { return nil, errors.New("not found") }
	t.Cleanup(func() { kfdProcRoot, nvidiaComputeApps = oldRoot, oldSMI })

	if _, ok := ProcessVRAMMiB(4242); ok {
		t.Error("ok without any tool")
	}
	if _, ok := ProcessVRAMMiB(0); ok {
		t.Error("ok for PID 0")
	}
}
