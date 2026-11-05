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
	oldRoot, oldProc, oldSMI := kfdProcRoot, procRoot, nvidiaComputeApps
	kfdProcRoot, procRoot = t.TempDir(), t.TempDir()
	nvidiaComputeApps = func() ([]byte, error) {
		return []byte("1234, 8000\n4242, 6100\n4242, 512\n"), nil
	}
	t.Cleanup(func() { kfdProcRoot, procRoot, nvidiaComputeApps = oldRoot, oldProc, oldSMI })

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
	oldRoot, oldProc, oldSMI := kfdProcRoot, procRoot, nvidiaComputeApps
	kfdProcRoot, procRoot = t.TempDir(), t.TempDir()
	nvidiaComputeApps = func() ([]byte, error) { return nil, errors.New("not found") }
	t.Cleanup(func() { kfdProcRoot, procRoot, nvidiaComputeApps = oldRoot, oldProc, oldSMI })

	if _, ok := ProcessVRAMMiB(4242); ok {
		t.Error("ok without any tool")
	}
	if _, ok := ProcessVRAMMiB(0); ok {
		t.Error("ok for PID 0")
	}
}

// Inside a container the kfd files list host PIDs, so the amdgpu
// driver's fdinfo is read: each DRM client once, other drivers' clients
// and other files left out. The figures are asmodean's (R9700, kernel
// 7.2) on 2026-11-05: a 4B model's llama-server, and an idle sd-server.
func TestProcessVRAMFromDRMFdinfo(t *testing.T) {
	oldRoot, oldProc, oldSMI := kfdProcRoot, procRoot, nvidiaComputeApps
	kfdProcRoot, procRoot = t.TempDir(), t.TempDir()
	nvidiaComputeApps = func() ([]byte, error) { return nil, errors.New("not found") }
	t.Cleanup(func() { kfdProcRoot, procRoot, nvidiaComputeApps = oldRoot, oldProc, oldSMI })

	write := func(pid, fd, body string) {
		dir := filepath.Join(procRoot, pid, "fdinfo")
		os.MkdirAll(dir, 0o755)
		os.WriteFile(filepath.Join(dir, fd), []byte(body), 0o444)
	}
	client := "pos:\t0\nflags:\t02100002\ndrm-driver:\tamdgpu\ndrm-client-id:\t284290\ndrm-pdev:\t0000:11:00.0\ndrm-memory-vram:\t10258240 KiB\n"
	write("843", "7", client)
	write("843", "9", client) // a second file of the same client
	write("843", "3", "pos:\t0\nflags:\t02\n")
	write("843", "11", "drm-driver:\ti915\ndrm-client-id:\t7\ndrm-memory-vram:\t4096 KiB\n")
	if mib, ok := ProcessVRAMMiB(843); !ok || mib != 10017 {
		t.Errorf("ProcessVRAMMiB = %d, %v; want 10017, true", mib, ok)
	}

	// An idle backend holds next to nothing: known, and 0.
	write("793", "7", "drm-driver:\tamdgpu\ndrm-client-id:\t284272\ndrm-resident-vram:\t52 KiB\n")
	if mib, ok := ProcessVRAMMiB(793); !ok || mib != 0 {
		t.Errorf("idle backend = %d, %v; want 0, true", mib, ok)
	}

	// No amdgpu client: not known.
	write("12", "3", "pos:\t0\n")
	if _, ok := ProcessVRAMMiB(12); ok {
		t.Error("ok for a process with no GPU client")
	}
}
