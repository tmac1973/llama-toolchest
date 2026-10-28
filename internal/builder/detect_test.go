package builder

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"
)

// covFakeTool writes an executable shell script called name into dir. The
// script prints stdout and exits with exitCode. It uses only shell
// built-in commands, so it works even when PATH holds nothing but dir.
func covFakeTool(t *testing.T, dir, name, stdout string, exitCode int) {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	outFile := filepath.Join(dir, name+".out")
	if err := os.WriteFile(outFile, []byte(stdout), 0o644); err != nil {
		t.Fatal(err)
	}
	script := fmt.Sprintf("#!/bin/sh\nwhile IFS= read -r line || [ -n \"$line\" ]; do printf '%%s\\n' \"$line\"; done < '%s'\nexit %d\n", outFile, exitCode)
	if err := os.WriteFile(filepath.Join(dir, name), []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
}

// covResetDetectCache empties the DetectBackends cache now and again when
// the test ends, so fake tool output never reaches another test.
func covResetDetectCache(t *testing.T) {
	t.Helper()
	reset := func() {
		detectMu.Lock()
		detectCache = nil
		detectMu.Unlock()
	}
	reset()
	t.Cleanup(reset)
}

// IsIGPUArch keeps integrated GPUs out of the default build targets, so a
// discrete card must never be reported as integrated, and the match is
// exact (gfx IDs are lower case in rocminfo output).
func TestIsIGPUArch(t *testing.T) {
	cases := []struct {
		gfx  string
		want bool
	}{
		{"gfx1151", true}, // Strix Halo
		{"gfx1103", true}, // Phoenix
		{"gfx90c", true},  // Renoir/Cezanne
		{"gfx1036", true}, // Raphael
		{"gfx1201", false},
		{"gfx1100", false},
		{"gfx942", false},
		{"", false},
		{"GFX1151", false},
		{"gfx11510", false},
	}
	for _, c := range cases {
		if got := IsIGPUArch(c.gfx); got != c.want {
			t.Errorf("IsIGPUArch(%q) = %v, want %v", c.gfx, got, c.want)
		}
	}
}

// RunningInContainer reads only the two marker files Docker and Podman
// create. The result on this machine must follow those files exactly.
// The marker paths are fixed, so the "inside a container" answer can be
// checked only on a machine that has one of them.
func TestRunningInContainerFollowsTheMarkerFiles(t *testing.T) {
	want := false
	for _, p := range []string{"/.dockerenv", "/run/.containerenv"} {
		if _, err := os.Stat(p); err == nil {
			want = true
		}
	}
	if got := RunningInContainer(); got != want {
		t.Errorf("RunningInContainer = %v, want %v from the marker files", got, want)
	}
}

const covROCmInfoOutput = `ROCk module is loaded
=====================
HSA Agents
==========
*******
Agent 1
*******
  Name:                    AMD Ryzen 9 7950X 16-Core Processor
  Marketing Name:          AMD Ryzen 9 7950X 16-Core Processor
  Device Type:             CPU
*******
Agent 2
*******
  Name:                    gfx1201
  Marketing Name:          AMD Radeon RX 9070 XT
  Device Type:             GPU
  ISA Info:
    ISA 1
      Name:                    amdgcn-amd-amdhsa--gfx1201
*******
Agent 3
*******
  Name:                    gfx1201
  Marketing Name:          AMD Radeon RX 9070 XT
  Device Type:             GPU
*******
Agent 4
*******
  Name:                    gfx11-generic
  Device Type:             GPU
*******
Agent 5
*******
  Name:                    aie2
  Device Type:             GPU
`

// detectROCm lists only short gfx IDs as build targets. Generic targets
// ("gfx11-generic"), non-gfx agents and CPU agents must not become
// GPU_TARGETS entries, and the nested ISA "Name:" line must not replace
// the agent's own name.
func TestDetectROCmListsOnlyShortGfxTargets(t *testing.T) {
	bin := t.TempDir()
	covFakeTool(t, bin, "rocminfo", covROCmInfoOutput, 0)
	t.Setenv("PATH", bin)

	b := detectROCm()
	if !b.Available {
		t.Fatalf("rocm not available: %+v", b)
	}
	if want := []string{"gfx1201", "gfx1201"}; !slices.Equal(b.GPUs, want) {
		t.Errorf("GPUs = %v, want %v", b.GPUs, want)
	}
	if b.Info != "gfx1201, gfx1201" {
		t.Errorf("Info = %q", b.Info)
	}
}

// A rocminfo that runs but reports only CPU agents, or that fails, must
// leave ROCm unavailable with a message that says which case it was.
func TestDetectROCmWithoutUsableGPUs(t *testing.T) {
	t.Run("only a CPU agent", func(t *testing.T) {
		bin := t.TempDir()
		covFakeTool(t, bin, "rocminfo", "Agent 1\n  Name: AMD Ryzen\n  Device Type: CPU\n", 0)
		t.Setenv("PATH", bin)
		b := detectROCm()
		if b.Available || len(b.GPUs) != 0 || b.Info != "rocminfo found but no GPU agents detected" {
			t.Errorf("detectROCm = %+v", b)
		}
	})
	t.Run("rocminfo fails", func(t *testing.T) {
		bin := t.TempDir()
		covFakeTool(t, bin, "rocminfo", covROCmInfoOutput, 1)
		t.Setenv("PATH", bin)
		b := detectROCm()
		if b.Available || len(b.GPUs) != 0 || b.Info != "rocminfo failed" {
			t.Errorf("detectROCm = %+v", b)
		}
	})
	t.Run("rocminfo missing", func(t *testing.T) {
		// FindROCmTool always checks /opt/rocm/bin as well, and that
		// path cannot be replaced in a test.
		if _, err := os.Stat("/opt/rocm/bin/rocminfo"); err == nil {
			t.Skip("/opt/rocm/bin/rocminfo exists on this machine")
		}
		t.Setenv("PATH", t.TempDir())
		t.Setenv("ROCM_PATH", t.TempDir())
		b := detectROCm()
		if b.Available || !strings.HasPrefix(b.Info, "rocminfo not found") {
			t.Errorf("detectROCm = %+v", b)
		}
	})
}

// detectCUDA reads one GPU name per nvidia-smi line and ignores blank
// lines. Each failure mode must leave CUDA unavailable.
func TestDetectCUDA(t *testing.T) {
	cases := []struct {
		name      string
		output    string
		exitCode  int
		noTool    bool
		available bool
		gpus      []string
		info      string
	}{
		{name: "three cards", output: "NVIDIA RTX A4000\n  NVIDIA RTX A4000  \n\nNVIDIA RTX A4000\n",
			available: true, gpus: []string{"NVIDIA RTX A4000", "NVIDIA RTX A4000", "NVIDIA RTX A4000"},
			info: "NVIDIA RTX A4000, NVIDIA RTX A4000, NVIDIA RTX A4000"},
		{name: "no cards listed", output: "\n\n", info: "nvidia-smi found but no GPUs detected"},
		{name: "nvidia-smi fails", output: "NVIDIA-SMI has failed\n", exitCode: 9, info: "nvidia-smi not found or failed"},
		{name: "nvidia-smi missing", noTool: true, info: "nvidia-smi not found or failed"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			bin := t.TempDir()
			if !c.noTool {
				covFakeTool(t, bin, "nvidia-smi", c.output, c.exitCode)
			}
			t.Setenv("PATH", bin)
			b := detectCUDA()
			if b.Available != c.available || !slices.Equal(b.GPUs, c.gpus) || b.Info != c.info {
				t.Errorf("detectCUDA = %+v, want available=%v gpus=%v info=%q", b, c.available, c.gpus, c.info)
			}
		})
	}
}

// detectVulkan keeps hardware devices only: a machine whose only Vulkan
// device is the llvmpipe software renderer must not offer a Vulkan build
// that runs on the CPU. Inside a container the probe is skipped.
func TestDetectVulkan(t *testing.T) {
	if RunningInContainer() {
		bin := t.TempDir()
		covFakeTool(t, bin, "vulkaninfo", "deviceName = AMD Radeon\n", 0)
		t.Setenv("PATH", bin)
		if b := detectVulkan(); b.Available || b.Info != "vulkan builds are only supported in host mode" {
			t.Errorf("detectVulkan in a container = %+v", b)
		}
		return
	}
	cases := []struct {
		name      string
		output    string
		exitCode  int
		noTool    bool
		available bool
		gpus      []string
		info      string
	}{
		{name: "hardware and software devices",
			output: "Devices:\n========\nGPU0:\n\tdeviceName        = AMD Radeon RX 9070 XT (RADV GFX1201)\n" +
				"GPU1:\n\tdeviceName        = llvmpipe (LLVM 20.1.0, 256 bits)\n\tdeviceName =\n",
			available: true, gpus: []string{"AMD Radeon RX 9070 XT (RADV GFX1201)"}, info: "AMD Radeon RX 9070 XT (RADV GFX1201)"},
		{name: "only llvmpipe", output: "\tdeviceName = llvmpipe (LLVM 20.1.0)\n",
			info: "vulkaninfo found but no hardware Vulkan devices"},
		{name: "vulkaninfo fails", output: "ERROR_INITIALIZATION_FAILED\n", exitCode: 1, info: "vulkaninfo failed to run"},
		{name: "vulkaninfo missing", noTool: true, info: "vulkaninfo not found"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			bin := t.TempDir()
			if !c.noTool {
				covFakeTool(t, bin, "vulkaninfo", c.output, c.exitCode)
			}
			t.Setenv("PATH", bin)
			b := detectVulkan()
			if b.Available != c.available || !slices.Equal(b.GPUs, c.gpus) || b.Info != c.info {
				t.Errorf("detectVulkan = %+v, want available=%v gpus=%v info=%q", b, c.available, c.gpus, c.info)
			}
		})
	}
}

// Metal exists only on macOS; anywhere else it must be unavailable.
func TestDetectMetal(t *testing.T) {
	b := detectMetal()
	if b.Name != "metal" {
		t.Errorf("Name = %q", b.Name)
	}
	if b.Available != (runtime.GOOS == "darwin") {
		t.Errorf("detectMetal on %s = %+v", runtime.GOOS, b)
	}
}

// DetectBackends returns every backend in a fixed order with CPU always
// available, and caches the answer: a second call shortly after must not
// run the probes again, even when the tools now answer differently.
func TestDetectBackendsOrderAndCache(t *testing.T) {
	covResetDetectCache(t)
	bin := t.TempDir()
	covFakeTool(t, bin, "rocminfo", covROCmInfoOutput, 0)
	covFakeTool(t, bin, "nvidia-smi", "NVIDIA RTX A4000\n", 0)
	t.Setenv("PATH", bin)

	got := DetectBackends()
	var names []string
	for _, b := range got {
		names = append(names, b.Name)
	}
	if want := []string{"rocm", "cuda", "vulkan", "metal", "cpu"}; !slices.Equal(names, want) {
		t.Fatalf("backend order = %v, want %v", names, want)
	}
	if !got[0].Available || !got[1].Available || !slices.Equal(got[1].GPUs, []string{"NVIDIA RTX A4000"}) {
		t.Errorf("rocm/cuda = %+v / %+v", got[0], got[1])
	}
	if !got[4].Available {
		t.Errorf("cpu backend unavailable: %+v", got[4])
	}

	// Change what the tool answers; the cached result must not move.
	covFakeTool(t, bin, "nvidia-smi", "NVIDIA GeForce RTX 3090\nNVIDIA GeForce RTX 3090\n", 0)
	again := DetectBackends()
	if !slices.Equal(again[1].GPUs, []string{"NVIDIA RTX A4000"}) {
		t.Errorf("second call probed again: cuda = %+v", again[1])
	}

	// Once the cache is cleared, the new answer is read.
	covResetDetectCache(t)
	fresh := DetectBackends()
	if len(fresh[1].GPUs) != 2 {
		t.Errorf("after the cache was cleared, cuda = %+v, want the two new cards", fresh[1])
	}
}

// exitErrDetail adds a failed command's stderr to an error message, also
// through wrapping, and adds nothing for errors that carry no stderr.
func TestExitErrDetail(t *testing.T) {
	_, withStderr := exec.Command("/bin/sh", "-c", "echo '  fatal: not a git repository  ' >&2; exit 128").Output()
	_, noStderr := exec.Command("/bin/sh", "-c", "exit 3").Output()
	var ee *exec.ExitError
	if !errors.As(withStderr, &ee) || !errors.As(noStderr, &ee) {
		t.Fatalf("expected exit errors, got %v and %v", withStderr, noStderr)
	}

	cases := []struct {
		name string
		err  error
		want string
	}{
		{"nil", nil, ""},
		{"plain error", errors.New("boom"), ""},
		{"exit error with stderr", withStderr, ": fatal: not a git repository"},
		{"wrapped exit error", fmt.Errorf("listing tags: %w", withStderr), ": fatal: not a git repository"},
		{"exit error without stderr", noStderr, ""},
	}
	for _, c := range cases {
		if got := exitErrDetail(c.err); got != c.want {
			t.Errorf("%s: exitErrDetail = %q, want %q", c.name, got, c.want)
		}
	}
}

// filterReleaseTags keeps only real release tags, in their input order.
// Tags that only start like a release ("backup/...", "vulkan-fix") must
// be dropped, because "latest" once built a backup snapshot.
func TestFilterReleaseTags(t *testing.T) {
	cases := []struct {
		name   string
		family string
		lines  []string
		want   []string
	}{
		{"nightlies", "b", []string{"b10", " b2 ", "backup/ui-shell", "b", "bx1", "", "b3\r", "b4-rc"},
			[]string{"b10", "b2", "b3"}},
		{"releases", "v", []string{"v0.1.0", "v1", "v1.2.3.4", "v1.", "vulkan-fix", "v1.2-rc1", "V1.0", "v"},
			[]string{"v0.1.0", "v1", "v1.2.3.4"}},
		{"empty git output", "b", []string{""}, nil},
		{"no input", "v", nil, nil},
	}
	for _, c := range cases {
		if got := filterReleaseTags(c.family, c.lines); !slices.Equal(got, c.want) {
			t.Errorf("%s: filterReleaseTags = %q, want %q", c.name, got, c.want)
		}
	}
}

// findCUDAHostCompiler must pick the highest side-by-side g++ that
// exists. The candidate paths are fixed under /usr/bin, so this test can
// only compare the answer with what this machine has installed.
func TestFindCUDAHostCompilerPrefersTheHighestVersion(t *testing.T) {
	want := ""
	for _, c := range []string{"/usr/bin/g++-15", "/usr/bin/g++-14", "/usr/bin/g++-13", "/usr/bin/g++-12"} {
		if st, err := os.Stat(c); err == nil && !st.IsDir() {
			want = c
			break
		}
	}
	if got := findCUDAHostCompiler(); got != want {
		t.Errorf("findCUDAHostCompiler = %q, want %q", got, want)
	}
}

// findHIPCompiler asks hipconfig first; when hipconfig names a directory
// without clang++, or fails, the ROCM_PATH of the build environment is
// searched next. The later fallbacks (/opt/rocm, /usr/lib/llvm-N) are
// fixed paths and are covered through hipClangInRoots and
// newestDistroHIPClang instead.
func TestFindHIPCompilerOrder(t *testing.T) {
	t.Run("hipconfig names a clang", func(t *testing.T) {
		bin := t.TempDir()
		llvm := filepath.Join(t.TempDir(), "llvm", "bin")
		mkclang(t, filepath.Join(llvm, "clang++"))
		covFakeTool(t, bin, "hipconfig", llvm+"\n", 0)
		t.Setenv("PATH", bin)

		// A ROCM_PATH clang also exists; hipconfig's answer must win.
		root := t.TempDir()
		mkclang(t, filepath.Join(root, "llvm", "bin", "clang++"))
		env := []string{"ROCM_PATH=" + root}
		if got, want := findHIPCompiler(env), filepath.Join(llvm, "clang++"); got != want {
			t.Errorf("findHIPCompiler = %q, want %q", got, want)
		}
	})
	t.Run("hipconfig names a directory without clang", func(t *testing.T) {
		bin := t.TempDir()
		covFakeTool(t, bin, "hipconfig", t.TempDir()+"\n", 0)
		t.Setenv("PATH", bin)
		root := t.TempDir()
		mkclang(t, filepath.Join(root, "llvm", "bin", "clang++"))
		if got, want := findHIPCompiler([]string{"ROCM_PATH=" + root}), filepath.Join(root, "llvm", "bin", "clang++"); got != want {
			t.Errorf("findHIPCompiler = %q, want %q", got, want)
		}
	})
	t.Run("hipconfig fails", func(t *testing.T) {
		bin := t.TempDir()
		covFakeTool(t, bin, "hipconfig", "/somewhere\n", 1)
		t.Setenv("PATH", bin)
		root := t.TempDir()
		mkclang(t, filepath.Join(root, "llvm", "bin", "clang++"))
		if got, want := findHIPCompiler([]string{"ROCM_PATH=" + root}), filepath.Join(root, "llvm", "bin", "clang++"); got != want {
			t.Errorf("findHIPCompiler = %q, want %q", got, want)
		}
	})
	t.Run("clang without execute bit is skipped", func(t *testing.T) {
		bin := t.TempDir()
		llvm := t.TempDir()
		if err := os.WriteFile(filepath.Join(llvm, "clang++"), []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
		covFakeTool(t, bin, "hipconfig", llvm+"\n", 0)
		t.Setenv("PATH", bin)
		root := t.TempDir()
		mkclang(t, filepath.Join(root, "llvm", "bin", "clang++"))
		if got, want := findHIPCompiler([]string{"ROCM_PATH=" + root}), filepath.Join(root, "llvm", "bin", "clang++"); got != want {
			t.Errorf("findHIPCompiler = %q, want %q", got, want)
		}
	})
}
