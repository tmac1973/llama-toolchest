package builder

import (
	"os"
	"path/filepath"
	"testing"
)

// mkclang creates an executable stub at path, plus its parent dirs.
func mkclang(t *testing.T, path string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("#!/bin/sh\n"), 0o755); err != nil {
		t.Fatal(err)
	}
}

// Debian/Ubuntu package the ROCm device libs per LLVM version
// (rocm-device-libs-17 -> /usr/lib/llvm-17/lib/clang/17/amdgcn/bitcode),
// so the newest clang on the box is regularly the wrong one to hand cmake.
// Pair on the bitcode, not on the version number alone.
func TestNewestDistroHIPClangSkipsClangWithoutDeviceLibs(t *testing.T) {
	lib := t.TempDir()
	mkclang(t, filepath.Join(lib, "llvm-17", "bin", "clang++"))
	if err := os.MkdirAll(filepath.Join(lib, "llvm-17", "lib", "clang", "17", "amdgcn", "bitcode"), 0o755); err != nil {
		t.Fatal(err)
	}
	// Newer, but no device libs — this is the trap.
	mkclang(t, filepath.Join(lib, "llvm-21", "bin", "clang++"))

	got := newestDistroHIPClang(lib)
	want := filepath.Join(lib, "llvm-17", "bin", "clang++")
	if got != want {
		t.Errorf("newestDistroHIPClang() = %q, want %q", got, want)
	}
}

func TestNewestDistroHIPClangPrefersHighestWithDeviceLibs(t *testing.T) {
	lib := t.TempDir()
	for _, v := range []string{"17", "21"} {
		mkclang(t, filepath.Join(lib, "llvm-"+v, "bin", "clang++"))
		if err := os.MkdirAll(filepath.Join(lib, "llvm-"+v, "lib", "clang", v, "amdgcn", "bitcode"), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	got := newestDistroHIPClang(lib)
	want := filepath.Join(lib, "llvm-21", "bin", "clang++")
	if got != want {
		t.Errorf("newestDistroHIPClang() = %q, want %q", got, want)
	}
}

func TestNewestDistroHIPClangEmptyWhenNoneUsable(t *testing.T) {
	lib := t.TempDir()
	mkclang(t, filepath.Join(lib, "llvm-21", "bin", "clang++"))
	if got := newestDistroHIPClang(lib); got != "" {
		t.Errorf("newestDistroHIPClang() = %q, want \"\" (no device libs anywhere)", got)
	}
}

// A ROCm install that ships its own llvm is authoritative; the roots are
// consulted in order so $ROCM_PATH beats the fallback locations.
func TestHIPClangInRootsUsesFirstMatch(t *testing.T) {
	dir := t.TempDir()
	rocmPath := filepath.Join(dir, "rocm")
	optRocm := filepath.Join(dir, "opt-rocm")
	mkclang(t, filepath.Join(optRocm, "llvm", "bin", "clang++"))

	// $ROCM_PATH has no llvm of its own — fall through to the next root.
	if got, want := hipClangInRoots([]string{rocmPath, optRocm}), filepath.Join(optRocm, "llvm", "bin", "clang++"); got != want {
		t.Errorf("hipClangInRoots() = %q, want %q", got, want)
	}

	mkclang(t, filepath.Join(rocmPath, "llvm", "bin", "clang++"))
	if got, want := hipClangInRoots([]string{rocmPath, optRocm}), filepath.Join(rocmPath, "llvm", "bin", "clang++"); got != want {
		t.Errorf("hipClangInRoots() = %q, want %q (first root wins)", got, want)
	}
}

func TestHIPClangInRootsEmptyWhenAbsent(t *testing.T) {
	if got := hipClangInRoots([]string{t.TempDir()}); got != "" {
		t.Errorf("hipClangInRoots() = %q, want \"\"", got)
	}
}

// An explicit -DCMAKE_HIP_COMPILER in the build's extra cmake flags must
// win over anything we detect.
func TestHasCMakeArg(t *testing.T) {
	args := []string{"..", "-G", "Ninja", "-DGGML_HIP=ON", "-DCMAKE_HIP_COMPILER=/custom/clang++"}
	if !hasCMakeArg(args, "CMAKE_HIP_COMPILER") {
		t.Error("hasCMakeArg did not see an explicit CMAKE_HIP_COMPILER")
	}
	if hasCMakeArg(args, "CMAKE_CUDA_COMPILER") {
		t.Error("hasCMakeArg matched a flag that isn't present")
	}
	// Prefix-only matches must not count.
	if hasCMakeArg([]string{"-DCMAKE_HIP_COMPILER_LAUNCHER=ccache"}, "CMAKE_HIP_COMPILER") {
		t.Error("hasCMakeArg matched CMAKE_HIP_COMPILER_LAUNCHER as CMAKE_HIP_COMPILER")
	}
}

func TestEnvValue(t *testing.T) {
	env := []string{"PATH=/usr/bin", "ROCM_PATH=/opt/rocm", "HIP_PATH=/opt/rocm"}
	if got := envValue(env, "ROCM_PATH"); got != "/opt/rocm" {
		t.Errorf("envValue(ROCM_PATH) = %q, want /opt/rocm", got)
	}
	if got := envValue(env, "MISSING"); got != "" {
		t.Errorf("envValue(MISSING) = %q, want \"\"", got)
	}
}
