package builder

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// fakeFS lays out files and relative symlinks under a temp root. Real installs
// use absolute links through /etc/alternatives; relative ones keep the tree
// inside the temp dir and resolve the same way.
type fakeFS struct {
	t    *testing.T
	root string
}

func newFakeFS(t *testing.T) *fakeFS { return &fakeFS{t: t, root: t.TempDir()} }

func (f *fakeFS) file(p, body string) {
	f.t.Helper()
	full := filepath.Join(f.root, p)
	if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
		f.t.Fatal(err)
	}
	if err := os.WriteFile(full, []byte(body), 0o755); err != nil {
		f.t.Fatal(err)
	}
}

func (f *fakeFS) link(p, target string) {
	f.t.Helper()
	full := filepath.Join(f.root, p)
	if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
		f.t.Fatal(err)
	}
	if err := os.Symlink(target, full); err != nil {
		f.t.Fatal(err)
	}
}

// amdInstall lays out the parts of an AMD install detection looks at.
func (f *fakeFS) amdInstall(dir, version string) {
	f.file(dir+"/.info/version", version+"\n")
	f.file(dir+"/lib/cmake/hip-lang/hip-lang-config.cmake", "")
	f.file(dir+"/lib/llvm/bin/clang++", "")
	f.file(dir+"/lib/llvm/lib/clang/22/lib/amdgcn/bitcode/ocml.bc", "")
	f.file(dir+"/include/hip/hip_runtime.h", "")
}

// distroInstall lays out Ubuntu's ROCm in /usr.
func (f *fakeFS) distroInstall(hipVersion string) {
	f.file("/usr/lib/x86_64-linux-gnu/cmake/hip/hip-config-version.cmake",
		`set(PACKAGE_VERSION "`+hipVersion+`")`)
	f.file("/usr/lib/x86_64-linux-gnu/cmake/hip-lang/hip-lang-config.cmake", "")
	f.file("/usr/lib/llvm-21/bin/clang++", "")
	f.file("/usr/lib/llvm-21/lib/clang/21/amdgcn/bitcode/ocml.bc", "")
}

func roots(installs []ROCmInstall) []string {
	var out []string
	for _, i := range installs {
		out = append(out, i.Root+" "+i.Version)
	}
	return out
}

// Ubuntu 26.04 with its own ROCm 7.1, then AMD's 10.1: /opt/rocm is a
// directory of alternatives links into core-10.1.
func TestDetectDistroPlusCore(t *testing.T) {
	f := newFakeFS(t)
	f.distroInstall("7.1.52801")
	f.amdInstall("/opt/rocm/core-10.1", "10.1.0")
	f.link("/opt/rocm/core", "core-10.1")
	f.link("/opt/rocm/core-10", "core-10.1")
	f.link("/opt/rocm/lib", "core-10.1/lib")
	f.link("/opt/rocm/llvm", "core-10.1/lib/llvm")

	got := detectROCmInstalls(f.root, "")
	want := []string{"/opt/rocm/core-10.1 10.1.0", "/usr 7.1"}
	if !reflect.DeepEqual(roots(got), want) {
		t.Fatalf("got %v, want %v", roots(got), want)
	}
	if !got[1].Distro || got[0].Distro {
		t.Errorf("only /usr is the distro install: %+v", got)
	}
	if got[0].Damaged != "" || got[1].Damaged != "" {
		t.Errorf("nothing is damaged here: %+v", got)
	}
}

// Ubuntu 24.04 with AMD's 7.2.4, then 10.1 — the user's machine. /opt/rocm
// links to /opt/rocm-7.2.4, so 10.1 lands inside it, and its llvm and amdgcn
// links get taken over.
func TestDetectOldLinePlusCore(t *testing.T) {
	f := newFakeFS(t)
	f.amdInstall("/opt/rocm-7.2.4", "7.2.4")
	f.link("/opt/rocm", "rocm-7.2.4")
	f.amdInstall("/opt/rocm-7.2.4/core-10.1", "10.1.0")
	f.link("/opt/rocm-7.2.4/core-10", "core-10.1")
	f.link("/opt/rocm-7.2.4/llvm", "core-10.1/lib/llvm")
	f.file("/opt/rocm-7.2.4/core-10.1/amdgcn/bitcode/ocml.bc", "")
	f.link("/opt/rocm-7.2.4/amdgcn", "core-10.1/amdgcn")

	got := detectROCmInstalls(f.root, "")
	want := []string{"/opt/rocm-7.2.4/core-10.1 10.1.0", "/opt/rocm-7.2.4 7.2.4"}
	if !reflect.DeepEqual(roots(got), want) {
		t.Fatalf("got %v, want %v", roots(got), want)
	}
	if got[1].Damaged == "" {
		t.Error("7.2.4's taken-over links are not reported")
	}
	if got[0].Damaged != "" {
		t.Errorf("10.1 itself is not damaged: %q", got[0].Damaged)
	}

	// Pinned to 7.2.4, the compiler and device library are its own, not
	// what the taken-over links lead to.
	c := got[1].hipCompiler(f.root)
	if c != "/opt/rocm-7.2.4/lib/llvm/bin/clang++" {
		t.Errorf("7.2.4 compiler = %q", c)
	}
	if d := deviceLibDir(f.root, c); d != "/opt/rocm-7.2.4/lib/llvm/lib/clang/22/lib/amdgcn/bitcode" {
		t.Errorf("7.2.4 device libs = %q", d)
	}
}

// A single self-contained /opt/rocm (Arch) and the Fedora container's AMD
// 7.2.4 behind an /opt/rocm link are each found once.
func TestDetectSingleInstalls(t *testing.T) {
	arch := newFakeFS(t)
	arch.amdInstall("/opt/rocm", "6.4.1")
	if got := roots(detectROCmInstalls(arch.root, "")); !reflect.DeepEqual(got, []string{"/opt/rocm 6.4.1"}) {
		t.Errorf("arch: %v", got)
	}

	linked := newFakeFS(t)
	linked.amdInstall("/opt/rocm-7.2.4", "7.2.4")
	linked.link("/opt/rocm", "rocm-7.2.4")
	if got := roots(detectROCmInstalls(linked.root, "")); !reflect.DeepEqual(got, []string{"/opt/rocm-7.2.4 7.2.4"}) {
		t.Errorf("linked: %v", got)
	}
}

// Runtime-only installs cannot build, and no ROCm at all is an empty list.
func TestDetectSkipsRuntimeOnly(t *testing.T) {
	f := newFakeFS(t)
	f.file("/opt/rocm/core-10.1/.info/version", "10.1.0\n")
	f.file("/opt/rocm/core-10.1/lib/libamdhip64.so.7", "")
	if got := detectROCmInstalls(f.root, ""); len(got) != 0 {
		t.Errorf("got %v", roots(got))
	}
}

// $ROCM_PATH names an install outside the usual places.
func TestDetectROCMPath(t *testing.T) {
	f := newFakeFS(t)
	f.amdInstall("/home/me/therock/install", "10.2.0")
	got := detectROCmInstalls(f.root, "/home/me/therock/install")
	if !reflect.DeepEqual(roots(got), []string{"/home/me/therock/install 10.2.0"}) {
		t.Errorf("got %v", roots(got))
	}
}

func TestVersionLess(t *testing.T) {
	for _, c := range []struct {
		a, b string
		want bool
	}{
		{"7.2.4", "10.1.0", true},
		{"10.1.0", "7.2.4", false},
		{"7.1", "7.2.4", true},
		{"10.0.0", "10.1.0", true},
		{"", "7.1", true},
		{"7.1", "", false},
	} {
		if got := versionLess(c.a, c.b); got != c.want {
			t.Errorf("versionLess(%q, %q) = %v", c.a, c.b, got)
		}
	}
}

func TestChooseROCm(t *testing.T) {
	two := []ROCmInstall{{Root: "/opt/rocm/core-10.1", Version: "10.1.0"}, {Root: "/usr", Version: "7.1", Distro: true}}

	if inst, pinned, err := chooseROCm(nil, ""); inst != nil || pinned || err != nil {
		t.Errorf("none: %v %v %v", inst, pinned, err)
	}
	// One install: recorded, not pinned — builds as before.
	if inst, pinned, _ := chooseROCm(two[1:], ""); inst == nil || inst.Root != "/usr" || pinned {
		t.Errorf("one: %v %v", inst, pinned)
	}
	if inst, pinned, _ := chooseROCm(two, ""); inst.Root != "/opt/rocm/core-10.1" || !pinned {
		t.Errorf("two, default: %v %v", inst, pinned)
	}
	if inst, pinned, _ := chooseROCm(two, "/usr"); inst.Root != "/usr" || !pinned {
		t.Errorf("two, chosen: %v %v", inst, pinned)
	}
	if _, _, err := chooseROCm(two, "/opt/rocm-6.4.0"); err == nil {
		t.Error("an unknown root is accepted")
	}
}

func TestROCmStampRoundTrip(t *testing.T) {
	s := ROCmStamp(ROCmInstall{Root: "/opt/rocm/core-10.1", Version: "10.1.0"})
	if s != "rocm 10.1.0 @ /opt/rocm/core-10.1" {
		t.Fatalf("stamp = %q", s)
	}
	if v, r, ok := ROCmStampInstall(s); !ok || v != "10.1.0" || r != "/opt/rocm/core-10.1" {
		t.Errorf("split = %q %q %v", v, r, ok)
	}
	for _, old := range []string{"", "rocm 7.2.4", "cuda 12.4"} {
		if _, _, ok := ROCmStampInstall(old); ok {
			t.Errorf("%q parsed as an install stamp", old)
		}
	}
}

func TestInstallGone(t *testing.T) {
	here := []ROCmInstall{{Root: "/opt/rocm/core-10.1", Version: "10.1.0"}}
	for _, c := range []struct {
		stamp         string
		installs      []ROCmInstall
		gone, applies bool
	}{
		{"rocm 10.1.0 @ /opt/rocm/core-10.1", here, false, true},
		{"rocm 10.0.0 @ /opt/rocm/core-10.1", here, true, true}, // upgraded in place
		{"rocm 7.2.4 @ /opt/rocm-7.2.4", here, true, true},      // removed
		{"rocm 7.2.4 @ /opt/rocm-7.2.4", nil, false, true},      // nothing detected: unknown
		{"rocm 7.2.4", here, false, false},                      // old stamp: not judged here
	} {
		gone, applies := InstallGone(c.stamp, c.installs)
		if gone != c.gone || applies != c.applies {
			t.Errorf("InstallGone(%q) = %v, %v; want %v, %v", c.stamp, gone, applies, c.gone, c.applies)
		}
	}
}

// The flags for a pinned build: everything from the one install, merged
// with, not over, what the user typed.
func TestPinCMakeFlags(t *testing.T) {
	core := ROCmInstall{Root: "/opt/rocm/core-10.1", Version: "10.1.0"}
	flags := map[string]string{"CMAKE_CXX_FLAGS": "-O3", "CMAKE_HIP_COMPILER": "/my/clang++"}
	pinCMakeFlags(flags, core, "/opt/rocm/core-10.1/lib/llvm/bin/clang++")
	want := map[string]string{
		"CMAKE_HIP_COMPILER":           "/my/clang++", // the user's choice stands
		"CMAKE_HIP_COMPILER_ROCM_ROOT": "/opt/rocm/core-10.1",
		"CMAKE_HIP_FLAGS":              "-isystem /opt/rocm/core-10.1/include",
		"CMAKE_CXX_FLAGS":              "-isystem /opt/rocm/core-10.1/include -O3",
		"CMAKE_BUILD_RPATH":            "/opt/rocm/core-10.1/lib",
	}
	if !reflect.DeepEqual(flags, want) {
		t.Errorf("core:\n got %v\nwant %v", flags, want)
	}

	distro := map[string]string{}
	pinCMakeFlags(distro, ROCmInstall{Root: "/usr", Distro: true}, "/usr/lib/llvm-21/bin/clang++")
	want = map[string]string{
		"CMAKE_HIP_COMPILER":           "/usr/lib/llvm-21/bin/clang++",
		"CMAKE_HIP_COMPILER_ROCM_ROOT": "/usr",
		"CMAKE_PREFIX_PATH":            "/usr",
	}
	if !reflect.DeepEqual(distro, want) {
		t.Errorf("distro:\n got %v\nwant %v", distro, want)
	}
}

// The environment for a pinned build overrides an inherited ROCM_PATH, which
// with two installs is as likely to name the wrong one.
func TestPinEnv(t *testing.T) {
	base := []string{"PATH=/usr/bin", "ROCM_PATH=/opt/rocm", "HIP_DEVICE_LIB_PATH=/opt/rocm/amdgcn/bitcode"}
	core := ROCmInstall{Root: "/opt/rocm/core-10.1"}
	env := pinEnv(append([]string(nil), base...), core, "/opt/rocm/core-10.1/lib/llvm/bin/clang++")
	for k, v := range map[string]string{
		"PATH":           "/opt/rocm/core-10.1/bin:/usr/bin",
		"ROCM_PATH":      "/opt/rocm/core-10.1",
		"HIP_PATH":       "/opt/rocm/core-10.1",
		"HIP_CLANG_PATH": "/opt/rocm/core-10.1/lib/llvm/bin",
	} {
		if got := envValue(env, k); got != v {
			t.Errorf("%s = %q, want %q", k, got, v)
		}
	}
	// No device libs at that path in the test, so the inherited one must not
	// survive to point at the other install.
	if got := envValue(env, "HIP_DEVICE_LIB_PATH"); got != "" {
		t.Errorf("HIP_DEVICE_LIB_PATH = %q", got)
	}

	distro := pinEnv(append([]string(nil), base...), ROCmInstall{Root: "/usr", Distro: true}, "")
	if envValue(distro, "ROCM_PATH") != "" || envValue(distro, "HIP_DEVICE_LIB_PATH") != "" {
		t.Errorf("distro build inherits another install: %v", distro)
	}
}
