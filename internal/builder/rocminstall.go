package builder

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

// ROCmInstall is one ROCm SDK found on this machine.
//
// A host can hold several, from three packaging lines that lay themselves out
// differently:
//
//   - the distro's own packages (Ubuntu 26.04, Fedora, Debian), spread across
//     /usr;
//   - AMD's repo.radeon.com packages, up to 7.2.4, each in /opt/rocm-X.Y.Z
//     with /opt/rocm a link to the current one;
//   - AMD's stable.repo.amd.com packages, 10.x on, each in /opt/rocm/core-X.Y.
//
// The 10.x packages point the shared names — /opt/rocm/{bin,lib,llvm,amdgcn},
// /usr/bin/{hipconfig,hipcc,rocminfo} — at the newest 10.x through
// update-alternatives, skipping any that a real file is in the way of. So
// beside another install those names end up half switched, with hipconfig
// from one install and hipcc from another. Nothing here is found through
// PATH or those names for that reason: each install is looked up by its own
// directory. plan/rocm-multi-install.md has the layouts as measured.
type ROCmInstall struct {
	// Root is the install's real directory, symlinks resolved: "/usr",
	// "/opt/rocm-7.2.4", "/opt/rocm/core-10.1".
	Root string
	// Version is the ROCm release, "10.1.0" or "7.2.4". A distro install has
	// no release file, so it is HIP's major.minor ("7.1"), which tracks the
	// release on the distro line. Empty when unreadable.
	Version string
	// Distro is true for the /usr install, which the build treats
	// differently (see pinROCm).
	Distro bool
	// Damaged explains why another install has been written into this one,
	// or is empty. Builds still work when pinned, but AMD does not support
	// the state and other tools on the machine may pick up the wrong ROCm.
	Damaged string
}

// Label is how the Builds page names the install.
func (i ROCmInstall) Label() string {
	v := i.Version
	if v == "" {
		v = "unknown version"
	}
	if i.Distro {
		return "ROCm " + v + " (distro, /usr)"
	}
	return "ROCm " + v + " (" + i.Root + ")"
}

// DetectROCmInstalls lists the ROCm installs on this machine that a build can
// use, newest first.
func DetectROCmInstalls() []ROCmInstall {
	return detectROCmInstalls("", os.Getenv("ROCM_PATH"))
}

// detectROCmInstalls does the work against a filesystem rooted at fsRoot, so
// tests can lay out fake installs. fsRoot is "" in production. rocmPath is
// $ROCM_PATH, already joined to nothing.
func detectROCmInstalls(fsRoot, rocmPath string) []ROCmInstall {
	at := func(p string) string { return filepath.Join(fsRoot, p) }

	var candidates []string
	add := func(globs ...string) {
		for _, g := range globs {
			m, _ := filepath.Glob(at(g))
			candidates = append(candidates, m...)
		}
	}
	add("/opt/rocm-*", "/opt/rocm/core-*", "/opt/rocm-*/core-*")
	// /opt/rocm itself only when it is a self-contained install (Arch, or a
	// source build). Under 10.x it is a directory of alternatives links, and
	// under the old AMD line a link to an /opt/rocm-X.Y.Z already listed.
	if st, err := os.Lstat(at("/opt/rocm/lib")); err == nil && st.IsDir() {
		candidates = append(candidates, at("/opt/rocm"))
	}
	if rocmPath != "" {
		candidates = append(candidates, at(rocmPath))
	}

	seen := map[string]bool{}
	var out []ROCmInstall
	for _, c := range candidates {
		// Skip the links: /opt/rocm/core-10 is an alternatives name for
		// whichever core-10.x is current, already listed as itself.
		if st, err := os.Lstat(c); err != nil || st.Mode()&os.ModeSymlink != 0 {
			if c != at(rocmPath) {
				continue
			}
		}
		real, err := filepath.EvalSymlinks(c)
		if err != nil || seen[real] || !hasHIPRuntime(real) {
			continue
		}
		seen[real] = true
		root := strings.TrimPrefix(real, fsRoot)
		out = append(out, ROCmInstall{
			Root:    root,
			Version: releaseVersion(real),
			Damaged: damagedReason(real, fsRoot),
		})
	}

	// The distro install lives in /usr alongside everything else, so it is
	// recognised by its HIP CMake package rather than by a directory of its
	// own.
	if v := distroHIPVersion(at("/usr")); v != "" && !seen[at("/usr")] {
		out = append(out, ROCmInstall{Root: "/usr", Version: v, Distro: true})
	}

	sort.SliceStable(out, func(a, b int) bool {
		return versionLess(out[b].Version, out[a].Version)
	})
	return out
}

// FindROCmInstall returns the install at root among those detected now.
func FindROCmInstall(installs []ROCmInstall, root string) (ROCmInstall, bool) {
	for _, i := range installs {
		if i.Root == root {
			return i, true
		}
	}
	return ROCmInstall{}, false
}

// hasHIPRuntime reports whether dir holds the HIP CMake package llama.cpp's
// enable_language(HIP) needs — the part a runtime-only install lacks.
func hasHIPRuntime(dir string) bool {
	for _, lib := range []string{"lib", "lib64"} {
		if fileExists(filepath.Join(dir, lib, "cmake", "hip-lang", "hip-lang-config.cmake")) {
			return true
		}
	}
	return false
}

// releaseVersion reads an AMD install's release file: .info/version in the
// install directory on both AMD lines.
func releaseVersion(dir string) string {
	b, err := os.ReadFile(filepath.Join(dir, ".info", "version"))
	if err != nil {
		return ""
	}
	if f := strings.Fields(string(b)); len(f) > 0 {
		return f[0]
	}
	return ""
}

var hipPackageVersion = regexp.MustCompile(`set\(PACKAGE_VERSION "([0-9]+)\.([0-9]+)\.`)

// distroHIPVersion returns HIP's major.minor from the distro's HIP CMake
// package under usr, or "" when there is none. Fedora keeps it in lib64,
// Debian and Ubuntu in the multiarch directory.
func distroHIPVersion(usr string) string {
	var files []string
	for _, g := range []string{"lib", "lib64", "lib/*-linux-gnu"} {
		m, _ := filepath.Glob(filepath.Join(usr, g, "cmake", "hip", "hip-config-version.cmake"))
		files = append(files, m...)
	}
	for _, f := range files {
		// A file reached through a link out of /usr belongs to some other
		// install, not the distro's.
		if real, err := filepath.EvalSymlinks(f); err != nil || !strings.HasPrefix(real, usr+"/") {
			continue
		}
		b, err := os.ReadFile(f)
		if err != nil {
			continue
		}
		if m := hipPackageVersion.FindSubmatch(b); m != nil {
			return string(m[1]) + "." + string(m[2])
		}
	}
	return ""
}

// damagedReason reports an old-line install whose llvm or amdgcn link has been
// taken over by a 10.x install. Those names are symlinks inside
// /opt/rocm-X.Y.Z, and with /opt/rocm pointing there, the 10.x
// update-alternatives replaces them. Measured on Ubuntu 24.04 with AMD 7.2.4
// and then 10.1.
func damagedReason(dir, fsRoot string) string {
	var taken []string
	for _, name := range []string{"llvm", "amdgcn"} {
		p := filepath.Join(dir, name)
		st, err := os.Lstat(p)
		if err != nil || st.Mode()&os.ModeSymlink == 0 {
			continue
		}
		real, err := filepath.EvalSymlinks(p)
		if err != nil {
			continue
		}
		// Outside this install, or into a 10.x install nested inside it —
		// where one lands when /opt/rocm links here.
		rel, inside := strings.CutPrefix(real, dir+"/")
		if !inside || strings.HasPrefix(rel, "core-") {
			taken = append(taken, strings.TrimPrefix(p, fsRoot))
		}
	}
	if len(taken) == 0 {
		return ""
	}
	return strings.Join(taken, " and ") + " now point into another ROCm install. " +
		"AMD's ROCm 10 packages took them over: AMD does not support installing " +
		"10.x alongside its 7.2.4-or-older packages. Builds pinned to one install " +
		"still work, but to fix the machine remove AMD's old ROCm packages, or use " +
		"your distro's ROCm alongside 10.x instead."
}

// hipCompiler returns the clang++ that belongs to install, never one found
// through hipconfig or the shared /opt/rocm/llvm name. "" when there is none.
func (i ROCmInstall) hipCompiler(fsRoot string) string {
	at := func(p string) string { return filepath.Join(fsRoot, p) }
	if i.Distro {
		if c := hipClangInRoots([]string{at("/usr/lib64/rocm"), at("/usr/lib/rocm")}); c != "" {
			return strings.TrimPrefix(c, fsRoot)
		}
		return strings.TrimPrefix(newestDistroHIPClang(at("/usr/lib")), fsRoot)
	}
	// <root>/lib/llvm is the real directory on both AMD lines; <root>/llvm
	// is a link to it, and the one a 10.x install can take over.
	if c := filepath.Join(at(i.Root), "lib", "llvm", "bin", "clang++"); isExecutable(c) {
		return strings.TrimPrefix(c, fsRoot)
	}
	return ""
}

// deviceLibDir returns the AMD device bitcode beside compiler, inside the
// compiler's own clang resource directory. Not <root>/amdgcn, which is a
// link a 10.x install can take over. "" when there is none.
func deviceLibDir(fsRoot, compiler string) string {
	llvm := filepath.Dir(filepath.Dir(filepath.Join(fsRoot, compiler)))
	for _, g := range []string{"lib/clang/*/lib/amdgcn/bitcode", "lib/clang/*/amdgcn/bitcode", "amdgcn/bitcode"} {
		if m, _ := filepath.Glob(filepath.Join(llvm, g)); len(m) > 0 {
			return strings.TrimPrefix(m[len(m)-1], fsRoot)
		}
	}
	return ""
}

// versionLess compares dotted version strings numerically; an empty version
// sorts lowest.
func versionLess(a, b string) bool {
	pa, pb := strings.Split(a, "."), strings.Split(b, ".")
	for k := 0; k < len(pa) || k < len(pb); k++ {
		var x, y int
		if k < len(pa) {
			x, _ = strconv.Atoi(pa[k])
		}
		if k < len(pb) {
			y, _ = strconv.Atoi(pb[k])
		}
		if x != y {
			return x < y
		}
	}
	return a == "" && b != ""
}

func fileExists(path string) bool {
	st, err := os.Stat(path)
	return err == nil && !st.IsDir()
}

// pinCMakeFlags returns the cmake flags that build against inst alone, merged
// over flags (the profile's, including any the user typed). Everything comes
// from inst's own directory; each step was needed in testing on a host with
// two installs (plan/rocm-multi-install.md):
//
//   - the compiler and ROCm root, which cmake otherwise takes from hipconfig —
//     the one name most likely to belong to the other install;
//   - for the distro, /usr first on the prefix path, because llama.cpp's
//     ggml-hip falls back to /opt/rocm when ROCM_PATH is unset;
//   - otherwise, inst's headers ahead of /usr/include, which clang searches
//     before the ROCm include directory: with distro HIP headers there, a
//     10.1 build compiles against 7.1's;
//   - otherwise, inst's lib on the binaries' run path. 7.x and 10.x ship the
//     same sonames, so without it a 10.1 build silently loads the distro's
//     7.1 libraries, and 10.x registers nothing with ldconfig.
//
// The flags are recorded on the build, so its info shows what it was built
// against, and two builds of one ref against different installs get distinct
// IDs rather than a replace prompt.
func pinCMakeFlags(flags map[string]string, inst ROCmInstall, compiler string) {
	setDefault := func(k, v string) {
		if _, ok := flags[k]; !ok {
			flags[k] = v
		}
	}
	prepend := func(k, v, sep string) {
		if cur := flags[k]; cur != "" {
			flags[k] = v + sep + cur
		} else {
			flags[k] = v
		}
	}
	setDefault("CMAKE_HIP_COMPILER", compiler)
	setDefault("CMAKE_HIP_COMPILER_ROCM_ROOT", inst.Root)
	if inst.Distro {
		prepend("CMAKE_PREFIX_PATH", "/usr", ";")
		return
	}
	inc := "-isystem " + filepath.Join(inst.Root, "include")
	prepend("CMAKE_HIP_FLAGS", inc, " ")
	prepend("CMAKE_CXX_FLAGS", inc, " ")
	prepend("CMAKE_BUILD_RPATH", filepath.Join(inst.Root, "lib"), ";")
}

// PinROCmFlags merges the flags that build against inst alone into flags (see
// pinCMakeFlags) and returns inst's compiler. ok is false when inst has no HIP
// compiler, and flags is then untouched. Shared by the build and the Builds
// form's flag preview, so the preview shows what will actually run.
func PinROCmFlags(flags map[string]string, inst ROCmInstall) (compiler string, ok bool) {
	compiler = inst.hipCompiler("")
	if compiler == "" {
		return "", false
	}
	pinCMakeFlags(flags, inst, compiler)
	return compiler, true
}

// pinEnv returns env set up to build against inst alone. ROCM_PATH and friends
// are overridden, not defaulted: a value exported around the service names
// some install, and with two of them it is as likely to be the wrong one.
func pinEnv(env []string, inst ROCmInstall, compiler string) []string {
	if inst.Distro {
		// The distro build relies on these being unset (see runBuild): with
		// ROCM_PATH=/usr clang looks for device bitcode where the distro
		// does not keep it, and pointing elsewhere is the mix being avoided.
		return unsetEnv(env, "ROCM_PATH", "HIP_PATH", "HIP_DEVICE_LIB_PATH", "HIP_CLANG_PATH")
	}
	env = prependPath(env, filepath.Join(inst.Root, "bin"))
	env = setEnv(env, "ROCM_PATH", inst.Root)
	env = setEnv(env, "HIP_PATH", inst.Root)
	env = setEnv(env, "HIP_CLANG_PATH", filepath.Dir(compiler))
	if dl := deviceLibDir("", compiler); dl != "" {
		env = setEnv(env, "HIP_DEVICE_LIB_PATH", dl)
	} else {
		env = unsetEnv(env, "HIP_DEVICE_LIB_PATH")
	}
	return env
}

// setEnv returns env with KEY=value, replacing any existing KEY.
func setEnv(env []string, key, value string) []string {
	return append(unsetEnv(env, key), key+"="+value)
}

// unsetEnv returns a copy of env without the named keys.
func unsetEnv(env []string, keys ...string) []string {
	out := make([]string, 0, len(env))
outer:
	for _, kv := range env {
		for _, k := range keys {
			if strings.HasPrefix(kv, k+"=") {
				continue outer
			}
		}
		out = append(out, kv)
	}
	return out
}

// rocmPin is the install a build is pinned to, and its compiler.
type rocmPin struct {
	install  ROCmInstall
	compiler string
}

// chooseROCm picks the install a rocm build uses. root is the Builds form's
// choice, "" for the default (the newest).
//
// pinned is true with more than one install, and for a 10.x install even on
// its own. Otherwise a machine with one install keeps building as it always
// has, so the setups that work today (Arch, Fedora, the 7.2.4 container) are
// unchanged; only the record of which install was used is new. A lone 10.x
// install can't be left to that: its packages point /usr/bin/hipconfig at
// themselves, and the hipconfig-based setup then takes /usr for the ROCm root
// and fails in cmake — measured on Debian 13 and Rocky 10 with 10.1 alone. (The
// ROCm 10 container only escaped because its PATH puts /opt/rocm/bin first.)
// inst is nil when nothing was detected, and the build is left to the
// hipconfig-based setup and fails there as before.
func chooseROCm(installs []ROCmInstall, root string) (inst *ROCmInstall, pinned bool, err error) {
	if root != "" {
		i, ok := FindROCmInstall(installs, root)
		if !ok {
			return nil, false, fmt.Errorf("ROCm install %s not found on this machine", root)
		}
		return &i, len(installs) > 1 || i.isCore(), nil
	}
	if len(installs) == 0 {
		return nil, false, nil
	}
	return &installs[0], len(installs) > 1 || installs[0].isCore(), nil
}

// isCore reports a 10.x install, from AMD's stable.repo.amd.com packages or
// image: one in a core-X.Y directory.
func (i ROCmInstall) isCore() bool {
	return strings.HasPrefix(filepath.Base(i.Root), "core-")
}

// ROCmStamp is BuiltAgainst for a build made with inst:
// "rocm 10.1.0 @ /opt/rocm/core-10.1".
func ROCmStamp(inst ROCmInstall) string {
	v := inst.Version
	if v == "" {
		v = "unknown"
	}
	return "rocm " + v + " @ " + inst.Root
}

// ROCmStampInstall splits a stamp written by ROCmStamp. ok is false for every
// other stamp, including the "rocm 7.2.4" of builds from before installs were
// recorded.
func ROCmStampInstall(stamp string) (version, root string, ok bool) {
	head, root, ok := strings.Cut(stamp, " @ ")
	if !ok {
		return "", "", false
	}
	backend, version, ok := strings.Cut(head, " ")
	if !ok || backend != "rocm" || root == "" {
		return "", "", false
	}
	return version, root, true
}

// InstallGone reports whether a build stamped with its install (see ROCmStamp)
// was made against an install that is no longer here at that version: removed,
// upgraded, or a container image with a different ROCm. applies is false for
// stamps without an install, which StampMismatch still judges.
//
// Nothing detected at all counts as unknown, not gone, matching StampMismatch's
// rule that an unknown side is never reported as a mismatch.
func InstallGone(stamp string, installs []ROCmInstall) (gone, applies bool) {
	version, root, ok := ROCmStampInstall(stamp)
	if !ok {
		return false, false
	}
	if len(installs) == 0 {
		return false, true
	}
	i, found := FindROCmInstall(installs, root)
	return !found || (i.Version != version && version != "unknown"), true
}

var (
	installsMu   sync.Mutex
	installsAt   time.Time
	installsMemo []ROCmInstall
)

// CachedROCmInstalls is DetectROCmInstalls, reused for a few seconds. The
// Builds and Server pages judge every build against it on each render and poll;
// a few seconds stale is harmless, since an install appearing or vanishing is
// rare and the next render catches up.
func CachedROCmInstalls() []ROCmInstall {
	installsMu.Lock()
	defer installsMu.Unlock()
	if installsMemo == nil || time.Since(installsAt) > 5*time.Second {
		installsMemo = DetectROCmInstalls()
		if installsMemo == nil {
			installsMemo = []ROCmInstall{}
		}
		installsAt = time.Now()
	}
	return installsMemo
}
