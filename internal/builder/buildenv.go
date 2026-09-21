package builder

import (
	"os"
	"os/exec"
	"regexp"
	"strings"
	"sync"
)

// rocmVersionPaths are the files that state the installed ROCm release, in the
// order they are tried. Two, because the two container images this project
// builds lay ROCm out differently:
//
//   - /opt/rocm/.info/version is the RPM layout, used by the Fedora image
//     (Dockerfile.rocm). It reads "7.2.4" there, and does not exist in the
//     AMD-published Ubuntu images.
//   - /opt/rocm/core/.info/version is the layout AMD's TheRock-built images use
//     (Dockerfile.rocm-next). It reads "10.0.0" on the ROCm 10 image and
//     "7.14.1" on the 7.14 one, and does not exist in the Fedora image.
//     /opt/rocm/core is a symlink through /etc/alternatives, so the path names
//     no release and keeps working across them.
//
// Deliberately NOT a fallback: `hipconfig --version`. It reports HIP's own
// component version, which is 7.15.x on ROCm 10.0.0 — a build stamped from it
// would claim to be a 7.x ROCm and make the Builds page's mismatch comparison
// wrong in the most confusing way available. It happens to agree with the
// release on the 7.14 image, so using it would have tested clean there and been
// silently wrong on ROCm 10.
var rocmVersionPaths = []string{
	"/opt/rocm/.info/version",
	"/opt/rocm/core/.info/version",
}

// nvccRelease matches the version in `nvcc --version`, e.g. "release 12.4, V…".
var nvccRelease = regexp.MustCompile(`release ([0-9]+\.[0-9]+)`)

// backendVersion reports the version of a GPU toolchain as installed here, or
// "" when it cannot be determined. rocmPaths is taken as an argument so the
// file reading can be tested without a ROCm installation.
//
// Any failure returns "": an unreadable toolchain must never fail a build, and
// "unknown" is a state the callers already handle.
func backendVersion(backend string, rocmPaths []string) string {
	switch backend {
	case "rocm":
		for _, p := range rocmPaths {
			b, err := os.ReadFile(p)
			if err != nil {
				continue
			}
			// First whitespace-separated field, so a longer line than
			// expected cannot become the version.
			if v := strings.Fields(string(b)); len(v) > 0 {
				return v[0]
			}
		}
		return ""
	case "cuda":
		out, err := exec.Command("nvcc", "--version").Output()
		if err != nil {
			return ""
		}
		if m := nvccRelease.FindSubmatch(out); m != nil {
			return string(m[1])
		}
		return ""
	default:
		// vulkan, cpu and anything else: no version worth comparing.
		return ""
	}
}

// BuildEnvStamp is what to record on a build made here, as
// "<backend> <version>" — e.g. "rocm 10.0.0". Empty when the backend has no
// version to read.
func BuildEnvStamp(backend string) string {
	v := backendVersion(backend, rocmVersionPaths)
	if v == "" {
		return ""
	}
	return backend + " " + v
}

var (
	currentEnvMu sync.Mutex
	currentEnv   = map[string]string{}
)

// CurrentBuildEnv is BuildEnvStamp for the running process, memoised per
// backend. Neither version file can change while the process lives, and the
// CUDA case shells out to nvcc, so this is worth not repeating per page render.
//
// A map guarded by a mutex rather than sync.Once, which cannot be keyed.
func CurrentBuildEnv(backend string) string {
	currentEnvMu.Lock()
	defer currentEnvMu.Unlock()
	if v, ok := currentEnv[backend]; ok {
		return v
	}
	v := BuildEnvStamp(backend)
	currentEnv[backend] = v
	return v
}

// StampMismatch reports whether a build was compiled against a different
// version of the same backend than the one running now — the case where a
// llama-server will not load, because it is linked against libraries that are
// no longer there.
//
// False whenever either side is unknown: a build from before stamping existed
// must not be reported as mismatched, because it may well be fine. Also false
// across different backends — a CUDA build listed on a ROCm host is a separate
// situation, and flagging it here would be noise.
func StampMismatch(buildStamp, currentStamp string) bool {
	if buildStamp == "" || currentStamp == "" {
		return false
	}
	bBackend, bVersion, okB := strings.Cut(buildStamp, " ")
	cBackend, cVersion, okC := strings.Cut(currentStamp, " ")
	if !okB || !okC || bBackend != cBackend {
		return false
	}
	return bVersion != cVersion
}
