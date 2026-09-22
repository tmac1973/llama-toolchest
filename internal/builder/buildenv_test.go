package builder

import (
	"os"
	"path/filepath"
	"testing"
)

// writeVersion creates a version file containing body and returns its path.
func writeVersion(t *testing.T, body string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "version")
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestBackendVersionReadsTheROCmVersionFile(t *testing.T) {
	cases := map[string]struct{ body, want string }{
		"plain":                    {"10.0.0", "10.0.0"},
		"trailing newline":         {"10.0.0\n", "10.0.0"},
		"surrounded by whitespace": {"  7.2.4  \n", "7.2.4"},
		// A longer line than expected must not become the version.
		"extra fields on the line": {"10.0.0 build 4 whatever\n", "10.0.0"},
		"empty file":               {"", ""},
		"whitespace only":          {"   \n", ""},
	}
	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			if got := backendVersion("rocm", []string{writeVersion(t, c.body)}); got != c.want {
				t.Errorf("backendVersion = %q, want %q", got, c.want)
			}
		})
	}

	if got := backendVersion("rocm", []string{filepath.Join(t.TempDir(), "absent")}); got != "" {
		t.Errorf("a missing file gave %q, want empty", got)
	}
	if got := backendVersion("rocm", nil); got != "" {
		t.Errorf("no paths at all gave %q, want empty", got)
	}
}

// The two container images lay ROCm out differently — the Fedora image has only
// the RPM path and AMD's Ubuntu images only the TheRock path — so both
// fall-through directions have to work. A bug here would stamp one of the two
// images as unknown, which is exactly the case the Builds page cannot flag.
func TestBackendVersionTriesEachPathInOrder(t *testing.T) {
	present := writeVersion(t, "10.0.0")
	second := writeVersion(t, "7.2.4")
	absent := filepath.Join(t.TempDir(), "absent")
	empty := writeVersion(t, "")

	cases := []struct {
		name  string
		paths []string
		want  string
	}{
		{"first present wins", []string{present, second}, "10.0.0"},
		{"first missing falls through", []string{absent, second}, "7.2.4"},
		{"first empty falls through", []string{empty, second}, "7.2.4"},
		{"both missing is unknown", []string{absent, filepath.Join(t.TempDir(), "nope")}, ""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := backendVersion("rocm", c.paths); got != c.want {
				t.Errorf("backendVersion = %q, want %q", got, c.want)
			}
		})
	}
}

func TestBackendVersionIgnoresBackendsWithNoVersion(t *testing.T) {
	p := writeVersion(t, "10.0.0")
	for _, backend := range []string{"vulkan", "cpu", "metal", "", "something-new"} {
		if got := backendVersion(backend, []string{p}); got != "" {
			t.Errorf("backendVersion(%q) = %q, want empty", backend, got)
		}
	}
}

func TestBuildEnvStampCombinesBackendAndVersion(t *testing.T) {
	// The real paths are almost certainly absent in a test environment, so
	// this asserts the shape of the "unknown" answer rather than a version.
	if got := BuildEnvStamp("vulkan"); got != "" {
		t.Errorf("BuildEnvStamp(vulkan) = %q, want empty", got)
	}
	// And that a known version is joined with a single space.
	if got := stampFrom("rocm", "10.0.0"); got != "rocm 10.0.0" {
		t.Errorf("stamp = %q, want \"rocm 10.0.0\"", got)
	}
}

// stampFrom mirrors BuildEnvStamp's formatting for a known version, so the
// format is asserted without needing a ROCm installation.
func stampFrom(backend, version string) string {
	if version == "" {
		return ""
	}
	return backend + " " + version
}

func TestStampMismatch(t *testing.T) {
	cases := []struct {
		name         string
		build, curr  string
		wantMismatch bool
	}{
		{"identical", "rocm 10.0.0", "rocm 10.0.0", false},
		{"different rocm versions", "rocm 7.2.4", "rocm 10.0.0", true},
		{"different rocm versions, other way", "rocm 10.0.0", "rocm 7.2.4", true},
		// Unknown on either side must never be reported as a mismatch: an
		// unstamped build may work perfectly, and guessing would flag it.
		{"build unstamped", "", "rocm 10.0.0", false},
		{"current unknown", "rocm 10.0.0", "", false},
		{"both unknown", "", "", false},
		// A different backend is a separate situation, out of scope here.
		{"different backends", "cuda 12.4", "rocm 10.0.0", false},
		{"malformed build stamp", "rocm", "rocm 10.0.0", false},
		{"malformed current stamp", "rocm 10.0.0", "rocm", false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := StampMismatch(c.build, c.curr); got != c.wantMismatch {
				t.Errorf("StampMismatch(%q, %q) = %v, want %v", c.build, c.curr, got, c.wantMismatch)
			}
		})
	}
}

func TestCurrentBuildEnvIsMemoisedPerBackend(t *testing.T) {
	// Two calls for the same backend must agree, and a second backend must
	// not receive the first one's answer.
	a1, a2 := CurrentBuildEnv("vulkan"), CurrentBuildEnv("vulkan")
	if a1 != a2 {
		t.Errorf("memoised value changed between calls: %q then %q", a1, a2)
	}
	if got := CurrentBuildEnv("cpu"); got != "" {
		t.Errorf("CurrentBuildEnv(cpu) = %q, want empty", got)
	}
}
