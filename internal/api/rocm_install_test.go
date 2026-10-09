package api

import (
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/builder"
)

var twoInstalls = []builder.ROCmInstall{
	{Root: "/opt/rocm/core-10.1", Version: "10.1.0"},
	{Root: "/opt/rocm-7.2.4", Version: "7.2.4", Damaged: "llvm now points into another ROCm install."},
}

func renderBuildOptions(t *testing.T, s *Server, query string) string {
	t.Helper()
	req := httptest.NewRequest("GET", "/api/builds/options?"+query, nil)
	req.Header.Set("HX-Request", "true")
	w := httptest.NewRecorder()
	s.handleProfileOptions(w, req)
	return w.Body.String()
}

// The picker appears only for a rocm build on a machine with a choice.
func TestBuildOptionsROCmPickerOnlyWithAChoice(t *testing.T) {
	s := newTestServer(t)
	if out := renderBuildOptions(t, s, "profile=rocm"); strings.Contains(out, `name="rocm_root"`) {
		t.Error("picker shown with no installs")
	}
	s.rocmInstallsFn = func() []builder.ROCmInstall { return twoInstalls[:1] }
	if out := renderBuildOptions(t, s, "profile=rocm"); strings.Contains(out, `name="rocm_root"`) {
		t.Error("picker shown with one install")
	}
	s.rocmInstallsFn = func() []builder.ROCmInstall { return twoInstalls }
	if out := renderBuildOptions(t, s, "profile=cuda"); strings.Contains(out, `name="rocm_root"`) {
		t.Error("picker shown for a cuda build")
	}
	out := renderBuildOptions(t, s, "profile=rocm")
	for _, want := range []string{`name="rocm_root"`, `value="/opt/rocm/core-10.1" selected`, "ROCm 7.2.4 (/opt/rocm-7.2.4)"} {
		if !strings.Contains(out, want) {
			t.Errorf("picker lacks %q:\n%s", want, out)
		}
	}
	// The re-fetch must also include the picker, or a toggle change would
	// reset the choice.
	if !strings.Contains(out, "#build-options select") {
		t.Error("the options re-fetch does not send the picker's value")
	}
	if strings.Contains(out, `role="alert"`) {
		t.Error("warning shown for an undamaged install")
	}
}

// Choosing the damaged install keeps it selected and says what is wrong.
func TestBuildOptionsROCmPickerWarnsOnDamagedInstall(t *testing.T) {
	s := newTestServer(t)
	s.rocmInstallsFn = func() []builder.ROCmInstall { return twoInstalls }
	out := renderBuildOptions(t, s, "profile=rocm&rocm_root=/opt/rocm-7.2.4")
	if !strings.Contains(out, `value="/opt/rocm-7.2.4" selected`) {
		t.Errorf("choice not kept:\n%s", out)
	}
	if !strings.Contains(out, "llvm now points into another ROCm install.") {
		t.Errorf("no warning for the damaged install:\n%s", out)
	}
}

// A build stamped with its install is judged by whether that install is
// still here, not by comparing against one "current" version.
func TestBuildRowJudgesInstallStamps(t *testing.T) {
	s := newTestServer(t)
	s.currentBuildEnv = func(string) string { return "rocm 10.1.0" }
	b := &builder.BuildResult{ID: "x", Profile: "rocm", BuiltAgainst: "rocm 7.2.4 @ /opt/rocm-7.2.4"}

	s.rocmInstallsFn = func() []builder.ROCmInstall { return twoInstalls }
	if row := s.buildRowFor(b); row.Mismatch {
		t.Errorf("a build for an install still here is flagged: %+v", row)
	}

	s.rocmInstallsFn = func() []builder.ROCmInstall { return twoInstalls[:1] }
	row := s.buildRowFor(b)
	if !row.Mismatch || !strings.Contains(row.BuiltAgainstTitle, "no longer on this machine") {
		t.Errorf("a build for a removed install is not flagged: %+v", row)
	}

	// Nothing detected is unknown, never a mismatch.
	s.rocmInstallsFn = func() []builder.ROCmInstall { return nil }
	if row := s.buildRowFor(b); row.Mismatch {
		t.Errorf("flagged with no installs detected: %+v", row)
	}
}
