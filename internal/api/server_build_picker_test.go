package api

import (
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/builder"
	"github.com/tmac1973/llama-toolchest/internal/config"
	"github.com/tmac1973/llama-toolchest/internal/models"
)

// pickerServer is a Server whose builder holds the given builds, with a fixed
// idea of the running toolchain and a chosen active build.
func pickerServer(t *testing.T, current, active string, builds []builder.BuildResult) *Server {
	t.Helper()
	s := stampServer(t, current, builds)
	s.cfg = &config.Config{ActiveBuild: active}
	return s
}

func serverPage(t *testing.T, s *Server) string {
	t.Helper()
	rec := httptest.NewRecorder()
	s.handleServerPage(rec, httptest.NewRequest("GET", "/server", nil))
	if rec.Code != 200 {
		t.Fatalf("server page: HTTP %d — %s", rec.Code, rec.Body.String())
	}
	return rec.Body.String()
}

// optionFor returns the <option> element for a build id.
func optionFor(t *testing.T, page, id string) string {
	t.Helper()
	for _, chunk := range strings.Split(page, "<option")[1:] {
		end := strings.Index(chunk, "</option>")
		if end < 0 {
			continue
		}
		if strings.Contains(chunk[:end], ">"+id+" (") || strings.Contains(chunk[:end], `value="`+id+`"`) {
			return chunk[:end]
		}
	}
	t.Fatalf("no <option> for %q in:\n%s", id, page)
	return ""
}

// A build compiled in another image is struck through and unselectable; one
// that matches is left alone. This is the case that produced a loader error
// with nothing on screen explaining it.
func TestServerPickerDisablesBuildsFromAnotherImage(t *testing.T) {
	s := pickerServer(t, "rocm 10.0.0", "", []builder.BuildResult{
		{ID: "b-here", Profile: "rocm", GitSHA: "aaaaaaaaaa", GitRef: "v0.4.1", Status: builder.BuildStatusSuccess, BuiltAgainst: "rocm 10.0.0"},
		{ID: "b-elsewhere", Profile: "rocm", GitSHA: "bbbbbbbbbb", GitRef: "v0.4.1", Status: builder.BuildStatusSuccess, BuiltAgainst: "rocm 7.2.4"},
	})
	page := serverPage(t, s)

	here := optionFor(t, page, "b-here")
	if strings.Contains(here, "disabled") || strings.Contains(here, "line-through") {
		t.Errorf("a matching build was refused:\n%s", here)
	}

	elsewhere := optionFor(t, page, "b-elsewhere")
	if !strings.Contains(elsewhere, "disabled") {
		t.Errorf("a build from another image is selectable:\n%s", elsewhere)
	}
	if !strings.Contains(elsewhere, "line-through") {
		t.Errorf("a build from another image is not struck through:\n%s", elsewhere)
	}
	if !strings.Contains(elsewhere, "rocm 7.2.4") {
		t.Errorf("the option does not say what it was built against:\n%s", elsewhere)
	}

	// The reason must be readable without hovering, because an <option>
	// title is not shown by every browser.
	if !strings.Contains(page, "Struck-through builds were compiled in a different container image") {
		t.Errorf("no visible explanation under the picker:\n%s", page)
	}
	if !strings.Contains(page, "rocm 10.0.0") {
		t.Errorf("the explanation does not name what is running:\n%s", page)
	}
}

// Refusing every build would leave nothing that can be started, which is worse
// than letting one be tried. They stay selectable and the hint says so.
func TestServerPickerKeepsBuildsWhenAllOfThemMismatch(t *testing.T) {
	s := pickerServer(t, "rocm 10.0.0", "", []builder.BuildResult{
		{ID: "b-one", Profile: "rocm", GitSHA: "aaaaaaaaaa", GitRef: "v0.4.1", Status: builder.BuildStatusSuccess, BuiltAgainst: "rocm 7.2.4"},
		{ID: "b-two", Profile: "rocm", GitSHA: "bbbbbbbbbb", GitRef: "v0.4.0", Status: builder.BuildStatusSuccess, BuiltAgainst: "rocm 7.2.4"},
	})
	page := serverPage(t, s)

	for _, id := range []string{"b-one", "b-two"} {
		if opt := optionFor(t, page, id); strings.Contains(opt, "disabled") {
			t.Errorf("%s was refused although every build mismatches:\n%s", id, opt)
		}
	}
	if !strings.Contains(page, "would leave nothing to start") {
		t.Errorf("the hint does not explain why they are still selectable:\n%s", page)
	}
}

// A build with no stamp cannot be judged, so it must never be refused.
func TestServerPickerNeverRefusesAnUnstampedBuild(t *testing.T) {
	s := pickerServer(t, "rocm 10.0.0", "", []builder.BuildResult{
		{ID: "b-stamped", Profile: "rocm", GitSHA: "aaaaaaaaaa", GitRef: "v0.4.1", Status: builder.BuildStatusSuccess, BuiltAgainst: "rocm 7.2.4"},
		{ID: "b10453-legacy", Profile: "rocm", GitSHA: "cccccccccc", GitRef: "b10453", Status: builder.BuildStatusSuccess},
	})
	page := serverPage(t, s)

	legacy := optionFor(t, page, "b10453-legacy")
	if strings.Contains(legacy, "disabled") || strings.Contains(legacy, "line-through") {
		t.Errorf("an unstamped build was refused:\n%s", legacy)
	}
	// The stamped mismatch alongside it is still refused.
	if !strings.Contains(optionFor(t, page, "b-stamped"), "disabled") {
		t.Error("the stamped mismatch should still be refused")
	}
}

// A disabled selected <option> cannot be submitted, so the form would post a
// different value on the next change. The active build is therefore never
// refused, however badly it matches.
func TestServerPickerNeverRefusesTheSelectedBuild(t *testing.T) {
	s := pickerServer(t, "rocm 10.0.0", "b-chosen", []builder.BuildResult{
		{ID: "b-chosen", Profile: "rocm", GitSHA: "aaaaaaaaaa", GitRef: "v0.4.1", Status: builder.BuildStatusSuccess, BuiltAgainst: "rocm 7.2.4"},
		{ID: "b-other", Profile: "rocm", GitSHA: "bbbbbbbbbb", GitRef: "v0.4.0", Status: builder.BuildStatusSuccess, BuiltAgainst: "rocm 7.2.4"},
	})
	page := serverPage(t, s)
	chosen := optionFor(t, page, "b-chosen")
	if strings.Contains(chosen, "disabled") {
		t.Errorf("the selected build was refused, which breaks the form:\n%s", chosen)
	}
	// It is still marked, so the user can see the problem.
	if !strings.Contains(chosen, "line-through") {
		t.Errorf("the selected mismatch is not marked at all:\n%s", chosen)
	}
}

// "Auto (newest ref)" looks like the safe choice; if it resolves to a build
// that cannot run here, that has to be said on the option itself.
func TestServerPickerWarnsWhenAutoPicksABadBuild(t *testing.T) {
	s := pickerServer(t, "rocm 10.0.0", "", []builder.BuildResult{
		{ID: "b-elsewhere", Profile: "rocm", GitSHA: "bbbbbbbbbb", GitRef: "v0.4.1", Status: builder.BuildStatusSuccess, BuiltAgainst: "rocm 7.2.4"},
	})
	if page := serverPage(t, s); !strings.Contains(page, "picks a build that cannot run here") {
		t.Errorf("Auto does not warn that it would pick an unusable build:\n%s", page)
	}

	// And stays quiet when Auto is fine.
	s = pickerServer(t, "rocm 10.0.0", "", []builder.BuildResult{
		{ID: "b-here", Profile: "rocm", GitSHA: "aaaaaaaaaa", GitRef: "v0.4.1", Status: builder.BuildStatusSuccess, BuiltAgainst: "rocm 10.0.0"},
	})
	if page := serverPage(t, s); strings.Contains(page, "picks a build that cannot run here") {
		t.Errorf("Auto warned when the build it picks is fine:\n%s", page)
	}
}

// With no builds at all the page must still render, and say nothing.
func TestServerPickerWithNoBuilds(t *testing.T) {
	s := pickerServer(t, "rocm 10.0.0", "", nil)
	page := serverPage(t, s)
	if strings.Contains(page, "Struck-through builds") || strings.Contains(page, "cannot run here") {
		t.Errorf("the page warned about builds that do not exist:\n%s", page)
	}
	if !strings.Contains(page, "Auto (newest ref)") {
		t.Errorf("the picker did not render:\n%s", page)
	}
}

var _ = models.NewRegistry
