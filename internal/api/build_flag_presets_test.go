package api

import (
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/builder"
)

func postFlagPreset(t *testing.T, h http.HandlerFunc, form url.Values) string {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/", strings.NewReader(form.Encode()))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	rec := httptest.NewRecorder()
	h(rec, req)
	return rec.Body.String()
}

// Saving records every option of the profile: a checked box as on and a
// missing one as off, so applying the set later does not fall back to the
// defaults for the off ones.
func TestSaveFlagPresetRecordsEveryOption(t *testing.T) {
	s := newTestServer(t)
	opts := builder.ProfileOptions("cpu")
	if len(opts) < 2 {
		t.Fatalf("the cpu profile has %d options; the test needs two", len(opts))
	}
	on, off := opts[0].Flag, opts[1].Flag
	body := postFlagPreset(t, s.handleSaveFlagPreset, url.Values{
		"profile": {"cpu"}, "tag": {"fast"}, "opt_" + on: {"on"}, "extra_cmake": {"  -DX=1 "},
	})
	if !strings.Contains(body, "Saved &#34;fast&#34;.") {
		t.Errorf("no saved message in:\n%s", body)
	}
	p, ok := s.builder.FindFlagPreset(builder.ProjectLlamaCpp, "fast")
	if !ok {
		t.Fatal("the preset was not saved")
	}
	if got, ok := p.Options[on]; !ok || !got {
		t.Errorf("checked option %s = %v, %v; want on", on, got, ok)
	}
	if got, ok := p.Options[off]; !ok || got {
		t.Errorf("unchecked option %s = %v, %v; want recorded as off", off, got, ok)
	}
	if len(p.Options) != len(opts) || p.ExtraCMake != "-DX=1" {
		t.Errorf("preset = %+v; want all %d options and trimmed extra flags", p, len(opts))
	}
}

// Without a Build Tag there is no name to save under.
func TestSaveFlagPresetNeedsATag(t *testing.T) {
	s := newTestServer(t)
	body := postFlagPreset(t, s.handleSaveFlagPreset, url.Values{"profile": {"cpu"}, "tag": {"  "}})
	if !strings.Contains(body, "Enter a Build Tag first") {
		t.Errorf("no tag message in:\n%s", body)
	}
	if len(s.builder.FlagPresets("", "")) != 0 {
		t.Error("a preset was saved without a name")
	}
}

// Delete needs a selected set, says when there is no such set, and
// removes one that exists.
func TestDeleteFlagPresetHandler(t *testing.T) {
	s := newTestServer(t)
	if body := postFlagPreset(t, s.handleDeleteFlagPreset, url.Values{"profile": {"cpu"}}); !strings.Contains(body, "Select a saved set to delete.") {
		t.Errorf("no selection: got\n%s", body)
	}
	if body := postFlagPreset(t, s.handleDeleteFlagPreset, url.Values{"profile": {"cpu"}, "preset": {"none"}}); !strings.Contains(body, "No saved set named &#34;none&#34;.") {
		t.Errorf("unknown set: got\n%s", body)
	}
	if err := s.builder.SaveFlagPreset(builder.FlagPreset{Name: "fast", Profile: "cpu", Options: map[string]bool{}}); err != nil {
		t.Fatal(err)
	}
	if body := postFlagPreset(t, s.handleDeleteFlagPreset, url.Values{"profile": {"cpu"}, "preset": {"fast"}}); !strings.Contains(body, "Deleted &#34;fast&#34;.") {
		t.Errorf("existing set: got\n%s", body)
	}
	if _, ok := s.builder.FindFlagPreset(builder.ProjectLlamaCpp, "fast"); ok {
		t.Error("the set survived the delete")
	}
}

// A set is saved for the chosen project, with that project's toggles,
// and the row lists only that project's sets.
func TestFlagPresetsFollowTheProject(t *testing.T) {
	s := newTestServer(t)
	postFlagPreset(t, s.handleSaveFlagPreset, url.Values{"project": {builder.ProjectSDCpp}, "profile": {"cpu"}, "tag": {"sd-set"}})
	postFlagPreset(t, s.handleSaveFlagPreset, url.Values{"profile": {"cpu"}, "tag": {"llama-set"}})

	sd, ok := s.builder.FindFlagPreset(builder.ProjectSDCpp, "sd-set")
	if !ok {
		t.Fatal("the sd.cpp set was not saved for sd.cpp")
	}
	if want := len(builder.ProjectOptions(builder.ProjectSDCpp, "cpu")); len(sd.Options) != want {
		t.Errorf("sd.cpp set has %d options; want sd.cpp's %d", len(sd.Options), want)
	}
	for flag := range sd.Options {
		if strings.HasPrefix(flag, "LLAMA_") {
			t.Errorf("sd.cpp set holds the llama.cpp toggle %s", flag)
		}
	}
	if _, ok := s.builder.FindFlagPreset(builder.ProjectLlamaCpp, "llama-set"); !ok {
		t.Error("a save with no project choice should be a llama.cpp set")
	}

	row := func(project string) string {
		req := httptest.NewRequest(http.MethodGet, "/?profile=cpu&project="+url.QueryEscape(project), nil)
		rec := httptest.NewRecorder()
		s.handleFlagPresetRow(rec, req)
		return rec.Body.String()
	}
	if body := row(builder.ProjectSDCpp); !strings.Contains(body, `value="sd-set"`) || strings.Contains(body, `value="llama-set"`) {
		t.Errorf("sd.cpp row:\n%s", body)
	}
	if body := row(""); strings.Contains(body, `value="sd-set"`) || !strings.Contains(body, `value="llama-set"`) {
		t.Errorf("llama.cpp row:\n%s", body)
	}
}

// Applying a set by name never loads another project's set: the toggles
// stay the chosen project's own, at their defaults.
func TestApplyFlagPresetRefusesAnotherProject(t *testing.T) {
	s := newTestServer(t)
	if err := s.builder.SaveFlagPreset(builder.FlagPreset{
		Name: "llama-set", Project: builder.ProjectLlamaCpp, Profile: "cpu",
		ExtraCMake: "-DFROM_LLAMA=1", Version: builder.FlagPresetVersion,
	}); err != nil {
		t.Fatal(err)
	}
	get := func(project string) string {
		req := httptest.NewRequest(http.MethodGet, "/?profile=cpu&preset=llama-set&project="+url.QueryEscape(project), nil)
		req.Header.Set("HX-Request", "true")
		rec := httptest.NewRecorder()
		s.handleProfileOptions(rec, req)
		return rec.Body.String()
	}
	if body := get(builder.ProjectSDCpp); strings.Contains(body, "-DFROM_LLAMA=1") {
		t.Errorf("sd.cpp applied the llama.cpp set:\n%s", body)
	}
	if body := get(builder.ProjectLlamaCpp); !strings.Contains(body, "-DFROM_LLAMA=1") {
		t.Errorf("llama.cpp did not apply its own set:\n%s", body)
	}
}
