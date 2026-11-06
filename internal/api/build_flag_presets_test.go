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
	p, ok := s.builder.FindFlagPreset("fast")
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
	if len(s.builder.FlagPresets("")) != 0 {
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
	if _, ok := s.builder.FindFlagPreset("fast"); ok {
		t.Error("the set survived the delete")
	}
}
