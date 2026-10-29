package api

import (
	"context"
	"encoding/json"
	"net/url"
	"slices"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/builder"
	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/testutil"
)

// The levels come from the loaded model's /props, asked with
// autoload=false so a model unloaded in the meantime is not loaded again.
func TestReadReasoningLevels(t *testing.T) {
	s, f, m := startFakeRouter(t)
	routerName := s.registry.RouterName(m.ID)
	cases := []struct {
		name, props string
		levels      []string
		known, err  bool
	}{
		{"levels", `{"chat_template_caps":{"supports_reasoning_effort":true,"reasoning_effort_levels":["low","medium","xhigh"]}}`, []string{"low", "medium", "xhigh"}, true, false},
		{"none accepted", `{"chat_template_caps":{"reasoning_effort_levels":[]}}`, []string{}, true, false},
		{"build too old to say", `{"chat_template_caps":{"supports_reasoning_effort":false}}`, nil, false, false},
		{"no caps", `{"model_alias":"x"}`, nil, false, true},
		{"not answered", ``, nil, false, true},
	}
	for _, c := range cases {
		f.mu.Lock()
		f.props, f.propsQueries = c.props, nil
		f.mu.Unlock()
		levels, known, err := s.readReasoningLevels(context.Background(), routerName)
		if (err != nil) != c.err || known != c.known || !slices.Equal(levels, c.levels) || (c.levels != nil && levels == nil) {
			t.Errorf("%s: levels %#v known %v err %v", c.name, levels, known, err)
		}
		f.mu.Lock()
		q := f.propsQueries
		f.mu.Unlock()
		if len(q) != 1 {
			t.Fatalf("%s: %d /props calls", c.name, len(q))
		}
		vals, _ := url.ParseQuery(q[0])
		if vals.Get("model") != routerName || vals.Get("autoload") != "false" {
			t.Errorf("%s: query %q", c.name, q[0])
		}
	}
}

// Once a model is seen loaded, its levels are read and saved with the
// active build, and not read again on that build.
func TestNoteModelLoadedSavesLevels(t *testing.T) {
	s, f, m := startFakeRouter(t)
	s.builder = testBuilder(t, builder.BuildResult{ID: "b1", Status: builder.BuildStatusSuccess, BinaryPath: "/x/llama-server"})
	s.cfg.ActiveBuild = "b1"
	f.mu.Lock()
	f.props = `{"chat_template_caps":{"reasoning_effort_levels":["high","max"]}}`
	f.mu.Unlock()

	reg, _ := s.registry.Get(m.ID)
	s.noteModelLoaded(reg)
	testutil.Eventually(t, func() bool {
		got, _ := s.registry.Get(m.ID)
		return got.ReasoningLevelsBuild == "b1"
	}, "the levels were not saved")
	got, _ := s.registry.Get(m.ID)
	if !slices.Equal(got.EffortLevels(), []string{"high", "max"}) {
		t.Errorf("levels %v", got.EffortLevels())
	}

	s.noteModelLoaded(got)
	f.mu.Lock()
	calls := len(f.propsQueries)
	f.mu.Unlock()
	if calls != 1 {
		t.Errorf("/props asked %d times, want once per build", calls)
	}
}

// A saved default effort fills in requests that do not choose their own
// reasoning: not when the request sets reasoning_effort or
// chat_template_kwargs.
func TestInjectReasoningEffortDefault(t *testing.T) {
	s := newTestServer(t)
	addProxyModel(t, s, models.ModelConfig{Enabled: true, ReasoningEffort: "low"})
	for _, c := range []struct {
		name, extra string
		want        any
	}{
		{"added", ``, "low"},
		{"client's level kept", `,"reasoning_effort":"high"`, "high"},
		{"left out beside chat_template_kwargs", `,"chat_template_kwargs":{"enable_thinking":false}`, nil},
	} {
		body := `{"model":"` + proxyModelID + `","messages":[]` + c.extra + `}`
		var got map[string]any
		if err := json.Unmarshal(s.injectSamplingDefaults([]byte(body)), &got); err != nil {
			t.Fatal(err)
		}
		if got["reasoning_effort"] != c.want {
			t.Errorf("%s: reasoning_effort = %v, want %v", c.name, got["reasoning_effort"], c.want)
		}
	}
}

// The dialog saves only a level the template accepts, or the template
// default, and a form without the dropdown keeps the saved value.
func TestSaveReasoningEffort(t *testing.T) {
	s := newProfileServer(t, "")
	cfg, _ := s.registry.GetConfig(profTestID)
	cfg.ReasoningOverride = &models.ReasoningCapability{Supported: true, Toggle: models.ReasoningToggleReasoningEffort}
	if err := s.registry.SetConfig(profTestID, cfg); err != nil {
		t.Fatal(err)
	}
	if err := s.registry.SetReasoningLevels(profTestID, []string{"low", "medium", "xhigh"}, true, "b1"); err != nil {
		t.Fatal(err)
	}
	save := func(form url.Values) string {
		t.Helper()
		if code, body := s.doProfileRequest(t, "PUT", "/api/models/"+profTestID+"/config", form); code != 200 {
			t.Fatalf("save: %d %s", code, body)
		}
		cfg, _ := s.registry.GetConfig(profTestID)
		return cfg.ReasoningEffort
	}
	if got := save(url.Values{"reasoning_effort": {"medium"}}); got != "medium" {
		t.Errorf("a listed level: %q", got)
	}
	if got := save(url.Values{"gpu_layers": {"0"}}); got != "medium" {
		t.Errorf("a form without the dropdown changed it: %q", got)
	}
	if got := save(url.Values{"reasoning_effort": {"bogus"}}); got != "" {
		t.Errorf("an unlisted level was saved: %q", got)
	}

	// The dialog lists the levels.
	_, body := s.doProfileRequest(t, "GET", "/api/models/"+profTestID+"/config", nil)
	if !strings.Contains(body, "Default Reasoning Effort") || !strings.Contains(body, `<option value="xhigh"`) {
		t.Errorf("the dialog does not list the levels")
	}
}
