package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/config"
)

func TestRouterLimits(t *testing.T) {
	const (
		off = config.EvictionOff
		lc  = config.EvictionLlamaCpp
		tc  = config.EvictionToolchest
	)
	cases := []struct {
		name                  string
		max                   int
		eviction              string
		job, hasOpt           bool
		wantMax, wantMarginMB int
		wantMode              memoryMode
	}{
		{"off", 1, off, false, true, 1, 0, memoryCount},
		{"off, benchmark", 3, off, true, true, 3, 0, memoryCount},
		{"llama.cpp for serving", 0, lc, false, true, 0, 1311, memoryRouter},
		{"llama.cpp, max kept", 3, lc, false, true, 3, 1311, memoryRouter},
		{"llama.cpp, benchmark", 0, lc, true, true, 1, 0, memoryCount},
		{"llama.cpp, build too old", 0, lc, false, false, 0, 0, memoryCount},
		{"toolchest", 3, tc, false, true, 0, 0, memoryToolchest},
		{"toolchest, build too old", 3, tc, false, false, 0, 0, memoryToolchest},
		{"toolchest, benchmark", 0, tc, true, true, 1, 0, memoryCount},
	}
	for _, c := range cases {
		gotMax, gotMargin, gotMode := routerLimits(c.max, c.eviction, c.job, c.hasOpt, 1311)
		if gotMax != c.wantMax || gotMargin != c.wantMarginMB || gotMode != c.wantMode {
			t.Errorf("%s: max %d margin %d mode %s, want %d %d %s", c.name, gotMax, gotMargin, gotMode, c.wantMax, c.wantMarginMB, c.wantMode)
		}
	}
}

// putSettings sends a settings form as the Server and Settings pages do.
func putSettings(t *testing.T, s *Server, form url.Values) string {
	t.Helper()
	req := httptest.NewRequest(http.MethodPut, "/api/settings", strings.NewReader(form.Encode()))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("HX-Request", "true")
	rec := httptest.NewRecorder()
	s.handleUpdateSettings(rec, req)
	return rec.Body.String()
}

// putSettingsJSON sends a JSON settings update as an API client does.
func putSettingsJSON(t *testing.T, s *Server, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPut, "/api/settings", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	s.handleUpdateSettings(rec, req)
	return rec
}

// Leaving off with Max Loaded at 1 lifts the count limit. llama.cpp stays
// chosen when generation backends are turned on, and can be chosen while
// they are: toolchest decides then. An unknown mode is refused.
func TestEvictionSetting(t *testing.T) {
	s := newSettingsServer(t, &config.Config{ModelsMax: 1, Eviction: config.EvictionOff})

	putSettings(t, s, url.Values{"eviction": {"llamacpp"}})
	if s.cfg.Eviction != config.EvictionLlamaCpp || s.cfg.ModelsMax != 0 {
		t.Fatalf("llama.cpp: eviction %q, max %d; want llamacpp and 0", s.cfg.Eviction, s.cfg.ModelsMax)
	}
	// The card posts Max loaded models with the mode; a value set there
	// is kept.
	putSettings(t, s, url.Values{"eviction": {"llamacpp"}, "models_max": {"3"}})
	if s.cfg.ModelsMax != 3 {
		t.Errorf("max = %d, want 3", s.cfg.ModelsMax)
	}
	putSettings(t, s, url.Values{"models_max": {"-2"}})
	if s.cfg.ModelsMax != 3 {
		t.Errorf("negative max saved: %d", s.cfg.ModelsMax)
	}

	putSettings(t, s, url.Values{"generation_touched": {"1"}, "generation_enabled": {"on"}})
	if !s.cfg.GenerationEnabled || s.cfg.Eviction != config.EvictionLlamaCpp {
		t.Fatalf("generation on: eviction %q, want llamacpp still", s.cfg.Eviction)
	}
	if e := s.cfg.EffectiveEviction(); e != config.EvictionToolchest {
		t.Errorf("generation on: effective %q, want toolchest", e)
	}

	putSettings(t, s, url.Values{"eviction": {"off"}, "models_max": {"1"}})
	body := putSettings(t, s, url.Values{"eviction": {"toolchest"}, "models_max": {"1"}})
	if s.cfg.Eviction != config.EvictionToolchest || s.cfg.ModelsMax != 0 || strings.Contains(body, "Settings not saved") {
		t.Errorf("toolchest: eviction %q, max %d, answer %q; want toolchest and 0", s.cfg.Eviction, s.cfg.ModelsMax, body)
	}

	req := httptest.NewRequest(http.MethodPut, "/api/settings", strings.NewReader(url.Values{"eviction": {"sideways"}}.Encode()))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	rec := httptest.NewRecorder()
	s.handleUpdateSettings(rec, req)
	if rec.Code != http.StatusBadRequest || s.cfg.Eviction != config.EvictionToolchest {
		t.Errorf("unknown mode: %d, eviction %q; want 400 and unchanged", rec.Code, s.cfg.Eviction)
	}
}

// The JSON API takes eviction, and still the older router_memory switch.
func TestEvictionSettingJSON(t *testing.T) {
	s := newSettingsServer(t, &config.Config{Eviction: config.EvictionOff})
	if rec := putSettingsJSON(t, s, `{"eviction":"toolchest","models_max":2}`); rec.Code != http.StatusOK ||
		s.cfg.Eviction != config.EvictionToolchest || s.cfg.ModelsMax != 2 {
		t.Fatalf("eviction: %d %s; eviction %q max %d", rec.Code, rec.Body, s.cfg.Eviction, s.cfg.ModelsMax)
	}
	var got settingsResponse
	if err := json.Unmarshal(putSettingsJSON(t, s, `{"router_memory":false}`).Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if s.cfg.Eviction != config.EvictionOff || got.Eviction != config.EvictionOff {
		t.Errorf("router_memory false: eviction %q, answer %q; want off", s.cfg.Eviction, got.Eviction)
	}
	putSettingsJSON(t, s, `{"router_memory":true}`)
	if s.cfg.Eviction != config.EvictionLlamaCpp {
		t.Errorf("router_memory true: eviction %q, want llamacpp", s.cfg.Eviction)
	}
	if rec := putSettingsJSON(t, s, `{"models_max":-1}`); rec.Code != http.StatusBadRequest {
		t.Errorf("negative models_max: %d, want 400", rec.Code)
	}
}

// The view says why llama.cpp cannot decide with backends or forks on,
// that it gave way when it was chosen, and the router's margin only when
// the router decides.
func TestEvictionInfo(t *testing.T) {
	s := newTestServer(t)
	s.cfg.Eviction = config.EvictionLlamaCpp
	if v := s.evictionInfo(); v.Checked != config.EvictionLlamaCpp || v.LlamaCppBlocked != "" || v.GaveWay || v.Restart {
		t.Errorf("generation off, router stopped: %+v", v)
	}
	s.cfg.GenerationEnabled = true
	v := s.evictionInfo()
	if v.Checked != config.EvictionToolchest || !v.GaveWay || v.Margin != "" || v.Note != "" ||
		v.LlamaCppBlocked != "Not with generation backends on" || v.Label != "llama-toolchest" {
		t.Errorf("generation on: %+v; want toolchest checked, gave way, no margin", v)
	}
	s.cfg.ForksEnabled = true
	if v := s.evictionInfo(); v.LlamaCppBlocked != "Not with generation backends or forks on" {
		t.Errorf("both on: %q", v.LlamaCppBlocked)
	}
	s.cfg.Eviction = config.EvictionToolchest
	if v := s.evictionInfo(); v.GaveWay {
		t.Error("toolchest chosen: says llama.cpp gave way")
	}
}

// With llama.cpp chosen and generation on, the router starts in toolchest
// mode: no count limit and no margin of its own. With generation off the
// router decides again, and the view asks for a restart until it does.
func TestStartRouterEvictionModes(t *testing.T) {
	s, proc := evalStopServer(t)
	s.builder = testBuilder(t, testBuild("b1", fakeLlamaServerDir(t)))
	s.cfg.Eviction, s.cfg.GenerationEnabled, s.cfg.ModelsMax = config.EvictionLlamaCpp, true, 3
	if err := s.startRouterWith(routerOptions{}); err != nil {
		t.Fatal(err)
	}
	waitRouterRunning(t, proc)
	if !s.toolchestMemory() {
		t.Errorf("mode = %s, want toolchest", memoryMode(s.memoryMode.Load()))
	}
	if v := s.evictionInfo(); v.Restart || !v.Running {
		t.Errorf("running in its mode: %+v", v)
	}
	s.cfg.GenerationEnabled = false
	if v := s.evictionInfo(); !v.Restart {
		t.Error("no restart asked for after generation was turned off")
	}
	if err := proc.Stop(); err != nil {
		t.Fatal(err)
	}
	s.cfg.Eviction = config.EvictionOff
	if err := s.startRouterWith(routerOptions{}); err != nil {
		t.Fatal(err)
	}
	waitRouterRunning(t, proc)
	if mode := memoryMode(s.memoryMode.Load()); mode != memoryCount {
		t.Errorf("eviction off: mode = %s, want count", mode)
	}
	if err := proc.Stop(); err != nil {
		t.Fatal(err)
	}
	// Toolchest decides also with generation and forks off.
	s.cfg.Eviction = config.EvictionToolchest
	if err := s.startRouterWith(routerOptions{}); err != nil {
		t.Fatal(err)
	}
	waitRouterRunning(t, proc)
	if !s.toolchestMemory() {
		t.Errorf("toolchest with generation off: mode = %s, want toolchest", memoryMode(s.memoryMode.Load()))
	}
}

// The Server page has a line with the mode in effect and a link to the
// Settings card, in place of the old controls.
func TestServerPageEvictionLine(t *testing.T) {
	s := newTestServer(t)
	s.cfg.Eviction, s.cfg.GenerationEnabled, s.cfg.ModelsMax = config.EvictionLlamaCpp, true, 0
	rec := httptest.NewRecorder()
	s.handleServerPage(rec, httptest.NewRequest(http.MethodGet, "/server", nil))
	body := rec.Body.String()
	for _, want := range []string{"Eviction: llama-toolchest", "Max loaded models: no limit", `href="/settings#eviction"`} {
		if !strings.Contains(body, want) {
			t.Errorf("Server page is missing %q", want)
		}
	}
	for _, gone := range []string{`name="models_max"`, `name="router_memory"`} {
		if strings.Contains(body, gone) {
			t.Errorf("Server page still has %s", gone)
		}
	}
}

// The card checks the mode in effect, disables llama.cpp with the reason,
// and shows the take-turns switches only under llama-toolchest with
// generation on; the prompt cache switch only with taking turns on.
func TestEvictionCardRenders(t *testing.T) {
	s := newTestServer(t)
	render := func() string {
		rec := httptest.NewRecorder()
		s.handleEvictionCard(rec, httptest.NewRequest(http.MethodGet, "/api/settings/eviction", nil))
		return rec.Body.String()
	}
	radio := func(page, value string) string {
		i := strings.Index(page, `value="`+value+`"`)
		if i < 0 {
			t.Fatalf("radio %s missing", value)
		}
		return page[i : i+strings.IndexByte(page[i:], '>')]
	}

	s.cfg.Eviction = config.EvictionOff
	page := render()
	if !strings.Contains(radio(page, "off"), "checked") || strings.Contains(radio(page, "llamacpp"), "disabled") {
		t.Error("off, generation off: off not checked or llama.cpp disabled")
	}
	if strings.Contains(page, `name="gen_evict"`) {
		t.Error("take-turns switch shown with generation off")
	}

	s.cfg.Eviction, s.cfg.GenerationEnabled = config.EvictionLlamaCpp, true
	page = render()
	if !strings.Contains(radio(page, "toolchest"), "checked") || !strings.Contains(radio(page, "llamacpp"), "disabled") ||
		!strings.Contains(page, "Not with generation backends on") || !strings.Contains(page, "You chose llama.cpp") {
		t.Errorf("llama.cpp with generation on: card wrong:\n%s", page)
	}
	if !strings.Contains(page, `name="gen_evict"`) || strings.Contains(page, `name="gen_keep_prompt_cache"`) {
		t.Error("take turns off: want its switch, no prompt cache switch")
	}
	// The switches are in a form of their own: a change there must not
	// send the checked radio, which would replace the saved llama.cpp.
	forms := strings.Split(page, "<form")
	if len(forms) != 3 || !strings.Contains(forms[1], `name="eviction"`) || strings.Contains(forms[2], `name="eviction"`) ||
		!strings.Contains(forms[2], `name="gen_evict"`) || !strings.Contains(forms[2], `name="models_max"`) {
		t.Error("the mode radios and the other fields are not in separate forms")
	}
	s.cfg.GenEvict = true
	if page = render(); !strings.Contains(page, `name="gen_keep_prompt_cache"`) {
		t.Error("take turns on: prompt cache switch missing")
	}
	if strings.Contains(page, "Restart server") {
		t.Error("Restart button shown with the router stopped")
	}
}

// A saved setting tells the Eviction card to fetch itself again.
func TestSettingsSaveTriggersCardRefresh(t *testing.T) {
	s := newSettingsServer(t, &config.Config{Eviction: config.EvictionOff})
	req := httptest.NewRequest(http.MethodPut, "/api/settings", strings.NewReader(url.Values{"eviction": {"toolchest"}}.Encode()))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("HX-Request", "true")
	rec := httptest.NewRecorder()
	s.handleUpdateSettings(rec, req)
	if got := rec.Header().Get("HX-Trigger"); got != "settingsSaved" {
		t.Errorf("HX-Trigger = %q, want settingsSaved", got)
	}
}
