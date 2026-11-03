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
	cases := []struct {
		name                     string
		max                      int
		on, outside, job, hasOpt bool
		wantMax, wantMarginMB    int
		wantMode                 memoryMode
	}{
		{"off", 1, false, false, false, true, 1, 0, memoryCount},
		{"off, generation on", 1, false, true, false, true, 1, 0, memoryCount},
		{"on for serving", 0, true, false, false, true, 0, 1311, memoryRouter},
		{"on, max kept", 3, true, false, false, true, 3, 1311, memoryRouter},
		{"on, benchmark", 0, true, false, true, true, 1, 0, memoryCount},
		{"on, benchmark, generation on", 0, true, true, true, true, 1, 0, memoryCount},
		{"on, build too old", 0, true, false, false, false, 0, 0, memoryCount},
		{"on, generation on", 3, true, true, false, true, 0, 0, memoryToolchest},
		{"on, generation on, build too old", 3, true, true, false, false, 0, 0, memoryToolchest},
	}
	for _, c := range cases {
		gotMax, gotMargin, gotMode := routerLimits(c.max, c.on, c.outside, c.job, c.hasOpt, 1311)
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

// Turning the option on with Max Loaded at 1 lifts the count limit. It
// stays on when generation backends are turned on, and can be turned on
// while they are: toolchest decides then.
func TestRouterMemorySetting(t *testing.T) {
	s := newSettingsServer(t, &config.Config{ModelsMax: 1})

	putSettings(t, s, url.Values{"router_memory_touched": {"1"}, "router_memory": {"on"}})
	if !s.cfg.RouterMemory || s.cfg.ModelsMax != 0 {
		t.Fatalf("on: router %v, max %d; want on and 0", s.cfg.RouterMemory, s.cfg.ModelsMax)
	}
	// A later form from the Server page keeps a Max Loaded the user set.
	putSettings(t, s, url.Values{"router_memory_touched": {"1"}, "router_memory": {"on"}, "models_max": {"3"}})
	if s.cfg.ModelsMax != 3 {
		t.Errorf("max = %d, want 3", s.cfg.ModelsMax)
	}

	putSettings(t, s, url.Values{"generation_touched": {"1"}, "generation_enabled": {"on"}})
	if !s.cfg.GenerationEnabled || !s.cfg.RouterMemory {
		t.Fatalf("generation on: router %v, want still on", s.cfg.RouterMemory)
	}

	putSettings(t, s, url.Values{"router_memory_touched": {"1"}, "models_max": {"3"}})
	if s.cfg.RouterMemory {
		t.Fatal("not turned off")
	}
	body := putSettings(t, s, url.Values{"router_memory_touched": {"1"}, "router_memory": {"on"}, "models_max": {"1"}})
	if !s.cfg.RouterMemory || s.cfg.ModelsMax != 0 || strings.Contains(body, "Settings not saved") {
		t.Errorf("on with generation: router %v, max %d, answer %q; want on and 0", s.cfg.RouterMemory, s.cfg.ModelsMax, body)
	}

	req := httptest.NewRequest(http.MethodPut, "/api/settings", strings.NewReader(`{"router_memory":false}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	s.handleUpdateSettings(rec, req)
	req = httptest.NewRequest(http.MethodPut, "/api/settings", strings.NewReader(`{"router_memory":true}`))
	req.Header.Set("Content-Type", "application/json")
	rec = httptest.NewRecorder()
	s.handleUpdateSettings(rec, req)
	var got map[string]any
	json.Unmarshal(rec.Body.Bytes(), &got)
	if rec.Code != http.StatusOK || !s.cfg.RouterMemory {
		t.Errorf("JSON with generation on: %d %s", rec.Code, rec.Body)
	}
}

// The Server page says who decides, shows the router's margin only when
// the router decides, and asks for a restart when the running router was
// started in another mode.
func TestRouterMemoryServerPage(t *testing.T) {
	s := newTestServer(t)
	s.cfg.RouterMemory = true
	if v := s.routerMemoryInfo(); v.Toolchest || !v.On || v.Restart {
		t.Errorf("generation off, router stopped: %+v", v)
	}
	s.cfg.GenerationEnabled = true
	if v := s.routerMemoryInfo(); !v.Toolchest || v.Margin != "" || v.Note != "" {
		t.Errorf("generation on: %+v; want toolchest, no margin or note", v)
	}
}

// With generation on and the option on, the router starts in toolchest
// mode: no count limit and no margin of its own. A stop puts the mode
// back to count; with generation off the router decides again.
func TestStartRouterMemoryModes(t *testing.T) {
	s, proc := evalStopServer(t)
	s.builder = testBuilder(t, testBuild("b1", fakeLlamaServerDir(t)))
	s.cfg.RouterMemory, s.cfg.GenerationEnabled, s.cfg.ModelsMax = true, true, 3
	if err := s.startRouterWith(routerOptions{}); err != nil {
		t.Fatal(err)
	}
	waitRouterRunning(t, proc)
	if !s.toolchestMemory() {
		t.Errorf("mode = %s, want toolchest", memoryMode(s.memoryMode.Load()))
	}
	if v := s.routerMemoryInfo(); v.Restart {
		t.Error("asks for a restart in the mode it runs in")
	}
	s.cfg.GenerationEnabled = false
	if v := s.routerMemoryInfo(); !v.Restart {
		t.Error("no restart asked for after generation was turned off")
	}
	if err := proc.Stop(); err != nil {
		t.Fatal(err)
	}
	s.cfg.RouterMemory = false
	if err := s.startRouterWith(routerOptions{}); err != nil {
		t.Fatal(err)
	}
	waitRouterRunning(t, proc)
	if mode := memoryMode(s.memoryMode.Load()); mode != memoryCount {
		t.Errorf("option off: mode = %s, want count", mode)
	}
}

// The switch is labelled for both modes and says who decides.
func TestRouterMemorySwitchRenders(t *testing.T) {
	s := newTestServer(t)
	s.cfg.RouterMemory, s.cfg.GenerationEnabled = true, true
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/server", nil)
	s.handleServerPage(rec, req)
	body := rec.Body.String()
	for _, want := range []string{"Keep several models loaded", "Toolchest decides, because generation backends or forks are on.", "Toolchest decides, with the fit margin of each GPU."} {
		if !strings.Contains(body, want) {
			t.Errorf("Server page is missing %q", want)
		}
	}
}
