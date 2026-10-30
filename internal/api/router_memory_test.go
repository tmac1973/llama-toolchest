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
		name                  string
		max                   int
		on, job, hasOption    bool
		wantMax, wantMarginMB int
	}{
		{"off", 1, false, false, true, 1, 0},
		{"on for serving", 0, true, false, true, 0, 1311},
		{"on, benchmark", 0, true, true, true, 1, 0},
		{"on, build too old", 0, true, false, false, 0, 0},
	}
	for _, c := range cases {
		gotMax, gotMargin := routerLimits(c.max, c.on, c.job, c.hasOption, 1311)
		if gotMax != c.wantMax || gotMargin != c.wantMarginMB {
			t.Errorf("%s: max %d margin %d, want %d %d", c.name, gotMax, gotMargin, c.wantMax, c.wantMarginMB)
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

// The router option and generation backends exclude each other: it cannot
// be turned on while they are on, and turning them on turns it off.
// Turning it on with Max Loaded at 1 lifts the count limit.
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
	if !s.cfg.GenerationEnabled || s.cfg.RouterMemory {
		t.Fatalf("generation on: router %v, want off", s.cfg.RouterMemory)
	}

	body := putSettings(t, s, url.Values{"router_memory_touched": {"1"}, "router_memory": {"on"}, "models_max": {"5"}})
	if s.cfg.RouterMemory || s.cfg.ModelsMax != 3 || !strings.Contains(body, "Settings not saved") {
		t.Errorf("on with generation: router %v, max %d, answer %q; want refused and nothing changed", s.cfg.RouterMemory, s.cfg.ModelsMax, body)
	}

	// JSON clients get a 400.
	req := httptest.NewRequest(http.MethodPut, "/api/settings", strings.NewReader(`{"router_memory":true}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	s.handleUpdateSettings(rec, req)
	if rec.Code != http.StatusBadRequest || s.cfg.RouterMemory {
		t.Errorf("JSON with generation on: %d, router %v", rec.Code, s.cfg.RouterMemory)
	}
	// Both in one request: generation off and the router on is allowed.
	req = httptest.NewRequest(http.MethodPut, "/api/settings", strings.NewReader(`{"generation_enabled":false,"router_memory":true}`))
	req.Header.Set("Content-Type", "application/json")
	rec = httptest.NewRecorder()
	s.handleUpdateSettings(rec, req)
	var got map[string]any
	json.Unmarshal(rec.Body.Bytes(), &got)
	if rec.Code != http.StatusOK || !s.cfg.RouterMemory || s.cfg.GenerationEnabled {
		t.Errorf("generation off and router on: %d %s", rec.Code, rec.Body)
	}
}

// The Server page greys the switch out while generation backends are on,
// and shows the margin while it is on.
func TestRouterMemoryServerPage(t *testing.T) {
	s := newTestServer(t)
	s.cfg.GenerationEnabled = true
	if v := s.routerMemoryInfo(); v.Blocked == "" {
		t.Error("not blocked with generation on")
	}
	s.cfg.GenerationEnabled = false
	s.cfg.RouterMemory = true
	if v := s.routerMemoryInfo(); v.Blocked != "" || !v.On {
		t.Errorf("generation off: %+v", v)
	}
}
