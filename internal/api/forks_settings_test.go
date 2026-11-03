package api

import (
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/backup"
	"github.com/tmac1973/llama-toolchest/internal/config"
)

func TestForksSwitchSavesAndReloads(t *testing.T) {
	s := newSettingsServer(t, &config.Config{LlamaPort: 8080, ListenAddr: ":3000"})
	put := func(form url.Values) string {
		req := httptest.NewRequest("PUT", "/api/settings", strings.NewReader(form.Encode()))
		req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
		req.Header.Set("HX-Request", "true")
		rec := httptest.NewRecorder()
		s.handleUpdateSettings(rec, req)
		return rec.Header().Get("HX-Refresh")
	}
	on := url.Values{"forks_touched": {"1"}, "forks_enabled": {"on"}}

	if got := put(on); got != "true" || !s.cfg.ForksEnabled {
		t.Fatalf("switched on: forks %v, HX-Refresh %q", s.cfg.ForksEnabled, got)
	}
	cfg, err := config.Load(s.configPath)
	if err != nil {
		t.Fatal(err)
	}
	if !cfg.ForksEnabled {
		t.Error("ForksEnabled not persisted to the config file")
	}
	if got := put(on); got != "" {
		t.Errorf("saved while on: HX-Refresh %q, want none", got)
	}
	// An unchecked switch posts only the marker.
	if got := put(url.Values{"forks_touched": {"1"}}); got != "true" || s.cfg.ForksEnabled {
		t.Errorf("switched off: forks %v, HX-Refresh %q", s.cfg.ForksEnabled, got)
	}
}

func TestForksSwitchJSON(t *testing.T) {
	s := newSettingsServer(t, &config.Config{LlamaPort: 8080, ListenAddr: ":3000"})
	req := httptest.NewRequest(http.MethodPut, "/api/settings", strings.NewReader(`{"forks_enabled":true}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	s.handleUpdateSettings(rec, req)
	if rec.Code != http.StatusOK || !s.cfg.ForksEnabled {
		t.Fatalf("PUT: %d, forks %v", rec.Code, s.cfg.ForksEnabled)
	}
	rec = httptest.NewRecorder()
	s.handleGetSettings(rec, httptest.NewRequest(http.MethodGet, "/api/settings", nil))
	if !strings.Contains(rec.Body.String(), `"forks_enabled":true`) {
		t.Errorf("GET does not report forks_enabled: %s", rec.Body)
	}
}

// Forks hold GPUs the router cannot see, as generation backends do. The
// option stays on with forks on, can be turned on then, and a restore
// keeps it: toolchest decides what to unload from the next start.
func TestForksKeepRouterMemoryOn(t *testing.T) {
	s := newSettingsServer(t, &config.Config{ModelsMax: 1})
	putSettings(t, s, url.Values{"router_memory_touched": {"1"}, "router_memory": {"on"}})
	putSettings(t, s, url.Values{"forks_touched": {"1"}, "forks_enabled": {"on"}})
	if !s.cfg.ForksEnabled || !s.cfg.RouterMemory {
		t.Fatalf("forks on: forks %v, router %v; want both on", s.cfg.ForksEnabled, s.cfg.RouterMemory)
	}
	if v := s.routerMemoryInfo(); !v.Toolchest {
		t.Error("Server page does not say toolchest decides with forks on")
	}

	putSettings(t, s, url.Values{"router_memory_touched": {"1"}})
	if s.cfg.RouterMemory {
		t.Fatal("not turned off")
	}
	req := httptest.NewRequest(http.MethodPut, "/api/settings", strings.NewReader(`{"router_memory":true}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	s.handleUpdateSettings(rec, req)
	if rec.Code != http.StatusOK || !s.cfg.RouterMemory {
		t.Errorf("JSON with forks on: %d, router %v; want on", rec.Code, s.cfg.RouterMemory)
	}

	s.cfg.RouterMemory = false
	on := true
	if _, err := s.restoreDeps().ApplySettings(backup.Settings{RouterMemory: &on}); err != nil {
		t.Fatal(err)
	}
	if !s.cfg.RouterMemory {
		t.Error("restore did not turn the option on with forks on")
	}
}

func TestSettingsPageForksSwitch(t *testing.T) {
	out := renderSettings(t, false, false)
	for _, want := range []string{`id="forks"`, `name="forks_touched"`, `name="forks_enabled"`, "llama.cpp forks (experimental)"} {
		if !strings.Contains(out, want) {
			t.Errorf("settings page lacks %s", want)
		}
	}
}
