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

// Forks hold GPUs the router cannot see, as generation backends do, so
// router memory is off while they are on and cannot be turned on.
func TestForksTurnRouterMemoryOff(t *testing.T) {
	s := newSettingsServer(t, &config.Config{ModelsMax: 1})
	putSettings(t, s, url.Values{"router_memory_touched": {"1"}, "router_memory": {"on"}})
	if !s.cfg.RouterMemory {
		t.Fatal("router memory not on")
	}

	putSettings(t, s, url.Values{"forks_touched": {"1"}, "forks_enabled": {"on"}})
	if !s.cfg.ForksEnabled || s.cfg.RouterMemory {
		t.Fatalf("forks on: router %v, want off", s.cfg.RouterMemory)
	}
	if v := s.routerMemoryInfo(); v.Blocked == "" {
		t.Error("Server page switch not blocked with forks on")
	}

	body := putSettings(t, s, url.Values{"router_memory_touched": {"1"}, "router_memory": {"on"}})
	if s.cfg.RouterMemory || !strings.Contains(body, "Settings not saved") {
		t.Errorf("on with forks: router %v, answer %q; want refused", s.cfg.RouterMemory, body)
	}

	req := httptest.NewRequest(http.MethodPut, "/api/settings", strings.NewReader(`{"router_memory":true}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	s.handleUpdateSettings(rec, req)
	if rec.Code != http.StatusBadRequest || s.cfg.RouterMemory {
		t.Errorf("JSON with forks on: %d, router %v", rec.Code, s.cfg.RouterMemory)
	}

	// A restore does not turn it back on either.
	on := true
	changed, err := s.restoreDeps().ApplySettings(backup.Settings{RouterMemory: &on})
	if err != nil {
		t.Fatal(err)
	}
	if s.cfg.RouterMemory {
		t.Errorf("restore turned router memory on with forks on (changed %v)", changed)
	}

	// Forks off, in the same request as the router: allowed.
	req = httptest.NewRequest(http.MethodPut, "/api/settings", strings.NewReader(`{"forks_enabled":false,"router_memory":true}`))
	req.Header.Set("Content-Type", "application/json")
	rec = httptest.NewRecorder()
	s.handleUpdateSettings(rec, req)
	if rec.Code != http.StatusOK || !s.cfg.RouterMemory || s.cfg.ForksEnabled {
		t.Errorf("forks off and router on: %d %s", rec.Code, rec.Body)
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
