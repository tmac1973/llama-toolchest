package api

import (
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/config"
)

// loadWithEnvOverrides loads a config file holding external_url and
// llama_port while both are overridden from the environment, the way a
// secure container install runs.
func loadWithEnvOverrides(t *testing.T) *config.Config {
	t.Helper()
	path := filepath.Join(t.TempDir(), "cfg.yaml")
	if err := os.WriteFile(path, []byte("external_url: http://myhost:3000\nllama_port: 8080\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("LLAMA_TOOLCHEST_EXTERNAL_URL", "https://localhost:8443")
	t.Setenv("LLAMA_TOOLCHEST_LLAMA_PORT", "9090")
	t.Setenv("LLAMA_TOOLCHEST_GEN_PUBLIC_PORT_START", "18081")
	cfg, err := config.Load(path)
	if err != nil {
		t.Fatal(err)
	}
	return cfg
}

// reloadSaved loads the file the server saved with the environment
// overrides gone, as after switching a container install to plain mode.
func reloadSaved(t *testing.T, s *Server) *config.Config {
	t.Helper()
	t.Setenv("LLAMA_TOOLCHEST_EXTERNAL_URL", "")
	t.Setenv("LLAMA_TOOLCHEST_LLAMA_PORT", "")
	t.Setenv("LLAMA_TOOLCHEST_GEN_PUBLIC_PORT_START", "")
	cfg, err := config.Load(s.configPath)
	if err != nil {
		t.Fatalf("reload saved config: %v", err)
	}
	return cfg
}

// Regression: saving an unrelated setting wrote the overridden
// external_url into the file, so links kept pointing at the secure URL
// after the variable was gone.
func TestSaveKeepsFileValuesOfEnvOverrides(t *testing.T) {
	s := newSettingsServer(t, loadWithEnvOverrides(t))

	req := httptest.NewRequest("PUT", "/api/settings", strings.NewReader(`{"api_key":"sk-test"}`))
	req.Header.Set("Content-Type", "application/json")
	s.handleUpdateSettings(httptest.NewRecorder(), req)

	if s.cfg.ExternalURL != "https://localhost:8443" || s.cfg.LlamaPort != 9090 {
		t.Errorf("in use: %q, %d; want the environment values", s.cfg.ExternalURL, s.cfg.LlamaPort)
	}
	saved := reloadSaved(t, s)
	if saved.ExternalURL != "http://myhost:3000" {
		t.Errorf("saved external_url = %q, want the file's http://myhost:3000", saved.ExternalURL)
	}
	if saved.LlamaPort != 8080 {
		t.Errorf("saved llama_port = %d, want the file's 8080", saved.LlamaPort)
	}
	if saved.GenPublicPortStart != 0 {
		t.Errorf("saved gen_public_port_start = %d, want it left unset", saved.GenPublicPortStart)
	}
	if saved.APIKey != "sk-test" {
		t.Errorf("saved api_key = %q, want the change itself saved", saved.APIKey)
	}
}

// Changing an overridden setting saves the new value to the file, while
// the environment variable keeps setting the value in use.
func TestChangingEnvOverriddenSettingSavesToFile(t *testing.T) {
	s := newSettingsServer(t, loadWithEnvOverrides(t))

	form := url.Values{"external_url": {"http://newhost:3000"}}
	req := httptest.NewRequest("PUT", "/api/settings", strings.NewReader(form.Encode()))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("HX-Request", "true")
	rec := httptest.NewRecorder()
	s.handleUpdateSettings(rec, req)

	if s.cfg.ExternalURL != "https://localhost:8443" {
		t.Errorf("in use = %q, want the environment value", s.cfg.ExternalURL)
	}
	if got := s.cfg.SavedExternalURL(); got != "http://newhost:3000" {
		t.Errorf("SavedExternalURL = %q, want the new value", got)
	}
	if !strings.Contains(rec.Body.String(), "LLAMA_TOOLCHEST_EXTERNAL_URL") {
		t.Errorf("response does not say the variable still wins:\n%s", rec.Body.String())
	}

	jreq := httptest.NewRequest("PUT", "/api/settings", strings.NewReader(`{"llama_port":8181}`))
	jreq.Header.Set("Content-Type", "application/json")
	jrec := httptest.NewRecorder()
	s.handleUpdateSettings(jrec, jreq)
	if s.cfg.LlamaPort != 9090 {
		t.Errorf("llama port in use = %d, want the environment value", s.cfg.LlamaPort)
	}
	if !strings.Contains(jrec.Body.String(), `"llama_port":"LLAMA_TOOLCHEST_LLAMA_PORT"`) {
		t.Errorf("response does not list the override:\n%s", jrec.Body.String())
	}

	saved := reloadSaved(t, s)
	if saved.ExternalURL != "http://newhost:3000" || saved.LlamaPort != 8181 {
		t.Errorf("saved = %q, %d; want the values changed in Settings", saved.ExternalURL, saved.LlamaPort)
	}
}
