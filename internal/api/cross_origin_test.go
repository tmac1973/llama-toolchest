package api

import (
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/config"
)

// A state-changing request from another site is refused; the same request
// from the toolchest's own pages, from a non-browser client, or to the API
// key routes passes, and so does any GET.
func TestCrossOriginProtection(t *testing.T) {
	h := crossOriginProtection(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	cases := []struct {
		method, path string
		headers      map[string]string
		want         int
	}{
		{"POST", "/api/settings", map[string]string{"Sec-Fetch-Site": "cross-site"}, http.StatusForbidden},
		{"PUT", "/api/settings", map[string]string{"Origin": "http://evil.example"}, http.StatusForbidden},
		{"POST", "/api/restore", map[string]string{"Sec-Fetch-Site": "same-origin"}, http.StatusOK},
		{"POST", "/api/settings", nil, http.StatusOK},
		{"GET", "/api/backup", map[string]string{"Sec-Fetch-Site": "cross-site"}, http.StatusOK},
		{"POST", "/v1/chat/completions", map[string]string{"Sec-Fetch-Site": "cross-site"}, http.StatusOK},
		{"POST", "/release_task", map[string]string{"Sec-Fetch-Site": "cross-site"}, http.StatusOK},
	}
	for _, c := range cases {
		req := httptest.NewRequest(c.method, "http://toolchest:3000"+c.path, strings.NewReader(""))
		for k, v := range c.headers {
			req.Header.Set(k, v)
		}
		w := httptest.NewRecorder()
		h.ServeHTTP(w, req)
		if w.Code != c.want {
			t.Errorf("%s %s %v: status %d, want %d", c.method, c.path, c.headers, w.Code, c.want)
		}
	}
}

// The router uses the protection.
func TestRouterRefusesCrossSiteSettings(t *testing.T) {
	s := newTestServer(t)
	req := httptest.NewRequest("PUT", "/api/settings", strings.NewReader("log_level=debug"))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("Sec-Fetch-Site", "cross-site")
	w := httptest.NewRecorder()
	s.buildRouter().ServeHTTP(w, req)
	if w.Code != http.StatusForbidden {
		t.Errorf("status %d, want 403", w.Code)
	}
}

// Saving the settings form with a secret left masked keeps its stored
// value, and a changed line is saved as typed.
func TestSettingsSaveKeepsMaskedEnvValue(t *testing.T) {
	s := newSettingsServer(t, &config.Config{RuntimeEnvExtra: "HF_TOKEN=hf_stored\nFOO=1"})
	form := url.Values{
		"runtime_env_touched": {"1"},
		"runtime_env_extra":   {"HF_TOKEN=" + config.EnvSecretMask + "\nFOO=2"},
	}
	req := httptest.NewRequest("PUT", "/api/settings", strings.NewReader(form.Encode()))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("HX-Request", "true")
	w := httptest.NewRecorder()
	s.buildRouter().ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("status %d: %s", w.Code, w.Body.String())
	}
	if got := s.cfg.RuntimeEnvExtra; got != "HF_TOKEN=hf_stored\nFOO=2" {
		t.Errorf("saved extra = %q (status %d: %s)", got, w.Code, w.Body.String())
	}
}
