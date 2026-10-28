package api

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/config"
)

// Settings saves change the API key, the external URL and the generation
// switch while requests read them. Run with -race, this fails if any of
// these readers takes a setting without cfgMu: the API key check on /v1,
// the settings and server pages, the dashboard, the settings JSON, and
// the generation checks.
func TestSettingsSaveRacesNoReader(t *testing.T) {
	s := newSettingsServer(t, &config.Config{APIKey: "sk-start", ExternalURL: "http://host:3000"})
	router := s.buildRouter()

	var wg sync.WaitGroup
	stop := make(chan struct{})
	save := func(i int) {
		form := url.Values{
			"api_key":            {fmt.Sprintf("sk-%d", i)},
			"external_url":       {fmt.Sprintf("http://host%d:3000", i)},
			"generation_touched": {"1"},
		}
		if i%2 == 0 {
			form.Set("generation_enabled", "on")
		}
		req := httptest.NewRequest("PUT", "/api/settings", strings.NewReader(form.Encode()))
		req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
		router.ServeHTTP(httptest.NewRecorder(), req)
	}
	read := func() {
		defer wg.Done()
		for {
			select {
			case <-stop:
				return
			default:
			}
			for _, path := range []string{"/v1/models", "/settings", "/server", "/api/dashboard", "/api/settings"} {
				req := httptest.NewRequest("GET", path, nil)
				req.Header.Set("Authorization", "Bearer sk-start")
				router.ServeHTTP(httptest.NewRecorder(), req)
			}
			s.genEnabled()
			s.genEvictOn()
			s.proxyEndpoint()
		}
	}
	for range 2 {
		wg.Add(1)
		go read()
	}
	for i := range 20 {
		save(i)
	}
	close(stop)
	wg.Wait()

	// The last save is what the readers now see.
	rec := httptest.NewRecorder()
	req := httptest.NewRequest("GET", "/v1/models", nil)
	req.Header.Set("Authorization", "Bearer sk-19")
	router.ServeHTTP(rec, req)
	if rec.Code == http.StatusUnauthorized {
		t.Errorf("the last saved API key is refused: %s", rec.Body)
	}
	if got := s.proxyEndpoint(); got != "http://host19:3000/v1" {
		t.Errorf("proxyEndpoint = %q, want the last saved URL", got)
	}
}
