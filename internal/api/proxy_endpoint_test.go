package api

import (
	"encoding/json"
	"net/http/httptest"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/config"
)

// The settings API reported "http://localhost127.0.0.1:3000/v1" when the
// listen address included a host. It now reports the same external URL
// the settings page and dashboard show.
func TestSettingsProxyEndpointUsesExternalURL(t *testing.T) {
	s := newSettingsServer(t, &config.Config{
		ListenAddr:  "127.0.0.1:3917",
		ExternalURL: "http://compute2:3000/",
	})
	rec := httptest.NewRecorder()
	s.handleGetSettings(rec, httptest.NewRequest("GET", "/api/settings", nil))

	var got settingsResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.ProxyEndpoint != "http://compute2:3000/v1" {
		t.Errorf("proxy_endpoint = %q, want http://compute2:3000/v1", got.ProxyEndpoint)
	}
}
