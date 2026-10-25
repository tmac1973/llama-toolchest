package api

import (
	"net/http/httptest"
	"strings"
	"testing"
)

// The dashboard's chat link uses the chat port as a browser reaches it,
// which in a container can differ from the port llama-server listens on.
func TestDashboardChatLinkPort(t *testing.T) {
	cases := []struct {
		name       string
		publicPort int
		want       string
	}{
		{"llama-server port", 0, `href="https://box:8080"`},
		{"published port", 9443, `href="https://box:9443"`},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			s := newTestServer(t)
			s.cfg.ExternalURL = "https://box:8443"
			s.cfg.LlamaPort = 8080
			s.cfg.ChatPublicPort = c.publicPort
			rec := httptest.NewRecorder()
			s.handleDashboard(rec, httptest.NewRequest("GET", "/api/dashboard", nil))
			if body := rec.Body.String(); !strings.Contains(body, c.want) {
				t.Errorf("chat link: want %s in:\n%s", c.want, body)
			}
		})
	}
}
