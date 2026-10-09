package api

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/go-chi/chi/v5"
)

// The base URL comes from the dialog's field when given, else the external
// URL, and always ends in /v1.
func TestAgentBaseURL(t *testing.T) {
	s := newTestServer(t)
	s.cfg.ExternalURL = "http://gpu-box:3000"
	for q, want := range map[string]string{
		"":                                   "http://gpu-box:3000/v1",
		"base=http://10.0.0.5:3000":          "http://10.0.0.5:3000/v1",
		"base=http://10.0.0.5:3000/v1/":      "http://10.0.0.5:3000/v1",
		"base=https://llm.example.com/v1":    "https://llm.example.com/v1",
		"base=https://llm.example.com/x?y=1": "https://llm.example.com/x/v1",
	} {
		got, err := s.agentBaseURL(httptest.NewRequest("GET", "/api/agent-configs?"+q, nil))
		if err != nil || got != want {
			t.Errorf("%q: got %q, %v; want %q", q, got, err, want)
		}
	}
	// No external URL: the address the page was reached on.
	s.cfg.ExternalURL = ""
	if got, _ := s.agentBaseURL(httptest.NewRequest("GET", "http://box.lan:3000/api/agent-configs", nil)); got != "http://box.lan:3000/v1" {
		t.Errorf("no external URL: got %q", got)
	}
	for _, q := range []string{"base=ftp://h/", "base=not a url", "base=http://"} {
		if _, err := s.agentBaseURL(httptest.NewRequest("GET", "/api/agent-configs?"+strings.ReplaceAll(q, " ", "%20"), nil)); err == nil {
			t.Errorf("%q accepted", q)
		}
	}
}

func TestIsLocalOnly(t *testing.T) {
	for base, want := range map[string]bool{
		"http://localhost:3000/v1":    true,
		"http://127.0.0.1:3000/v1":    true,
		"http://127.1.2.3:3000/v1":    true,
		"http://[::1]:3000/v1":        true,
		"http://192.168.1.50:3000/v1": false,
		"https://llm.example.com/v1":  false,
	} {
		if got := isLocalOnly(base); got != want {
			t.Errorf("isLocalOnly(%q) = %v", base, got)
		}
	}
}

// With the router stopped there is nothing to list: the dialog says so and
// offers no downloads, and a download asks for the server to be started.
func TestAgentConfigsWithNothingServed(t *testing.T) {
	s := newTestServer(t)
	req := httptest.NewRequest("GET", "/api/agent-configs", nil)
	req.Header.Set("HX-Request", "true")
	w := httptest.NewRecorder()
	s.handleAgentConfigs(w, req)
	out := w.Body.String()
	if !strings.Contains(out, "isn't serving any chat models") || strings.Contains(out, "Download ") {
		t.Errorf("dialog with nothing served:\n%s", out)
	}

	w = downloadAgentConfig(s, "opencode")
	if w.Code != http.StatusConflict {
		t.Errorf("download with nothing served: %d %s", w.Code, w.Body.String())
	}
	if w = downloadAgentConfig(s, "no-such-agent"); w.Code != http.StatusNotFound {
		t.Errorf("unknown agent: %d", w.Code)
	}
}

func downloadAgentConfig(s *Server, agent string) *httptest.ResponseRecorder {
	r := chi.NewRouter()
	r.Get("/api/agent-configs/{agent}", s.handleAgentConfigDownload)
	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest("GET", "/api/agent-configs/"+agent, nil))
	return w
}

// The key line names the variable but never the key: Settings keeps the key
// write-only, and the dialog must not be the place it leaks.
func TestAgentConfigsKeyExport(t *testing.T) {
	d := agentConfigsData{KeyEnv: "LLAMA_TOOLCHEST_API_KEY"}
	if got := d.KeyExport(); got != "export LLAMA_TOOLCHEST_API_KEY='your API key'" {
		t.Errorf("KeyExport = %q", got)
	}
}

// The Server tab links the dialog from the API Endpoint card.
func TestDashboardLinksAgentConfigs(t *testing.T) {
	out := renderDashboardCards(t, dashboardCardsData{APIURL: "http://h:3000/v1"})
	if !strings.Contains(out, "openAgentConfigs()") {
		t.Errorf("API Endpoint card lacks the Agent Configs link:\n%s", out)
	}
}
