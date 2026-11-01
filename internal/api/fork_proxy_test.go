package api

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/tmac1973/llama-toolchest/internal/forks"
)

// fakeFork stands in for a fork's llama-server: /health answers 200,
// /props gives one 256K slot, and every other request is recorded and answered with a short stream.
type fakeFork struct {
	srv  *httptest.Server
	port int
	mu   sync.Mutex
	got  []fakeForkRequest
}

type fakeForkRequest struct {
	Path string
	Body map[string]any
}

func newFakeFork(t *testing.T) *fakeFork {
	t.Helper()
	f := &fakeFork{}
	f.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			return
		case "/props":
			fmt.Fprint(w, `{"default_generation_settings":{"n_ctx":262144},"total_slots":1,"modalities":{"vision":false}}`)
			return
		}
		var body map[string]any
		raw, _ := io.ReadAll(r.Body)
		json.Unmarshal(raw, &body)
		f.mu.Lock()
		f.got = append(f.got, fakeForkRequest{Path: r.URL.Path, Body: body})
		f.mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n")
		w.(http.Flusher).Flush()
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	t.Cleanup(f.srv.Close)
	_, port, _ := net.SplitHostPort(strings.TrimPrefix(f.srv.URL, "http://"))
	fmt.Sscan(port, &f.port)
	return f
}

func (f *fakeFork) requests() []fakeForkRequest {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]fakeForkRequest(nil), f.got...)
}

// runningFork is forkInstanceServer with one fork instance, "qwen-next",
// running against a fake fork server.
func runningFork(t *testing.T) (*Server, *fakeFork) {
	t.Helper()
	s, _ := forkInstanceServer(t)
	fake := newFakeFork(t)
	if err := s.forks.AddInstance(forks.Instance{ID: "qwen-next", ForkID: "avalon", GPU: 1, Port: fake.port}); err != nil {
		t.Fatal(err)
	}
	if _, err := s.startForkInstance("qwen-next"); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := s.forkMgr.WaitRunning(ctx, "qwen-next"); err != nil {
		t.Fatal(err)
	}
	return s, fake
}

// withChiParam adds one chi URL parameter to a request.
func withChiParam(r *http.Request, key, value string) *http.Request {
	rc := chi.NewRouteContext()
	rc.URLParams.Add(key, value)
	return r.WithContext(context.WithValue(r.Context(), chi.RouteCtxKey, rc))
}

func proxyPost(s *Server, path, body string) *httptest.ResponseRecorder {
	req := httptest.NewRequest("POST", path, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	s.newProxyHandler().ServeHTTP(rec, req)
	return rec
}

func TestForkChatGoesToTheForkInstance(t *testing.T) {
	s, fake := runningFork(t)
	rec := proxyPost(s, "/v1/chat/completions", `{"model":"qwen-next","stream":true,"temperature":0.2,"messages":[{"role":"user","content":"hi"}]}`)
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), `"content":"hi"`) || !strings.Contains(rec.Body.String(), "[DONE]") {
		t.Fatalf("chat: %d %q", rec.Code, rec.Body)
	}
	got := fake.requests()
	if len(got) != 1 || got[0].Path != "/v1/chat/completions" {
		t.Fatalf("fork got %+v", got)
	}
	body := got[0].Body
	// The client's temperature is kept; the fork's other defaults are added.
	if body["temperature"] != 0.2 || body["top_k"] != float64(20) || body["top_p"] != 0.95 {
		t.Errorf("sampling = temperature %v, top_k %v, top_p %v", body["temperature"], body["top_k"], body["top_p"])
	}

	// Other POST paths go there too, without chat defaults.
	proxyPost(s, "/v1/completions", `{"model":"qwen-next","prompt":"x"}`)
	if got := fake.requests(); len(got) != 2 || got[1].Path != "/v1/completions" || got[1].Body["top_k"] != nil {
		t.Errorf("completions request = %+v", got[len(got)-1])
	}
}

// Any other model still goes to the router (which is not running here).
func TestOtherModelsStillGoToTheRouter(t *testing.T) {
	s, fake := runningFork(t)
	rec := proxyPost(s, "/v1/chat/completions", `{"model":"some-llm","messages":[]}`)
	if !strings.Contains(rec.Body.String(), "router") {
		t.Errorf("other model: %d %s", rec.Code, rec.Body)
	}
	if len(fake.requests()) != 0 {
		t.Error("the fork got another model's request")
	}
	// With forks off, a fork's name is just an unknown model.
	s.cfg.ForksEnabled = false
	proxyPost(s, "/v1/chat/completions", `{"model":"qwen-next","messages":[]}`)
	if len(fake.requests()) != 0 {
		t.Error("the fork got a request while forks are off")
	}
}

func TestForkNotRunningOrLoading(t *testing.T) {
	s, port := forkInstanceServer(t)
	s.forks.AddInstance(forks.Instance{ID: "stopped", ForkID: "avalon", GPU: 0, Port: port})
	rec := proxyPost(s, "/v1/chat/completions", `{"model":"stopped","messages":[]}`)
	if rec.Code != http.StatusServiceUnavailable || !strings.Contains(rec.Body.String(), "is not running. Start it on the Server page") {
		t.Errorf("stopped: %d %s", rec.Code, rec.Body)
	}

	// Starting: its health check does not answer yet (nothing listens on
	// this port), so it stays loading.
	closed := newFakeFork(t)
	closed.srv.Close()
	s.forks.AddInstance(forks.Instance{ID: "loading", ForkID: "avalon", GPU: 1, Port: closed.port})
	if _, err := s.startForkInstance("loading"); err != nil {
		t.Fatal(err)
	}
	rec = proxyPost(s, "/v1/chat/completions", `{"model":"loading","messages":[]}`)
	if rec.Code != http.StatusServiceUnavailable || rec.Header().Get("Retry-After") != "30" || !strings.Contains(rec.Body.String(), "is loading") {
		t.Errorf("loading: %d %s %v", rec.Code, rec.Body, rec.Header())
	}
}

func TestV1ModelsListsRunningForks(t *testing.T) {
	s, _ := runningFork(t)
	s.forks.AddInstance(forks.Instance{ID: "stopped", ForkID: "avalon", GPU: 0, Port: 1})
	list := func(query string) []map[string]any {
		rec := httptest.NewRecorder()
		s.handleV1Models(rec, httptest.NewRequest("GET", "/v1/models"+query, nil))
		var out struct {
			Data []map[string]any `json:"data"`
		}
		json.Unmarshal(rec.Body.Bytes(), &out)
		return out.Data
	}
	ids := func(data []map[string]any) []string {
		var out []string
		for _, d := range data {
			out = append(out, d["id"].(string))
		}
		return out
	}
	all := list("")
	if got := ids(all); len(got) != 1 || got[0] != "qwen-next" {
		t.Fatalf("models = %v", got)
	}
	meta := all[0]["meta"].(map[string]any)
	if all[0]["owned_by"] != "fork:avalon" || meta["kind"] != "text" || meta["fork"] != "avalon" || meta["state"] != "running" {
		t.Errorf("entry = %+v", all[0])
	}
	if got := ids(list("?kind=text")); len(got) != 1 {
		t.Errorf("kind=text = %v", got)
	}
	if got := ids(list("?kind=image")); len(got) != 0 {
		t.Errorf("kind=image = %v", got)
	}

	one := func(id string) int {
		rec := httptest.NewRecorder()
		req := httptest.NewRequest("GET", "/v1/models/"+id, nil)
		s.handleV1Model(rec, withChiParam(req, "model", id))
		return rec.Code
	}
	if code := one("qwen-next"); code != http.StatusOK {
		t.Errorf("single running fork: %d", code)
	}
	if code := one("stopped"); code != http.StatusNotFound {
		t.Errorf("single stopped fork: %d", code)
	}
	s.cfg.ForksEnabled = false
	if got := list(""); len(got) != 0 {
		t.Errorf("forks off: %v", ids(got))
	}
}

// Haruspex finds a toolchest's models in /api/service/loaded-models, so a
// running fork is listed there, with its context from /props, even while
// the router is stopped. A stopped instance is not listed.
func TestLoadedModelsListsRunningForks(t *testing.T) {
	s, _ := runningFork(t)
	s.forks.AddInstance(forks.Instance{ID: "stopped", ForkID: "avalon", GPU: 0, Port: 1})
	rec := httptest.NewRecorder()
	s.handleLoadedModels(rec, httptest.NewRequest("GET", "/api/service/loaded-models", nil))
	var out struct {
		Running bool `json:"running"`
		Models  []struct {
			ID           string         `json:"id"`
			Status       string         `json:"status"`
			PublicName   string         `json:"public_name"`
			Capabilities map[string]any `json:"capabilities"`
		} `json:"models"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	if s.process.IsRunning() {
		t.Fatal("the router runs in this test")
	}
	if !out.Running || len(out.Models) != 1 {
		t.Fatalf("loaded-models = %s", rec.Body)
	}
	m := out.Models[0]
	if m.ID != "qwen-next" || m.Status != "loaded" || m.PublicName != "qwen-next" {
		t.Errorf("entry = %+v", m)
	}
	if m.Capabilities["context_per_request"] != float64(262144) || m.Capabilities["parallel"] != float64(1) {
		t.Errorf("capabilities = %v", m.Capabilities)
	}
}
