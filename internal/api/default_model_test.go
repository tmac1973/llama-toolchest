package api

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/config"
	"github.com/tmac1973/llama-toolchest/internal/forks"
	"github.com/tmac1973/llama-toolchest/internal/models"
)

func TestIsChatPath(t *testing.T) {
	for _, p := range []string{"/v1/chat/completions", "/chat/completions", "/v1/completions", "/completions", "/v1/messages", "/v1/responses"} {
		if !isChatPath(p) {
			t.Errorf("%s is not a chat path", p)
		}
	}
	for _, p := range []string{"/v1/embeddings", "/v1/rerank", "/v1/audio/speech", "/v1/messages/count_tokens", "/v1/models"} {
		if isChatPath(p) {
			t.Errorf("%s is a chat path", p)
		}
	}
}

func TestForkUse(t *testing.T) {
	var u forkUse
	if !u.lastUse("a").IsZero() {
		t.Error("an unused fork has a last use")
	}
	t0 := time.Date(2026, 11, 8, 12, 0, 0, 0, time.UTC)
	now := t0
	u.now = func() time.Time { return now }
	u.note("a")
	now = t0.Add(time.Minute)
	u.note("a")
	if got := u.lastUse("a"); !got.Equal(now) {
		t.Errorf("last use %v, want %v", got, now)
	}
}

// A chat request to a chat model is noted as the model used last; a
// request on another path is not.
func TestLoadAndCountRequestNotesChatUse(t *testing.T) {
	s, f, m := startFakeRouter(t)
	f.reset(map[string]string{m.ID: "loaded"}, nil)
	body := []byte(`{"model":"` + m.ID + `"}`)

	end, err := s.loadAndCountRequest(context.Background(), "/v1/embeddings", body)
	if err != nil {
		t.Fatal(err)
	}
	end()
	if got := s.defaultState.get(); got.ID != "" {
		t.Errorf("an embeddings request was noted: %+v", got)
	}

	end, err = s.loadAndCountRequest(context.Background(), "/v1/chat/completions", body)
	if err != nil {
		t.Fatal(err)
	}
	end()
	if got := s.defaultState.get(); got.ID != m.ID || got.Fork {
		t.Errorf("after a chat request: %+v, want %s", got, m.ID)
	}
}

// addChatModel registers one more model beside the fake router's.
func addChatModel(t *testing.T, s *Server, id string, cfg models.ModelConfig, embedding bool) *models.Model {
	t.Helper()
	m := &models.Model{
		ID: id, ModelID: "org/" + id, Quant: "Q4_K_M", Embedding: embedding,
		Filename: id + ".gguf", FilePath: filepath.Join(t.TempDir(), id+".gguf"),
	}
	if err := s.registry.Add(m); err != nil {
		t.Fatal(err)
	}
	if err := s.registry.SetConfig(m.ID, &cfg); err != nil {
		t.Fatal(err)
	}
	return m
}

func mustResolve(t *testing.T, s *Server) defaultTarget {
	t.Helper()
	got, ok := s.resolveDefaultModel()
	if !ok {
		t.Fatal("no target")
	}
	return got
}

func TestResolveDefaultModel(t *testing.T) {
	s, f, a := startFakeRouter(t)
	b := addChatModel(t, s, "beta", models.ModelConfig{Enabled: true}, false)
	emb := addChatModel(t, s, "embedder", models.ModelConfig{Enabled: true}, true)
	t0 := time.Date(2026, 11, 8, 12, 0, 0, 0, time.UTC)
	now := t0
	s.llmUse.now = func() time.Time { return now }

	t.Run("nothing loaded, no history", func(t *testing.T) {
		f.reset(map[string]string{}, nil)
		if got, ok := s.resolveDefaultModel(); ok {
			t.Errorf("target %+v, want none", got)
		}
	})
	t.Run("nothing loaded: the model used last", func(t *testing.T) {
		s.defaultState.note(b.ID, false, t0)
		got := mustResolve(t, s)
		if got.ID != b.ID || got.Reason != defaultReasonLastUsed || got.Name != b.PublicName() {
			t.Errorf("target %+v, want %s used last", got, b.ID)
		}
	})
	t.Run("tie: the model used last before the restart", func(t *testing.T) {
		f.reset(map[string]string{a.ID: "loaded", b.ID: "loaded", emb.ID: "loaded"}, nil)
		if got := mustResolve(t, s); got.ID != b.ID || got.Reason != defaultReasonLoaded {
			t.Errorf("target %+v, want %s loaded", got, b.ID)
		}
		s.defaultState.note(emb.ID, false, t0) // not a chat model: no tie-break
		want := a
		if b.PublicName() < a.PublicName() {
			want = b
		}
		if got := mustResolve(t, s); got.ID != want.ID {
			t.Errorf("target %s, want the lower name %s", got.ID, want.ID)
		}
	})
	t.Run("the loaded model used last", func(t *testing.T) {
		now = t0.Add(time.Minute)
		s.llmUse.touch(a.ID)
		now = t0.Add(2 * time.Minute)
		s.llmUse.touch(b.ID)
		if got := mustResolve(t, s); got.ID != b.ID {
			t.Errorf("target %s, want %s", got.ID, b.ID)
		}
		now = t0.Add(3 * time.Minute)
		s.llmUse.touch(a.ID)
		if got := mustResolve(t, s); got.ID != a.ID {
			t.Errorf("after a use of %s: target %s", a.ID, got.ID)
		}
		now = t0.Add(4 * time.Minute)
		s.llmUse.touch(emb.ID)
		if got := mustResolve(t, s); got.ID != a.ID {
			t.Errorf("an embedding model was picked: %s", got.ID)
		}
	})
	t.Run("a pin wins", func(t *testing.T) {
		s.cfg.DefaultModel.Pin = b.ID
		if got := mustResolve(t, s); got.ID != b.ID || got.Reason != defaultReasonPinned {
			t.Errorf("target %+v, want %s pinned", got, b.ID)
		}
	})
	t.Run("a stale pin is skipped, and cleared only by the cleanup", func(t *testing.T) {
		for _, pin := range []string{"gone", emb.ID} {
			s.cfg.DefaultModel.Pin = pin
			if got := mustResolve(t, s); got.ID != a.ID || got.Reason != defaultReasonLoaded {
				t.Errorf("pin %s: target %+v, want %s loaded", pin, got, a.ID)
			}
			if s.cfg.DefaultModel.Pin != pin {
				t.Errorf("pin %s: resolving changed the config", pin)
			}
			s.clearStaleDefaultPin()
			if s.cfg.DefaultModel.Pin != "" {
				t.Errorf("pin %s: not cleared", pin)
			}
		}
		s.registry.SetConfig(b.ID, &models.ModelConfig{Enabled: false})
		s.cfg.DefaultModel.Pin = b.ID
		s.clearStaleDefaultPin()
		if s.cfg.DefaultModel.Pin != "" {
			t.Error("a pin to a disabled model was not cleared")
		}
		s.cfg.DefaultModel.Pin = a.ID
		s.clearStaleDefaultPin()
		if s.cfg.DefaultModel.Pin != a.ID {
			t.Error("a good pin was cleared")
		}
	})
}

func TestResolveDefaultModelForks(t *testing.T) {
	s, _ := runningFork(t) // "qwen-next", running; no router
	s.forks.AddInstance(forks.Instance{ID: "stopped", ForkID: "avalon", GPU: 0, Port: 1})

	if got := mustResolve(t, s); !got.Fork || got.ID != "qwen-next" || got.Reason != defaultReasonLoaded {
		t.Errorf("unpinned: %+v, want the running fork", got)
	}
	s.cfg.DefaultModel.Pin = config.DefaultModelForkPrefix + "stopped"
	if got := mustResolve(t, s); !got.Fork || got.ID != "stopped" || got.Reason != defaultReasonPinned {
		t.Errorf("pinned to a stopped fork: %+v", got)
	}

	// Forks off: the pin is skipped but kept.
	s.cfg.ForksEnabled = false
	if got, ok := s.resolveDefaultModel(); ok {
		t.Errorf("forks off: target %+v, want none", got)
	}
	s.clearStaleDefaultPin()
	if s.cfg.DefaultModel.Pin == "" {
		t.Error("forks off: the fork pin was cleared")
	}

	// The instance removed: the pin goes.
	s.cfg.ForksEnabled = true
	if err := s.forks.DeleteInstance("stopped"); err != nil {
		t.Fatal(err)
	}
	s.clearStaleDefaultPin()
	if s.cfg.DefaultModel.Pin != "" {
		t.Error("a pin to a removed fork instance was not cleared")
	}
}

// The virtual name clashes with any model name or alias, enabled or not,
// and with a fork instance name while forks are off too.
func TestDefaultNameClash(t *testing.T) {
	s, _ := forkInstanceServer(t)
	m := addChatModel(t, s, "alpha", models.ModelConfig{Enabled: false, Aliases: []string{"default"}}, false)
	if s.defaultNameClash("default") == "" {
		t.Error("an alias of a disabled model does not clash")
	}
	if s.defaultNameClash(m.ID) == "" || s.defaultNameClash(m.PublicName()) == "" {
		t.Error("a model's ID or public name does not clash")
	}
	s.forks.AddInstance(forks.Instance{ID: "chat", ForkID: "avalon", GPU: 0, Port: 1})
	s.cfg.ForksEnabled = false
	if s.defaultNameClash("chat") == "" {
		t.Error("a fork instance name does not clash while forks are off")
	}
	if got := s.defaultNameClash("other"); got != "" {
		t.Errorf("no clash expected: %s", got)
	}
	s.forks = nil
	if got := s.defaultNameClash("chat"); got != "" {
		t.Errorf("no fork store: %s", got)
	}
}

// defaultProxy is a proxy in front of a fake router with the test model
// loaded. It returns the bodies the router got on /v1.
func defaultProxy(t *testing.T) (*Server, *fakeRouter, *models.Model, func() []map[string]json.RawMessage) {
	t.Helper()
	s, f, m := startFakeRouter(t)
	f.reset(map[string]string{m.ID: "loaded"}, nil)
	var mu sync.Mutex
	var got []map[string]json.RawMessage
	f.mu.Lock()
	f.v1 = func(w http.ResponseWriter, r *http.Request) {
		var body map[string]json.RawMessage
		json.NewDecoder(r.Body).Decode(&body)
		mu.Lock()
		got = append(got, body)
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"model":"real"}`)
	}
	f.mu.Unlock()
	return s, f, m, func() []map[string]json.RawMessage {
		mu.Lock()
		defer mu.Unlock()
		return append([]map[string]json.RawMessage(nil), got...)
	}
}

func TestProxyResolvesDefaultModel(t *testing.T) {
	s, f, m, bodies := defaultProxy(t)
	want := `"` + m.PublicName() + `"`

	for _, body := range []string{
		`{"model":"default","seed":12345678901234567,"messages":[]}`,
		`{"seed":12345678901234567,"messages":[]}`,
		`{"model":null,"seed":12345678901234567,"messages":[]}`,
		`{"model":"","seed":12345678901234567,"messages":[]}`,
	} {
		rec := proxyPost(s, "/v1/chat/completions", body)
		got := bodies()
		if rec.Code != http.StatusOK || len(got) == 0 {
			t.Fatalf("%s: %d %s", body, rec.Code, rec.Body)
		}
		last := got[len(got)-1]
		if string(last["model"]) != want || string(last["seed"]) != "12345678901234567" {
			t.Errorf("%s: router got model %s seed %s", body, last["model"], last["seed"])
		}
	}
	if got := s.defaultState.get(); got.ID != m.ID {
		t.Errorf("the resolved model was not noted as used: %+v", got)
	}

	t.Run("other paths", func(t *testing.T) {
		rec := proxyPost(s, "/v1/embeddings", `{"model":"default","input":"x"}`)
		if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), "default_model_chat_only") {
			t.Errorf("default on embeddings: %d %s", rec.Code, rec.Body)
		}
		n := len(bodies())
		proxyPost(s, "/v1/embeddings", `{"input":"x"}`)
		if got := bodies(); len(got) != n+1 || got[n]["model"] != nil {
			t.Errorf("no model on embeddings: router got %v", got[n:])
		}
	})

	t.Run("switched off", func(t *testing.T) {
		s.cfg.DefaultModel.Off = true
		defer func() { s.cfg.DefaultModel.Off = false }()
		proxyPost(s, "/v1/chat/completions", `{"model":"default","messages":[]}`)
		if got := bodies(); string(got[len(got)-1]["model"]) != `"default"` {
			t.Errorf("off: router got model %s", got[len(got)-1]["model"])
		}
		if rec := proxyPost(s, "/v1/embeddings", `{"model":"default"}`); rec.Code == http.StatusBadRequest {
			t.Error("off: a non-chat request to the name was refused")
		}
	})

	t.Run("a real model's alias wins", func(t *testing.T) {
		other := addChatModel(t, s, "aliased", models.ModelConfig{Enabled: true, Aliases: []string{"default"}}, false)
		defer s.registry.Remove(other.ID)
		f.mu.Lock()
		f.status[other.ID] = "loaded"
		f.mu.Unlock()
		proxyPost(s, "/v1/chat/completions", `{"model":"default","messages":[]}`)
		if got := bodies(); string(got[len(got)-1]["model"]) != `"default"` {
			t.Errorf("router got model %s, want the alias as sent", got[len(got)-1]["model"])
		}
	})

	t.Run("no target", func(t *testing.T) {
		f.reset(map[string]string{}, nil)
		s.defaultState = loadDefaultModelState(filepath.Join(t.TempDir(), "default-model.json"))
		rec := proxyPost(s, "/v1/chat/completions", `{"model":"default","messages":[]}`)
		if rec.Code != http.StatusServiceUnavailable || !strings.Contains(rec.Body.String(), "no_default_model") {
			t.Errorf("%d %s", rec.Code, rec.Body)
		}
	})
}

// A pinned model that is not loaded loads as its own name would, and its
// sampling defaults are added.
func TestProxyDefaultModelLoadsPin(t *testing.T) {
	s, f, _, bodies := defaultProxy(t)
	p := addChatModel(t, s, "pinned", models.ModelConfig{Enabled: true, Temperature: ptr(0.3)}, false)
	s.cfg.DefaultModel.Pin = p.ID
	f.reset(map[string]string{p.ID: "unloaded"}, func(f *fakeRouter, n int) {
		if f.status[p.ID] == "loading" {
			f.status[p.ID] = "loaded"
		}
	})
	rec := proxyPost(s, "/v1/chat/completions", `{"model":"default","messages":[]}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("%d %s", rec.Code, rec.Body)
	}
	if loads := f.loadCalls(); len(loads) != 1 || loads[0] != p.ID {
		t.Errorf("loads = %v, want %s", loads, p.ID)
	}
	got := bodies()
	last := got[len(got)-1]
	if string(last["model"]) != `"`+p.PublicName()+`"` || string(last["temperature"]) != "0.3" {
		t.Errorf("router got model %s temperature %s", last["model"], last["temperature"])
	}
}

// A pinned fork that is stopped answers as the fork does.
func TestProxyDefaultModelStoppedForkPin(t *testing.T) {
	s, _ := forkInstanceServer(t)
	s.forks.AddInstance(forks.Instance{ID: "stopped", ForkID: "avalon", GPU: 0, Port: 1})
	s.cfg.DefaultModel.Pin = config.DefaultModelForkPrefix + "stopped"
	rec := proxyPost(s, "/v1/chat/completions", `{"model":"default","messages":[]}`)
	if rec.Code != http.StatusServiceUnavailable || !strings.Contains(rec.Body.String(), "Fork instance stopped is not running") {
		t.Errorf("%d %s", rec.Code, rec.Body)
	}
}
