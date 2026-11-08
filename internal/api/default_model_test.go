package api

import (
	"context"
	"testing"
	"time"
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
