package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/go-chi/chi/v5"

	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/presets"
)

// presetServer is a test server whose preset fetcher talks to a local
// server that has nothing (404 for every request), and counts the
// requests it gets.
func presetServer(t *testing.T) (*Server, *atomic.Int32) {
	t.Helper()
	var hits atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		http.NotFound(w, r)
	}))
	t.Cleanup(upstream.Close)
	s := newTestServer(t)
	f := presets.NewFetcher("", "")
	f.HFBase, f.DocsBase = upstream.URL, upstream.URL
	s.presets = f
	return s, &hits
}

func addPresetModel(t *testing.T, s *Server, id, repo string, embedding bool) {
	t.Helper()
	if err := s.registry.Add(&models.Model{ID: id, ModelID: repo, Filename: "m.gguf", Embedding: embedding}); err != nil {
		t.Fatal(err)
	}
}

func checkedAt(t *testing.T, s *Server, id string) bool {
	t.Helper()
	m, err := s.registry.Get(id)
	if err != nil {
		t.Fatal(err)
	}
	return !m.PresetsCheckedAt.IsZero()
}

// A chat model is looked up upstream; a miss still stamps the attempt, so
// the backfill does not try it again at every start.
func TestEnrichModelPresetsStampsAMiss(t *testing.T) {
	s, hits := presetServer(t)
	addPresetModel(t, s, "org--chat--m", "org/chat", false)
	s.enrichModelPresets("org--chat--m")
	if hits.Load() == 0 {
		t.Error("a chat model was not looked up upstream")
	}
	if !checkedAt(t, s, "org--chat--m") {
		t.Error("the attempt was not stamped")
	}
}

// An embedding model has no sampling presets: it is stamped without a
// request.
func TestEnrichModelPresetsSkipsNonChatModels(t *testing.T) {
	s, hits := presetServer(t)
	addPresetModel(t, s, "org--embed--m", "org/embed", true)
	s.enrichModelPresets("org--embed--m")
	if hits.Load() != 0 {
		t.Errorf("an embedding model made %d upstream requests", hits.Load())
	}
	if !checkedAt(t, s, "org--embed--m") {
		t.Error("the embedding model was not stamped")
	}
}

// An unknown model, for example one deleted meanwhile, is left alone.
func TestEnrichModelPresetsUnknownModel(t *testing.T) {
	s, hits := presetServer(t)
	s.enrichModelPresets("org--gone--m")
	if hits.Load() != 0 {
		t.Error("an unknown model was looked up upstream")
	}
}

// The backfill does every model never attempted, pausing between them
// but not before the first.
func TestBackfillPresets(t *testing.T) {
	s, _ := presetServer(t)
	addPresetModel(t, s, "org--a--m", "org/a", false)
	addPresetModel(t, s, "org--b--m", "org/b", true)
	pauses := 0
	old := presetBackfillPause
	presetBackfillPause = func() { pauses++ }
	t.Cleanup(func() { presetBackfillPause = old })

	s.backfillPresets()
	if pauses != 1 {
		t.Errorf("pauses = %d, want 1 between two models", pauses)
	}
	if !checkedAt(t, s, "org--a--m") || !checkedAt(t, s, "org--b--m") {
		t.Error("the backfill left a model unstamped")
	}
	if ids := s.registry.ListNeedingPresetFetch(); len(ids) != 0 {
		t.Errorf("still needing a fetch: %v", ids)
	}
}

// Refresh answers 404 for an unknown model and the presets as JSON for a
// known one.
func TestHandleRefreshPresets(t *testing.T) {
	s, _ := presetServer(t)
	addPresetModel(t, s, "org--chat--m", "org/chat", false)
	r := chi.NewRouter()
	r.Post("/api/models/{id}/refresh-presets", s.handleRefreshPresets)

	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/api/models/org--none--m/refresh-presets", nil))
	if rec.Code != http.StatusNotFound {
		t.Errorf("unknown model: status %d, want 404", rec.Code)
	}

	rec = httptest.NewRecorder()
	r.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/api/models/org--chat--m/refresh-presets", nil))
	var body map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil || rec.Code != http.StatusOK {
		t.Fatalf("known model: status %d, body %q", rec.Code, rec.Body.String())
	}
	if _, ok := body["presets"]; !ok {
		t.Errorf("JSON reply has no presets field: %v", body)
	}
}
