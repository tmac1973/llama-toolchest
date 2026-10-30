package api

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/go-chi/chi/v5"

	"github.com/tmac1973/llama-toolchest/internal/models"
)

// ngramServer is a profile server whose model uses N-gram Cache, and the
// path of its learned n-gram file.
func ngramServer(t *testing.T) (*Server, string) {
	t.Helper()
	s := newProfileServer(t, "")
	cfg, _ := s.registry.GetConfig(profTestID)
	cfg.SpecAssist = "ngram-cache"
	if err := s.registry.SetConfig(profTestID, cfg); err != nil {
		t.Fatal(err)
	}
	path := s.registry.NgramCacheFile(profTestID, cfg, models.Target{})
	if path == "" {
		t.Fatal("no file for an N-gram Cache model")
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	return s, path
}

// Configure shows the file's size and the Clear button, and the button
// deletes the file.
func TestNgramCacheDialogAndClear(t *testing.T) {
	s, path := ngramServer(t)
	_, body := s.doProfileRequest(t, "GET", "/api/models/"+profTestID+"/config", nil)
	if !strings.Contains(body, "Nothing learned yet") || strings.Contains(body, "Clear learned n-grams") {
		t.Errorf("before a save: want the nothing-learned text and no button")
	}
	if err := os.WriteFile(path, make([]byte, 3*1024), 0o644); err != nil {
		t.Fatal(err)
	}
	_, body = s.doProfileRequest(t, "GET", "/api/models/"+profTestID+"/config", nil)
	if !strings.Contains(body, "Learned n-grams: 3 KB") || !strings.Contains(body, "Clear learned n-grams") {
		t.Errorf("with a file: no size or button")
	}

	r := chi.NewRouter()
	r.Post("/api/models/{id}/ngram-cache/clear", s.handleClearNgramCache)
	req := httptest.NewRequest("POST", "/api/models/"+profTestID+"/ngram-cache/clear", nil)
	req.Header.Set("HX-Request", "true")
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), "Nothing learned yet") {
		t.Errorf("clear: %d %q", rec.Code, rec.Body.String())
	}
	if _, err := os.Stat(path); err == nil {
		t.Error("clear left the file")
	}
}

// A broken file is deleted right before a load.
func TestPrepareNgramCacheForLoad(t *testing.T) {
	s, path := ngramServer(t)
	if err := os.WriteFile(path, []byte{1, 2, 3, 4, 5}, 0o644); err != nil {
		t.Fatal(err)
	}
	s.prepareNgramCacheForLoad(profTestID)
	if _, err := os.Stat(path); err == nil {
		t.Error("a broken file was kept before the load")
	}
}

// An older build says it keeps nothing, and a model on another method
// shows nothing.
func TestNgramCacheViewNotes(t *testing.T) {
	s, _ := ngramServer(t)
	cfg, _ := s.registry.GetConfig(profTestID)
	other := *cfg
	other.SpecAssist = "ngram-mod"
	if v := s.ngramCacheView(profTestID, &other); v.Show {
		t.Errorf("another method: %+v", v)
	}
	other.SpecAssist = "ngram-cache"
	other.ExtraFlags = "--lookup-cache-dynamic /x.bin"
	if v := s.ngramCacheView(profTestID, &other); !strings.Contains(v.Note, "extra flags") {
		t.Errorf("own path in extra flags: %+v", v)
	}
}
