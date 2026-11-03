package api

import (
	"net/http"
	"net/http/httptest"
	"net/url"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/huggingface"
	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/modelsource"
	"github.com/tmac1973/llama-toolchest/internal/testutil"
)

// The "Get reranking models" card lists every curated reranker with its
// context, licence and size, and marks the ones already downloaded.
func TestRerankPresetsCard(t *testing.T) {
	s := newTestServer(t)
	if err := s.registry.Add(&models.Model{ID: "b", ModelID: "gpustack/bge-reranker-v2-m3-GGUF",
		Filename: "bge-reranker-v2-m3-Q8_0.gguf"}); err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodGet, "/api/models/rerank-presets", nil)
	req.Header.Set("HX-Request", "true")
	rec := httptest.NewRecorder()
	s.handleRerankPresets(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	body := rec.Body.String()
	for _, p := range models.CuratedRerankModels() {
		if !strings.Contains(body, p.Name) {
			t.Errorf("card does not list %s", p.Name)
		}
	}
	for _, want := range []string{"40960", "Apache-2.0", "CC-BY-NC-4.0", "4.3 GB", "305 MB", "Downloaded", "/api/models/rerank-presets/download"} {
		if !strings.Contains(body, want) {
			t.Errorf("card does not show %q", want)
		}
	}
	if n := strings.Count(body, "Downloaded"); n != 1 {
		t.Errorf("%d models marked downloaded, want 1", n)
	}
}

// Only a curated reranker's file is downloaded from the card's route.
func TestDownloadRerankPreset(t *testing.T) {
	s := newTestServer(t)
	var mu sync.Mutex
	var fetched []string
	fake := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		fetched = append(fetched, r.URL.Path)
		mu.Unlock()
		w.Write([]byte("gguf"))
	}))
	t.Cleanup(fake.Close)
	s.downloader = huggingface.NewDownloader(s.cfg.DataDir, filepath.Join(s.cfg.DataDir, "models"), "")
	s.downloader.RegisterProvider(modelsource.SourceHuggingFace, huggingface.Provider{
		URL: func(repo, file string) string { return fake.URL + "/" + repo + "/" + file },
	})
	post := func(repo, file string) *httptest.ResponseRecorder {
		form := url.Values{"repo": {repo}, "filename": {file}}
		r := httptest.NewRequest(http.MethodPost, "/api/models/rerank-presets/download", strings.NewReader(form.Encode()))
		r.Header.Set("Content-Type", "application/x-www-form-urlencoded")
		rec := httptest.NewRecorder()
		s.handleDownloadRerankPreset(rec, r)
		return rec
	}

	for _, bad := range [][2]string{
		{"org/Some-Chat-GGUF", "chat.gguf"},
		// A curated repo, but a file the card does not offer.
		{"mradermacher/Qwen3-Reranker-4B-GGUF", "Qwen3-Reranker-4B.Q8_0.gguf"},
		{"gpustack/bge-reranker-v2-m3-GGUF", "bge-reranker-v2-m3-Q2_K.gguf"},
	} {
		if rec := post(bad[0], bad[1]); rec.Code != http.StatusBadRequest {
			t.Errorf("%s/%s: status %d, want 400", bad[0], bad[1], rec.Code)
		}
	}

	p := models.CuratedRerankModels()[0]
	rec := post(p.Repo, p.Filename)
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), "download_id") {
		t.Fatalf("curated %s: status %d: %s", p.Name, rec.Code, rec.Body)
	}
	testutil.Eventually(t, func() bool {
		mu.Lock()
		defer mu.Unlock()
		return len(fetched) > 0
	}, "the curated file was not fetched")
	mu.Lock()
	defer mu.Unlock()
	for _, f := range fetched {
		if f != "/"+p.Repo+"/"+p.Filename {
			t.Errorf("fetched %s, want only %s/%s", f, p.Repo, p.Filename)
		}
	}
}
