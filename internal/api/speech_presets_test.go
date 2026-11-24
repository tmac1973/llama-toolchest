package api

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/builder"
	"github.com/tmac1973/llama-toolchest/internal/huggingface"
	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/modelsource"
	"github.com/tmac1973/llama-toolchest/internal/testutil"
)

// The "Get speech models" card lists every curated speech model, and
// says so when the active llama.cpp build is older than one needs.
func TestSpeechPresetsCard(t *testing.T) {
	p := models.CuratedSpeechModels()[0]
	// The newest build any card needs: below it, at least one card warns.
	newest := 0
	for _, q := range models.CuratedSpeechModels() {
		newest = max(newest, q.MinBuild)
	}
	render := func(t *testing.T, s *Server) string {
		req := httptest.NewRequest(http.MethodGet, "/api/models/speech-presets", nil)
		req.Header.Set("HX-Request", "true")
		rec := httptest.NewRecorder()
		s.handleSpeechPresets(rec, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("status %d: %s", rec.Code, rec.Body)
		}
		return rec.Body.String()
	}

	t.Run("no build", func(t *testing.T) {
		body := render(t, newTestServer(t))
		for _, want := range []string{p.Name, p.Languages, p.VRAMLabel(), p.SizeLabel(), "/api/models/speech-presets/download"} {
			if !strings.Contains(body, want) {
				t.Errorf("card does not show %q", want)
			}
		}
		if strings.Contains(body, "Needs llama.cpp") {
			t.Error("warns about a build when none is known")
		}
	})

	for _, c := range []struct {
		name  string
		build int
		warn  bool
	}{
		{"older build", newest - 1, true},
		{"same build", newest, false},
	} {
		t.Run(c.name, func(t *testing.T) {
			s := stampServer(t, "", []builder.BuildResult{{ID: "b", Profile: "cpu", GitRef: fmt.Sprintf("b%d", c.build),
				GitSHA: "aaaaaaaaaa", Status: builder.BuildStatusSuccess, CommitCount: c.build}})
			body := render(t, s)
			if got := strings.Contains(body, "Needs llama.cpp"); got != c.warn {
				t.Errorf("build b%d: warning shown %v, want %v", c.build, got, c.warn)
			}
		})
	}

	t.Run("downloaded", func(t *testing.T) {
		s := newTestServer(t)
		if err := s.registry.Add(&models.Model{ID: "q", ModelID: p.Repo, Filename: p.Filename}); err != nil {
			t.Fatal(err)
		}
		if body := render(t, s); !strings.Contains(body, "Downloaded") {
			t.Error("a downloaded model is not marked")
		}
	})
}

// The card's route downloads a curated speech model and its projector,
// and nothing else.
func TestDownloadSpeechPreset(t *testing.T) {
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
	post := func(repo string) *httptest.ResponseRecorder {
		form := url.Values{"repo": {repo}}
		r := httptest.NewRequest(http.MethodPost, "/api/models/speech-presets/download", strings.NewReader(form.Encode()))
		r.Header.Set("Content-Type", "application/x-www-form-urlencoded")
		rec := httptest.NewRecorder()
		s.handleDownloadSpeechPreset(rec, r)
		return rec
	}

	for _, bad := range []string{"", "ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF", "org/Some-Chat-GGUF"} {
		if rec := post(bad); rec.Code != http.StatusBadRequest {
			t.Errorf("%q: status %d, want 400", bad, rec.Code)
		}
	}

	p := models.CuratedSpeechModels()[0]
	// The newest build any card needs: below it, at least one card warns.
	newest := 0
	for _, q := range models.CuratedSpeechModels() {
		newest = max(newest, q.MinBuild)
	}
	rec := post(p.Repo)
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), "mmproj_download_id") {
		t.Fatalf("curated %s: status %d: %s", p.Name, rec.Code, rec.Body)
	}
	want := []string{"/" + p.Repo + "/" + p.Filename, "/" + p.Repo + "/" + p.MmprojFilename}
	testutil.Eventually(t, func() bool {
		mu.Lock()
		defer mu.Unlock()
		return slices.Contains(fetched, want[0]) && slices.Contains(fetched, want[1])
	}, "the model and its projector were not both fetched")
	mu.Lock()
	defer mu.Unlock()
	for _, f := range fetched {
		if !slices.Contains(want, f) {
			t.Errorf("fetched %s, want only %v", f, want)
		}
	}
}
