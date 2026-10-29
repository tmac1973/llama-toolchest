package api

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/go-chi/chi/v5"

	"github.com/tmac1973/llama-toolchest/internal/huggingface"
	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/modelsource"
)

func sha256Hex(s string) string {
	h := sha256.Sum256([]byte(s))
	return hex.EncodeToString(h[:])
}

// updateFixture is a ModelScope model on disk, org/m with m.gguf and a
// linked mmproj, and a fake ModelScope that lists the repository with
// the given contents and serves them.
type updateFixture struct {
	s        *Server
	id       string
	modelDir string
	router   chi.Router
}

func newUpdateFixture(t *testing.T, onDisk, upstream map[string]string) *updateFixture {
	t.Helper()
	s := hfhServer(t)
	modelsDir := filepath.Join(s.cfg.DataDir, "models")
	s.cfg.ModelsDir = modelsDir
	s.downloader = huggingface.NewDownloader(s.cfg.DataDir, modelsDir, "")
	s.downloader.SetHashStore(s.hashes)
	s.downloader.SetOnComplete(s.onDownloadComplete)

	modelDir := filepath.Join(modelsDir, "org--m")
	os.MkdirAll(modelDir, 0o755)
	for name, body := range onDisk {
		if err := os.WriteFile(filepath.Join(modelDir, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	m := &models.Model{
		ID: "org--m--m", ModelID: "org/m", Filename: "m.gguf", Source: modelsource.SourceModelScope,
		FilePath: filepath.Join(modelDir, "m.gguf"), HelperRole: true,
	}
	if err := s.registry.Add(m); err != nil {
		t.Fatal(err)
	}
	cfg, _ := s.registry.GetConfig(m.ID)
	cfg.MmprojPath = filepath.Join(modelDir, "mmproj-F16.gguf")
	s.registry.SetConfig(m.ID, cfg)

	var listing []string
	for name, body := range upstream {
		listing = append(listing, fmt.Sprintf(`{"Name":%q,"Path":%q,"Size":%d,"Type":"blob","Sha256":%q}`,
			name, name, len(body), sha256Hex(body)))
	}
	hfhModelScope(t, s, func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, `{"Code":200,"Success":true,"Data":{"Files":[%s]}}`, strings.Join(listing, ","))
	})
	files := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, upstream[strings.TrimPrefix(r.URL.Path, "/org/m/")])
	}))
	t.Cleanup(files.Close)
	s.downloader.RegisterProvider(modelsource.SourceModelScope, huggingface.Provider{
		URL: func(modelID, filename string) string { return files.URL + "/" + modelID + "/" + filename },
	})

	r := chi.NewRouter()
	r.Post("/api/models/{id}/update/check", s.handleModelUpdateCheck)
	r.Get("/api/models/{id}/update", s.handleModelUpdateStatus)
	r.Post("/api/models/{id}/update", s.handleModelUpdateApply)
	return &updateFixture{s: s, id: m.ID, modelDir: modelDir, router: r}
}

func (f *updateFixture) do(t *testing.T, method, path string) string {
	t.Helper()
	req := httptest.NewRequest(method, "/api/models/"+f.id+path, nil)
	req.Header.Set("HX-Request", "true")
	rec := httptest.NewRecorder()
	f.router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("%s %s: HTTP %d: %s", method, path, rec.Code, rec.Body.String())
	}
	return rec.Body.String()
}

// check starts a check and waits for its result.
func (f *updateFixture) check(t *testing.T) string {
	t.Helper()
	f.do(t, http.MethodPost, "/update/check")
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if body := f.do(t, http.MethodGet, "/update"); !strings.Contains(body, `hx-trigger="every 1s"`) {
			return body
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("check did not finish")
	return ""
}

// The check finds the one changed file, and the update replaces only
// that file, keeping what the old record knew that the file cannot say.
func TestModelUpdateReplacesOnlyChangedFiles(t *testing.T) {
	f := newUpdateFixture(t,
		map[string]string{"m.gguf": "old model", "mmproj-F16.gguf": "projector"},
		map[string]string{"m.gguf": "new model, fixed template", "mmproj-F16.gguf": "projector", "README.md": "readme"})

	body := f.check(t)
	if !strings.Contains(body, "newer version of 1 file") || !strings.Contains(body, "<code>m.gguf</code>") {
		t.Fatalf("check result does not list m.gguf as the one changed file:\n%s", body)
	}
	if strings.Contains(body, "<code>mmproj-F16.gguf</code>") {
		t.Errorf("unchanged mmproj listed as changed:\n%s", body)
	}
	mmproj := filepath.Join(f.modelDir, "mmproj-F16.gguf")
	if sha, ok := f.s.hashes.Lookup(mmproj); !ok || sha != sha256Hex("projector") {
		t.Errorf("the mmproj's hash was not saved by the check: %q, %v", sha, ok)
	}

	body = f.do(t, http.MethodPost, "/update")
	if !strings.Contains(body, "Downloading the update") {
		t.Fatalf("apply did not start:\n%s", body)
	}
	main := filepath.Join(f.modelDir, "m.gguf")
	deadline := time.Now().Add(5 * time.Second)
	for {
		got, _ := os.ReadFile(main)
		if string(got) == "new model, fixed template" {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("m.gguf = %q, want the new version", got)
		}
		time.Sleep(10 * time.Millisecond)
	}
	if got, _ := os.ReadFile(mmproj); string(got) != "projector" {
		t.Errorf("mmproj changed: %q", got)
	}
	// The completion callback runs after the rename; wait for it.
	deadline = time.Now().Add(5 * time.Second)
	for {
		if m, err := f.s.registry.Get(f.id); err == nil && m.SizeBytes == int64(len("new model, fixed template")) {
			if !m.HelperRole {
				t.Error("the update dropped the model's helper role")
			}
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("the model was not registered again after the update")
		}
		time.Sleep(10 * time.Millisecond)
	}

	if body := f.check(t); !strings.Contains(body, "up to date") {
		t.Errorf("a check after the update does not report up to date:\n%s", body)
	}
}

// A file the repository no longer has is reported and left alone.
func TestModelUpdateReportsFilesGoneFromRepo(t *testing.T) {
	f := newUpdateFixture(t,
		map[string]string{"m.gguf": "model", "mmproj-F16.gguf": "projector"},
		map[string]string{"m.gguf": "model"})
	body := f.check(t)
	if !strings.Contains(body, "up to date") || !strings.Contains(body, "Not in the repository any more: <code>mmproj-F16.gguf</code>") {
		t.Errorf("result:\n%s", body)
	}
}

// A model that did not come from a repository has no Update button.
func TestModelCardHidesUpdateForLocalModels(t *testing.T) {
	s := newTestServer(t)
	s.cfg.ModelsDir = filepath.Join(s.cfg.DataDir, "models")
	repo := &models.Model{ID: "a", ModelID: "org/m", FilePath: filepath.Join(s.cfg.ModelsDir, "org--m", "m.gguf")}
	local := &models.Model{ID: "b", ModelID: "m", FilePath: "/elsewhere/m.gguf"}
	if _, ok := s.updateTargets(repo); !ok {
		t.Error("a model in its repository folder cannot be updated")
	}
	if _, ok := s.updateTargets(local); ok {
		t.Error("a model from outside the models folder can be updated")
	}
}
