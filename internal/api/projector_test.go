package api

import (
	"bytes"
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/go-chi/chi/v5"

	"github.com/tmac1973/llama-toolchest/internal/huggingface"
	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/modelsource"
)

// The Projector field is offered to every chat model without built-in
// vision, with the projectors on disk to pick from, and asks for a
// download offer when the model has none. Built-in vision and encoder
// models keep their own rows.
func TestProjectorFieldForEveryChatModel(t *testing.T) {
	base := testTemplates(t)
	render := func(data modelConfigPanelData) string {
		data.ModelID = "m"
		data.DraftModes, data.AssistModes = models.DraftModes(), models.AssistModes()
		var buf bytes.Buffer
		if err := base.ExecuteTemplate(&buf, "model_config", data); err != nil {
			t.Fatalf("execute: %v", err)
		}
		return buf.String()
	}

	none := render(modelConfigPanelData{
		Config:           &models.ModelConfig{},
		ProjectorChoices: []projectorChoice{{Path: "/models/b--X-GGUF/X.mmproj-Q8_0.gguf", Label: "b--X-GGUF/X.mmproj-Q8_0.gguf"}},
		ProjectorLookup:  true,
	})
	for _, want := range []string{
		`name="mmproj_path"`, `name="mmproj_enabled"`, `list="mmproj-choices-m"`,
		`<option value="/models/b--X-GGUF/X.mmproj-Q8_0.gguf">b--X-GGUF/X.mmproj-Q8_0.gguf</option>`,
		`hx-get="/api/models/m/projector-sources"`,
	} {
		if !strings.Contains(none, want) {
			t.Errorf("model without a projector: panel missing %s", want)
		}
	}

	set := render(modelConfigPanelData{Config: &models.ModelConfig{MmprojPath: "/p/mmproj.gguf"}, HasMMProj: true})
	if !strings.Contains(set, `value="/p/mmproj.gguf"`) || strings.Contains(set, "projector-sources") {
		t.Error("model with a projector: wrong Projector field")
	}
	if builtin := render(modelConfigPanelData{Config: &models.ModelConfig{}, HasBuiltinVision: true}); strings.Contains(builtin, `name="mmproj_path"`) {
		t.Error("built-in vision model shows the Projector field")
	}
	if enc := render(modelConfigPanelData{Config: &models.ModelConfig{}, IsEncoder: true, IsEmbedding: true}); strings.Contains(enc, `name="mmproj_path"`) {
		t.Error("embedding model shows the Projector field")
	}
}

// A path picked or typed in the Projector field is saved, and the switch
// beside it stays on.
func TestProjectorFieldSavesAPath(t *testing.T) {
	s := newTestServer(t)
	m := &models.Model{ID: "a--M-GGUF--m", ModelID: "a/M-GGUF", Filename: "m.gguf", FilePath: "/models/a--M-GGUF/m.gguf"}
	if err := s.registry.Add(m); err != nil {
		t.Fatal(err)
	}
	form := url.Values{"mmproj_path": {"/models/b--M-GGUF/M.mmproj-f16.gguf"}, "mmproj_enabled": {"on"}}
	req := httptest.NewRequest(http.MethodPut, "/api/models/"+m.ID+"/config", strings.NewReader(form.Encode()))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	rctx := chi.NewRouteContext()
	rctx.URLParams.Add("id", m.ID)
	req = req.WithContext(context.WithValue(req.Context(), chi.RouteCtxKey, rctx))
	s.handleUpdateModelConfig(httptest.NewRecorder(), req)

	cfg, _ := s.registry.GetConfig(m.ID)
	if cfg.MmprojPath != "/models/b--M-GGUF/M.mmproj-f16.gguf" || cfg.MmprojDisabled {
		t.Errorf("config = path %q, disabled %v", cfg.MmprojPath, cfg.MmprojDisabled)
	}
}

// projectorChoices lists the projector GGUFs in the models folder, and
// nothing else.
func TestProjectorChoicesListsProjectorsOnDisk(t *testing.T) {
	dir := t.TempDir()
	for _, p := range []string{"a--M/M.mmproj-Q8_0.gguf", "a--M/M.Q4_K_M.gguf", "b--N/Q8_0/mmproj-F16.gguf", "b--N/notes-mmproj.txt"} {
		os.MkdirAll(filepath.Join(dir, filepath.Dir(p)), 0o755)
		os.WriteFile(filepath.Join(dir, p), []byte("x"), 0o644)
	}
	got := projectorChoices(dir)
	var labels []string
	for _, c := range got {
		labels = append(labels, c.Label)
	}
	if strings.Join(labels, ",") != "a--M/M.mmproj-Q8_0.gguf,b--N/Q8_0/mmproj-F16.gguf" {
		t.Errorf("choices = %v", labels)
	}
}

// projectorFixture is a model from mradermacher's imatrix repository on
// disk, with no projector, a fake lookup that finds the static sibling,
// and a fake host that serves its projector.
type projectorFixture struct {
	s       *Server
	m       *models.Model
	lookups atomic.Int32
	router  chi.Router
}

func newProjectorFixture(t *testing.T, lookupErr error) *projectorFixture {
	t.Helper()
	s := newTestServer(t)
	modelsDir := filepath.Join(s.cfg.DataDir, "models")
	s.cfg.ModelsDir = modelsDir
	s.downloader = huggingface.NewDownloader(s.cfg.DataDir, modelsDir, "")
	files := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/mradermacher/ThinkingCap-Qwen3.8-27B-GGUF/ThinkingCap-Qwen3.8-27B.mmproj-Q8_0.gguf" {
			http.NotFound(w, r)
			return
		}
		w.Write([]byte("projector"))
	}))
	t.Cleanup(files.Close)
	s.downloader.RegisterProvider(modelsource.SourceHuggingFace, huggingface.Provider{
		URL: func(modelID, filename string) string { return files.URL + "/" + modelID + "/" + filename },
	})

	dir := filepath.Join(modelsDir, "mradermacher--ThinkingCap-Qwen3.8-27B-i1-GGUF")
	os.MkdirAll(dir, 0o755)
	m := &models.Model{
		ID: "mradermacher--ThinkingCap-Qwen3.8-27B-i1-GGUF--ThinkingCap-Qwen3.8-27B.i1-Q4_K_M.gguf", ModelID: "mradermacher/ThinkingCap-Qwen3.8-27B-i1-GGUF",
		Filename: "ThinkingCap-Qwen3.8-27B.i1-Q4_K_M.gguf", FilePath: filepath.Join(dir, "ThinkingCap-Qwen3.8-27B.i1-Q4_K_M.gguf"),
		BaseModelRepo: "bottlecapai/ThinkingCap-Qwen3.8-27B",
	}
	os.WriteFile(m.FilePath, []byte("model"), 0o644)
	if err := s.registry.Add(m); err != nil {
		t.Fatal(err)
	}
	f := &projectorFixture{s: s, m: m}
	s.findProjector = func(_ context.Context, repo, base string) (*huggingface.ProjectorRepo, error) {
		f.lookups.Add(1)
		if repo != m.ModelID || base != m.BaseModelRepo {
			t.Errorf("lookup for %q (base %q)", repo, base)
		}
		if lookupErr != nil {
			return nil, lookupErr
		}
		return &huggingface.ProjectorRepo{Repo: "mradermacher/ThinkingCap-Qwen3.8-27B-GGUF", Files: []huggingface.ModelFile{
			{Filename: "ThinkingCap-Qwen3.8-27B.mmproj-Q8_0.gguf", Quant: "Q8_0", Size: 9, IsMMProj: true},
			{Filename: "ThinkingCap-Qwen3.8-27B.mmproj-f16.gguf", Quant: "F16", Size: 928 << 20, IsMMProj: true},
		}}, nil
	}
	r := chi.NewRouter()
	r.Get("/api/models/{id}/projector-sources", s.handleProjectorSources)
	r.Post("/api/models/{id}/projector-download", s.handleProjectorDownload)
	f.router = r
	return f
}

func (f *projectorFixture) do(t *testing.T, method, path string, form url.Values) string {
	t.Helper()
	req := httptest.NewRequest(method, "/api/models/"+f.m.ID+path, strings.NewReader(form.Encode()))
	req.Header.Set("HX-Request", "true")
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	rec := httptest.NewRecorder()
	f.router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("%s %s: HTTP %d: %s", method, path, rec.Code, rec.Body.String())
	}
	return rec.Body.String()
}

// With the i1 repository installed, Configure offers the projector from
// the static repository; one click downloads it into the model's folder
// and pairs it.
func TestProjectorOfferDownloadsAndPairs(t *testing.T) {
	f := newProjectorFixture(t, nil)
	offer := f.do(t, http.MethodGet, "/projector-sources", nil)
	for _, want := range []string{
		"This repository has no projector.",
		`href="https://huggingface.co/mradermacher/ThinkingCap-Qwen3.8-27B-GGUF"`,
		"Q8_0 (", "F16 (928.0 MiB)",
		`hx-post="/api/models/` + f.m.ID + `/projector-download"`,
	} {
		if !strings.Contains(offer, want) {
			t.Errorf("offer missing %q:\n%s", want, offer)
		}
	}
	// The lookup is cached: a second render does not ask again.
	f.do(t, http.MethodGet, "/projector-sources", nil)
	if n := f.lookups.Load(); n != 1 {
		t.Errorf("%d lookups, want 1", n)
	}

	reply := f.do(t, http.MethodPost, "/projector-download", url.Values{
		"repo": {"mradermacher/ThinkingCap-Qwen3.8-27B-GGUF"}, "file": {"ThinkingCap-Qwen3.8-27B.mmproj-Q8_0.gguf"},
	})
	if !strings.Contains(reply, "Downloading ThinkingCap-Qwen3.8-27B.mmproj-Q8_0.gguf") {
		t.Fatalf("reply = %s", reply)
	}
	want := filepath.Join(filepath.Dir(f.m.FilePath), "ThinkingCap-Qwen3.8-27B.mmproj-Q8_0.gguf")
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		if cfg, _ := f.s.registry.GetConfig(f.m.ID); cfg.MmprojPath == want {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if cfg, _ := f.s.registry.GetConfig(f.m.ID); cfg.MmprojPath != want {
		t.Fatalf("projector = %q, want %q", cfg.MmprojPath, want)
	}
	if b, _ := os.ReadFile(want); string(b) != "projector" {
		t.Errorf("projector file = %q", b)
	}
	// Paired: the offer is not made any more.
	if body := f.do(t, http.MethodGet, "/projector-sources", nil); strings.TrimSpace(body) != "" {
		t.Errorf("offer after pairing: %s", body)
	}
}

// Only a file the lookup offered can be downloaded.
func TestProjectorDownloadRefusesAnUnofferedFile(t *testing.T) {
	f := newProjectorFixture(t, nil)
	reply := f.do(t, http.MethodPost, "/projector-download", url.Values{
		"repo": {"evil/repo"}, "file": {"../../etc/passwd"},
	})
	if !strings.Contains(reply, "no longer offered") {
		t.Errorf("reply = %s", reply)
	}
	if active := f.s.downloader.ListActive(); len(active) != 0 {
		t.Errorf("downloads started: %+v", active)
	}
}

// A failed lookup renders nothing, and is not repeated on the next
// render.
func TestProjectorLookupFailsQuietly(t *testing.T) {
	f := newProjectorFixture(t, errors.New("HTTP 503"))
	for range 2 {
		if body := f.do(t, http.MethodGet, "/projector-sources", nil); strings.TrimSpace(body) != "" {
			t.Errorf("failed lookup rendered %q", body)
		}
	}
	if n := f.lookups.Load(); n != 1 {
		t.Errorf("%d lookups, want 1", n)
	}
}
