package api

import (
	"context"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/tmac1973/llama-toolchest/internal/generation"
	"github.com/tmac1973/llama-toolchest/internal/huggingface"
	"github.com/tmac1973/llama-toolchest/internal/modelsource"
)

// genServer returns a test server with generation backends on, an empty
// store, and a downloader whose Hugging Face files come from a fake server
// that answers every file with "<repo>/<file>".
func genServer(t *testing.T) *Server {
	t.Helper()
	s := newTestServer(t)
	s.cfg.GenerationEnabled = true
	store, err := generation.Load(s.cfg.DataDir)
	if err != nil {
		t.Fatal(err)
	}
	s.gen = store
	fake := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(strings.TrimPrefix(r.URL.Path, "/")))
	}))
	t.Cleanup(fake.Close)
	s.downloader = huggingface.NewDownloader(s.cfg.DataDir, filepath.Join(s.cfg.DataDir, "models"), "")
	s.downloader.RegisterProvider(modelsource.SourceHuggingFace, huggingface.Provider{
		URL: func(repo, file string) string { return fake.URL + "/" + repo + "/" + file },
	})
	return s
}

// withID adds a chi URL parameter to a request.
func withID(r *http.Request, id string) *http.Request {
	rc := chi.NewRouteContext()
	rc.URLParams.Add("id", id)
	return r.WithContext(context.WithValue(r.Context(), chi.RouteCtxKey, rc))
}

func htmxReq(method, target string, form url.Values) *http.Request {
	var body *strings.Reader
	if form != nil {
		body = strings.NewReader(form.Encode())
	} else {
		body = strings.NewReader("")
	}
	r := httptest.NewRequest(method, target, body)
	r.Header.Set("HX-Request", "true")
	if form != nil {
		r.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	}
	return r
}

func TestGenModelRoutesHiddenWhileOff(t *testing.T) {
	s := genServer(t)
	s.cfg.GenerationEnabled = false
	rec := httptest.NewRecorder()
	s.handleGenModels(rec, htmxReq("GET", "/api/gen/models", nil))
	if rec.Code != http.StatusNotFound {
		t.Errorf("GET with the feature off = %d, want 404", rec.Code)
	}
}

// A catalog download fetches every file into gen-models (a folder of its
// own inside the LLM models dir), then marks the set complete.
func TestGenCatalogDownloadCompletes(t *testing.T) {
	s := genServer(t)
	// The fake serves short bodies, so the expected sizes are not checked
	// here; take the catalog entry as it is.
	rec := httptest.NewRecorder()
	s.handleGenCatalogDownload(rec, withID(htmxReq("POST", "/", nil), "flux2-klein-4b"))
	if rec.Code != http.StatusOK {
		t.Fatalf("download: %d %s", rec.Code, rec.Body.String())
	}

	deadline := time.Now().Add(10 * time.Second)
	var set generation.ModelSet
	for time.Now().Before(deadline) {
		set, _ = s.gen.ModelSet("flux2-klein-4b")
		if set.Complete {
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	if !set.Complete {
		t.Fatalf("set never completed: %+v", set)
	}
	root := generation.ModelsDir(s.cfg)
	for _, c := range set.Components {
		if !strings.HasPrefix(c.Path, root) {
			t.Errorf("%s landed at %s, outside %s", c.Slot, c.Path, root)
		}
		if got, _ := os.ReadFile(c.Path); string(got) != c.Repo+"/"+c.File {
			t.Errorf("%s holds %q", c.Slot, got)
		}
	}
	if entries, _ := os.ReadDir(filepath.Join(s.cfg.DataDir, "models")); len(entries) != 1 || entries[0].Name() != "gen-models" {
		t.Errorf("models dir holds %v, want only gen-models", entries)
	}
	if !strings.Contains(rec.Body.String(), "FLUX.2 klein 4B") {
		t.Error("response does not render the section")
	}

	again := httptest.NewRecorder()
	s.handleGenCatalogDownload(again, withID(htmxReq("POST", "/", nil), "flux2-klein-4b"))
	if again.Code != http.StatusConflict {
		t.Errorf("downloading a complete set again = %d, want 409", again.Code)
	}
}

// Deleting one ACE-Step set keeps the files the other still uses.
func TestGenDeleteKeepsSharedFiles(t *testing.T) {
	s := genServer(t)
	root := generation.ModelsDir(s.cfg)
	for _, id := range []string{"ace-step-turbo", "ace-step-xl-sft"} {
		e, _ := generation.LookupCatalog(id)
		set := generation.ModelSet{ID: e.ID, Name: e.Name, Kind: e.Kind, CatalogID: e.ID, Complete: true}
		for _, c := range e.Components {
			c.Path = huggingface.FilePath(root, c.Repo, c.File)
			if err := os.MkdirAll(filepath.Dir(c.Path), 0o755); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(c.Path, []byte("x"), 0o644); err != nil {
				t.Fatal(err)
			}
			set.Components = append(set.Components, c)
		}
		if err := s.gen.AddModelSet(set); err != nil {
			t.Fatal(err)
		}
	}

	rec := httptest.NewRecorder()
	s.handleGenDeleteSet(rec, withID(htmxReq("DELETE", "/", nil), "ace-step-turbo"))
	if rec.Code != http.StatusOK {
		t.Fatalf("delete: %d %s", rec.Code, rec.Body.String())
	}
	turbo, _ := generation.LookupCatalog("ace-step-turbo")
	for _, c := range turbo.Components {
		_, err := os.Stat(huggingface.FilePath(root, c.Repo, c.File))
		if c.Slot == "dit" && !os.IsNotExist(err) {
			t.Error("the turbo DiT was kept")
		}
		if c.Slot != "dit" && err != nil {
			t.Errorf("shared %s file deleted: %v", c.Slot, err)
		}
	}
}

func TestGenCustomSet(t *testing.T) {
	s := genServer(t)
	dir := t.TempDir()
	var lines []string
	for _, slot := range []string{"diffusion_model", "llm", "vae"} {
		p := filepath.Join(dir, slot+".gguf")
		if err := os.WriteFile(p, []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
		lines = append(lines, slot+" = "+p)
	}
	form := url.Values{
		"name": {"Qwen Turbo Local"}, "kind": {"sd"}, "recipe": {"qwen-image-21-turbo"},
		"files": {strings.Join(lines, "\n")}, "launch_args": {"--sigmas qwen-image-2.1-turbo"},
	}
	rec := httptest.NewRecorder()
	s.handleGenCreateSet(rec, htmxReq("POST", "/", form))
	if rec.Code != http.StatusOK {
		t.Fatalf("create: %d %s", rec.Code, rec.Body.String())
	}
	set, err := s.gen.ModelSet("qwen-turbo-local")
	if err != nil || !set.Complete || len(set.Components) != 3 || set.CatalogID != "" || set.DefaultWidth != 1024 {
		t.Errorf("stored set = %+v, %v", set, err)
	}

	for name, f := range map[string]url.Values{
		"missing file": {"name": {"a"}, "kind": {"sd"}, "recipe": {"sdxl-checkpoint"}, "files": {"model = " + dir + "/nope.safetensors"}},
		"missing slot": {"name": {"b"}, "kind": {"sd"}, "recipe": {"qwen-image-21-turbo"}, "files": {"diffusion_model = " + dir + "/vae.gguf"}},
		"denied arg":   {"name": {"c"}, "kind": {"sd"}, "recipe": {"sdxl-checkpoint"}, "files": {"model = " + dir + "/vae.gguf"}, "launch_args": {"--max-vram 4"}},
		"bad line":     {"name": {"d"}, "kind": {"sd"}, "recipe": {"sdxl-checkpoint"}, "files": {dir + "/vae.gguf"}},
		"same name":    form,
	} {
		rec := httptest.NewRecorder()
		s.handleGenCreateSet(rec, htmxReq("POST", "/", f))
		if rec.Code < 400 {
			t.Errorf("%s: %d, want an error", name, rec.Code)
		}
	}

	// Removing a custom set never touches the user's files.
	rec = httptest.NewRecorder()
	s.handleGenDeleteSet(rec, withID(htmxReq("DELETE", "/", nil), "qwen-turbo-local"))
	if rec.Code != http.StatusOK {
		t.Fatalf("delete: %d", rec.Code)
	}
	if _, err := os.Stat(filepath.Join(dir, "llm.gguf")); err != nil {
		t.Errorf("custom set's file was deleted: %v", err)
	}
}

func TestGenDeleteRefusedWhileInUse(t *testing.T) {
	s := genServer(t)
	if err := s.gen.AddModelSet(generation.ModelSet{ID: "m", Kind: generation.KindSD}); err != nil {
		t.Fatal(err)
	}
	if err := s.gen.AddInstance(generation.Instance{ID: "i", ModelSetID: "m"}); err != nil {
		t.Fatal(err)
	}
	rec := httptest.NewRecorder()
	s.handleGenDeleteSet(rec, withID(htmxReq("DELETE", "/", nil), "m"))
	if rec.Code != http.StatusConflict {
		t.Errorf("delete of a set in use = %d, want 409", rec.Code)
	}
}

// The JSON view uses snake_case keys throughout.
func TestGenModelsJSONKeys(t *testing.T) {
	s := genServer(t)
	if err := s.gen.AddModelSet(generation.ModelSet{ID: "m", Kind: generation.KindSD, Complete: true}); err != nil {
		t.Fatal(err)
	}
	rec := httptest.NewRecorder()
	s.handleGenModels(rec, httptest.NewRequest("GET", "/api/gen/models", nil))
	body := rec.Body.String()
	for _, key := range []string{`"catalog"`, `"sets"`, `"id":"m"`, `"complete":true`, `"percent":100`, `"size_gb"`} {
		if !strings.Contains(body, key) {
			t.Errorf("JSON lacks %s:\n%.400s", key, body)
		}
	}
}

// The catalog card lives on the Download Models tab and the sets on the
// Models tab. A download started from the catalog answers with the card
// and asks the downloads list to refresh; the Models tab's Resume gets
// the sets back.
func TestGenCatalogCardIsSeparateFromSets(t *testing.T) {
	s := genServer(t)
	rec := httptest.NewRecorder()
	s.handleGenCatalog(rec, htmxReq("GET", "/api/gen/models/catalog", nil))
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), "FLUX.2 klein 4B") {
		t.Fatalf("catalog card: %d %s", rec.Code, rec.Body.String())
	}
	if strings.Contains(rec.Body.String(), "Add a custom set") {
		t.Error("the catalog card carries the custom set form")
	}

	sets := httptest.NewRecorder()
	s.handleGenModels(sets, htmxReq("GET", "/api/gen/models", nil))
	if strings.Contains(sets.Body.String(), "/download?view=catalog") {
		t.Error("the Models tab section still carries the catalog")
	}

	dl := httptest.NewRecorder()
	s.handleGenCatalogDownload(dl, withID(htmxReq("POST", "/?view=catalog", nil), "flux2-klein-4b"))
	if dl.Code != http.StatusOK {
		t.Fatalf("download: %d %s", dl.Code, dl.Body.String())
	}
	if got := dl.Header().Get("HX-Trigger"); got != "dlPanelRefresh" {
		t.Errorf("HX-Trigger = %q, want dlPanelRefresh", got)
	}
	if strings.Contains(dl.Body.String(), "Add a custom set") || !strings.Contains(dl.Body.String(), "Generation models") {
		t.Errorf("a catalog download did not answer with the catalog card:\n%s", dl.Body.String())
	}
}

// A catalog row says what its set is doing: Download before it is
// started, Resume when files are missing, On this machine when complete.
func TestGenCatalogRowStatus(t *testing.T) {
	s := genServer(t)
	entry, _ := generation.LookupCatalog("flux2-klein-4b")
	row := func() genCatalogRow {
		for _, r := range s.genModelsView(s.gen).Catalog {
			if r.ID == entry.ID {
				return r
			}
		}
		t.Fatal("catalog entry missing")
		return genCatalogRow{}
	}
	if r := row(); r.Installed || r.Incomplete || r.Downloading {
		t.Errorf("before download: %+v", r)
	}
	set := generation.ModelSet{ID: entry.ID, Name: entry.Name, Kind: entry.Kind, CatalogID: entry.ID, Recipe: entry.Recipe, Components: entry.Components}
	if err := s.gen.AddModelSet(set); err != nil {
		t.Fatal(err)
	}
	if r := row(); !r.Incomplete || r.Installed {
		t.Errorf("with files missing: installed=%v incomplete=%v", r.Installed, r.Incomplete)
	}
	set.Complete = true
	if err := s.gen.UpdateModelSet(set); err != nil {
		t.Fatal(err)
	}
	if r := row(); !r.Installed || r.Incomplete {
		t.Errorf("complete: installed=%v incomplete=%v", r.Installed, r.Incomplete)
	}
}
