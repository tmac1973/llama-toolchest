package api

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/testutil"
)

// mdlModelsDir is the models folder of a newTestServer registry.
func mdlModelsDir(s *Server) string { return filepath.Join(s.cfg.DataDir, "models") }

// mdlAddFileModel writes a GGUF file under the models folder, in a
// folder of its own as downloads do, and registers it. Its presets are
// marked as checked, so a scan never starts a network fetch for it.
func mdlAddFileModel(t *testing.T, s *Server, repo, file string) *models.Model {
	t.Helper()
	dir := filepath.Join(mdlModelsDir(s), strings.ReplaceAll(repo, "/", "--"))
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, file)
	if err := os.WriteFile(path, []byte("not a real gguf"), 0o644); err != nil {
		t.Fatal(err)
	}
	m := &models.Model{
		ID: strings.ReplaceAll(repo, "/", "--") + "--" + strings.TrimSuffix(file, ".gguf"), ModelID: repo,
		Quant: "Q4_K_M", Filename: file, FilePath: path, PresetsCheckedAt: time.Now(),
	}
	if err := s.registry.Add(m); err != nil {
		t.Fatal(err)
	}
	cfg := models.DefaultConfig()
	cfg.Enabled = true
	if err := s.registry.SetConfig(m.ID, &cfg); err != nil {
		t.Fatal(err)
	}
	return m
}

// mdlDelete calls handleDeleteModel for the given name.
func mdlDelete(s *Server, name, query string, htmx bool) *httptest.ResponseRecorder {
	r := httptest.NewRequest(http.MethodDelete, "/api/models/"+name+query, nil)
	if htmx {
		r.Header.Set("HX-Request", "true")
	}
	rc := chi.NewRouteContext()
	rc.URLParams.Add("id", name)
	r = r.WithContext(context.WithValue(r.Context(), chi.RouteCtxKey, rc))
	rec := httptest.NewRecorder()
	s.handleDeleteModel(rec, r)
	return rec
}

// Deleting a model removes its registry entry, its file and the folder
// the file leaves empty, and writes the preset again so the router no
// longer lists the model. A JSON caller gets 204.
func TestMdlDeleteModelRemovesEntryAndFiles(t *testing.T) {
	s := newTestServer(t)
	m := mdlAddFileModel(t, s, "org/Gone-GGUF", "Gone-Q4_K_M.gguf")
	keep := mdlAddFileModel(t, s, "org/Kept-GGUF", "Kept-Q4_K_M.gguf")

	rec := mdlDelete(s, m.ID, "", false)
	if rec.Code != http.StatusNoContent {
		t.Fatalf("status %d, want 204: %s", rec.Code, rec.Body.String())
	}
	if _, err := s.registry.Get(m.ID); err == nil {
		t.Error("the model is still in the registry")
	}
	if _, err := os.Stat(m.FilePath); !os.IsNotExist(err) {
		t.Errorf("the model file is still on disk (stat err %v)", err)
	}
	if _, err := os.Stat(filepath.Dir(m.FilePath)); !os.IsNotExist(err) {
		t.Error("the model's empty folder was not removed")
	}
	if _, err := os.Stat(mdlModelsDir(s)); err != nil {
		t.Errorf("the models folder itself was removed: %v", err)
	}
	preset := readPreset(t, s, models.PresetFileName)
	if strings.Contains(preset, m.FilePath) || !strings.Contains(preset, keep.FilePath) {
		t.Errorf("the preset was not written again without the deleted model:\n%s", preset)
	}
}

// keep_files=true removes the model from the list but leaves its file,
// for a user who wants to keep the download for another program.
func TestMdlDeleteModelKeepFiles(t *testing.T) {
	s := newTestServer(t)
	m := mdlAddFileModel(t, s, "org/Kept-GGUF", "Kept-Q4_K_M.gguf")
	if rec := mdlDelete(s, m.ID, "?keep_files=true", false); rec.Code != http.StatusNoContent {
		t.Fatalf("status %d, want 204: %s", rec.Code, rec.Body.String())
	}
	if _, err := s.registry.Get(m.ID); err == nil {
		t.Error("the model is still in the registry")
	}
	if _, err := os.Stat(m.FilePath); err != nil {
		t.Errorf("keep_files=true removed the file: %v", err)
	}
}

// From the page (HTMX) a delete answers 200 with an HX-Trigger that tells
// the model list and the GPU map to reload. The model may be named by its
// public name as well as its registry ID.
func TestMdlDeleteModelFromPageTriggersRefresh(t *testing.T) {
	s := newTestServer(t)
	m := mdlAddFileModel(t, s, "org/Page-GGUF", "Page-Q4_K_M.gguf")
	rec := mdlDelete(s, m.PublicName(), "", true)
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d, want 200: %s", rec.Code, rec.Body.String())
	}
	var trig map[string]bool
	if err := json.Unmarshal([]byte(rec.Header().Get("HX-Trigger")), &trig); err != nil {
		t.Fatalf("HX-Trigger %q is not JSON: %v", rec.Header().Get("HX-Trigger"), err)
	}
	if !trig["modelsChanged"] || !trig["gpuMapChanged"] {
		t.Errorf("HX-Trigger = %v, want modelsChanged and gpuMapChanged", trig)
	}
	if _, err := s.registry.Get(m.ID); err == nil {
		t.Error("deleting by public name did not remove the model")
	}
}

// An unknown model is a 404. A read-only registry is a 409, and the file
// is kept: deleting the file without being able to record it would leave
// an entry for a missing file.
func TestMdlDeleteModelErrors(t *testing.T) {
	s := newTestServer(t)
	for _, q := range []string{"", "?keep_files=true"} {
		if rec := mdlDelete(s, "no-such-model", q, false); rec.Code != http.StatusNotFound {
			t.Errorf("unknown model%s: status %d, want 404", q, rec.Code)
		}
	}

	m := mdlAddFileModel(t, s, "org/Locked-GGUF", "Locked-Q4_K_M.gguf")
	// A models.json that cannot be parsed makes a registry read-only.
	path := filepath.Join(s.cfg.DataDir, "config", "models.json")
	if err := os.WriteFile(path, []byte("{broken"), 0o644); err != nil {
		t.Fatal(err)
	}
	s.registry = models.NewRegistry(s.cfg.DataDir, mdlModelsDir(s))
	if s.registry.ReadOnlyReason() == "" {
		t.Fatal("setup: the registry is not read-only")
	}
	// The read-only registry has no entries, so name a model it would
	// find if it could: the refusal comes before the lookup.
	if rec := mdlDelete(s, m.ID, "", false); rec.Code != http.StatusConflict {
		t.Errorf("read-only registry: status %d, want 409: %s", rec.Code, rec.Body.String())
	}
	if _, err := os.Stat(m.FilePath); err != nil {
		t.Errorf("a refused delete removed the file: %v", err)
	}
}

// A scan registers GGUF files found in the models folder that are not
// registered yet, reports how many, and a second scan finds none. A new
// model starts the preset fetch; for an embedding model that only stamps
// the attempt, so no network is used.
func TestMdlScanModelsRegistersNewFiles(t *testing.T) {
	s := newTestServer(t)
	dir := filepath.Join(mdlModelsDir(s), "nomic-ai--nomic-embed-text-v1.5-GGUF")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	file := filepath.Join(dir, "nomic-embed-text-v1.5.Q8_0.gguf")
	if err := os.WriteFile(file, []byte("not a real gguf"), 0o644); err != nil {
		t.Fatal(err)
	}
	// Files a scan must leave alone.
	for _, name := range []string{"half.gguf.part", "notes.txt", "mmproj-f16.gguf"} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	scan := func() int {
		t.Helper()
		rec := httptest.NewRecorder()
		s.handleScanModels(rec, httptest.NewRequest(http.MethodPost, "/api/models/scan", nil))
		if rec.Code != http.StatusOK {
			t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
		}
		var out map[string]int
		if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
			t.Fatalf("not JSON: %v\n%s", err, rec.Body.String())
		}
		return out["new_models"]
	}
	if n := scan(); n != 1 {
		t.Fatalf("first scan found %d new models, want 1", n)
	}
	var found *models.Model
	for _, m := range s.registry.List() {
		if m.FilePath == file {
			found = m
		}
	}
	if found == nil {
		t.Fatal("the scanned file is not in the registry")
	}
	if found.ModelID != "nomic-ai/nomic-embed-text-v1.5-GGUF" {
		t.Errorf("ModelID = %q, want the repo from the folder name", found.ModelID)
	}
	testutil.Eventually(t, func() bool {
		m, err := s.registry.Get(found.ID)
		return err == nil && !m.PresetsCheckedAt.IsZero()
	}, "the preset fetch was not started for the new model")

	if n := scan(); n != 0 {
		t.Errorf("second scan found %d new models, want 0", n)
	}
	if n := len(s.registry.List()); n != 1 {
		t.Errorf("registry holds %d models, want 1", n)
	}
}

// From the page (HTMX) a scan answers with the chat model list, so the
// page shows the models without a second request.
func TestMdlScanModelsFromPageRendersList(t *testing.T) {
	s := newTestServer(t)
	m := mdlAddFileModel(t, s, "org/Chat-GGUF", "Chat-Q4_K_M.gguf")
	rec := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodPost, "/api/models/scan", nil)
	r.Header.Set("HX-Request", "true")
	s.handleScanModels(rec, r)
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	if ct := rec.Header().Get("Content-Type"); !strings.HasPrefix(ct, "text/html") {
		t.Errorf("Content-Type = %q, want HTML", ct)
	}
	body := rec.Body.String()
	if !strings.Contains(body, "model-card-list") || !strings.Contains(body, m.ID) {
		t.Errorf("the answer is not the model list with %s:\n%s", m.ID, body)
	}
}

// filterModels splits the list by kind and keeps the order: chat models
// and embedding models are shown in separate sections, and a model must
// never appear in both or in neither.
func TestMdlFilterModels(t *testing.T) {
	chatA := &models.Model{ID: "a", ModelID: "org/Llama-GGUF"}
	emb := &models.Model{ID: "e", ModelID: "nomic-ai/nomic-embed-text-v1.5-GGUF"}
	chatB := &models.Model{ID: "b", ModelID: "org/Qwen-GGUF"}
	// Detected from the registry ID when the repo name says nothing.
	embByID := &models.Model{ID: "bge-small", ModelID: "org/small"}
	all := []*models.Model{chatA, emb, chatB, embByID}

	if got := filterModels(all, false); !reflect.DeepEqual(got, []*models.Model{chatA, chatB}) {
		t.Errorf("chat models = %v, want a and b in order", mdlIDs(got))
	}
	if got := filterModels(all, true); !reflect.DeepEqual(got, []*models.Model{emb, embByID}) {
		t.Errorf("embedding models = %v, want e and bge-small in order", mdlIDs(got))
	}
	if got := filterModels(nil, false); len(got) != 0 {
		t.Errorf("filterModels(nil) = %v, want empty", mdlIDs(got))
	}
}

func mdlIDs(list []*models.Model) []string {
	out := make([]string, 0, len(list))
	for _, m := range list {
		out = append(out, m.ID)
	}
	return out
}

// routerStateFor answers with the state of the first name the router
// knows, in the order given, so the router's own section name wins over
// an alias that may belong to an older entry.
func TestMdlRouterStateFor(t *testing.T) {
	states := map[string]string{"section": "loaded", "alias": "unloaded"}
	cases := []struct {
		names     []string
		want      string
		wantFound bool
	}{
		{[]string{"section", "alias"}, "loaded", true},
		{[]string{"alias", "section"}, "unloaded", true},
		{[]string{"missing", "alias"}, "unloaded", true},
		{[]string{"missing"}, "", false},
		{nil, "", false},
	}
	for _, c := range cases {
		got, found := routerStateFor(states, c.names...)
		if got != c.want || found != c.wantFound {
			t.Errorf("routerStateFor(%v) = %q, %v; want %q, %v", c.names, got, found, c.want, c.wantFound)
		}
	}
	if _, found := routerStateFor(nil, "section"); found {
		t.Error("a nil map (router down) reported a state")
	}
}
