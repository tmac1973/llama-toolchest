package api

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/generation"
	"github.com/tmac1973/llama-toolchest/internal/huggingface"
	"github.com/tmac1973/llama-toolchest/internal/modelsource"
	"github.com/tmac1973/llama-toolchest/internal/testutil"
)

const testCommit = "1111111111111111111111111111111111111111"

// hfFake is a Hugging Face repository API with fixed files; every branch
// points to testCommit.
type hfFake struct {
	mu       sync.Mutex
	files    map[string][]modelsource.RepoFile // repo → files
	err      error
	commits  int
	listings int
	// denied maps "repo/file" to the error a download of it would get.
	denied   map[string]error
	accesses int
}

func (f *hfFake) FileAccess(ctx context.Context, repo, rev, file string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.accesses++
	return f.denied[repo+"/"+file]
}

func (f *hfFake) RepoCommit(ctx context.Context, repo, rev string) (string, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.commits++
	if f.err != nil {
		return "", f.err
	}
	if _, ok := f.files[repo]; !ok {
		return "", fmt.Errorf("Hugging Face cannot find %s", repo)
	}
	if len(rev) == 40 {
		return rev, nil
	}
	return testCommit, nil
}

func (f *hfFake) RepoFilesAt(ctx context.Context, repo, rev string) ([]modelsource.RepoFile, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.listings++
	return f.files[repo], nil
}

// linkServer is genServer with a fake Hugging Face API and a download
// server that serves fakeFileBody, or body when set, and records the
// revisions asked for; revs returns them.
func linkServer(t *testing.T, body []byte) (s *Server, fake *hfFake, revs func() []string) {
	t.Helper()
	s = genServer(t)
	upscaler := fakeFileBody("org/up", "x4.pth")
	if body != nil {
		upscaler = body
	}
	fake = &hfFake{files: map[string][]modelsource.RepoFile{
		"org/up":  {{Path: "x4.pth", Size: int64(len(upscaler))}, {Path: "README.md", Size: 10}},
		"org/big": {{Path: "huge.safetensors", Size: 1 << 60}},
	}}
	s.hfRepoFake = fake
	var mu sync.Mutex
	var seen []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		parts := strings.SplitN(strings.TrimPrefix(r.URL.Path, "/"), "/", 4) // org/repo/rev/file
		mu.Lock()
		seen = append(seen, parts[2])
		mu.Unlock()
		w.Write(upscaler)
	}))
	t.Cleanup(srv.Close)
	s.downloader.RegisterProvider(modelsource.SourceHuggingFace, huggingface.Provider{
		URL: func(repo, file string) string { return srv.URL + "/" + repo + "/main/" + file },
		RevisionURL: func(repo, rev, file string) string {
			return srv.URL + "/" + repo + "/" + rev + "/" + file
		},
	})
	revs = func() []string {
		mu.Lock()
		defer mu.Unlock()
		return slices.Clone(seen)
	}
	return s, fake, revs
}

const upLink = "https://huggingface.co/org/up/blob/main/x4.pth"

func TestFormComponentsReadsLinks(t *testing.T) {
	local := filepath.Join(t.TempDir(), "vae.safetensors")
	r := htmxReq("POST", "/", url.Values{"slot_diffusion_model": {upLink}, "slot_vae": {local}})
	r.ParseForm()
	comps, err := formComponents(r)
	if err != nil {
		t.Fatal(err)
	}
	want := []generation.Component{
		{Slot: "diffusion_model", Repo: "org/up", File: "x4.pth", Revision: "main"},
		{Slot: "vae", Path: local},
	}
	if !slices.Equal(comps, want) {
		t.Errorf("components = %+v, want %+v", comps, want)
	}

	r = htmxReq("POST", "/", url.Values{"slot_lora": {upLink}})
	r.ParseForm()
	if _, err := formComponents(r); err == nil || !strings.Contains(err.Error(), "LoRA files must be local") {
		t.Errorf("a LoRA link: %v", err)
	}
}

func TestResolveHFComponents(t *testing.T) {
	s, fake, _ := linkServer(t, nil)
	in := []generation.Component{{Slot: "upscaler", Repo: "org/up", File: "x4.pth", Revision: "main"}, {Slot: "vae", Path: "/m/v.gguf"}}
	out, err := s.resolveHFComponents(context.Background(), in, nil)
	if err != nil {
		t.Fatal(err)
	}
	size := int64(len(fakeFileBody("org/up", "x4.pth")))
	if c := out[0]; c.Revision != testCommit || c.Bytes != size || c.Path != "" {
		t.Errorf("resolved %+v", c)
	}
	if out[1] != in[1] {
		t.Errorf("a local file changed: %+v", out[1])
	}

	// A file already at its download path is taken.
	path := huggingface.FilePath(generation.ModelsDir(s.cfg), "org/up", "x4.pth")
	os.MkdirAll(filepath.Dir(path), 0o755)
	os.WriteFile(path, fakeFileBody("org/up", "x4.pth"), 0o644)
	out, err = s.resolveHFComponents(context.Background(), in[:1], nil)
	if err != nil || out[0].Path != path {
		t.Errorf("on disk: %+v, %v", out, err)
	}

	// The same pinned file of the set being edited needs no request.
	before := fake.commits
	pinned := out[0]
	again := []generation.Component{{Slot: "upscaler", Repo: "org/up", File: "x4.pth", Revision: testCommit}}
	out, err = s.resolveHFComponents(context.Background(), again, []generation.Component{pinned})
	if err != nil || out[0] != pinned || fake.commits != before {
		t.Errorf("known file: %+v, %v, %d requests", out, err, fake.commits-before)
	}

	if _, err := s.resolveHFComponents(context.Background(),
		[]generation.Component{{Slot: "upscaler", Repo: "org/up", File: "x2.pth", Revision: "main"}}, nil); err == nil ||
		!strings.Contains(err.Error(), "x2.pth is not in org/up at main") {
		t.Errorf("missing file: %v", err)
	}
	fake.err = errors.New("org/gated is a gated model. Open https://huggingface.co/org/gated")
	if _, err := s.resolveHFComponents(context.Background(),
		[]generation.Component{{Slot: "upscaler", Repo: "org/up", File: "x4.pth"}}, nil); err == nil ||
		!strings.Contains(err.Error(), "upscaler: org/gated is a gated model") {
		t.Errorf("gated: %v", err)
	}
}

// A gated file is listed like any other, so Add set asks for access
// first: the set is refused before it is saved, and the size line says
// why. A file already on disk is not asked about.
func TestCreateSetRefusesGatedFile(t *testing.T) {
	s, fake, revs := linkServer(t, nil)
	fake.denied = map[string]error{"org/up/x4.pth": errors.New("org/up is a gated model. Open https://huggingface.co/org/up")}
	form := url.Values{"name": {"Gated"}, "kind": {"upscale"}, "slot_upscaler": {upLink}}

	rec := httptest.NewRecorder()
	s.handleGenCreateSet(rec, htmxReq("POST", "/", form))
	if rec.Code == http.StatusOK || !strings.Contains(rec.Body.String(), "upscaler: org/up is a gated model") {
		t.Errorf("add: HTTP %d %s", rec.Code, rec.Body)
	}
	if _, err := s.gen.ModelSet("gated"); !errors.Is(err, generation.ErrNotFound) {
		t.Errorf("the set was saved: %v", err)
	}
	if len(revs()) != 0 {
		t.Error("a download started")
	}

	rec = httptest.NewRecorder()
	s.handleGenSetFormSize(rec, htmxReq("GET", "/?"+form.Encode(), nil))
	if !strings.Contains(rec.Body.String(), "gated model") || !strings.Contains(rec.Body.String(), "pico-del-color") {
		t.Errorf("size line: %s", rec.Body)
	}

	path := huggingface.FilePath(generation.ModelsDir(s.cfg), "org/up", "x4.pth")
	os.MkdirAll(filepath.Dir(path), 0o755)
	os.WriteFile(path, fakeFileBody("org/up", "x4.pth"), 0o644)
	before := fake.accesses
	if _, err := s.resolveHFComponents(context.Background(), []generation.Component{{Slot: "upscaler", Repo: "org/up", File: "x4.pth"}}, nil); err != nil || fake.accesses != before {
		t.Errorf("on disk: %v, %d access checks", err, fake.accesses-before)
	}
}

func TestCreateSetFromLinkDownloads(t *testing.T) {
	s, _, revs := linkServer(t, nil)
	rec := httptest.NewRecorder()
	s.handleGenCreateSet(rec, htmxReq("POST", "/", url.Values{"name": {"Up Link"}, "kind": {"upscale"}, "slot_upscaler": {upLink}}))
	if rec.Code != http.StatusOK {
		t.Fatalf("HTTP %d %s", rec.Code, rec.Body)
	}
	if rec.Header().Get("HX-Trigger") != "dlPanelRefresh" {
		t.Error("no download panel refresh")
	}
	var set generation.ModelSet
	testutil.EventuallyMsg(t, func() bool {
		set, _ = s.gen.ModelSet("up-link")
		return set.Complete
	}, func() string { return fmt.Sprintf("set never completed: %+v", set) })
	c := set.Components[0]
	if c.Revision != testCommit || c.Path != huggingface.FilePath(generation.ModelsDir(s.cfg), "org/up", "x4.pth") || c.Error != "" {
		t.Errorf("component %+v", c)
	}
	if got := revs(); !slices.Equal(got, []string{testCommit}) {
		t.Errorf("downloaded revisions %v, want the pinned commit", got)
	}
}

func TestCreateSetRefusesWhatDoesNotFit(t *testing.T) {
	s, _, _ := linkServer(t, nil)
	if s.downloader.AvailableForDownload() < 0 {
		t.Skip("free space is unknown here")
	}
	rec := httptest.NewRecorder()
	s.handleGenCreateSet(rec, htmxReq("POST", "/", url.Values{"name": {"Big"}, "kind": {"upscale"},
		"slot_upscaler": {"https://huggingface.co/org/big/blob/main/huge.safetensors"}}))
	if rec.Code != http.StatusInsufficientStorage || !strings.Contains(rec.Body.String(), "GB is free in the models folder") {
		t.Errorf("HTTP %d %s", rec.Code, rec.Body)
	}
	if _, err := s.gen.ModelSet("big"); !errors.Is(err, generation.ErrNotFound) {
		t.Error("a set that does not fit was saved")
	}
}

// A downloaded file that is not a model file is deleted, its reason kept,
// and the row says so; Resume clears it and downloads again.
func TestDownloadedWrongFileIsRefused(t *testing.T) {
	s, _, revs := linkServer(t, []byte("<!DOCTYPE html><html>not found</html>"))
	rec := httptest.NewRecorder()
	s.handleGenCreateSet(rec, htmxReq("POST", "/", url.Values{"name": {"Bad"}, "kind": {"upscale"}, "slot_upscaler": {upLink}}))
	if rec.Code != http.StatusOK {
		t.Fatalf("HTTP %d %s", rec.Code, rec.Body)
	}
	var set generation.ModelSet
	testutil.EventuallyMsg(t, func() bool {
		set, _ = s.gen.ModelSet("bad")
		return set.Components[0].Error != ""
	}, func() string { return fmt.Sprintf("no error recorded: %+v", set) })
	if set.Complete || set.Components[0].Path != "" {
		t.Errorf("a refused file left %+v", set)
	}
	if _, err := os.Stat(huggingface.FilePath(generation.ModelsDir(s.cfg), "org/up", "x4.pth")); !errors.Is(err, os.ErrNotExist) {
		t.Errorf("the refused file was kept: %v", err)
	}
	testutil.EventuallyMsg(t, func() bool { return !s.setDownloading("bad") }, func() string { return "still downloading" })
	row := s.genSetRowFor(set)
	if !strings.Contains(row.FileError, "is not a model file") || !row.Resumable {
		t.Errorf("row %+v", row)
	}
	var page strings.Builder
	s.renderPartial(&httptestWriter{&page}, "gen_model_sets", s.genModelsView(s.gen))
	if !strings.Contains(page.String(), "Incomplete (wrong file)") || !strings.Contains(page.String(), "/api/gen/models/sets/bad/download") {
		t.Error("the row does not show the wrong file and its Resume")
	}

	rec = httptest.NewRecorder()
	s.handleGenSetDownload(rec, withID(htmxReq("POST", "/", nil), "bad"))
	if rec.Code != http.StatusOK {
		t.Fatalf("resume: HTTP %d %s", rec.Code, rec.Body)
	}
	// The server still serves the bad file, so the resumed download is
	// refused again: proof that Resume downloaded it once more.
	testutil.EventuallyMsg(t, func() bool {
		set, _ = s.gen.ModelSet("bad")
		return len(revs()) == 2 && !s.setDownloading("bad") && set.Components[0].Error != ""
	}, func() string {
		return fmt.Sprintf("resume did not download again: %d downloads, %+v", len(revs()), set)
	})
}

func TestResumeRefusals(t *testing.T) {
	s, _, _ := linkServer(t, nil)
	local := filepath.Join(t.TempDir(), "x4.pth")
	os.WriteFile(local, fakeGGUF, 0o644)
	sets := []generation.ModelSet{
		{ID: "done", Name: "Done", Kind: generation.KindUpscale, Complete: true,
			Components: []generation.Component{{Slot: "upscaler", Path: local, Bytes: int64(len(fakeGGUF))}}},
		{ID: "catalog", Name: "Catalog", Kind: generation.KindUpscale, CatalogID: "x",
			Components: []generation.Component{{Slot: "upscaler", Repo: "org/up", File: "x4.pth"}}},
		{ID: "local", Name: "Local", Kind: generation.KindUpscale,
			Components: []generation.Component{{Slot: "upscaler"}}},
	}
	for _, set := range sets {
		if err := s.gen.AddModelSet(set); err != nil {
			t.Fatal(err)
		}
	}
	for id, want := range map[string]int{"done": http.StatusConflict, "catalog": http.StatusBadRequest,
		"local": http.StatusBadRequest, "none": http.StatusNotFound} {
		rec := httptest.NewRecorder()
		s.handleGenSetDownload(rec, withID(htmxReq("POST", "/", nil), id))
		if rec.Code != want {
			t.Errorf("%s: HTTP %d, want %d", id, rec.Code, want)
		}
	}
	if s.genSetRowFor(sets[2]).Resumable {
		t.Error("a set with only local files offers Resume")
	}
}

// Delete removes a custom set's downloaded file and its emptied folder,
// keeps one another set uses, and never touches a local file.
func TestDeleteCustomSetFiles(t *testing.T) {
	s, _, _ := linkServer(t, nil)
	root := generation.ModelsDir(s.cfg)
	own := huggingface.FilePath(root, "org/up", "x4.pth")
	shared := huggingface.FilePath(root, "org/vae", "ae.safetensors")
	local := filepath.Join(t.TempDir(), "mine.gguf")
	for _, p := range []string{own, shared, local} {
		os.MkdirAll(filepath.Dir(p), 0o755)
		os.WriteFile(p, fakeGGUF, 0o644)
	}
	n := int64(len(fakeGGUF))
	for _, set := range []generation.ModelSet{
		{ID: "mixed", Name: "Mixed", Kind: generation.KindSD, Recipe: "z-image-turbo", Complete: true, Components: []generation.Component{
			{Slot: "diffusion_model", Repo: "org/up", File: "x4.pth", Path: own, Bytes: n, Revision: testCommit},
			{Slot: "vae", Repo: "org/vae", File: "ae.safetensors", Path: shared, Bytes: n},
			{Slot: "llm", Path: local, Bytes: n},
		}},
		{ID: "other", Name: "Other", Kind: generation.KindSD, Recipe: "z-image-turbo", Complete: true, Components: []generation.Component{
			{Slot: "vae", Repo: "org/vae", File: "ae.safetensors", Path: shared, Bytes: n},
		}},
	} {
		if err := s.gen.AddModelSet(set); err != nil {
			t.Fatal(err)
		}
	}
	rec := httptest.NewRecorder()
	s.handleGenDeleteSet(rec, withID(htmxReq("DELETE", "/", nil), "mixed"))
	if rec.Code != http.StatusOK {
		t.Fatalf("HTTP %d %s", rec.Code, rec.Body)
	}
	// The emptied repository folder goes too; the folder of the shared
	// file and the gen-models folder stay.
	for p, keep := range map[string]bool{own: false, shared: true, local: true,
		filepath.Dir(own): false, filepath.Dir(shared): true, root: true} {
		if _, err := os.Stat(p); (err == nil) != keep {
			t.Errorf("%s: kept=%v, want %v", p, err == nil, keep)
		}
	}
}

// The edit form shows a downloaded file as its link, so saving the edit
// keeps it a downloaded file; the save needs no Hugging Face request.
func TestEditKeepsLinks(t *testing.T) {
	s, fake, _ := linkServer(t, nil)
	path := huggingface.FilePath(generation.ModelsDir(s.cfg), "org/up", "x4.pth")
	body := fakeFileBody("org/up", "x4.pth")
	os.MkdirAll(filepath.Dir(path), 0o755)
	os.WriteFile(path, body, 0o644)
	set := generation.ModelSet{ID: "up", Name: "Up", Kind: generation.KindUpscale, Scale: 4, Complete: true,
		Components: []generation.Component{{Slot: "upscaler", Repo: "org/up", File: "x4.pth", Revision: testCommit, Path: path, Bytes: int64(len(body))}}}
	if err := s.gen.AddModelSet(set); err != nil {
		t.Fatal(err)
	}
	f, code, err := s.genSetEditForm(htmxReq("GET", "/?edit=up", nil), s.gen)
	if err != nil {
		t.Fatalf("%d %v", code, err)
	}
	link := generation.HFLink("org/up", testCommit, "x4.pth")
	if f.Slots[0].Value != link {
		t.Errorf("edit form holds %q, want %q", f.Slots[0].Value, link)
	}

	before := fake.commits + fake.listings
	rec := httptest.NewRecorder()
	s.handleGenUpdateSet(rec, withID(htmxReq("PUT", "/", url.Values{"slot_upscaler": {link}, "scale": {"2"}}), "up"))
	if rec.Code != http.StatusOK {
		t.Fatalf("HTTP %d %s", rec.Code, rec.Body)
	}
	got, _ := s.gen.ModelSet("up")
	if got.Scale != 2 || !got.Complete || got.Components[0] != set.Components[0] {
		t.Errorf("after the edit: %+v", got)
	}
	if fake.commits+fake.listings != before {
		t.Error("an edit that kept the file asked Hugging Face")
	}
}

// httptestWriter lets renderPartial write into a strings.Builder.
type httptestWriter struct{ b *strings.Builder }

func (w *httptestWriter) Header() http.Header         { return http.Header{} }
func (w *httptestWriter) Write(p []byte) (int, error) { return w.b.Write(p) }
func (w *httptestWriter) WriteHeader(int)             {}
