package api

import (
	"bytes"
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/go-chi/chi/v5"
	"github.com/tmac1973/llama-toolchest/internal/forks"
	"github.com/tmac1973/llama-toolchest/internal/huggingface"
	"github.com/tmac1973/llama-toolchest/internal/modelsource"
	"github.com/tmac1973/llama-toolchest/internal/testutil"
)

// testFork is a small stand-in for Avalon: three shards, a draft model in
// a subfolder and a ranking file. It keeps the ID "avalon" so the fork
// store, which checks IDs against the real list, accepts its instances.
func testFork() forks.Fork {
	f, _ := forks.Find("avalon")
	f.ModelRepo = "test/fork-model"
	f.ModelRevision = "1111111111111111111111111111111111111111"
	f.ModelFiles = []forks.ModelFile{
		{Path: "m-00001-of-00003.gguf", Bytes: 10},
		{Path: "m-00002-of-00003.gguf", Bytes: 20},
		{Path: "m-00003-of-00003.gguf", Bytes: 30},
		{Path: "MTP/draft.gguf", Bytes: 5},
		{Path: "rank.bin", Bytes: 3},
	}
	f.Recipe.ModelFileArg, f.Recipe.DraftFileArg = "m-00001-of-00003.gguf", "MTP/draft.gguf"
	return f
}

// forkServer is a test server with forks on, an empty fork store, the
// test fork as the only curated one, and a downloader whose files come
// from a fake server. The fake answers each file with as many bytes as
// the fork lists for it, counts the requests per file, and records the
// revision each request named.
type forkFake struct {
	mu        sync.Mutex
	gets      map[string]int
	revisions map[string]bool
}

func forkServer(t *testing.T) (*Server, *forkFake) {
	t.Helper()
	f := testFork()
	old := curatedForks
	curatedForks = func() []forks.Fork { return []forks.Fork{f} }
	t.Cleanup(func() { curatedForks = old })

	s := newTestServer(t)
	s.cfg.ForksEnabled = true
	store, err := forks.Load(s.cfg.DataDir)
	if err != nil {
		t.Fatal(err)
	}
	s.forks = store
	fake := &forkFake{gets: map[string]int{}, revisions: map[string]bool{}}
	sizes := map[string]int64{}
	for _, m := range f.ModelFiles {
		sizes[m.Path] = m.Bytes
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// /<repo>/<revision>/<file>
		parts := strings.SplitN(strings.TrimPrefix(r.URL.Path, "/"), "/", 4)
		rev, file := parts[2], parts[3]
		size, ok := sizes[file]
		if !ok {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Length", fmt.Sprint(size))
		if r.Method == http.MethodGet {
			fake.mu.Lock()
			fake.gets[file]++
			fake.revisions[rev] = true
			fake.mu.Unlock()
			w.Write(bytes.Repeat([]byte("x"), int(size)))
		}
	}))
	t.Cleanup(srv.Close)
	s.downloader = huggingface.NewDownloader(s.cfg.DataDir, filepath.Join(s.cfg.DataDir, "models"), "")
	s.downloader.RegisterProvider(modelsource.SourceHuggingFace, huggingface.Provider{
		URL: func(repo, file string) string { return srv.URL + "/" + repo + "/main/" + file },
		RevisionURL: func(repo, rev, file string) string {
			return srv.URL + "/" + repo + "/" + rev + "/" + file
		},
	})
	return s, fake
}

func withFork(r *http.Request, id string) *http.Request {
	rc := chi.NewRouteContext()
	rc.URLParams.Add("fork", id)
	return r.WithContext(context.WithValue(r.Context(), chi.RouteCtxKey, rc))
}

func TestForkModelRoutesHiddenWhileOff(t *testing.T) {
	s, _ := forkServer(t)
	s.cfg.ForksEnabled = false
	for name, h := range map[string]http.HandlerFunc{
		"list": s.handleForkModels, "download": s.handleForkModelDownload, "delete": s.handleForkModelDelete,
	} {
		rec := httptest.NewRecorder()
		h(rec, withFork(htmxReq("POST", "/", nil), "avalon"))
		if rec.Code != http.StatusNotFound {
			t.Errorf("%s with forks off = %d, want 404", name, rec.Code)
		}
	}
}

// A download fetches every file from the pinned revision into
// fork-models, with one download for all the shards, then marks the set
// complete.
func TestForkModelDownloadCompletes(t *testing.T) {
	s, fake := forkServer(t)
	f := testFork()
	rec := httptest.NewRecorder()
	s.handleForkModelDownload(rec, withFork(htmxReq("POST", "/", nil), "avalon"))
	if rec.Code != http.StatusOK {
		t.Fatalf("download: %d %s", rec.Code, rec.Body)
	}
	testutil.EventuallyMsg(t, func() bool {
		set, _ := s.forks.ModelSet("avalon")
		return set.Complete
	}, func() string { return "set never completed" })

	dir := s.forkSetDir(f)
	if !strings.HasPrefix(dir, filepath.Join(s.cfg.DataDir, "models", "fork-models")) {
		t.Errorf("set folder %s is not under fork-models", dir)
	}
	for _, m := range f.ModelFiles {
		if !forkFileOnDisk(dir, m) {
			t.Errorf("%s not in place", m.Path)
		}
	}
	fake.mu.Lock()
	defer fake.mu.Unlock()
	for file, n := range fake.gets {
		if n != 1 {
			t.Errorf("%s fetched %d times, want once", file, n)
		}
	}
	if len(fake.revisions) != 1 || !fake.revisions[f.ModelRevision] {
		t.Errorf("revisions fetched = %v, want only the pinned one", fake.revisions)
	}

	again := httptest.NewRecorder()
	s.handleForkModelDownload(again, withFork(htmxReq("POST", "/", nil), "avalon"))
	if again.Code != http.StatusConflict {
		t.Errorf("second download = %d, want 409", again.Code)
	}
	list := httptest.NewRecorder()
	s.handleForkModels(list, htmxReq("GET", "/", nil))
	if !strings.Contains(list.Body.String(), "Ready") {
		t.Errorf("section does not show the set as ready:\n%s", list.Body)
	}
}

// Shards go in one group, so the downloader (which fetches every shard
// when given one) is started once for them.
func TestForkFileGroups(t *testing.T) {
	groups := forkFileGroups(testFork())
	if len(groups) != 3 {
		t.Fatalf("groups = %+v, want shards, draft, ranking", groups)
	}
	if g := groups[0]; g.First != "m-00001-of-00003.gguf" || len(g.Files) != 3 || g.Bytes != 60 {
		t.Errorf("shard group = %+v", g)
	}
	real, _ := forks.Find("avalon")
	if g := forkFileGroups(real); len(g) != 3 || len(g[0].Files) != 33 {
		t.Errorf("Avalon groups: %d, first has %d files", len(g), len(g[0].Files))
	}
}

// Files already in place at their size are kept; a wrong-sized file is
// fetched again, and the set is not complete until it is.
func TestForkModelResumeKeepsGoodFiles(t *testing.T) {
	s, fake := forkServer(t)
	f := testFork()
	dir := s.forkSetDir(f)
	os.MkdirAll(dir, 0o755)
	os.WriteFile(filepath.Join(dir, "rank.bin"), []byte("xxx"), 0o644)
	os.MkdirAll(filepath.Join(dir, "MTP"), 0o755)
	os.WriteFile(filepath.Join(dir, "MTP", "draft.gguf"), []byte("xx"), 0o644)
	os.WriteFile(filepath.Join(dir, "m-00001-of-00003.gguf"), bytes.Repeat([]byte("x"), 10), 0o644)
	os.WriteFile(filepath.Join(dir, "m-00002-of-00003.gguf"), []byte("x"), 0o644)

	rec := httptest.NewRecorder()
	s.handleForkModelDownload(rec, withFork(htmxReq("POST", "/", nil), "avalon"))
	if rec.Code != http.StatusOK {
		t.Fatalf("download: %d %s", rec.Code, rec.Body)
	}
	testutil.EventuallyMsg(t, func() bool {
		set, _ := s.forks.ModelSet("avalon")
		return set.Complete
	}, func() string { return "set never completed" })
	fake.mu.Lock()
	defer fake.mu.Unlock()
	if fake.gets["rank.bin"] != 0 {
		t.Error("a file already in place was fetched again")
	}
	if fake.gets["MTP/draft.gguf"] != 1 {
		t.Errorf("the wrong-sized file was fetched %d times, want once", fake.gets["MTP/draft.gguf"])
	}
	if fake.gets["m-00001-of-00003.gguf"] != 0 {
		t.Error("a shard already in place was fetched again")
	}
	if fake.gets["m-00002-of-00003.gguf"] != 1 {
		t.Errorf("the wrong-sized shard was fetched %d times, want once", fake.gets["m-00002-of-00003.gguf"])
	}
}

func TestForkModelDelete(t *testing.T) {
	s, _ := forkServer(t)
	f := testFork()
	dir := s.forkSetDir(f)
	os.MkdirAll(dir, 0o755)
	os.WriteFile(filepath.Join(dir, "rank.bin"), []byte("xxx"), 0o644)
	if err := s.forks.PutModelSet(forks.ModelSet{ForkID: "avalon", Dir: dir}); err != nil {
		t.Fatal(err)
	}
	if err := s.forks.AddInstance(forks.Instance{ID: "q", ForkID: "avalon"}); err != nil {
		t.Fatal(err)
	}
	rec := httptest.NewRecorder()
	s.handleForkModelDelete(rec, withFork(htmxReq("DELETE", "/", nil), "avalon"))
	if rec.Code != http.StatusConflict {
		t.Errorf("delete with an instance = %d, want 409", rec.Code)
	}

	s.forks.DeleteInstance("q")
	rec = httptest.NewRecorder()
	s.handleForkModelDelete(rec, withFork(htmxReq("DELETE", "/", nil), "avalon"))
	if rec.Code != http.StatusOK {
		t.Fatalf("delete: %d %s", rec.Code, rec.Body)
	}
	if _, err := os.Stat(dir); !os.IsNotExist(err) {
		t.Error("set folder still there")
	}
	if _, err := s.forks.ModelSet("avalon"); err == nil {
		t.Error("store entry still there")
	}
}

// A stored folder outside fork-models is never deleted.
func TestForkModelDeleteStaysInsideForkModels(t *testing.T) {
	s, _ := forkServer(t)
	outside := t.TempDir()
	keep := filepath.Join(outside, "keep")
	os.WriteFile(keep, []byte("x"), 0o644)
	for _, dir := range []string{outside, s.forkModelsDir(), filepath.Join(s.forkModelsDir(), "..")} {
		if err := s.forks.PutModelSet(forks.ModelSet{ForkID: "avalon", Dir: dir}); err != nil {
			t.Fatal(err)
		}
		rec := httptest.NewRecorder()
		s.handleForkModelDelete(rec, withFork(htmxReq("DELETE", "/", nil), "avalon"))
		if rec.Code != http.StatusInternalServerError {
			t.Errorf("delete of %s = %d, want refused", dir, rec.Code)
		}
	}
	if _, err := os.Stat(keep); err != nil {
		t.Error("a file outside fork-models was deleted")
	}
}

func TestForkModelDownloadNeedsSpace(t *testing.T) {
	s, _ := forkServer(t)
	big := testFork()
	big.ModelFiles = []forks.ModelFile{{Path: "huge.gguf", Bytes: 1 << 60}}
	curatedForks = func() []forks.Fork { return []forks.Fork{big} }
	// Free space is read from the models folder, so it must exist.
	os.MkdirAll(filepath.Join(s.cfg.DataDir, "models"), 0o755)
	rec := httptest.NewRecorder()
	s.handleForkModelDownload(rec, withFork(htmxReq("POST", "/", nil), "avalon"))
	if rec.Code != http.StatusInsufficientStorage || !strings.Contains(rec.Body.String(), "Not enough free disk space") {
		t.Errorf("download of a huge set = %d %s", rec.Code, rec.Body)
	}
}

func TestModelsPageForkSection(t *testing.T) {
	s := newTestServer(t)
	render := func() string {
		rec := httptest.NewRecorder()
		s.handleModelsPage(rec, httptest.NewRequest("GET", "/models", nil))
		return rec.Body.String()
	}
	if strings.Contains(render(), "fork-model-sets") {
		t.Error("fork section shown with forks off")
	}
	s.cfg.ForksEnabled = true
	if !strings.Contains(render(), `hx-get="/api/forks/models"`) {
		t.Error("fork section missing with forks on")
	}
}
