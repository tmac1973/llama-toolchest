package huggingface

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

const testSHA = "0123456789abcdef0123456789abcdef01234567"

func TestRepoCommit(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		switch r.URL.Path {
		case "/models/org/repo/revision/main":
			w.Write([]byte(`{"id":"org/repo","sha":"` + testSHA + `"}`))
		case "/models/org/gated/revision/main":
			w.Header().Set("X-Error-Code", "GatedRepo")
			w.WriteHeader(http.StatusForbidden)
		case "/models/org/nosha/revision/main":
			w.Write([]byte(`{"id":"org/nosha"}`))
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	c := NewClient("")
	c.apiBase = srv.URL
	ctx := context.Background()

	got, err := c.RepoCommit(ctx, "org/repo", "main")
	if err != nil || got != testSHA {
		t.Errorf("main: got %q, %v; want %q", got, err, testSHA)
	}

	// A full commit needs no request.
	before := calls.Load()
	if got, err := c.RepoCommit(ctx, "org/repo", testSHA); err != nil || got != testSHA {
		t.Errorf("commit: got %q, %v", got, err)
	}
	if calls.Load() != before {
		t.Error("a full commit SHA made a request")
	}

	// A gated repository gets the explained message.
	if _, err := c.RepoCommit(ctx, "org/gated", "main"); err == nil || !strings.Contains(err.Error(), "gated model") {
		t.Errorf("gated: got %v, want the gated-model message", err)
	}
	if _, err := c.RepoCommit(ctx, "org/nosha", "main"); err == nil {
		t.Error("an answer with no sha was taken")
	}
}

func TestRepoFilesAt(t *testing.T) {
	var paths []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		paths = append(paths, r.URL.Path)
		w.Write([]byte(`[{"type":"file","path":"split_files/vae/v.safetensors","size":42,"oid":"x","lfs":{"oid":"beef","size":42}}]`))
	}))
	defer srv.Close()
	c := NewClient("")
	c.apiBase = srv.URL

	files, err := c.RepoFilesAt(context.Background(), "org/repo", testSHA)
	if err != nil {
		t.Fatal(err)
	}
	if len(files) != 1 || files[0].Path != "split_files/vae/v.safetensors" || files[0].Size != 42 {
		t.Errorf("files = %+v", files)
	}
	if _, err := c.RepoFiles(context.Background(), "org/repo"); err != nil {
		t.Fatal(err)
	}
	want := []string{"/models/org/repo/tree/" + testSHA, "/models/org/repo/tree/main"}
	if strings.Join(paths, " ") != strings.Join(want, " ") {
		t.Errorf("requests = %v, want %v", paths, want)
	}
}

func TestFileAccess(t *testing.T) {
	var methods []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		methods = append(methods, r.Method+" "+r.URL.Path)
		switch r.URL.Path {
		case "/org/open/resolve/" + testSHA + "/big.gguf":
			http.Redirect(w, r, "http://127.0.0.1:1/cdn", http.StatusFound) // not followed
		case "/org/open/resolve/" + testSHA + "/small.json":
			w.WriteHeader(http.StatusOK)
		case "/org/gated/resolve/" + testSHA + "/m.safetensors":
			w.Header().Set("X-Error-Code", "GatedRepo")
			w.WriteHeader(http.StatusUnauthorized)
		default:
			w.Header().Set("X-Error-Code", "EntryNotFound")
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	c := NewClient("")
	c.siteBase = srv.URL
	ctx := context.Background()

	for _, f := range []string{"big.gguf", "small.json"} {
		if err := c.FileAccess(ctx, "org/open", testSHA, f); err != nil {
			t.Errorf("%s: %v", f, err)
		}
	}
	if err := c.FileAccess(ctx, "org/gated", testSHA, "m.safetensors"); err == nil || !strings.Contains(err.Error(), "gated model") {
		t.Errorf("gated: got %v, want the gated-model message", err)
	}
	if err := c.FileAccess(ctx, "org/open", testSHA, "gone.gguf"); err == nil || !strings.Contains(err.Error(), "not in org/open") {
		t.Errorf("missing file: got %v", err)
	}
	for _, m := range methods {
		if !strings.HasPrefix(m, "HEAD ") {
			t.Errorf("request %s is not a HEAD", m)
		}
	}
}
