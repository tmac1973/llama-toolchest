package huggingface

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

// projectorFake is a HuggingFace API with a few repositories: each path
// under /models/ answers with its JSON, and /models (the search) records
// its filters and answers with search.
type projectorFake struct {
	repos  map[string]string // "org/name" → repository record
	trees  map[string]string // "org/name" → tree listing
	search string

	mu      sync.Mutex
	filters [][]string
}

func (f *projectorFake) client(t *testing.T) *Client {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path := strings.TrimPrefix(r.URL.Path, "/models")
		switch {
		case path == "":
			f.mu.Lock()
			f.filters = append(f.filters, r.URL.Query()["filter"])
			f.mu.Unlock()
			w.Write([]byte(f.search))
		case strings.HasSuffix(path, "/tree/main"):
			body, ok := f.trees[strings.TrimSuffix(strings.TrimPrefix(path, "/"), "/tree/main")]
			if !ok {
				http.NotFound(w, r)
				return
			}
			w.Write([]byte(body))
		default:
			body, ok := f.repos[strings.TrimPrefix(path, "/")]
			if !ok {
				http.NotFound(w, r)
				return
			}
			w.Write([]byte(body))
		}
	}))
	t.Cleanup(srv.Close)
	c := NewClient("")
	c.apiBase = srv.URL
	return c
}

const (
	thinkingI1 = `{"id":"mradermacher/ThinkingCap-Qwen3.8-27B-i1-GGUF","author":"mradermacher",
		"cardData":{"base_model":"bottlecapai/ThinkingCap-Qwen3.8-27B"},
		"siblings":[{"rfilename":"ThinkingCap-Qwen3.8-27B.i1-Q4_K_M.gguf"}]}`
	thinkingStatic = `{"id":"mradermacher/ThinkingCap-Qwen3.8-27B-GGUF","author":"mradermacher",
		"siblings":[{"rfilename":"ThinkingCap-Qwen3.8-27B.Q4_K_M.gguf"},
		            {"rfilename":"ThinkingCap-Qwen3.8-27B.mmproj-f16.gguf"},
		            {"rfilename":"ThinkingCap-Qwen3.8-27B.mmproj-Q8_0.gguf"}]}`
	thinkingStaticTree = `[{"type":"file","path":"ThinkingCap-Qwen3.8-27B.Q4_K_M.gguf","size":16000000000},
		{"type":"file","path":"ThinkingCap-Qwen3.8-27B.mmproj-f16.gguf","size":928000000,"lfs":{"oid":"f16sha"}},
		{"type":"file","path":"ThinkingCap-Qwen3.8-27B.mmproj-Q8_0.gguf","size":600000000,"lfs":{"oid":"q8sha"}}]`
)

// mradermacher's imatrix repository has no projector; its static sibling
// does, and is found without a search. Q8_0 is offered first.
func TestFindProjectorRepoTriesThePublishersSibling(t *testing.T) {
	f := &projectorFake{
		repos: map[string]string{
			"mradermacher/ThinkingCap-Qwen3.8-27B-i1-GGUF": thinkingI1,
			"mradermacher/ThinkingCap-Qwen3.8-27B-GGUF":    thinkingStatic,
		},
		trees: map[string]string{"mradermacher/ThinkingCap-Qwen3.8-27B-GGUF": thinkingStaticTree},
	}
	got, err := f.client(t).FindProjectorRepo(context.Background(), "mradermacher/ThinkingCap-Qwen3.8-27B-i1-GGUF", "")
	if err != nil {
		t.Fatal(err)
	}
	if got == nil || got.Repo != "mradermacher/ThinkingCap-Qwen3.8-27B-GGUF" {
		t.Fatalf("repo = %+v, want the static sibling", got)
	}
	if len(got.Files) != 2 {
		t.Fatalf("files = %+v, want the two projectors", got.Files)
	}
	if q, f16 := got.Files[0], got.Files[1]; q.Quant != "Q8_0" || q.Size != 600000000 || q.OID != "q8sha" || f16.Quant != "F16" {
		t.Errorf("files = %+v, want Q8_0 (with size and hash) then F16", got.Files)
	}
	if len(f.filters) != 0 {
		t.Errorf("searched %v although the sibling had a projector", f.filters)
	}
}

// Without a sibling, the other quantizations of the base model are
// searched, and the same publisher's is taken over a more downloaded one.
func TestFindProjectorRepoSearchesTheBaseModelsQuantizations(t *testing.T) {
	f := &projectorFake{
		repos: map[string]string{
			"bartowski/Thinking-GGUF": `{"id":"bartowski/Thinking-GGUF","author":"bartowski",
				"cardData":{"base_model":["org/Thinking"]},"siblings":[{"rfilename":"Thinking-Q4_K_M.gguf"}]}`,
		},
		trees: map[string]string{
			"bartowski/Thinking-vision-GGUF": `[{"type":"file","path":"mmproj-Thinking-f16.gguf","size":900}]`,
		},
		search: `[{"id":"other/Thinking-GGUF","author":"other","siblings":[{"rfilename":"mmproj-f16.gguf"}]},
			{"id":"bartowski/Thinking-GGUF","author":"bartowski","siblings":[{"rfilename":"Thinking-Q4_K_M.gguf"}]},
			{"id":"bartowski/Thinking-vision-GGUF","author":"bartowski","siblings":[{"rfilename":"mmproj-Thinking-f16.gguf"}]}]`,
	}
	got, err := f.client(t).FindProjectorRepo(context.Background(), "bartowski/Thinking-GGUF", "")
	if err != nil {
		t.Fatal(err)
	}
	if got == nil || got.Repo != "bartowski/Thinking-vision-GGUF" {
		t.Fatalf("repo = %+v, want the same publisher's", got)
	}
	if len(f.filters) != 1 || strings.Join(f.filters[0], ",") != "base_model:quantized:org/Thinking,gguf" {
		t.Errorf("search filters = %v", f.filters)
	}
}

// A repository whose card names no base model is searched by the base
// the GGUF header gives.
func TestFindProjectorRepoFallsBackToTheHeadersBaseModel(t *testing.T) {
	f := &projectorFake{
		repos: map[string]string{"a/M-GGUF": `{"id":"a/M-GGUF","author":"a","siblings":[]}`},
		trees: map[string]string{"b/M-GGUF": `[{"type":"file","path":"mmproj-M-Q8_0.gguf","size":5}]`},
		search: `[{"id":"b/M-GGUF","author":"b","siblings":[{"rfilename":"mmproj-M-Q8_0.gguf"}]}]`,
	}
	got, err := f.client(t).FindProjectorRepo(context.Background(), "a/M-GGUF", "up/M")
	if err != nil || got == nil || got.Repo != "b/M-GGUF" {
		t.Fatalf("got %+v, %v; want b/M-GGUF", got, err)
	}
	if len(f.filters) != 1 || f.filters[0][0] != "base_model:quantized:up/M" {
		t.Errorf("search filters = %v", f.filters)
	}
}

// A projector in the model's own repository (one not downloaded) is
// offered from there; nothing at all is no result and no error; a
// repository HuggingFace does not know is an error.
func TestFindProjectorRepoOwnRepoAndNone(t *testing.T) {
	f := &projectorFake{
		repos: map[string]string{
			"a/Own-GGUF":  `{"id":"a/Own-GGUF","siblings":[{"rfilename":"mmproj-F16.gguf"}]}`,
			"a/Text-GGUF": `{"id":"a/Text-GGUF","cardData":{"base_model":"up/Text"},"siblings":[]}`,
		},
		trees:  map[string]string{"a/Own-GGUF": `[{"type":"file","path":"mmproj-F16.gguf","size":7}]`},
		search: `[]`,
	}
	c := f.client(t)
	if got, err := c.FindProjectorRepo(context.Background(), "a/Own-GGUF", ""); err != nil || got == nil || got.Repo != "a/Own-GGUF" {
		t.Errorf("own repository: got %+v, %v", got, err)
	}
	if got, err := c.FindProjectorRepo(context.Background(), "a/Text-GGUF", ""); err != nil || got != nil {
		t.Errorf("text-only model: got %+v, %v; want nil, nil", got, err)
	}
	if _, err := c.FindProjectorRepo(context.Background(), "a/Missing", ""); err == nil {
		t.Error("unknown repository: no error")
	}
}

// StartInto saves a file from one repository in another's folder and
// reports it to its own callback, not as a model download.
func TestStartIntoSavesInTheGivenFolder(t *testing.T) {
	var gotURL string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("projector"))
	}))
	t.Cleanup(srv.Close)
	dir := t.TempDir()
	d := NewDownloader(dir, filepath.Join(dir, "models"), "")
	d.RegisterProvider("fake", Provider{URL: func(modelID, filename string) string {
		gotURL = modelID + "/" + filename
		return srv.URL
	}})
	d.SetOnComplete(func(string, string, string, string, int64) { t.Error("model completion callback called") })
	dest := filepath.Join(dir, "models", "a--M-i1-GGUF")

	done := make(chan string, 1)
	id, err := d.StartInto("fake", "a/M-GGUF", "M.mmproj-Q8_0.gguf", dest, 9, func(path string, _ int64) { done <- path })
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(id, "mmproj--") {
		t.Errorf("download ID = %q", id)
	}
	select {
	case path := <-done:
		if want := filepath.Join(dest, "M.mmproj-Q8_0.gguf"); path != want {
			t.Errorf("path = %q, want %q", path, want)
		}
		if b, _ := os.ReadFile(path); string(b) != "projector" {
			t.Errorf("content = %q", b)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("download did not finish")
	}
	if gotURL != "a/M-GGUF/M.mmproj-Q8_0.gguf" {
		t.Errorf("fetched %q, want the file from its own repository", gotURL)
	}
	if _, err := os.Stat(filepath.Join(dir, "models", "a--M-GGUF")); err == nil {
		t.Error("a folder was made for the source repository")
	}
}
