package huggingface

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/models"
)

// hfcRequest is one request a fake HuggingFace server received, with the
// address the client meant to reach before it was sent to the fake.
type hfcRequest struct {
	Method string
	Host   string // host the client built the URL for
	Path   string
	Query  url.Values
	Auth   string
}

// hfcFake is a fake HuggingFace API. Each test sets handler to answer;
// every request is recorded.
type hfcFake struct {
	srv *httptest.Server

	mu   sync.Mutex
	reqs []hfcRequest
}

// hfcRedirect sends every request to the fake server, keeping the path,
// query and headers the client built, and remembers the original host
// in a header so the fake can record it.
type hfcRedirect struct {
	target *url.URL
}

func (r hfcRedirect) RoundTrip(req *http.Request) (*http.Response, error) {
	out := req.Clone(req.Context())
	out.Header.Set("X-Original-Host", req.URL.Host)
	out.URL.Scheme = r.target.Scheme
	out.URL.Host = r.target.Host
	out.Host = r.target.Host
	return http.DefaultTransport.RoundTrip(out)
}

// newHFCFake starts a fake API and returns a client whose requests all
// go to it. Search, GetModel and the tree listing build their URLs from
// the fixed huggingface.co address (they do not read apiBase), so the
// redirect happens in the transport.
func newHFCFake(t *testing.T, token string, handler http.HandlerFunc) (*Client, *hfcFake) {
	t.Helper()
	f := &hfcFake{}
	f.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		f.mu.Lock()
		f.reqs = append(f.reqs, hfcRequest{
			Method: r.Method,
			Host:   r.Header.Get("X-Original-Host"),
			Path:   r.URL.Path,
			Query:  r.URL.Query(),
			Auth:   r.Header.Get("Authorization"),
		})
		f.mu.Unlock()
		handler(w, r)
	}))
	t.Cleanup(f.srv.Close)
	target, _ := url.Parse(f.srv.URL)
	c := NewClient(token)
	c.httpClient = &http.Client{Transport: hfcRedirect{target: target}}
	return c, f
}

func (f *hfcFake) requests() []hfcRequest {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]hfcRequest(nil), f.reqs...)
}

// A search must ask for GGUF repositories only, most downloaded first,
// with the user's words escaped: an unescaped "&" would end the search
// term and add a parameter of its own.
func TestSearchBuildsTheQueryAndDecodesResults(t *testing.T) {
	c, fake := newHFCFake(t, "", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`[{"id":"unsloth/Qwen3-8B-GGUF","author":"unsloth","downloads":1200,"likes":30,"tags":["gguf"]},
		                 {"id":"bartowski/Qwen3-8B-GGUF","downloads":800}]`))
	})

	got, err := c.Search(context.Background(), "qwen 3&sort=likes")
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(got) != 2 || got[0].ID != "unsloth/Qwen3-8B-GGUF" || got[0].Downloads != 1200 ||
		got[0].Likes != 30 || got[0].Author != "unsloth" || got[1].ID != "bartowski/Qwen3-8B-GGUF" {
		t.Errorf("results = %+v", got)
	}

	reqs := fake.requests()
	if len(reqs) != 1 {
		t.Fatalf("%d requests, want 1", len(reqs))
	}
	r := reqs[0]
	if r.Method != http.MethodGet || r.Host != "huggingface.co" || r.Path != "/api/models" {
		t.Errorf("request = %s %s%s, want GET huggingface.co/api/models", r.Method, r.Host, r.Path)
	}
	want := map[string]string{
		"search": "qwen 3&sort=likes", "filter": "gguf", "sort": "downloads",
		"direction": "-1", "limit": "50",
	}
	for k, v := range want {
		if got := r.Query[k]; len(got) != 1 || got[0] != v {
			t.Errorf("query %s = %q, want exactly [%q]", k, got, v)
		}
	}
}

// An upstream error must come back as an error naming the status, not as
// an empty result list: an empty list reads as "no such model", which
// hides an expired token (401) or an outage (500).
func TestSearchReportsUpstreamStatus(t *testing.T) {
	for _, code := range []int{http.StatusUnauthorized, http.StatusNotFound, http.StatusInternalServerError} {
		c, _ := newHFCFake(t, "", func(w http.ResponseWriter, r *http.Request) {
			http.Error(w, `{"error":"nope"}`, code)
		})
		got, err := c.Search(context.Background(), "qwen")
		if err == nil {
			t.Errorf("HTTP %d: no error, results %+v", code, got)
			continue
		}
		if !strings.Contains(err.Error(), fmt.Sprintf("HTTP %d", code)) {
			t.Errorf("HTTP %d: error %q does not name the status", code, err)
		}
	}
}

// A 200 answer that is not the expected JSON is an error too, for the
// same reason: it must not look like an empty search.
func TestSearchRejectsAMalformedBody(t *testing.T) {
	c, _ := newHFCFake(t, "", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`<html>maintenance</html>`))
	})
	if got, err := c.Search(context.Background(), "qwen"); err == nil {
		t.Errorf("no error for an HTML body; results %+v", got)
	}
}

// hfcModelHandler answers the model endpoint with siblings and the tree
// endpoint with tree, which is how GetModel gets names and sizes from two
// separate requests.
func hfcModelHandler(t *testing.T, siblings, tree string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.HasSuffix(r.URL.Path, "/tree/main"):
			w.Write([]byte(tree))
		case strings.HasPrefix(r.URL.Path, "/api/models/"):
			w.Write([]byte(siblings))
		default:
			t.Errorf("unexpected request %s", r.URL.Path)
			http.NotFound(w, r)
		}
	}
}

// GetModel lists only GGUF files (in any letter case), marks vision
// projectors, and takes each file's size and content id from the tree
// listing. The LFS id is preferred over the git blob id, because the LFS
// sha256 is what identifies the file's content for the header cache.
func TestGetModelKeepsGGUFFilesAndFillsSizes(t *testing.T) {
	siblings := `{"id":"org/Model-GGUF","gguf":{"total":8030261248},"siblings":[
		{"rfilename":"README.md"},
		{"rfilename":"config.json"},
		{"rfilename":"Model-Q4_K_M.gguf"},
		{"rfilename":"Model-Q8_0.GGUF"},
		{"rfilename":"mmproj-F16.gguf"}]}`
	tree := `[
		{"path":"README.md","size":100,"oid":"readme-blob"},
		{"path":"Model-Q4_K_M.gguf","size":4920000000,"oid":"blob-q4","lfs":{"oid":"sha-q4"}},
		{"path":"Model-Q8_0.GGUF","size":8540000000,"oid":"blob-q8"},
		{"path":"mmproj-F16.gguf","size":600000000,"oid":"blob-mm","lfs":{"oid":"sha-mm"}}]`
	c, fake := newHFCFake(t, "", hfcModelHandler(t, siblings, tree))

	d, err := c.GetModel(context.Background(), "org/Model-GGUF")
	if err != nil {
		t.Fatalf("GetModel: %v", err)
	}
	if d.ID != "org/Model-GGUF" || d.ParamCount != 8030261248 {
		t.Errorf("detail = id %q, params %d", d.ID, d.ParamCount)
	}
	byName := map[string]ModelFile{}
	for _, f := range d.Files {
		byName[f.Filename] = f
	}
	if len(byName) != 3 {
		t.Fatalf("files = %+v, want the 3 GGUF files only", d.Files)
	}
	q4 := byName["Model-Q4_K_M.gguf"]
	if q4.Size != 4920000000 || q4.OID != "sha-q4" || q4.Quant != "Q4_K_M" || q4.IsMMProj {
		t.Errorf("Q4 file = %+v", q4)
	}
	if want := models.EstimateVRAM(4920000000); q4.VRAMEstGB != want {
		t.Errorf("Q4 VRAM estimate = %v, want %v from its size", q4.VRAMEstGB, want)
	}
	if q8 := byName["Model-Q8_0.GGUF"]; q8.Size != 8540000000 || q8.OID != "blob-q8" {
		t.Errorf("Q8 file (no LFS id) = %+v, want the blob id", q8)
	}
	if mm := byName["mmproj-F16.gguf"]; !mm.IsMMProj || mm.Size != 600000000 {
		t.Errorf("mmproj file = %+v", mm)
	}

	reqs := fake.requests()
	if len(reqs) != 2 {
		t.Fatalf("%d requests, want the model and the tree", len(reqs))
	}
	if reqs[0].Path != "/api/models/org/Model-GGUF" {
		t.Errorf("first request path %q", reqs[0].Path)
	}
	if reqs[1].Path != "/api/models/org/Model-GGUF/tree/main" || reqs[1].Query.Get("recursive") != "true" {
		t.Errorf("tree request = %s?%s, want a recursive tree of main", reqs[1].Path, reqs[1].Query.Encode())
	}
}

// The shards of a split file become one entry whose size is their sum,
// with the shard names and sizes kept in order, so the list shows one
// download of the real total and the header probe can read each shard.
func TestGetModelGroupsShards(t *testing.T) {
	siblings := `{"id":"org/Big-GGUF","siblings":[
		{"rfilename":"Q4/Big-Q4_K_M-00002-of-00002.gguf"},
		{"rfilename":"Q4/Big-Q4_K_M-00001-of-00002.gguf"}]}`
	tree := `[
		{"path":"Q4/Big-Q4_K_M-00001-of-00002.gguf","size":3000},
		{"path":"Q4/Big-Q4_K_M-00002-of-00002.gguf","size":2000}]`
	c, _ := newHFCFake(t, "", hfcModelHandler(t, siblings, tree))

	d, err := c.GetModel(context.Background(), "org/Big-GGUF")
	if err != nil {
		t.Fatalf("GetModel: %v", err)
	}
	if len(d.Files) != 1 {
		t.Fatalf("files = %+v, want one grouped entry", d.Files)
	}
	f := d.Files[0]
	if f.Filename != "Q4/Big-Q4_K_M-00001-of-00002.gguf" || f.Size != 5000 {
		t.Errorf("grouped file = %q size %d, want the first shard's name and size 5000", f.Filename, f.Size)
	}
	if len(f.Shards) != 2 || f.Shards[1] != "Q4/Big-Q4_K_M-00002-of-00002.gguf" ||
		len(f.ShardSizes) != 2 || f.ShardSizes[0] != 3000 || f.ShardSizes[1] != 2000 {
		t.Errorf("shards = %v sizes %v", f.Shards, f.ShardSizes)
	}
}

// When the model endpoint fails, GetModel returns the status as an error
// and does not go on to ask for the tree of a repository it could not
// read.
func TestGetModelReportsUpstreamStatus(t *testing.T) {
	for _, code := range []int{http.StatusUnauthorized, http.StatusNotFound, http.StatusInternalServerError} {
		c, fake := newHFCFake(t, "", func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(code)
		})
		d, err := c.GetModel(context.Background(), "org/Gated-GGUF")
		if err == nil || !strings.Contains(err.Error(), fmt.Sprintf("HTTP %d", code)) {
			t.Errorf("HTTP %d: detail %+v, error %v; want an error naming the status", code, d, err)
		}
		if n := len(fake.requests()); n != 1 {
			t.Errorf("HTTP %d: %d requests, want 1 (no tree request after a failure)", code, n)
		}
	}
}

// A failed tree listing must not fail the whole listing: the files are
// still shown, with unknown size (0), which the file list treats as
// "may fit" rather than hiding the file.
func TestGetModelSurvivesAFailedTreeListing(t *testing.T) {
	c, _ := newHFCFake(t, "", func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/tree/main") {
			http.Error(w, "internal error", http.StatusInternalServerError)
			return
		}
		w.Write([]byte(`{"id":"org/M","siblings":[{"rfilename":"M-Q4_0.gguf"}]}`))
	})
	d, err := c.GetModel(context.Background(), "org/M")
	if err != nil {
		t.Fatalf("GetModel: %v", err)
	}
	if len(d.Files) != 1 || d.Files[0].Filename != "M-Q4_0.gguf" || d.Files[0].Size != 0 {
		t.Errorf("files = %+v, want the one file with unknown size", d.Files)
	}
}

// The tree listing is paged with a Link header past 1,000 entries. A
// file on the second page must still get its size.
func TestPopulateFileSizesFollowsTheNextPage(t *testing.T) {
	c, fake := newHFCFake(t, "", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("cursor") == "" {
			w.Header().Set("Link", `<https://huggingface.co/api/models/org/M/tree/main?recursive=true&cursor=p2>; rel="next"`)
			w.Write([]byte(`[{"path":"a.gguf","size":10}]`))
			return
		}
		w.Write([]byte(`[{"path":"b.gguf","size":20,"lfs":{"oid":"sha-b"}}]`))
	})
	d := &ModelDetail{Files: []ModelFile{{Filename: "a.gguf"}, {Filename: "b.gguf"}, {Filename: "c.gguf"}}}
	c.populateFileSizes(context.Background(), "org/M", d)

	if d.Files[0].Size != 10 || d.Files[1].Size != 20 || d.Files[1].OID != "sha-b" {
		t.Errorf("files = %+v, want sizes from both pages", d.Files)
	}
	if d.Files[2].Size != 0 || d.Files[2].VRAMEstGB != 0 {
		t.Errorf("a file missing from the tree got size %d, estimate %v; want both unknown", d.Files[2].Size, d.Files[2].VRAMEstGB)
	}
	reqs := fake.requests()
	if len(reqs) != 2 || reqs[1].Query.Get("cursor") != "p2" {
		t.Errorf("requests = %+v, want the first page and then cursor p2", reqs)
	}
}

// The page loop is bounded, so a host that always names a next page
// cannot keep a listing running for ever.
func TestPopulateFileSizesStopsAfterTwentyPages(t *testing.T) {
	c, fake := newHFCFake(t, "", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Link", `<https://huggingface.co/api/models/org/M/tree/main?cursor=again>; rel="next"`)
		w.Write([]byte(`[]`))
	})
	c.populateFileSizes(context.Background(), "org/M", &ModelDetail{})
	if n := len(fake.requests()); n != 20 {
		t.Errorf("%d tree requests, want the limit of 20", n)
	}
}

// The Authorization header is sent only when a token is set, and a token
// set or cleared with SetToken applies to the next request on every
// endpoint. Sending "Bearer " with an empty token makes HuggingFace
// answer 401 even for public repositories.
func TestAuthorizationFollowsSetToken(t *testing.T) {
	c, fake := newHFCFake(t, "", hfcModelHandler(t, `{"id":"org/M","siblings":[{"rfilename":"m.gguf"}]}`, `[]`))
	ctx := context.Background()
	authOf := func() []string {
		var out []string
		for _, r := range fake.requests() {
			out = append(out, r.Auth)
		}
		fake.mu.Lock()
		fake.reqs = nil
		fake.mu.Unlock()
		return out
	}

	c.GetModel(ctx, "org/M")
	for i, a := range authOf() {
		if a != "" {
			t.Errorf("no token: request %d sent Authorization %q", i, a)
		}
	}

	c.SetToken("hf_secret")
	c.GetModel(ctx, "org/M")
	got := authOf()
	if len(got) != 2 {
		t.Fatalf("%d requests, want model and tree", len(got))
	}
	for i, a := range got {
		if a != "Bearer hf_secret" {
			t.Errorf("with token: request %d sent Authorization %q", i, a)
		}
	}

	c.SetToken("")
	c.GetModel(ctx, "org/M")
	for i, a := range authOf() {
		if a != "" {
			t.Errorf("token cleared: request %d still sent Authorization %q", i, a)
		}
	}
}

// A token given to NewClient is used from the first request, which is
// how a token saved before a restart is applied.
func TestNewClientTokenIsSentOnSearch(t *testing.T) {
	c, fake := newHFCFake(t, "hf_start", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`[]`))
	})
	if _, err := c.Search(context.Background(), "q"); err != nil {
		t.Fatal(err)
	}
	if reqs := fake.requests(); len(reqs) != 1 || reqs[0].Auth != "Bearer hf_start" {
		t.Errorf("requests = %+v, want one with the token from NewClient", reqs)
	}
}

// apiBase points every request at a test server: Search, GetModel and
// the tree listing as well as ListGGUF. A request that used the fixed
// huggingface.co address instead would reach the real host from a test.
func TestSearchAndGetModelUseAPIBase(t *testing.T) {
	var paths []string
	var mu sync.Mutex
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		paths = append(paths, r.URL.Path)
		mu.Unlock()
		switch {
		case r.URL.Path == "/models":
			w.Write([]byte(`[]`))
		case strings.HasSuffix(r.URL.Path, "/tree/main"):
			w.Write([]byte(`[]`))
		default:
			w.Write([]byte(`{"id":"org/M","siblings":[]}`))
		}
	}))
	defer srv.Close()

	c := NewClient("")
	c.apiBase = srv.URL
	if _, err := c.Search(context.Background(), "q"); err != nil {
		t.Errorf("Search: %v", err)
	}
	if _, err := c.GetModel(context.Background(), "org/M"); err != nil {
		t.Errorf("GetModel: %v", err)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(paths) != 3 {
		t.Errorf("test server saw %v, want the search, model and tree requests", paths)
	}
}
