package modelscope

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// newTestClient points a Client at a stub server by rewriting the request
// host, so the real URL-building code under test still runs.
func newTestClient(t *testing.T, handler http.HandlerFunc) (*Client, *httptest.Server) {
	t.Helper()
	srv := httptest.NewServer(handler)
	c := NewClient("")
	c.httpClient = srv.Client()
	c.httpClient.Transport = rewriteHost{base: srv.URL, rt: srv.Client().Transport}
	return c, srv
}

type rewriteHost struct {
	base string
	rt   http.RoundTripper
}

func (r rewriteHost) RoundTrip(req *http.Request) (*http.Response, error) {
	req.URL.Scheme = "http"
	req.URL.Host = strings.TrimPrefix(r.base, "http://")
	return r.rt.RoundTrip(req)
}

// The search body is the part most likely to break silently against the
// live service, so pin its exact shape: "Criterion" (not
// "SingleCriterion") and SortBy "DownloadsCount" (not "Downloads").
func TestSearchRequestShape(t *testing.T) {
	var got searchRequest
	var method, path string
	c, srv := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		method, path = r.Method, r.URL.Path
		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &got)
		// Assert on the raw JSON keys too — the struct would happily
		// decode a body that spelled the field differently.
		var rawKeys map[string]json.RawMessage
		json.Unmarshal(body, &rawKeys)
		if _, ok := rawKeys["Criterion"]; !ok {
			t.Errorf("request body has no Criterion key; keys=%v", keysOf(rawKeys))
		}
		if _, ok := rawKeys["SingleCriterion"]; ok {
			t.Error("request used SingleCriterion, which this endpoint ignores")
		}
		w.Write([]byte(`{"Code":200,"Success":true,"Data":{"Model":{"Models":[]}}}`))
	})
	defer srv.Close()

	if _, err := c.Search(context.Background(), "qwen3"); err != nil {
		t.Fatalf("Search: %v", err)
	}
	if method != http.MethodPut {
		t.Errorf("method = %s, want PUT (the endpoint rejects POST)", method)
	}
	if path != "/api/v1/dolphin/models" {
		t.Errorf("path = %s", path)
	}
	if got.SortBy != "DownloadsCount" {
		t.Errorf("SortBy = %q, want DownloadsCount (\"Downloads\" returns an empty list)", got.SortBy)
	}
	if got.Name != "qwen3" || got.PageSize != searchLimit {
		t.Errorf("Name/PageSize = %q/%d", got.Name, got.PageSize)
	}
	if len(got.Criterion) != 1 || got.Criterion[0].Category != "libraries" ||
		got.Criterion[0].Values[0] != "gguf" {
		t.Errorf("Criterion = %+v, want libraries contains gguf", got.Criterion)
	}
}

func keysOf(m map[string]json.RawMessage) []string {
	var out []string
	for k := range m {
		out = append(out, k)
	}
	return out
}

func TestSearchMapsFields(t *testing.T) {
	c, srv := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"Code":200,"Success":true,"Data":{"Model":{"Models":[
		  {"Name":"Qwen3-8B-GGUF","Path":"unsloth","Downloads":23553,"Stars":13,
		   "Tags":["unsloth","gguf"],"Libraries":["gguf","pytorch"],"License":"apache-2.0"}]}}}`))
	})
	defer srv.Close()

	got, err := c.Search(context.Background(), "qwen")
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("got %d results, want 1", len(got))
	}
	r := got[0]
	// ID is owner/name assembled from two fields — HuggingFace returns it
	// pre-joined, and everything downstream assumes that shape.
	if r.ID != "unsloth/Qwen3-8B-GGUF" {
		t.Errorf("ID = %q, want unsloth/Qwen3-8B-GGUF", r.ID)
	}
	if r.Author != "unsloth" || r.Downloads != 23553 || r.License != "apache-2.0" {
		t.Errorf("bad mapping: %+v", r)
	}
	// ModelScope has no "likes"; stars are the closest equivalent and are
	// what the results list renders.
	if r.Likes != 13 {
		t.Errorf("Likes = %d, want 13 (from Stars)", r.Likes)
	}
}

// If the server-side filter stops applying, non-GGUF repos come back with
// HTTP 200. Those must never reach the results list.
func TestSearchDropsNonGGUFResults(t *testing.T) {
	c, srv := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"Code":200,"Success":true,"Data":{"Model":{"Models":[
		  {"Name":"Meta-Llama-3.1-8B","Path":"LLM-Research","Libraries":["safetensors","pytorch"]},
		  {"Name":"Llama-3.2-3B-Instruct-GGUF","Path":"unsloth","Libraries":["gguf","pytorch"]}]}}}`))
	})
	defer srv.Close()

	got, err := c.Search(context.Background(), "llama")
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(got) != 1 || got[0].ID != "unsloth/Llama-3.2-3B-Instruct-GGUF" {
		t.Errorf("got %+v, want only the GGUF repo", got)
	}
}

// Filter failure across the board is an error, not an empty list: an empty
// list reads as "no such model" and would send people hunting for a
// spelling mistake.
func TestSearchReportsTotalFilterFailure(t *testing.T) {
	c, srv := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"Code":200,"Success":true,"Data":{"Model":{"Models":[
		  {"Name":"Meta-Llama-3.1-8B","Path":"LLM-Research","Libraries":["safetensors"]}]}}}`))
	})
	defer srv.Close()

	_, err := c.Search(context.Background(), "llama")
	if err == nil {
		t.Fatal("want an error when every result is non-GGUF")
	}
	if !strings.Contains(err.Error(), "filter") {
		t.Errorf("error should name the filter as the cause, got: %v", err)
	}
}

func TestGetModelFilesAndShards(t *testing.T) {
	var gotPath, gotQuery string
	c, srv := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		gotPath, gotQuery = r.URL.Path, r.URL.RawQuery
		w.Write([]byte(`{"Code":200,"Success":true,"Data":{"Files":[
		  {"Name":"README.md","Path":"README.md","Size":100,"Type":"blob"},
		  {"Name":"UD-IQ1_M","Path":"UD-IQ1_M","Size":0,"Type":"tree"},
		  {"Name":"m-UD-IQ1_M-00001-of-00002.gguf","Path":"UD-IQ1_M/m-UD-IQ1_M-00001-of-00002.gguf","Size":1000,"Type":"blob"},
		  {"Name":"m-UD-IQ1_M-00002-of-00002.gguf","Path":"UD-IQ1_M/m-UD-IQ1_M-00002-of-00002.gguf","Size":2000,"Type":"blob"},
		  {"Name":"mmproj-F16.gguf","Path":"mmproj-F16.gguf","Size":500,"Type":"blob"}]}}`))
	})
	defer srv.Close()

	detail, err := c.GetModel(context.Background(), "unsloth/Model-GGUF")
	if err != nil {
		t.Fatalf("GetModel: %v", err)
	}
	if gotPath != "/api/v1/models/unsloth/Model-GGUF/repo/files" {
		t.Errorf("path = %s", gotPath)
	}
	// ModelScope's default branch is master, not main.
	if !strings.Contains(gotQuery, "Revision=master") || !strings.Contains(gotQuery, "Recursive=True") {
		t.Errorf("query = %s", gotQuery)
	}

	// README dropped, tree entry dropped, the two shards grouped into one.
	if len(detail.Files) != 2 {
		t.Fatalf("got %d files, want 2 (grouped shard set + mmproj): %+v", len(detail.Files), detail.Files)
	}
	for _, f := range detail.Files {
		switch {
		case len(f.Shards) > 0:
			if f.Size != 3000 {
				t.Errorf("grouped shard size = %d, want 3000", f.Size)
			}
			if len(f.Shards) != 2 {
				t.Errorf("want 2 shards, got %d", len(f.Shards))
			}
			// The subdirectory has to survive: it is part of the download path.
			if !strings.HasPrefix(f.Shards[0], "UD-IQ1_M/") {
				t.Errorf("shard lost its subdirectory: %q", f.Shards[0])
			}
		case f.IsMMProj:
			if f.Filename != "mmproj-F16.gguf" {
				t.Errorf("mmproj filename = %q", f.Filename)
			}
		default:
			t.Errorf("unexpected file %+v", f)
		}
	}
}

func TestGetModelRejectsBadID(t *testing.T) {
	c := NewClient("")
	for _, id := range []string{"noslash", "/name", "owner/", "a/b/c"} {
		if _, err := c.GetModel(context.Background(), id); err == nil {
			t.Errorf("GetModel(%q) should have failed", id)
		}
	}
}

func TestURLs(t *testing.T) {
	c := NewClient("")
	got := c.DownloadURL("unsloth/Qwen3-8B-GGUF", "UD-IQ1_M/m-00001-of-00002.gguf")
	want := "https://modelscope.cn/api/v1/models/unsloth/Qwen3-8B-GGUF/repo?Revision=master&FilePath=UD-IQ1_M%2Fm-00001-of-00002.gguf"
	if got != want {
		t.Errorf("DownloadURL =\n  %s\nwant\n  %s", got, want)
	}
	if got := c.ModelURL("unsloth/Qwen3-8B-GGUF"); got != "https://modelscope.cn/models/unsloth/Qwen3-8B-GGUF" {
		t.Errorf("ModelURL = %s", got)
	}
	if got := c.ModelURL("nonsense"); got != "" {
		t.Errorf("ModelURL of a malformed id = %q, want empty", got)
	}
}

// A resumed download is told apart from a restarted one by this helper,
// because ModelScope labels a partial body 200. Getting it wrong appends
// a tail to a partial file and silently corrupts the result.
func TestResponseIsPartial(t *testing.T) {
	mk := func(status int, contentRange string) *http.Response {
		h := http.Header{}
		if contentRange != "" {
			h.Set("Content-Range", contentRange)
		}
		return &http.Response{StatusCode: status, Header: h}
	}
	tests := []struct {
		name string
		resp *http.Response
		want bool
	}{
		{"conforming 206", mk(206, "bytes 100-1123/2275379008"), true},
		{"ModelScope's 200 with Content-Range", mk(200, "bytes 100-1123/2275379008"), true},
		{"plain 200, whole file", mk(200, ""), false},
		{"206 without the header", mk(206, ""), true},
	}
	for _, tt := range tests {
		if got := ResponseIsPartial(tt.resp); got != tt.want {
			t.Errorf("%s: ResponseIsPartial = %v, want %v", tt.name, got, tt.want)
		}
	}
}
