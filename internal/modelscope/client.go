// Package modelscope searches and resolves GGUF models on ModelScope
// (modelscope.cn), Alibaba's model hub, as an alternative to HuggingFace.
//
// The endpoints here were established by probing the live service: the
// search call is the one modelscope.cn's own web UI issues, and is not
// part of a published REST contract. That has a practical consequence
// this package is built around — the API answers a malformed request with
// HTTP 200 and a plausible-looking result set rather than an error. A
// misspelled sort key returns zero models; an unrecognized filter key is
// ignored and unfiltered results come back. So responses are checked for
// the shape we asked for rather than trusted, and Search reports a filter
// that did not take effect instead of quietly handing back safetensors
// repositories.
//
// The same "answers 200 where it means something else" theme shows up in
// the download path, where a ranged request comes back as 200 with a
// Content-Range header. See DownloadURL.
package modelscope

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/modelsource"
)

const (
	baseURL = "https://modelscope.cn"
	apiBase = baseURL + "/api/v1"

	// revision is ModelScope's default branch name. HuggingFace calls the
	// equivalent "main".
	revision = "master"

	// searchLimit matches the HuggingFace client's page size so both
	// sources fill the results list to the same depth.
	searchLimit = 50
)

// Client is a ModelScope API client. An empty token is fine: the public
// repositories, which is all of them that matter here, need no auth.
type Client struct {
	httpClient *http.Client

	// token is guarded because Settings can replace it while a search or
	// a listing is in flight on another goroutine.
	mu    sync.RWMutex
	token string
}

// SetToken replaces the access token, so one saved in Settings applies to
// the next request rather than the next restart.
func (c *Client) SetToken(token string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.token = token
}

func (c *Client) authToken() string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.token
}

func NewClient(token string) *Client {
	return &Client{
		httpClient: &http.Client{Timeout: 30 * time.Second},
		token:      token,
	}
}

// searchRequest is the body modelscope.cn's web UI sends.
//
// The Criterion field name is load-bearing and easy to get wrong: some
// ModelScope SDK versions use "SingleCriterion", which this endpoint
// accepts and silently ignores, returning unfiltered results. Likewise
// SortBy must be "DownloadsCount" — "Downloads" is accepted and returns
// an empty list.
type searchRequest struct {
	Name       string      `json:"Name"`
	PageSize   int         `json:"PageSize"`
	PageNumber int         `json:"PageNumber"`
	SortBy     string      `json:"SortBy"`
	Criterion  []criterion `json:"Criterion"`
}

type criterion struct {
	Category  string   `json:"category"`
	Predicate string   `json:"predicate"`
	Values    []string `json:"values"`
}

type searchResponse struct {
	Code    int    `json:"Code"`
	Success bool   `json:"Success"`
	Message string `json:"Message"`
	Data    struct {
		Model struct {
			Models []struct {
				Name      string   `json:"Name"`
				Path      string   `json:"Path"` // the owner/org
				Downloads int      `json:"Downloads"`
				Stars     int      `json:"Stars"`
				Tags      []string `json:"Tags"`
				Libraries []string `json:"Libraries"`
				License   string   `json:"License"`
			} `json:"Models"`
		} `json:"Model"`
	} `json:"Data"`
}

// Search queries ModelScope for GGUF models, most-downloaded first.
func (c *Client) Search(ctx context.Context, query string) ([]modelsource.SearchResult, error) {
	body, err := json.Marshal(searchRequest{
		Name:       query,
		PageSize:   searchLimit,
		PageNumber: 1,
		SortBy:     "DownloadsCount",
		Criterion: []criterion{
			{Category: "libraries", Predicate: "contains", Values: []string{"gguf"}},
		},
	})
	if err != nil {
		return nil, err
	}

	// PUT, not POST — the endpoint rejects POST.
	req, err := http.NewRequestWithContext(ctx, http.MethodPut, apiBase+"/dolphin/models", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	c.setAuth(req)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ModelScope API returned %d", resp.StatusCode)
	}

	var raw searchResponse
	if err := json.NewDecoder(resp.Body).Decode(&raw); err != nil {
		return nil, err
	}
	if !raw.Success {
		return nil, fmt.Errorf("ModelScope search failed: %s", raw.Message)
	}

	found := raw.Data.Model.Models
	results := make([]modelsource.SearchResult, 0, len(found))
	dropped := 0
	for _, m := range found {
		// Defence against the silent-filter failure described above: drop
		// anything that isn't actually a GGUF repository, so a change to
		// the criterion contract degrades to fewer results rather than to
		// a list of models nothing here can load.
		if !hasLibrary(m.Libraries, "gguf") {
			dropped++
			continue
		}
		results = append(results, modelsource.SearchResult{
			ID:        m.Path + "/" + m.Name,
			Author:    m.Path,
			Downloads: m.Downloads,
			Likes:     m.Stars,
			Tags:      m.Tags,
			License:   m.License,
		})
	}
	// Every result being wrong means the server-side filter stopped
	// working, which is worth surfacing rather than showing an empty list
	// that reads as "no such model".
	if dropped > 0 && len(results) == 0 {
		return nil, fmt.Errorf("ModelScope returned %d models, none of them GGUF: the search filter is no longer being applied", dropped)
	}
	return results, nil
}

func hasLibrary(libs []string, want string) bool {
	for _, l := range libs {
		if strings.EqualFold(l, want) {
			return true
		}
	}
	return false
}

type filesResponse struct {
	Code    int    `json:"Code"`
	Success bool   `json:"Success"`
	Message string `json:"Message"`
	Data    struct {
		Files []struct {
			Name string `json:"Name"` // basename
			Path string `json:"Path"` // path within the repo, which is what we want
			Size int64  `json:"Size"`
			Type string `json:"Type"` // "blob" or "tree"
		} `json:"Files"`
	} `json:"Data"`
}

// GetModel fetches a repository's GGUF files. Unlike the HuggingFace
// client this needs a single request: ModelScope's file listing carries
// sizes, where HuggingFace returns names and sizes from separate
// endpoints.
func (c *Client) GetModel(ctx context.Context, modelID string) (*modelsource.Detail, error) {
	owner, name, ok := splitID(modelID)
	if !ok {
		return nil, fmt.Errorf("invalid ModelScope model id %q, want owner/name", modelID)
	}

	u := fmt.Sprintf("%s/models/%s/%s/repo/files?Revision=%s&Recursive=True",
		apiBase, url.PathEscape(owner), url.PathEscape(name), revision)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return nil, err
	}
	c.setAuth(req)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ModelScope API returned %d", resp.StatusCode)
	}

	var raw filesResponse
	if err := json.NewDecoder(resp.Body).Decode(&raw); err != nil {
		return nil, err
	}
	if !raw.Success {
		return nil, fmt.Errorf("ModelScope file listing failed: %s", raw.Message)
	}

	detail := &modelsource.Detail{ID: modelID}
	for _, f := range raw.Data.Files {
		if f.Type == "tree" {
			continue
		}
		// Path carries any subdirectory (quant folders like "UD-IQ1_M/"),
		// which is what the download URL and the shard grouping need.
		rel := f.Path
		if rel == "" {
			rel = f.Name
		}
		if !strings.HasSuffix(strings.ToLower(rel), ".gguf") {
			continue
		}
		detail.Files = append(detail.Files, modelsource.File{
			Filename:  rel,
			Size:      f.Size,
			Quant:     models.ParseQuant(rel),
			IsMMProj:  models.IsMMProjFile(rel),
			VRAMEstGB: modelsource.EstimateVRAM(f.Size),
		})
	}
	detail.Files = modelsource.GroupShards(detail.Files)
	return detail, nil
}

// DownloadURL returns a range-capable URL for one file.
//
// Two forms serve the same bytes and the choice between them is a real
// trade-off, measured rather than assumed:
//
//   - This one, /api/v1/.../repo, is served by modelscope.cn directly and
//     was reachable on every attempt from outside China.
//   - The browser-facing /models/<id>/resolve/<rev>/<file> redirects to a
//     signed cdn-lfs-cn-1.modelscope.cn link. That CDN answers a ranged
//     request with a correct 206, but timed out on one attempt in three
//     from here, and the signed URL expires.
//
// Reliability wins, but it comes with a caveat the caller must handle:
// this endpoint answers a ranged request with status 200 and a
// Content-Range header, where RFC 9110 requires 206. The body really is
// just the requested range. Code that resumes a download by sending
// Range and treats 200 as "the server ignored my Range and is sending
// the whole file" will silently append a tail to a partial file and
// produce a corrupt result — so a 200 carrying Content-Range must be
// read as partial. See ResponseIsPartial.
func (c *Client) DownloadURL(modelID, filename string) string {
	owner, name, ok := splitID(modelID)
	if !ok {
		return ""
	}
	return fmt.Sprintf("%s/models/%s/%s/repo?Revision=%s&FilePath=%s",
		apiBase, url.PathEscape(owner), url.PathEscape(name), revision, url.QueryEscape(filename))
}

// ModelURL returns the human-facing page for a repository, for the link
// next to a search result.
func (c *Client) ModelURL(modelID string) string {
	owner, name, ok := splitID(modelID)
	if !ok {
		return ""
	}
	return fmt.Sprintf("%s/models/%s/%s", baseURL, url.PathEscape(owner), url.PathEscape(name))
}

// splitID splits "owner/name". ModelScope ids have exactly one slash.
func splitID(modelID string) (owner, name string, ok bool) {
	owner, name, ok = strings.Cut(modelID, "/")
	if !ok || owner == "" || name == "" || strings.Contains(name, "/") {
		return "", "", false
	}
	return owner, name, true
}

// ResponseIsPartial reports whether a response to a ranged request
// carries only part of the file. It exists because ModelScope answers
// with 200 plus Content-Range rather than the 206 the RFC requires (see
// DownloadURL), so a status check alone is not enough to tell a resumed
// download from a restarted one.
func ResponseIsPartial(resp *http.Response) bool {
	return resp.StatusCode == http.StatusPartialContent || resp.Header.Get("Content-Range") != ""
}

func (c *Client) setAuth(req *http.Request) {
	if tok := c.authToken(); tok != "" {
		req.Header.Set("Authorization", "Bearer "+tok)
	}
}
