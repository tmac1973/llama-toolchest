package huggingface

import (
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

const baseURL = "https://huggingface.co/api"

// Client is a HuggingFace API client.
type Client struct {
	httpClient *http.Client

	// token is guarded because Settings can replace it while a search or
	// a listing is in flight on another goroutine.
	mu    sync.RWMutex
	token string

	// apiBase replaces the HuggingFace API address, for tests.
	apiBase string
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

// The search and file types are shared with the other model sources, so
// they are declared once in modelsource and aliased here. An alias, not a
// copy: huggingface.ModelFile and modelsource.File are the same type, so
// a client for another source can return values this package's callers
// accept without conversion.
type (
	ModelSearchResult = modelsource.SearchResult
	ModelFile         = modelsource.File
	ModelDetail       = modelsource.Detail
)

func NewClient(token string) *Client {
	return &Client{
		httpClient: &http.Client{Timeout: 30 * time.Second},
		token:      token,
	}
}

// Search queries HuggingFace for GGUF models.
func (c *Client) Search(ctx context.Context, query string) ([]ModelSearchResult, error) {
	u := fmt.Sprintf("%s/models?search=%s&filter=gguf&sort=downloads&direction=-1&limit=50",
		baseURL, url.QueryEscape(query))

	req, err := http.NewRequestWithContext(ctx, "GET", u, nil)
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
		return nil, fmt.Errorf("HF API returned %d", resp.StatusCode)
	}

	var results []ModelSearchResult
	if err := json.NewDecoder(resp.Body).Decode(&results); err != nil {
		return nil, err
	}
	return results, nil
}

// GetModel fetches model details and returns only GGUF files.
func (c *Client) GetModel(ctx context.Context, modelID string) (*ModelDetail, error) {
	u := fmt.Sprintf("%s/models/%s", baseURL, modelID)

	req, err := http.NewRequestWithContext(ctx, "GET", u, nil)
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
		return nil, fmt.Errorf("HF API returned %d", resp.StatusCode)
	}

	var raw struct {
		ID       string `json:"id"`
		Siblings []struct {
			Filename string `json:"rfilename"`
		} `json:"siblings"`
		GGUF struct {
			Total int64 `json:"total"`
		} `json:"gguf"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&raw); err != nil {
		return nil, err
	}

	detail := &ModelDetail{ID: raw.ID, ParamCount: raw.GGUF.Total}
	for _, s := range raw.Siblings {
		if !strings.HasSuffix(strings.ToLower(s.Filename), ".gguf") {
			continue
		}
		quant := models.ParseQuant(s.Filename)
		isMMProj := models.IsMMProjFile(s.Filename)
		// We don't have file sizes from the siblings list; fetch separately
		detail.Files = append(detail.Files, ModelFile{
			Filename: s.Filename,
			Quant:    quant,
			IsMMProj: isMMProj,
		})
	}

	// Fetch file sizes via tree API
	c.populateFileSizes(ctx, modelID, detail)

	// Group split/sharded GGUF files into single entries
	detail.Files = modelsource.GroupShards(detail.Files)

	return detail, nil
}

// populateFileSizes fetches file sizes from the HF tree API.
func (c *Client) populateFileSizes(ctx context.Context, modelID string, detail *ModelDetail) {
	type treeEntry struct {
		Path string `json:"path"`
		Size int64  `json:"size"`
		OID  string `json:"oid"`
		LFS  *struct {
			OID string `json:"oid"`
		} `json:"lfs"`
	}
	// The listing is paged past 1,000 entries, with the next page named in
	// a Link header. Repositories that large are rare (the biggest checked
	// held 395 files), but a missed page would leave sizes unknown.
	var tree []treeEntry
	next := fmt.Sprintf("%s/models/%s/tree/main?recursive=true", baseURL, modelID)
	for page := 0; next != "" && page < 20; page++ {
		req, err := http.NewRequestWithContext(ctx, "GET", next, nil)
		if err != nil {
			return
		}
		c.setAuth(req)
		resp, err := c.httpClient.Do(req)
		if err != nil {
			return
		}
		var part []treeEntry
		err = json.NewDecoder(resp.Body).Decode(&part)
		resp.Body.Close()
		if err != nil {
			break
		}
		tree = append(tree, part...)
		next = nextLink(resp.Header.Get("Link"))
	}

	type entry struct {
		size int64
		oid  string
	}
	byPath := map[string]entry{}
	for _, t := range tree {
		e := entry{size: t.Size, oid: t.OID}
		if t.LFS != nil && t.LFS.OID != "" {
			e.oid = t.LFS.OID
		}
		byPath[t.Path] = e
	}

	for i := range detail.Files {
		if e, ok := byPath[detail.Files[i].Filename]; ok {
			detail.Files[i].Size = e.size
			detail.Files[i].OID = e.oid
			detail.Files[i].VRAMEstGB = models.EstimateVRAM(e.size)
		}
	}
}

// DownloadURL returns the URL for one file in a repository. The same
// form the downloader uses, exposed so other callers (the header probe)
// do not have to rebuild it.
func (c *Client) DownloadURL(modelID, filename string) string {
	return "https://huggingface.co/" + modelID + "/resolve/main/" + filename
}

// ModelURL returns the human-facing page for a repository, for the link
// next to a search result.
func (c *Client) ModelURL(modelID string) string {
	if modelID == "" || strings.HasPrefix(modelID, "/") || !strings.Contains(modelID, "/") {
		return ""
	}
	return "https://huggingface.co/" + modelID
}

func (c *Client) setAuth(req *http.Request) {
	if tok := c.authToken(); tok != "" {
		req.Header.Set("Authorization", "Bearer "+tok)
	}
}

// nextLink returns the rel="next" URL of an HTTP Link header, or "".
func nextLink(h string) string {
	for _, part := range strings.Split(h, ",") {
		segs := strings.Split(part, ";")
		if len(segs) < 2 {
			continue
		}
		for _, p := range segs[1:] {
			if strings.ReplaceAll(strings.TrimSpace(p), " ", "") == `rel="next"` {
				return strings.Trim(strings.TrimSpace(segs[0]), "<>")
			}
		}
	}
	return ""
}
