package huggingface

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/modelsource"
)

const baseURL = "https://huggingface.co/api"

// Client is a HuggingFace API client.
type Client struct {
	httpClient *http.Client
	token      string
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
	}
	if err := json.NewDecoder(resp.Body).Decode(&raw); err != nil {
		return nil, err
	}

	detail := &ModelDetail{ID: raw.ID}
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
	detail.Files = groupShards(detail.Files)

	return detail, nil
}

// populateFileSizes fetches file sizes from the HF tree API.
func (c *Client) populateFileSizes(ctx context.Context, modelID string, detail *ModelDetail) {
	u := fmt.Sprintf("%s/models/%s/tree/main?recursive=true", baseURL, modelID)
	req, err := http.NewRequestWithContext(ctx, "GET", u, nil)
	if err != nil {
		return
	}
	c.setAuth(req)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return
	}
	defer resp.Body.Close()

	var tree []struct {
		Path string `json:"path"`
		Size int64  `json:"size"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&tree); err != nil {
		return
	}

	sizeMap := map[string]int64{}
	for _, t := range tree {
		sizeMap[t.Path] = t.Size
	}

	for i := range detail.Files {
		if size, ok := sizeMap[detail.Files[i].Filename]; ok {
			detail.Files[i].Size = size
			detail.Files[i].VRAMEstGB = estimateVRAM(size)
		}
	}
}

func (c *Client) setAuth(req *http.Request) {
	if c.token != "" {
		req.Header.Set("Authorization", "Bearer "+c.token)
	}
}

// The GGUF file helpers below are not HuggingFace-specific — shard naming
// and size arithmetic are properties of the files, not of the host — so
// they live in modelsource and are forwarded here for existing callers.
var (
	groupShards  = modelsource.GroupShards
	estimateVRAM = modelsource.EstimateVRAM
)

// ExpandShards returns all shard filenames for a split GGUF, or a
// single-element slice for an unsplit one.
func ExpandShards(filename string) []string { return modelsource.ExpandShards(filename) }
