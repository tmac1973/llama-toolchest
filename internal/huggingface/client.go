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

// SiteURL is the Hugging Face web site: repository pages and file
// downloads.
const SiteURL = "https://huggingface.co"

const baseURL = SiteURL + "/api"

// Client must stay interchangeable with the other model sources: the API
// layer picks one by source id without any handler, template or struct
// knowing which it got.
var _ modelsource.Client = (*Client)(nil)

// Client is a HuggingFace API client.
type Client struct {
	httpClient *http.Client

	// token is guarded because Settings can replace it while a search or
	// a listing is in flight on another goroutine.
	modelsource.BearerToken

	// apiBase replaces the HuggingFace API address, for tests.
	apiBase string
	// siteBase replaces SiteURL for file requests, for tests.
	siteBase string
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
	c := &Client{
		httpClient: &http.Client{Timeout: 30 * time.Second},
	}
	c.SetToken(token)
	return c
}

// Search queries HuggingFace for GGUF models.
func (c *Client) Search(ctx context.Context, query string) ([]ModelSearchResult, error) {
	u := fmt.Sprintf("%s/models?search=%s&filter=gguf&sort=downloads&direction=-1&limit=50",
		c.apiURL(), url.QueryEscape(query))

	req, err := http.NewRequestWithContext(ctx, "GET", u, nil)
	if err != nil {
		return nil, err
	}
	c.SetAuth(req)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Hugging Face API returned HTTP %d", resp.StatusCode)
	}

	var results []ModelSearchResult
	if err := json.NewDecoder(resp.Body).Decode(&results); err != nil {
		return nil, err
	}
	return results, nil
}

// GetModel fetches model details and returns only GGUF files.
func (c *Client) GetModel(ctx context.Context, modelID string) (*ModelDetail, error) {
	u := fmt.Sprintf("%s/models/%s", c.apiURL(), modelID)

	req, err := http.NewRequestWithContext(ctx, "GET", u, nil)
	if err != nil {
		return nil, err
	}
	c.SetAuth(req)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, statusError(resp, modelID, c.Token() != "")
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
		if !models.IsGGUFName(s.Filename) {
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
	modelsource.MarkMainlineCopies(detail.Files)

	return detail, nil
}

// isCommit reports whether rev is a full commit SHA (40 hex digits).
func isCommit(rev string) bool {
	if len(rev) != 40 {
		return false
	}
	for _, r := range rev {
		if !strings.ContainsRune("0123456789abcdef", r) {
			return false
		}
	}
	return true
}

// RepoCommit returns the commit that a branch or tag of a repository
// points to now, so a download can be pinned to it. A full commit SHA is
// returned as it is, without a request.
func (c *Client) RepoCommit(ctx context.Context, modelID, rev string) (string, error) {
	if isCommit(rev) {
		return rev, nil
	}
	u := fmt.Sprintf("%s/models/%s/revision/%s", c.apiURL(), modelID, url.PathEscape(rev))
	req, err := http.NewRequestWithContext(ctx, "GET", u, nil)
	if err != nil {
		return "", err
	}
	c.SetAuth(req)
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", statusError(resp, modelID, c.Token() != "")
	}
	var raw struct {
		SHA string `json:"sha"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&raw); err != nil {
		return "", err
	}
	if !isCommit(raw.SHA) {
		return "", fmt.Errorf("%s: Hugging Face gave no commit for %s", modelID, rev)
	}
	return raw.SHA, nil
}

// FileAccess checks that a file of a repository at a commit can be
// downloaded with the saved token. Hugging Face lists the files of a
// gated repository to anyone, and only the download is refused, so a
// listing alone does not show it. A HEAD request on the download address
// answers like the download would; its redirect to the file store is not
// followed. The error is the one statusError gives, for example the
// gated-model message.
func (c *Client) FileAccess(ctx context.Context, modelID, rev, filename string) error {
	site := SiteURL
	if c.siteBase != "" {
		site = c.siteBase
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodHead, site+"/"+modelID+"/resolve/"+url.PathEscape(rev)+"/"+filename, nil)
	if err != nil {
		return err
	}
	c.SetAuth(req)
	client := *c.httpClient
	client.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	resp.Body.Close()
	if resp.StatusCode >= 400 {
		return statusError(resp, modelID, c.Token() != "")
	}
	return nil
}

type treeEntry struct {
	Type string `json:"type"`
	Path string `json:"path"`
	Size int64  `json:"size"`
	OID  string `json:"oid"`
	LFS  *struct {
		OID string `json:"oid"`
	} `json:"lfs"`
}

// tree lists a repository's files through the HF tree API. The listing
// is paged past 1,000 entries, with the next page named in a Link
// header. Repositories that large are rare (the biggest checked held 395
// files), but a missed page would leave files out. A page that fails
// ends the listing early: err is set only when no page was read. rev
// is a branch, tag or commit.
func (c *Client) tree(ctx context.Context, modelID, rev string) ([]treeEntry, error) {
	var tree []treeEntry
	next := fmt.Sprintf("%s/models/%s/tree/%s?recursive=true", c.apiURL(), modelID, url.PathEscape(rev))
	for page := 0; next != "" && page < 20; page++ {
		req, err := http.NewRequestWithContext(ctx, "GET", next, nil)
		if err != nil {
			return tree, err
		}
		c.SetAuth(req)
		resp, err := c.httpClient.Do(req)
		if err != nil {
			if page == 0 {
				return nil, err
			}
			break
		}
		if resp.StatusCode != http.StatusOK {
			resp.Body.Close()
			if page == 0 {
				return nil, statusError(resp, modelID, c.Token() != "")
			}
			break
		}
		var part []treeEntry
		err = json.NewDecoder(resp.Body).Decode(&part)
		resp.Body.Close()
		if err != nil {
			if page == 0 {
				return nil, err
			}
			break
		}
		tree = append(tree, part...)
		next = nextLink(resp.Header.Get("Link"))
	}
	return tree, nil
}

// RepoFiles lists every file in a repository's main branch with its
// size and, for files kept in Git LFS, its SHA-256.
func (c *Client) RepoFiles(ctx context.Context, modelID string) ([]modelsource.RepoFile, error) {
	return c.RepoFilesAt(ctx, modelID, "main")
}

// RepoFilesAt is RepoFiles at one revision: a branch, tag or commit.
func (c *Client) RepoFilesAt(ctx context.Context, modelID, rev string) ([]modelsource.RepoFile, error) {
	tree, err := c.tree(ctx, modelID, rev)
	if err != nil {
		return nil, err
	}
	var out []modelsource.RepoFile
	for _, t := range tree {
		if t.Type == "directory" {
			continue
		}
		f := modelsource.RepoFile{Path: t.Path, Size: t.Size}
		// A plain git file's oid is a git blob SHA-1, not a SHA-256 of
		// the content, so only the LFS one is usable.
		if t.LFS != nil {
			f.SHA256 = t.LFS.OID
		}
		out = append(out, f)
	}
	return out, nil
}

// populateFileSizes fetches file sizes from the HF tree API.
func (c *Client) populateFileSizes(ctx context.Context, modelID string, detail *ModelDetail) {
	tree, _ := c.tree(ctx, modelID, "main")

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
	return hfRevisionURL(modelID, "main", filename)
}

// ModelURL returns the human-facing page for a repository, for the link
// next to a search result.
func (c *Client) ModelURL(modelID string) string {
	if modelID == "" || strings.HasPrefix(modelID, "/") || !strings.Contains(modelID, "/") {
		return ""
	}
	return RepoURL(modelID)
}

// RepoURL is the web page of a repository ("org/name").
func RepoURL(repoID string) string {
	return SiteURL + "/" + repoID
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
