package huggingface

import (
	"context"
	"net/url"
	"path"
	"slices"
	"strings"

	"github.com/tmac1973/llama-toolchest/internal/models"
)

// ProjectorRepo is a repository holding vision projector (mmproj) files
// for a model, with the projector files best first (see projectorRank).
type ProjectorRepo struct {
	Repo  string
	Files []ModelFile
}

// maxBaseModels caps how many of a repository's base models are searched
// for quantizations: a merge can list many, and the first is the one the
// quant was made from.
const maxBaseModels = 2

// repoInfo is the part of a repository's API record the projector lookup
// reads.
type repoInfo struct {
	ID       string `json:"id"`
	Author   string `json:"author"`
	CardData struct {
		BaseModel BaseModels `json:"base_model"`
	} `json:"cardData"`
	Siblings []struct {
		Filename string `json:"rfilename"`
	} `json:"siblings"`
}

func (ri *repoInfo) hasProjector() bool {
	for _, s := range ri.Siblings {
		if isProjectorFile(s.Filename) {
			return true
		}
	}
	return false
}

func isProjectorFile(name string) bool {
	return strings.HasSuffix(strings.ToLower(name), ".gguf") && models.IsMMProjFile(name)
}

func (c *Client) repoInfo(ctx context.Context, repo string) (*repoInfo, error) {
	var ri repoInfo
	if err := c.getJSON(ctx, c.apiURL()+"/models/"+repo, &ri); err != nil {
		return nil, err
	}
	if ri.ID == "" {
		ri.ID = repo
	}
	return &ri, nil
}

// FindProjectorRepo looks for the vision projector of a model downloaded
// from repo. It is the repository itself when that has one; otherwise the
// same publisher's sibling repository (mradermacher's "X-i1-GGUF" keeps
// its projector in "X-GGUF"), and then any other quantization of the same
// base model, the same publisher's first. base stands in for the
// repository's base model when its card names none (the GGUF header's
// general.base_model). A nil result with a nil error means none was found.
func (c *Client) FindProjectorRepo(ctx context.Context, repo, base string) (*ProjectorRepo, error) {
	self, err := c.repoInfo(ctx, repo)
	if err != nil {
		return nil, err
	}
	if self.hasProjector() {
		return c.projectorFiles(ctx, self.ID)
	}
	author := self.Author
	if author == "" {
		author, _, _ = strings.Cut(repo, "/")
	}

	for _, guess := range siblingGuesses(repo) {
		ri, err := c.repoInfo(ctx, guess)
		if err != nil || !ri.hasProjector() {
			continue
		}
		return c.projectorFiles(ctx, ri.ID)
	}

	bases := []string(self.CardData.BaseModel)
	if len(bases) == 0 && base != "" {
		bases = []string{base}
	}
	if len(bases) > maxBaseModels {
		bases = bases[:maxBaseModels]
	}
	var firstErr error
	for _, b := range bases {
		cands, err := c.quantizationsOf(ctx, b)
		if err != nil {
			if firstErr == nil {
				firstErr = err
			}
			continue
		}
		// The list is in download order; a stable sort puts the same
		// publisher's repositories first without disturbing the rest.
		slices.SortStableFunc(cands, func(a, b repoInfo) int {
			return boolRank(strings.EqualFold(a.Author, author)) - boolRank(strings.EqualFold(b.Author, author))
		})
		for _, cand := range cands {
			if strings.EqualFold(cand.ID, repo) || !cand.hasProjector() {
				continue
			}
			return c.projectorFiles(ctx, cand.ID)
		}
	}
	return nil, firstErr
}

func boolRank(first bool) int {
	if first {
		return 0
	}
	return 1
}

// siblingGuesses names the publisher's other repositories that may hold a
// projector for repo, by publishers' naming conventions.
func siblingGuesses(repo string) []string {
	lower := strings.ToLower(repo)
	for _, suffix := range []string{"-i1-gguf", "-imatrix-gguf"} {
		if strings.HasSuffix(lower, suffix) {
			return []string{repo[:len(repo)-len(suffix)] + "-GGUF"}
		}
	}
	return nil
}

// quantizationsOf lists the GGUF quantizations of a base model, most
// downloaded first, with their file lists.
func (c *Client) quantizationsOf(ctx context.Context, base string) ([]repoInfo, error) {
	v := url.Values{}
	v.Add("filter", "base_model:quantized:"+base)
	v.Add("filter", "gguf")
	v.Set("sort", "downloads")
	v.Set("direction", "-1")
	v.Set("limit", "50")
	for _, f := range []string{"author", "siblings"} {
		v.Add("expand[]", f)
	}
	var out []repoInfo
	if err := c.getJSON(ctx, c.apiURL()+"/models?"+v.Encode(), &out); err != nil {
		return nil, err
	}
	return out, nil
}

// projectorFiles lists a repository's projector files with their sizes.
func (c *Client) projectorFiles(ctx context.Context, repo string) (*ProjectorRepo, error) {
	tree, err := c.tree(ctx, repo)
	if err != nil {
		return nil, err
	}
	pr := &ProjectorRepo{Repo: repo}
	for _, t := range tree {
		if t.Type == "directory" || !isProjectorFile(t.Path) {
			continue
		}
		f := ModelFile{
			Filename: t.Path,
			Size:     t.Size,
			Quant:    projectorQuant(t.Path),
			IsMMProj: true,
			OID:      t.OID,
		}
		if t.LFS != nil && t.LFS.OID != "" {
			f.OID = t.LFS.OID
		}
		pr.Files = append(pr.Files, f)
	}
	if len(pr.Files) == 0 {
		return nil, nil
	}
	slices.SortStableFunc(pr.Files, func(a, b ModelFile) int {
		return projectorRank(a.Quant) - projectorRank(b.Quant)
	})
	return pr, nil
}

// projectorQuant is a projector file's precision as written in its name
// ("mmproj-Q8_0.gguf", "mmproj-model-f16.gguf"), upper-cased.
func projectorQuant(name string) string {
	base := strings.ToUpper(strings.TrimSuffix(path.Base(name), path.Ext(name)))
	for _, q := range []string{"Q8_0", "BF16", "F16", "F32", "Q6_K", "Q4_K_M"} {
		if strings.Contains(base, q) {
			return q
		}
	}
	return models.ParseQuant(name)
}

// projectorRank orders projector precisions: Q8_0 (small, near-lossless
// for a projector) first, then the 16-bit forms, then the rest.
func projectorRank(quant string) int {
	switch quant {
	case "Q8_0":
		return 0
	case "F16":
		return 1
	case "BF16":
		return 2
	case "F32":
		return 3
	}
	return 4
}
