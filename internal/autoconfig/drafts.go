package autoconfig

import (
	"context"
	"math"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/modelsource"
)

// Hub is the part of a model hub client draft suggestions need.
type Hub interface {
	Search(ctx context.Context, query string) ([]modelsource.SearchResult, error)
	GetModel(ctx context.Context, repo string) (*modelsource.Detail, error)
}

// DraftSuggestion is a file worth downloading for speculative decoding.
type DraftSuggestion struct {
	Repo      string
	Filename  string
	SizeBytes int64
	Mode      string // the draft method it serves
	Why       string
}

// sizeToken matches a parameter count in a repository name: "-9B",
// "-0.8B", "-500M". Everything from it on is dropped to get the family
// name: "Qwen3.5-9B-Instruct" → "Qwen3.5".
var sizeToken = regexp.MustCompile(`-(\d+(?:\.\d+)?)([BbMm])\b`)

// familyAndSize splits a repository name (the part after the owner) into
// its family and parameter count. ok is false when the name has no size.
func familyAndSize(name string) (family string, params float64, ok bool) {
	loc := sizeToken.FindStringSubmatchIndex(name)
	if loc == nil {
		return "", 0, false
	}
	n, err := strconv.ParseFloat(name[loc[2]:loc[3]], 64)
	if err != nil {
		return "", 0, false
	}
	switch strings.ToUpper(name[loc[4]:loc[5]]) {
	case "B":
		n *= 1e9
	case "M":
		n *= 1e6
	}
	return name[:loc[0]], n, true
}

func repoName(repo string) (owner, name string) {
	if i := strings.IndexByte(repo, '/'); i >= 0 {
		return repo[:i], repo[i+1:]
	}
	return "", repo
}

// preferredDraftFile picks the file to suggest from a repository: Q8_0,
// then Q4_K_M. Vision projectors and MTP heads are not draft models.
func preferredDraftFile(files []modelsource.File) (modelsource.File, bool) {
	for _, q := range []string{"Q8_0", "Q4_K_M"} {
		for _, f := range files {
			if f.Quant == q && !models.IsMMProjFile(f.Filename) && !strings.Contains(strings.ToLower(f.Filename), "mtp") {
				return f, true
			}
		}
	}
	return modelsource.File{}, false
}

// FindDraftSuggestions finds files for the draft method the model card
// recommends when none is installed (checked.WantDraft):
//
//   - the repository the card names, if it has a GGUF;
//   - for draft-mtp, a separate MTP head in the model's own repository;
//   - for a plain draft model, the smallest model of the same family from
//     the same publisher, below 2B parameters and at most a quarter of the
//     model's size.
//
// Whether a downloaded draft model's vocabulary matches is checked
// afterwards, by the rules that list draft candidates.
func FindDraftSuggestions(ctx context.Context, hub Hub, m *models.Model, checked Checked, installed func(repo, filename string) bool) []DraftSuggestion {
	if checked.WantDraft == "" || hub == nil {
		return nil
	}
	var out []DraftSuggestion
	add := func(repo string, f modelsource.File, why string) {
		if installed != nil && installed(repo, f.Filename) {
			return
		}
		out = append(out, DraftSuggestion{Repo: repo, Filename: f.Filename, SizeBytes: f.Size, Mode: checked.WantDraft, Why: why})
	}

	if checked.DraftRepo != "" && strings.Count(checked.DraftRepo, "/") == 1 {
		if d, err := hub.GetModel(ctx, checked.DraftRepo); err == nil {
			if f, ok := preferredDraftFile(d.Files); ok {
				add(checked.DraftRepo, f, "The model card names this repository for speculative decoding.")
				return out
			}
		}
	}

	switch checked.WantDraft {
	case "draft-mtp":
		if d, err := hub.GetModel(ctx, m.ModelID); err == nil {
			for _, f := range d.Files {
				if strings.Contains(strings.ToLower(f.Filename), "mtp") && !models.IsMMProjFile(f.Filename) {
					add(m.ModelID, f, "The model's multi-token prediction (MTP) head, published next to the model.")
					break
				}
			}
		}
	case "draft":
		if s, ok := familyDraft(ctx, hub, m); ok {
			add(s.Repo, modelsource.File{Filename: s.Filename, Size: s.SizeBytes}, s.Why)
		}
	}
	return out
}

func familyDraft(ctx context.Context, hub Hub, m *models.Model) (DraftSuggestion, bool) {
	owner, name := repoName(m.ModelID)
	baseName := name
	if m.BaseModelRepo != "" {
		_, baseName = repoName(m.BaseModelRepo)
	}
	family, params, ok := familyAndSize(baseName)
	if !ok {
		if family, params, ok = familyAndSize(name); !ok {
			return DraftSuggestion{}, false
		}
	}
	results, err := hub.Search(ctx, family+" GGUF")
	if err != nil {
		return DraftSuggestion{}, false
	}
	type cand struct {
		repo   string
		params float64
	}
	var cands []cand
	limit := math.Min(2e9, params/4)
	for _, r := range results {
		rOwner, rName := repoName(r.ID)
		if !strings.EqualFold(rOwner, owner) {
			continue
		}
		fam, p, ok := familyAndSize(rName)
		if !ok || !strings.EqualFold(fam, family) || p >= limit {
			continue
		}
		cands = append(cands, cand{r.ID, p})
	}
	sort.Slice(cands, func(i, j int) bool { return cands[i].params < cands[j].params })
	for _, c := range cands {
		d, err := hub.GetModel(ctx, c.repo)
		if err != nil {
			continue
		}
		if f, ok := preferredDraftFile(d.Files); ok {
			return DraftSuggestion{Repo: c.repo, Filename: f.Filename, SizeBytes: f.Size, Mode: "draft",
				Why: "A much smaller model of the same family from the same publisher, to propose tokens for the larger one."}, true
		}
	}
	return DraftSuggestion{}, false
}
