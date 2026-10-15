package autoconfig

import (
	"context"
	"fmt"
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
//   - the repository the card names, if it has a GGUF — or its -GGUF
//     conversion, and under the model's publisher when the card gives the
//     name without an owner;
//   - for draft-mtp, a separate MTP head in the model's own repository;
//   - for a converted head (EAGLE-3, DFlash, DSpark), a GGUF repository
//     named after the model and the method, the publisher's first;
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

	headBased := models.IsHeadBasedDraftMode(checked.WantDraft)
	pick := preferredDraftFile
	if headBased {
		pick = preferredHeadFile
	}

	if checked.DraftRepo != "" {
		for _, repo := range namedRepoCandidates(checked.DraftRepo, m) {
			d, err := hub.GetModel(ctx, repo)
			if err != nil {
				continue
			}
			if f, ok := pick(d.Files); ok {
				add(canonicalRepo(d, repo), f, "The model card names this repository for speculative decoding.")
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
	default:
		if headBased {
			if repo, f, ok := headByName(ctx, hub, m, checked.WantDraft); ok {
				add(repo, f, fmt.Sprintf("A %s head for this model, found by name on the hub.", models.SpecModeLabel(checked.WantDraft)))
			}
		}
	}
	return out
}

// baseRepo is the repository of the model this one was made from — the
// publisher's own, not a quantizer's — or, when the file does not say,
// the model's own repository without its -GGUF suffix.
func baseRepo(m *models.Model) (owner, name string) {
	if m.BaseModelRepo != "" {
		return repoName(m.BaseModelRepo)
	}
	owner, name = repoName(m.ModelID)
	for _, suf := range []string{"-GGUF", "-gguf"} {
		name = strings.TrimSuffix(name, suf)
	}
	return owner, name
}

// namedRepoCandidates are the repositories a name the card gives may
// mean, in the order worth trying: the name as given, then its GGUF
// conversion, since a card usually names the original weights. A name
// without an owner is tried under the model's publisher, then under the
// owner of the model's own repository.
func namedRepoCandidates(named string, m *models.Model) []string {
	var owners []string
	owner, name := repoName(strings.Trim(named, "/ "))
	if owner != "" {
		owners = []string{owner}
	} else {
		baseOwner, _ := baseRepo(m)
		modelOwner, _ := repoName(m.ModelID)
		owners = []string{baseOwner, modelOwner}
	}
	seen := map[string]bool{}
	var out []string
	for _, o := range owners {
		if o == "" {
			continue
		}
		for _, n := range []string{name, name + "-GGUF", name + "-gguf"} {
			// Both spellings of the suffix: Hugging Face matches either,
			// but a hub that matches case exactly finds only the real one.
			repo := o + "/" + n
			if !seen[repo] {
				seen[repo] = true
				out = append(out, repo)
			}
		}
	}
	return out
}

// canonicalRepo is the repository's name as the hub spells it. The hub
// finds "X-DSpark-GGUF" when the repository is "X-DSpark-gguf", but the
// download and the model's registry entry should carry the real name.
func canonicalRepo(d *modelsource.Detail, asked string) string {
	if d != nil && d.ID != "" {
		return d.ID
	}
	return asked
}

// headSearchWords is the word a hub search for a head of each method uses.
var headSearchWords = map[string]string{
	"draft-eagle3": "EAGLE3",
	"draft-dflash": "DFlash",
	"draft-dspark": "DSpark",
}

// headByName searches the hub for a converted head of the given method
// made for m: a GGUF repository whose name carries both the model's name
// and the method's, the way publishers name them ("MiniCPM-V-4.7-35B-A3B-
// DSpark-gguf"). The model's publisher's own comes first; then the rest,
// most downloaded first, which is the order the search returns them in.
func headByName(ctx context.Context, hub Hub, m *models.Model, mode string) (string, modelsource.File, bool) {
	owner, name := baseRepo(m)
	if name == "" {
		return "", modelsource.File{}, false
	}
	results, err := hub.Search(ctx, name+" "+headSearchWords[mode])
	if err != nil {
		return "", modelsource.File{}, false
	}
	want := strings.ToLower(name)
	var own, others []string
	for _, r := range results {
		rOwner, rName := repoName(r.ID)
		if !strings.Contains(strings.ToLower(rName), want) {
			continue // another model's head
		}
		if len(models.HeadsForMode(mode, []models.DraftCandidate{{ID: rName}})) == 0 {
			continue // not named for this method
		}
		if strings.EqualFold(rOwner, owner) {
			own = append(own, r.ID)
		} else {
			others = append(others, r.ID)
		}
	}
	for _, repo := range append(own, others...) {
		d, err := hub.GetModel(ctx, repo)
		if err != nil {
			continue
		}
		if f, ok := preferredHeadFile(d.Files); ok {
			return canonicalRepo(d, repo), f, true
		}
	}
	return "", modelsource.File{}, false
}

// preferredHeadFile picks the file to suggest from a head's repository.
// A head is small — hundreds of MiB — so it is taken at 8 bits where it
// can be, and at full precision before 4 bits, which costs a head more of
// its acceptance than it saves.
func preferredHeadFile(files []modelsource.File) (modelsource.File, bool) {
	for _, q := range []string{"Q8_0", "F16", "BF16", "Q6_K", "Q4_K_M"} {
		for _, f := range files {
			if strings.EqualFold(f.Quant, q) && !models.IsMMProjFile(f.Filename) {
				return f, true
			}
		}
	}
	return modelsource.File{}, false
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
