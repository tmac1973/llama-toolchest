package recommend

import (
	"cmp"
	"math"
	"slices"
	"strings"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/huggingface"
	"github.com/tmac1973/llama-toolchest/internal/models"
)

// Group is one base model and the repos that publish GGUFs of it. The
// first repo is the one a card names; the rest are listed beside it.
type Group struct {
	Key string
	// BaseRepo is the base model's repository ("Qwen/Qwen3-32B"), or ""
	// when the repos do not name one. Name is what the card calls it.
	BaseRepo string
	Name     string
	Repos    []huggingface.ListedModel

	Params     int64
	Arch       string
	TrainedCtx int
	Downloads  int
	Likes      int
	// Released is the earliest upload in the group, close to when the
	// model came out. Publishers re-upload often, so a repo's last change
	// says little about the model's age.
	Released time.Time
	Gated    bool

	// Filled in for finalists that verify.
	Vision      bool
	MMProjBytes int64
	Picks       map[models.ContextClass]*Pick

	// unsupported marks a group the active build cannot load.
	unsupported bool
	// paramsFromHub is false when HuggingFace's summary described some
	// other file in the repo (the image reader), so Params is not the
	// model's.
	paramsFromHub bool
	coarse        float64
}

// Repo is the repo the card names.
func (g *Group) Repo() huggingface.ListedModel { return g.Repos[0] }

// Alts are the other repos publishing the same model.
func (g *Group) Alts() []string {
	var out []string
	for _, r := range g.Repos[1:] {
		out = append(out, r.ID)
	}
	return out
}

// groupKey is the base model a repo quantizes, in lower case: the first
// cardData.base_model, or the repo's own name without a GGUF suffix.
func groupKey(r huggingface.ListedModel) (key, base string) {
	if len(r.CardData.BaseModel) > 0 && r.CardData.BaseModel[0] != "" {
		base = r.CardData.BaseModel[0]
		return strings.ToLower(base), base
	}
	name := r.ID
	if i := strings.LastIndex(name, "/"); i >= 0 {
		name = name[i+1:]
	}
	for _, suffix := range []string{"-gguf", "_gguf", ".gguf"} {
		if strings.HasSuffix(strings.ToLower(name), suffix) {
			name = name[:len(name)-len(suffix)]
		}
	}
	return strings.ToLower(name), ""
}

// groupRepos gathers repos by base model and orders each group's repos:
// the base model's own author first (an official quant), then trusted
// publishers in their order, then everyone else by downloads.
func groupRepos(repos []huggingface.ListedModel, p Profile) []*Group {
	byKey := map[string]*Group{}
	var order []string
	for _, r := range repos {
		key, base := groupKey(r)
		g, ok := byKey[key]
		if !ok {
			g = &Group{Key: key, BaseRepo: base}
			byKey[key] = g
			order = append(order, key)
		}
		g.Repos = append(g.Repos, r)
	}

	groups := make([]*Group, 0, len(order))
	for _, key := range order {
		g := byKey[key]
		baseAuthor := ""
		if i := strings.Index(g.BaseRepo, "/"); i > 0 {
			baseAuthor = g.BaseRepo[:i]
		}
		rank := func(r huggingface.ListedModel) int {
			switch {
			case baseAuthor != "" && strings.EqualFold(r.Author, baseAuthor):
				return -1
			case publisherRank(r.Author) >= 0:
				return publisherRank(r.Author)
			}
			return math.MaxInt
		}
		slices.SortStableFunc(g.Repos, func(a, b huggingface.ListedModel) int {
			if c := cmp.Compare(rank(a), rank(b)); c != 0 {
				return c
			}
			return cmp.Compare(b.Downloads, a.Downloads)
		})

		first := g.Repo()
		g.Name = displayName(g.BaseRepo, first.ID)
		g.Arch = first.GGUF.Architecture
		g.Params = first.GGUF.Total
		g.paramsFromHub = g.Arch != clipArch && g.Params > 0
		g.TrainedCtx = first.GGUF.ContextLength
		g.Gated = bool(first.Gated)
		for _, r := range g.Repos {
			g.Downloads += r.Downloads
			g.Likes += r.Likes
			if g.Released.IsZero() || (!r.CreatedAt.IsZero() && r.CreatedAt.Before(g.Released)) {
				g.Released = r.CreatedAt
			}
		}
		g.unsupported = g.Arch != clipArch && !p.supports(g.Arch)
		groups = append(groups, g)
	}
	return groups
}

// displayName is the name a card shows: the base model's, or the repo's
// without its GGUF suffix.
func displayName(baseRepo, repo string) string {
	name := baseRepo
	if name == "" {
		name = repo
		for _, suffix := range []string{"-GGUF", "-gguf", "_GGUF", ".gguf"} {
			name = strings.TrimSuffix(name, suffix)
		}
	}
	if i := strings.LastIndex(name, "/"); i >= 0 {
		name = name[i+1:]
	}
	return name
}

const (
	// finalistCount is how many groups get a full check: a file listing
	// and a header read each.
	finalistCount = 40
	// smallestBPW is the fewest bits per weight the feed suggests (about
	// IQ3_XXS); typicalBPW is a common 4-bit quant (Q4_K_M).
	smallestBPW = 3.0
	typicalBPW  = 4.8
)

// sizeBucket places a group by how much of the GPU budget a typical
// 4-bit quant of it takes.
type sizeBucket int

const (
	bucketSmall  sizeBucket = iota // under a quarter of the GPU budget
	bucketMedium                   // a quarter to 90%
	bucketLarge                    // over 90%: needs a low-bit quant or experts in system memory
)

// bucketPlaces reserves finalist places by size, so one kind of model
// cannot take every place. vllm-toolchest's first live build had models
// of hundreds of gigabytes filling half the list.
var bucketPlaces = map[sizeBucket]int{bucketSmall: 10, bucketMedium: 14, bucketLarge: 10}

func gib(bytes float64) float64 { return bytes / (1 << 30) }

// chooseFinalists drops groups too large for the machine, ranks the rest
// by a coarse score and returns the ones worth a full check.
func chooseFinalists(groups []*Group, hw models.Hardware) []*Group {
	vram, ram := models.FitBudgets(hw)
	var fit []*Group
	for _, g := range groups {
		// Whether a model can keep experts in system memory is not known
		// until its header is read, so system memory counts for every
		// model here; the quant picker refuses dense models that need it.
		if g.paramsFromHub && gib(float64(g.Params)*smallestBPW/8) > vram+ram {
			continue
		}
		fit = append(fit, g)
	}
	if len(fit) == 0 {
		return nil
	}

	downloads := make([]float64, len(fit))
	released := make([]float64, len(fit))
	likes := make([]float64, len(fit))
	for i, g := range fit {
		downloads[i] = float64(g.Downloads)
		released[i] = float64(g.Released.Unix())
		likes[i] = float64(g.Likes)
	}
	pd, pr, pl := percentiles(downloads), percentiles(released), percentiles(likes)
	for i, g := range fit {
		trusted := 0.0
		for _, r := range g.Repos {
			if publisherRank(r.Author) >= 0 {
				trusted = 1
				break
			}
		}
		g.coarse = 0.45*pd[i] + 0.30*pr[i] + 0.15*trusted + 0.10*pl[i]
	}
	slices.SortStableFunc(fit, func(a, b *Group) int { return cmp.Compare(b.coarse, a.coarse) })

	bucketOf := func(g *Group) sizeBucket {
		if !g.paramsFromHub || vram <= 0 {
			return bucketMedium
		}
		share := gib(float64(g.Params)*typicalBPW/8) / vram
		switch {
		case share < 0.25:
			return bucketSmall
		case share > 0.9:
			return bucketLarge
		}
		return bucketMedium
	}
	taken := map[*Group]bool{}
	var out []*Group
	for _, b := range []sizeBucket{bucketSmall, bucketMedium, bucketLarge} {
		n := 0
		for _, g := range fit {
			if n == bucketPlaces[b] {
				break
			}
			if bucketOf(g) == b {
				out = append(out, g)
				taken[g] = true
				n++
			}
		}
	}
	for _, g := range fit {
		if len(out) >= finalistCount {
			break
		}
		if !taken[g] {
			out = append(out, g)
		}
	}
	slices.SortStableFunc(out, func(a, b *Group) int { return cmp.Compare(b.coarse, a.coarse) })
	return out
}

// percentiles ranks each value from 0 (lowest) to 1 (highest). Equal
// values share their average rank.
func percentiles(values []float64) []float64 {
	n := len(values)
	out := make([]float64, n)
	if n <= 1 {
		for i := range out {
			out[i] = 1
		}
		return out
	}
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	slices.SortFunc(idx, func(a, b int) int { return cmp.Compare(values[a], values[b]) })
	for i := 0; i < n; {
		j := i
		for j+1 < n && values[idx[j+1]] == values[idx[i]] {
			j++
		}
		rank := float64(i+j) / 2 / float64(n-1)
		for k := i; k <= j; k++ {
			out[idx[k]] = rank
		}
		i = j + 1
	}
	return out
}
