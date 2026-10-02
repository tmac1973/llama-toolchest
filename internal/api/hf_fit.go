package api

import (
	"context"
	"fmt"
	"strings"

	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/modelsource"
)

// fileFit is what the Download Models table says about one file: how it
// would run on this machine, planned by the same models.PlanFit that
// Autoconfigure uses, so the two never disagree.
type fileFit struct {
	// Kind is "gpu" (every layer on the GPUs), "experts" (some MoE expert
	// weights in system memory), "partial" (a dense model with layers on
	// the CPU) or "none" (too large even with offloading).
	Kind string
	// Label is the cell text, e.g. "Up to 128K".
	Label string
	// Detail is one line per context size, for the tooltip.
	Detail string
	// VRAMGiB is the estimate at fitVRAMContext (or the model's limit,
	// when shorter) with every layer on the GPUs and a full-precision KV
	// cache: one fixed setting, so files can be compared by it.
	VRAMGiB float64
}

// fitVRAMContext is the context the VRAM column is estimated at.
const fitVRAMContext = 32768

// fitClasses are the context sizes the Fit column plans, largest first.
var fitClasses = []models.ContextClass{models.ContextMax, models.ContextLong, models.ContextMedium, models.ContextShort}

// metaProbeFile picks the file whose header describes the repository: the
// largest model file. Every quant carries the same description, and a
// header read costs the same whatever the file's size, so the largest is
// chosen because it is certainly the main model — a draft model or MTP
// head kept in the same repository is always smaller.
func metaProbeFile(files []modelsource.File) (modelsource.File, bool) {
	var best modelsource.File
	found := false
	for _, f := range files {
		if f.IsMMProj || f.Size <= 0 {
			continue
		}
		if !found || f.Size > best.Size {
			best, found = f, true
		}
	}
	return best, found
}

// repoMeta returns the model description for a repository, read from f
// (see metaProbeFile): from the cache or, when fetch is set, by reading
// one header from the host. Nil when there is none to be had.
func (s *Server) repoMeta(ctx context.Context, source, repo string, f modelsource.File, fetch bool) *models.GGUFMeta {
	key := modelsource.MetaKey(source, repo, f)
	if meta, ok := s.metaCache.Get(key); ok {
		return meta
	}
	if !fetch {
		return nil
	}
	first, firstSize := f.Filename, f.Size
	if len(f.Shards) > 0 && len(f.ShardSizes) == len(f.Shards) {
		first, firstSize = f.Shards[0], f.ShardSizes[0]
	}
	url := s.sourceClient(source).DownloadURL(repo, first)
	meta, err := modelsource.ProbeMeta(ctx, nil, s.sourceToken(source), url, firstSize)
	if err != nil {
		return nil
	}
	s.metaCache.Put(key, meta)
	return meta
}

// planFileFits plans every model file in a repository from its shared
// description. Files the description cannot speak for are left out, and
// keep the size-only label.
func (s *Server) planFileFits(detail *modelsource.Detail, meta *models.GGUFMeta) map[string]*fileFit {
	if meta == nil {
		return nil
	}
	hw := s.hardware()
	if len(hw.GPUs) == 0 {
		return nil
	}
	out := map[string]*fileFit{}
	for _, f := range detail.Files {
		if f.IsMMProj || f.Size <= 0 || !sameModel(f, detail.ParamCount) {
			continue
		}
		d := meta.DerivedFor(f.Size, detail.ParamCount)
		if f.StreamProbed {
			d.PLEBytes = f.StreamedBytes // measured beats derived
		}
		m := &models.Model{SizeBytes: f.Size}
		d.ApplyTo(m)
		out[f.Filename] = planFileFit(m, hw)
	}
	return out
}

// sameModel reports whether a file plausibly holds the model the
// repository's parameter count describes. A repository sometimes carries
// a small draft model beside the main one; its bits per weight against
// the main model's parameter count come out far below any real quant,
// and the shared description would misdescribe it.
func sameModel(f modelsource.File, params int64) bool {
	if params <= 0 {
		return true
	}
	bpw := float64(f.Size) * 8 / float64(params)
	return bpw >= 1.0 && bpw <= 34
}

// planFileFit plans one file at every context class and sums the result
// up as a label and a tooltip.
func planFileFit(m *models.Model, hw models.Hardware) *fileFit {
	trained := m.ContextLength
	base := models.DefaultConfig()

	var lines []string
	seen := map[int]bool{}
	bestGPU, bestExperts, anyPartial := 0, 0, false
	for _, class := range fitClasses {
		want := models.ContextClassTokens[class]
		if class == models.ContextMax || want == 0 || (trained > 0 && want > trained) {
			want = trained
		}
		if want <= 0 || seen[want] {
			continue
		}
		seen[want] = true

		r := models.PlanFit(m, base, hw, class)
		c := r.Config
		kind := fitKind(m, r)
		switch kind {
		case "gpu":
			bestGPU = max(bestGPU, c.ContextSize)
		case "experts":
			bestExperts = max(bestExperts, c.ContextSize)
		case "partial":
			anyPartial = true
		}
		lines = append(lines, tokensLabel(want)+": "+fitLine(m, r, kind, want))
	}

	fit := &fileFit{Detail: strings.Join(lines, "\n")}
	switch {
	case bestGPU > 0:
		fit.Kind, fit.Label = "gpu", "Up to "+tokensLabel(bestGPU)
	case bestExperts > 0:
		fit.Kind, fit.Label = "experts", "Experts in RAM · up to "+tokensLabel(bestExperts)
	case anyPartial:
		fit.Kind, fit.Label = "partial", "Partly on CPU"
	default:
		fit.Kind, fit.Label = "none", "Too large"
	}

	ctx := fitVRAMContext
	if trained > 0 && trained < ctx {
		ctx = trained
	}
	cfg := base
	cfg.ContextSize = ctx
	fit.VRAMGiB = models.VRAMEstimateForConfigOn(m, &cfg, discreteCards(hw))
	return fit
}

func fitKind(m *models.Model, r models.FitResult) string {
	c := r.Config
	switch {
	case !r.Fits:
		return "none"
	case c.GPULayers > 0 && c.GPULayers < m.NLayers:
		return "partial"
	case c.CPUMoE > 0:
		return "experts"
	}
	return "gpu"
}

// fitLine describes one context class's plan in a few words.
func fitLine(m *models.Model, r models.FitResult, kind string, want int) string {
	c := r.Config
	var parts []string
	switch kind {
	case "none":
		return "does not fit"
	case "partial":
		parts = append(parts, fmt.Sprintf("only %d of %d layers on the GPU", c.GPULayers, m.NLayers))
	case "experts":
		parts = append(parts, fmt.Sprintf("experts of %d layers in system memory", c.CPUMoE-m.ExpertLayerFirst))
	default:
		parts = append(parts, "all on the GPU")
	}
	if c.KVCacheQuant == "q8_0" {
		parts = append(parts, "8-bit KV cache")
	} else {
		parts = append(parts, "full-precision KV cache")
	}
	if c.ContextSize < want {
		parts = append(parts, "only "+tokensLabel(c.ContextSize)+" fits")
	}
	return strings.Join(parts, ", ")
}

// tokensLabel writes a context size the way models are described:
// 8K, 128K, 1M.
func tokensLabel(n int) string {
	if n >= 1<<20 && n%(1<<20) == 0 {
		return fmt.Sprintf("%dM", n>>20)
	}
	if n >= 1024 {
		return fmt.Sprintf("%dK", (n+512)/1024)
	}
	return fmt.Sprintf("%d", n)
}

// discreteCards is how many GPUs PlanFit spreads a model over: the
// dedicated ones, or the integrated GPU when it is the only one.
func discreteCards(hw models.Hardware) int {
	n := 0
	for _, g := range hw.GPUs {
		if !g.IsIGPU {
			n++
		}
	}
	return max(1, n)
}
