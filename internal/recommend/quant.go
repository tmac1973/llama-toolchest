package recommend

import (
	"cmp"
	"slices"

	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/modelsource"
)

// Classes are the context sizes the feed offers, smallest first.
var Classes = []models.ContextClass{models.ContextShort, models.ContextMedium, models.ContextLong, models.ContextMax}

// goodBPW is where a quant stops losing noticeable quality: about IQ4_XS
// and up. Below it, down to smallestBPW, a quant is still offered but
// ranks lower on quality.
const goodBPW = 4.2

// offloadMaxBPW bounds the quant chosen when experts go to system memory:
// Q4 and Q5 class. A larger quant keeps more experts in system memory,
// which costs speed for little quality.
const offloadMaxBPW = 6.0

// unquantizedBPW is where full-precision files begin (F16, BF16). Q8_0
// keeps practically all of a model's quality at half the size and twice
// the speed, so a full-precision file is suggested only when a repo has
// nothing smaller that runs.
const unquantizedBPW = 9.0

// Pick is the file suggested for one model at one context size, and how
// it would run.
type Pick struct {
	File      string // the first shard of a split file
	Quant     string
	SizeBytes int64
	BPW       float64

	Context   int
	KVQuant   string // "" (f16) or "q8_0"
	Placement models.Placement
	// EstimateGiB is the GPU memory the plan uses, of BudgetGiB.
	EstimateGiB float64
	BudgetGiB   float64
	// CPUMoE and CPURAMGiB describe experts kept in system memory.
	CPUMoE    int
	CPURAMGiB float64
	// FewestGPUs is the fewest cards the file runs on fully at this
	// context, of Cards in all. Equal when it needs them all.
	FewestGPUs int
	Cards      int

	// For the speed estimate: the expert weights, and how many experts
	// each token uses.
	ExpertBytes     int64
	ExpertUsedCount int
	ExpertCount     int
}

// candidateFile is one quant of a finalist, ready to plan.
type candidateFile struct {
	file  modelsource.File
	model *models.Model
	bpw   float64
}

// classTarget is the context a class asks of a model. A model trained for
// less than the class has no pick there: it cannot hold that much.
func classTarget(class models.ContextClass, trained int) (int, bool) {
	if class == models.ContextMax {
		return trained, trained > 0
	}
	t := models.ContextClassTokens[class]
	if trained > 0 && t > trained {
		return 0, false
	}
	return t, true
}

// pickTiers are the order a quant is chosen in, the first match winning
// and the largest file within a tier first. A dense model with layers on
// the CPU is never suggested.
//
// For a mixture-of-experts model, a Q4–Q5 quant with some experts in
// system memory comes before a 3-bit quant on the GPUs. Each token reads
// only a few experts, so the offload costs less speed than the low-bit
// quant costs quality, and it is how these models are commonly run on a
// card too small for them. The first live build bore this out: the
// 3-bit picks of 35B-A3B models filled the top of every list.
var pickTiers = []struct {
	minBPW, maxBPW float64
	placement      models.Placement
}{
	{goodBPW, unquantizedBPW, models.PlacementGPU},
	{unquantizedBPW, 1e9, models.PlacementGPU},
	{goodBPW, offloadMaxBPW, models.PlacementExperts},
	{smallestBPW, goodBPW, models.PlacementGPU},
	{smallestBPW, goodBPW, models.PlacementExperts},
}

// pickFor chooses the file to suggest at one context class, or nil when
// none runs well enough. files must be sorted largest first.
func pickFor(files []candidateFile, class models.ContextClass, hw models.Hardware) *Pick {
	if len(files) == 0 {
		return nil
	}
	want, ok := classTarget(class, files[0].model.ContextLength)
	if !ok {
		return nil
	}
	base := models.DefaultConfig()
	plans := make([]*models.FitResult, len(files))
	plan := func(i int) models.FitResult {
		if plans[i] == nil {
			r := models.PlanFit(files[i].model, base, hw, class)
			plans[i] = &r
		}
		return *plans[i]
	}
	for _, tier := range pickTiers {
		for i, f := range files {
			if f.bpw < tier.minBPW || f.bpw >= tier.maxBPW {
				continue
			}
			r := plan(i)
			if r.Config.ContextSize < want || r.Placement(f.model) != tier.placement {
				continue
			}
			return newPick(f, r, want, class, hw)
		}
	}
	return nil
}

func newPick(f candidateFile, r models.FitResult, want int, class models.ContextClass, hw models.Hardware) *Pick {
	m := f.model
	cards := models.PlanCards(hw)
	p := &Pick{
		File: f.file.Filename, Quant: f.file.Quant, SizeBytes: f.file.Size, BPW: f.bpw,
		Context: r.Config.ContextSize, KVQuant: r.Config.KVCacheQuant, Placement: r.Placement(m),
		EstimateGiB: r.EstimateGiB, BudgetGiB: r.BudgetGiB,
		CPUMoE: r.Config.CPUMoE, CPURAMGiB: r.CPURAMGiB,
		Cards: len(cards), FewestGPUs: len(cards),
		ExpertBytes: m.ExpertBytes, ExpertUsedCount: m.ExpertUsedCount, ExpertCount: m.ExpertCount,
	}
	if p.Placement == models.PlacementGPU && len(cards) > 1 {
		p.FewestGPUs = fewestGPUs(m, cards, want, class, hw)
	}
	return p
}

// fewestGPUs is the fewest cards, largest first, on which the model runs
// fully on the GPU at want tokens. The plan itself spreads a model over
// every card, as Autoconfigure does; this is for the card to say "fits on
// 1 of 3 GPUs".
func fewestGPUs(m *models.Model, cards []models.GPUSpec, want int, class models.ContextClass, hw models.Hardware) int {
	sorted := slices.Clone(cards)
	slices.SortStableFunc(sorted, func(a, b models.GPUSpec) int { return cmp.Compare(b.VRAMTotalMiB, a.VRAMTotalMiB) })
	for k := 1; k < len(sorted); k++ {
		sub := hw
		sub.GPUs = sorted[:k]
		r := models.PlanFit(m, models.DefaultConfig(), sub, class)
		if r.Placement(m) == models.PlacementGPU && r.Config.ContextSize >= want {
			return k
		}
	}
	return len(cards)
}

// nominalBPW is the usual bits per weight of a quant, for working out a
// model's parameter count from a file when HuggingFace's count is wrong.
// Only quants common as a repo's largest file are listed.
var nominalBPW = map[string]float64{
	"F32": 32, "F16": 16, "BF16": 16,
	"Q8_0": 8.5, "UD_Q8_K_XL": 8.5, "Q8_K_XL": 8.5,
	"Q6_K": 6.56, "Q6_K_L": 6.56, "UD_Q6_K_XL": 6.56,
	"Q5_K_M": 5.69, "Q5_K_S": 5.54, "Q4_K_M": 4.85,
}

// paramsFromFile estimates a model's parameter count from one file and
// its quant, or 0 when the quant's size is not known.
func paramsFromFile(f modelsource.File) int64 {
	bpw, ok := nominalBPW[f.Quant]
	if !ok {
		bpw, ok = nominalBPW[models.ParseQuant(f.Filename)]
	}
	if !ok || f.Size <= 0 {
		return 0
	}
	return int64(float64(f.Size) * 8 / bpw)
}
