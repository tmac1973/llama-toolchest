package models

import (
	"fmt"
	"math"
	"slices"
)

// ContextClass is the context size a user asks autoconfigure for, in
// plain terms rather than tokens.
type ContextClass string

const (
	ContextShort  ContextClass = "short"  // about 8,000 tokens
	ContextMedium ContextClass = "medium" // about 32,000 tokens
	ContextLong   ContextClass = "long"   // about 128,000 tokens
	ContextMax    ContextClass = "max"    // the model's trained limit
)

// ContextClassTokens is the target context of each class except
// ContextMax, which uses the model's trained context length.
var ContextClassTokens = map[ContextClass]int{
	ContextShort:  8192,
	ContextMedium: 32768,
	ContextLong:   131072,
}

// minFitContext is the smallest context the planner reduces to. Below it a
// model is hardly usable for chat, so the planner reports that it does not
// fit rather than offering it.
const minFitContext = 4096

// unknownContextFallback stands in for a trained context length that the
// GGUF did not state.
const unknownContextFallback = 32768

// GPUSpec is one GPU as the planner sees it.
type GPUSpec struct {
	Index        int
	Name         string
	VRAMTotalMiB int
	IsIGPU       bool
	// OtherUsedMiB is memory other programs hold on the card (a desktop,
	// another server), which a plan has to leave to them.
	OtherUsedMiB int
	// ReservedMiB is memory reserved on the card for generation backend
	// instances (image and music servers), whether they are running or
	// not. A plan leaves it to them, as it does other programs' memory.
	ReservedMiB int
}

// UsableMiB is what the card leaves for llama.cpp before the safety
// margin: its total less other programs' use and generation reservations.
func (g GPUSpec) UsableMiB() int {
	return max(0, g.VRAMTotalMiB-g.OtherUsedMiB-g.ReservedMiB)
}

// cardBudgetGiB is what a plan may use of a card: what other programs
// leave of it, less the safety margin, less generation reservations. The
// margin is worked out before the reservations, so reserving memory for a
// backend never shrinks the room left for the driver and the desktop.
func cardBudgetGiB(g GPUSpec) float64 {
	avail := float64(max(0, g.VRAMTotalMiB-g.OtherUsedMiB)) / 1024
	return max(0, avail-fitMarginGiB(avail)-float64(g.ReservedMiB)/1024)
}

// CardBudgetMiB is what the planner lets models use of a card, in MiB:
// cardBudgetGiB in the units the generation backend checks work in.
func CardBudgetMiB(g GPUSpec) int {
	return int(cardBudgetGiB(g) * 1024)
}

// Hardware is the machine a config has to fit. The caller builds it from
// the system monitor; this package does not read hardware itself, so the
// planner can be tested with any machine.
type Hardware struct {
	GPUs         []GPUSpec
	LogicalCores int
	RAMTotalMiB  int // 0 = unknown, and the system-memory check is skipped
	// Build is the llama.cpp build the plan is for. It decides which
	// options a plan may use; the zero value is current llama.cpp.
	Build Target
}

// FitResult is what PlanFit proposes.
type FitResult struct {
	// Config is the base config with the fit settings applied.
	Config ModelConfig
	// Notes explain each setting the planner changed, in plain language,
	// with Origin "hardware fit".
	Notes []ProfileNote
	// EstimateGiB is the GPU memory the proposed config is estimated to
	// use; BudgetGiB is what the planner allowed, after a safety margin.
	EstimateGiB float64
	BudgetGiB   float64
	// CPURAMGiB is what the config keeps in system memory: weights, and
	// the KV cache of layers left on the CPU.
	CPURAMGiB float64
	// Fits is false only when nothing the planner can change makes the
	// model fit; Config is then the most reduced attempt.
	Fits bool
}

// ThreadsFor is the default CPU thread count for a machine with cores
// logical cores: half of them, which is the physical core count on a
// machine with two threads per core. llama.cpp gains little from the
// second thread of a core and loses to contention with it.
func ThreadsFor(cores int) int {
	return max(1, cores/2)
}

// fitMarginGiB is the memory left free on each GPU: at least 1 GiB, or 8%
// of the card, for the driver, the desktop and whatever else is running.
func fitMarginGiB(totalGiB float64) float64 {
	return math.Max(1, totalGiB*0.08)
}

// ramMarginGiB is the system memory left free for the operating system
// and other programs: at least 4 GiB, or 10%.
func ramMarginGiB(totalGiB float64) float64 {
	return math.Max(4, totalGiB*0.10)
}

// PlanFit proposes the settings that make model m fit this machine at the
// context the user asked for: context size, KV cache type, expert offload
// for mixture-of-experts models, GPU layers for dense ones, GPU placement,
// and threads when part of the model runs on the CPU. Everything else in
// base is kept. The order of preference, first that fits wins:
//
//  1. at the requested context, then half of it and so on down to 4096:
//     a. everything on the GPU with a full-precision (f16) KV cache
//     b. an 8-bit (q8_0) KV cache
//     c. MoE only: the fewest expert layers in system memory that fits
//  2. dense only, when nothing above fits: the most GPU layers that fit,
//     at the requested context first
func PlanFit(m *Model, base ModelConfig, hw Hardware, class ContextClass) FitResult {
	cfg := cloneConfig(base)

	// Placement: the same "all (discrete) GPUs" choice the config form
	// offers by default, so an integrated GPU is left out when a real one
	// exists, less the cards with no room left (see planPlacement).
	opt, _ := planPlacement(hw.GPUs)
	if opt != nil {
		cfg.GPUAssign = opt.Value
		cfg.TensorSplit, cfg.SplitMode, cfg.MainGPU = ResolveGPUAssign(opt.Value, len(hw.GPUs))
	}
	budget, ramBudget := FitBudgets(hw)
	nCards := max(1, len(PlanCards(hw)))
	if opt != nil {
		budget, nCards = 0, max(1, len(opt.GPUs))
		for _, i := range opt.GPUs {
			budget += cardBudgetGiB(hw.GPUs[i])
		}
	}
	gpuMiB := GPUMiB(hw)

	requested := ContextClassTokens[class]
	trained := m.ContextLength
	if trained <= 0 {
		trained = unknownContextFallback
	}
	if class == ContextMax || requested == 0 {
		requested = trained
	}
	ctx := min(requested, trained)

	// Settings the planner owns start from known values: a plan does not
	// depend on what a previous hand edit left behind.
	cfg.BatchSize, cfg.UBatchSize = 0, 0
	cfg.FlashAttention = true
	// The MoE cache spends whatever GPU memory is left over, which is
	// exactly what the plan is deciding. Autotune measures it afterwards.
	cfg.MoECacheMiB = 0
	// The drafter's micro-batch is memory too: the plan sets one only
	// where it is what makes a setting fit.
	cfg.DraftUBatchSize = 0
	smallDraftUBatch := IsDraftMode(cfg.SpecType) && hw.Build.DraftUBatchOption()
	if cfg.Parallel > 1 {
		cfg.Parallel = 0
		cfg.SharedContext, cfg.ContextPerSlot = false, 0
	}

	fits := func(c *ModelConfig) (VRAMBreakdown, bool) {
		return fitsOn(m, c, nCards, budget, ramBudget, gpuMiB)
	}

	var last ModelConfig
	var lastB VRAMBreakdown
	try := func(ctx int, kv string, gpuLayers, cpuMoE int) bool {
		c := cfg
		c.ContextSize, c.KVCacheQuant, c.GPULayers, c.CPUMoE = ctx, kv, gpuLayers, cpuMoE
		b, ok := fits(&c)
		last, lastB = c, b
		if ok || !smallDraftUBatch {
			return ok
		}
		// Before giving up anything the model itself uses — context, cache
		// precision, layers on the GPU — try the drafter with a small
		// micro-batch of its own. It drafts a few tokens a step, so only
		// reading a prompt through it gets slower, and slightly.
		c.DraftUBatchSize = planDraftUBatch
		b, ok = fits(&c)
		last, lastB = c, b
		return ok
	}
	done := func(ok bool) FitResult {
		return finishFit(m, base, last, lastB, budget, requested, hw, ok)
	}
	var contexts []int
	for c := ctx; c >= minFitContext; c /= 2 {
		contexts = append(contexts, c)
	}
	if len(contexts) == 0 {
		contexts = []int{ctx}
	}

	// Every layer on the GPU, at the largest context that allows it.
	// Expert offload belongs here: each token reads only a few experts,
	// so keeping some in system memory costs far less speed than halving
	// the context costs usefulness.
	for _, c := range contexts {
		if try(c, "", 999, 0) || try(c, "q8_0", 999, 0) {
			return done(true)
		}
		if m.ExpertCount > 0 && m.ExpertLayers > 0 {
			// The smallest number that fits: each layer moved costs speed.
			for n := m.ExpertLayerFirst + 1; n <= m.ExpertLayerFirst+m.ExpertLayers; n++ {
				if try(c, "q8_0", 999, n) {
					return done(true)
				}
			}
		}
	}

	// A dense model that does not fit at any context: run part of it on
	// the CPU, which is slow, so only as a last resort — at the requested
	// context, since reducing it no longer avoids the slowdown. The
	// largest number of layers that fits. Zero is left out: the estimate
	// reads it as "not set" (see CPUWeightBytes).
	if m.ExpertCount == 0 && m.NLayers > 1 {
		for _, c := range contexts {
			for n := m.NLayers - 1; n >= 1; n-- {
				if try(c, "q8_0", n, 0) {
					return done(true)
				}
			}
		}
	}
	return done(false)
}

// spanAllOption returns the placement option that spans every discrete
// GPU (or the only GPU on an APU-only machine), or nil without GPUs.
func spanAllOption(gpus []GPUSpec) *GPUOption {
	igpu := make([]bool, len(gpus))
	for i, g := range gpus {
		igpu[i] = g.IsIGPU
	}
	for _, o := range GPUAssignOptions(len(gpus), igpu) {
		if o.IsSpanAll {
			return &o
		}
	}
	return nil
}

// minCardRoomGiB is the least room a card must have left, after other
// programs, generation reservations and the safety margin, for a plan to
// put part of a model on it. Each card a model spans also holds a compute
// buffer of its own, so a card with less than this could take at most a
// few layers, and in practice refuses the load.
const minCardRoomGiB = 1.0

// planPlacement returns the placement a plan uses: every discrete GPU,
// as spanAllOption gives, less the cards with less than minCardRoomGiB
// left, which it also returns. A card that generation backends have
// reserved nearly all of would otherwise be given a share of the model
// and refuse every load. When no card, or every card, has the room, the
// span over all of them is kept: leaving cards out then changes nothing.
func planPlacement(gpus []GPUSpec) (opt *GPUOption, leftOut []int) {
	span := spanAllOption(gpus)
	if span == nil {
		return nil, nil
	}
	var keep []int
	for _, i := range span.GPUs {
		if cardBudgetGiB(gpus[i]) >= minCardRoomGiB {
			keep = append(keep, i)
		} else {
			leftOut = append(leftOut, i)
		}
	}
	if len(keep) == 0 || len(leftOut) == 0 {
		return span, nil
	}
	igpu := make([]bool, len(gpus))
	for i, g := range gpus {
		igpu[i] = g.IsIGPU
	}
	// The config form's own option when it has one for these cards, so
	// the form shows the choice by name; otherwise the explicit set,
	// which ResolveGPUAssign reads the same way.
	for _, o := range GPUAssignOptions(len(gpus), igpu) {
		if !o.IsTensor && !o.IsSpanAll && slices.Equal(o.GPUs, keep) {
			return &o, leftOut
		}
	}
	return &GPUOption{Value: joinInts(keep), GPUs: keep}, leftOut
}

// planDraftUBatch is the drafter micro-batch PlanFit sets when it needs
// the memory. A drafter is fed the tokens it drafts, a handful per step,
// but also every prompt, which it reads in chunks of this size: 128 keeps
// a quarter of the default 512's compute buffer without making a long
// prompt a thousand small steps.
const planDraftUBatch = 128

// fitsOn reports whether c fits the budgets PlanFit plans against.
func fitsOn(m *Model, c *ModelConfig, nCards int, budget, ramBudget float64, gpuMiB []int) (VRAMBreakdown, bool) {
	b := VRAMBreakdownForConfigOn(m, c, nCards)
	if budget <= 0 || b.Total() > budget {
		return b, false
	}
	if ramBudget > 0 && b.CPURAM > ramBudget {
		return b, false
	}
	// With experts in system memory the layers differ in size, so the
	// pooled total can fit while one card cannot. Each card is checked
	// under the split the preset will write.
	if loads, budgets, ok := PlanCardLoads(m, c, gpuMiB); ok {
		for i := range loads {
			if loads[i] > budgets[i] {
				return b, false
			}
		}
	}
	return b, true
}

// moeCacheStepMiB is the granularity MaxMoECacheMiB works in.
const moeCacheStepMiB = 256

// MaxMoECacheMiB is the largest MoE cache, in MiB, that fits beside cfg on
// this machine by the same rules PlanFit plans with, or 0 when none does:
// cfg keeps no experts in system memory, there is no room, or the room
// left would not hold the experts of a single token, which llama.cpp
// refuses to load. It is never more than the experts there are to cache.
//
// The budget is the whole machine's, as PlanFit's is; a profile placed
// on fewer cards than that is not narrowed to them.
func MaxMoECacheMiB(m *Model, cfg ModelConfig, hw Hardware) int {
	experts := cpuExpertBytes(m, &cfg)
	if experts == 0 {
		return 0
	}
	budget, ramBudget := FitBudgets(hw)
	nCards := max(1, len(PlanCards(hw)))
	gpuMiB := GPUMiB(hw)
	ok := func(steps int) bool {
		c := cfg
		c.MoECacheMiB = steps * moeCacheStepMiB
		_, fits := fitsOn(m, &c, nCards, budget, ramBudget, gpuMiB)
		return fits
	}
	hi := int((experts>>20)/moeCacheStepMiB) + 1 // enough to cache every expert
	if !ok(0) {
		return 0 // the profile itself does not fit by the estimate
	}
	lo := 0
	for lo < hi { // largest steps in [0, hi] that fits
		mid := (lo + hi + 1) / 2
		if ok(mid) {
			lo = mid
		} else {
			hi = mid - 1
		}
	}
	mib := lo * moeCacheStepMiB
	if int64(mib)<<20 < moeCacheMinBytes(m) {
		return 0
	}
	return mib
}

// moeCacheMinBytes is about the smallest cache llama.cpp accepts: room
// for the experts one token uses in a layer, with some to spare, since
// below that it caches nothing and refuses to load.
func moeCacheMinBytes(m *Model) int64 {
	if m.ExpertLayers <= 0 || m.ExpertCount <= 0 {
		return 0
	}
	perExpert := m.ExpertBytes / int64(m.ExpertLayers) / int64(m.ExpertCount)
	return perExpert * int64(max(1, m.ExpertUsedCount)) * 4
}

// finishFit sets threads for CPU offload and writes one note per setting
// the plan changed from base.
func finishFit(m *Model, base, cfg ModelConfig, b VRAMBreakdown, budget float64, requested int, hw Hardware, fits bool) FitResult {
	onCPU := cfg.CPUMoE > 0 || (cfg.GPULayers > 0 && cfg.GPULayers < m.NLayers)
	if onCPU && hw.LogicalCores > 0 {
		cfg.Threads = ThreadsFor(hw.LogicalCores)
	}

	var notes []ProfileNote
	note := func(field, reason string) {
		notes = append(notes, ProfileNote{Field: field, Reason: reason, Origin: "hardware fit"})
	}

	if cfg.ContextSize != base.ContextSize {
		if cfg.ContextSize < requested {
			note("context_size", fmt.Sprintf("The requested %s tokens did not fit on this machine; reduced to %s tokens, the largest that fits.",
				GroupDigits(requested), GroupDigits(cfg.ContextSize)))
		} else {
			note("context_size", fmt.Sprintf("%s tokens, as requested. Longer conversations and documents need more context, and more GPU memory.",
				GroupDigits(cfg.ContextSize)))
		}
	}
	if cfg.KVCacheQuant != base.KVCacheQuant {
		if cfg.KVCacheQuant == "q8_0" {
			note("kv_cache_quant", "Stores the conversation memory (KV cache) at 8 bits so the requested context fits. The effect on answer quality is very small.")
		} else {
			note("kv_cache_quant", "Stores the conversation memory (KV cache) at full precision, because it fits.")
		}
	}
	if cfg.CPUMoE != base.CPUMoE {
		if cfg.CPUMoE > 0 {
			note("cpu_moe", fmt.Sprintf("Keeps the expert weights of the first %d layers in system memory so the model fits. Each token uses only a few experts, so this costs less speed than moving whole layers.", cfg.CPUMoE))
		} else {
			note("cpu_moe", "All expert weights fit on the GPU, so none are kept in system memory.")
		}
	}
	if cfg.DraftUBatchSize != base.DraftUBatchSize {
		if cfg.DraftUBatchSize > 0 {
			note("draft_ubatch_size", fmt.Sprintf("Gives the speculative drafter a micro-batch of %d tokens of its own, instead of the model's, so these settings fit. The drafter only drafts a few tokens per step, so generation is as fast as before; reading a long prompt takes slightly longer.", cfg.DraftUBatchSize))
		} else {
			note("draft_ubatch_size", "The speculative drafter uses the model's micro-batch again: these settings fit without a smaller one.")
		}
	}
	if cfg.MoECacheMiB != base.MoECacheMiB {
		note("moe_cache_mib", "The MoE cache is left off: it uses GPU memory the plan gives to the model first. Autotune measures whether a cache helps with the memory left over.")
	}
	if cfg.GPULayers != base.GPULayers {
		if cfg.GPULayers > 0 && cfg.GPULayers < m.NLayers {
			note("gpu_layers", fmt.Sprintf("Only %d of the model's %d layers fit on the GPU; the rest run on the CPU, which is slower.", cfg.GPULayers, m.NLayers))
		} else {
			note("gpu_layers", "Every layer runs on the GPU.")
		}
	}
	if opt, leftOut := planPlacement(hw.GPUs); len(leftOut) > 0 && opt != nil {
		where := "Puts the model on GPU " + joinInts(opt.GPUs)
		if len(opt.GPUs) > 1 {
			where = "Spreads the model over GPUs " + joinInts(opt.GPUs)
		}
		cards := "GPU " + joinInts(leftOut) + " is"
		if len(leftOut) > 1 {
			cards = "GPUs " + joinInts(leftOut) + " are"
		}
		note("gpu_assign", fmt.Sprintf("%s. %s left out: after the memory reserved for image and music backends and what other programs use, less than %.0f GiB is free there.",
			where, cards, minCardRoomGiB))
	} else if cfg.GPUAssign != base.GPUAssign {
		note("gpu_assign", "Spreads the model over every dedicated GPU in this machine. An integrated GPU is left out when a dedicated one is present, because it shares system memory and is slower.")
	}
	if cfg.FlashAttention != base.FlashAttention {
		note("flash_attention", "Flash attention uses less memory and is faster on most GPUs.")
	}
	if cfg.BatchSize != base.BatchSize || cfg.UBatchSize != base.UBatchSize {
		note("ubatch_size", "Batch sizes are left at llama.cpp's defaults. Autotune can measure better values for this machine.")
	}
	if cfg.Parallel != base.Parallel {
		note("parallel", "llama.cpp's default: conversations share the whole context, so one conversation can use all of it.")
	}
	if cfg.Threads != base.Threads {
		note("threads", fmt.Sprintf("Part of the model runs on the CPU, so it uses %d threads: one per physical core.", cfg.Threads))
	}
	if other := otherUsedGiB(hw); other >= 0.5 {
		note("", fmt.Sprintf("Other programs are using %.1f GiB of GPU memory right now (the desktop, another AI tool). These settings leave it to them. With that memory free, more context or fewer layers in system memory may fit.", other))
	}
	if reserved := reservedGiB(hw); reserved > 0 {
		note("", fmt.Sprintf("%.1f GiB of GPU memory is reserved for image and music backends. These settings leave it to them.", reserved))
	}
	if !fits {
		note("", "This model is too large for this machine even with the settings above. Choose a smaller quantization of it.")
	}

	return FitResult{
		Config:      cfg,
		Notes:       notes,
		EstimateGiB: b.Total(),
		BudgetGiB:   budget,
		CPURAMGiB:   b.CPURAM,
		Fits:        fits,
	}
}

// GroupDigits writes n with thousands separators: 131072 → "131,072".
// Used wherever a token count or context size is shown.
func GroupDigits(n int) string {
	s := fmt.Sprint(n)
	for i := len(s) - 3; i > 0; i -= 3 {
		s = s[:i] + "," + s[i:]
	}
	return s
}

// Placement is how a fit plan runs a model.
type Placement string

const (
	PlacementGPU     Placement = "gpu"     // every layer on the GPUs
	PlacementExperts Placement = "experts" // some MoE expert weights in system memory
	PlacementPartial Placement = "partial" // a dense model with layers on the CPU
	PlacementNone    Placement = "none"    // too large even with offloading
)

// Placement says how the plan runs m.
func (r FitResult) Placement(m *Model) Placement {
	c := r.Config
	switch {
	case !r.Fits:
		return PlacementNone
	case c.GPULayers > 0 && c.GPULayers < m.NLayers:
		return PlacementPartial
	case c.CPUMoE > 0:
		return PlacementExperts
	}
	return PlacementGPU
}

// PlanCards returns the GPUs of the machine a plan is made for: the
// dedicated ones, or the integrated GPU when it is the only one. PlanFit
// spreads a model over these, less any card with no room left on it
// (planPlacement).
func PlanCards(hw Hardware) []GPUSpec {
	opt := spanAllOption(hw.GPUs)
	if opt == nil {
		return nil
	}
	cards := make([]GPUSpec, 0, len(opt.GPUs))
	for _, i := range opt.GPUs {
		cards = append(cards, hw.GPUs[i])
	}
	return cards
}

// FitBudgets returns the GPU memory and system memory PlanFit allows a
// model on this machine, after its safety margins. ramGiB is 0 when the
// system memory is not known.
func FitBudgets(hw Hardware) (vramGiB, ramGiB float64) {
	for _, g := range PlanCards(hw) {
		vramGiB += cardBudgetGiB(g)
	}
	if hw.RAMTotalMiB > 0 {
		total := float64(hw.RAMTotalMiB) / 1024
		ramGiB = total - ramMarginGiB(total)
	}
	return vramGiB, ramGiB
}

// otherUsedGiB is what other programs hold on the cards a plan uses.
func otherUsedGiB(hw Hardware) float64 {
	var n int
	for _, g := range PlanCards(hw) {
		n += g.OtherUsedMiB
	}
	return float64(n) / 1024
}

// reservedGiB is what generation backends reserve on the cards a plan
// uses.
func reservedGiB(hw Hardware) float64 {
	var n int
	for _, g := range PlanCards(hw) {
		n += g.ReservedMiB
	}
	return float64(n) / 1024
}

// GPUMiB is what each GPU lets a model use, by position: CardBudgetMiB,
// its total less what other programs hold, the safety margin and
// generation reservations. MoESplitConfig, PlanCardLoads and CardLoadsMiB
// take it, so the split a launch writes, the planner's check of each card
// and the reservation check on loads all work from the same figure.
func GPUMiB(hw Hardware) []int {
	out := make([]int, len(hw.GPUs))
	for i, g := range hw.GPUs {
		out[i] = CardBudgetMiB(g)
	}
	return out
}
