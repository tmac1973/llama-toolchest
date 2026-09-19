package models

import (
	"fmt"
	"math"
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
}

// Hardware is the machine a config has to fit. The caller builds it from
// the system monitor; this package does not read hardware itself, so the
// planner can be tested with any machine.
type Hardware struct {
	GPUs         []GPUSpec
	LogicalCores int
	RAMTotalMiB  int // 0 = unknown, and the system-memory check is skipped
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
	// exists.
	var cards []GPUSpec
	if opt := spanAllOption(hw.GPUs); opt != nil {
		cfg.GPUAssign = opt.Value
		cfg.TensorSplit, cfg.SplitMode, cfg.MainGPU = ResolveGPUAssign(opt.Value, len(hw.GPUs))
		for _, i := range opt.GPUs {
			cards = append(cards, hw.GPUs[i])
		}
	}
	var budget float64
	for _, g := range cards {
		total := float64(g.VRAMTotalMiB) / 1024
		budget += total - fitMarginGiB(total)
	}
	nCards := max(1, len(cards))
	ramBudget := 0.0
	if hw.RAMTotalMiB > 0 {
		total := float64(hw.RAMTotalMiB) / 1024
		ramBudget = total - ramMarginGiB(total)
	}

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
	if cfg.Parallel > 1 {
		cfg.Parallel = 0
	}

	fits := func(c *ModelConfig) (VRAMBreakdown, bool) {
		b := VRAMBreakdownForConfigOn(m, c, nCards)
		if budget <= 0 || b.Total() > budget {
			return b, false
		}
		if ramBudget > 0 && b.CPURAM > ramBudget {
			return b, false
		}
		return b, true
	}

	var last ModelConfig
	var lastB VRAMBreakdown
	try := func(ctx int, kv string, gpuLayers, cpuMoE int) bool {
		c := cfg
		c.ContextSize, c.KVCacheQuant, c.GPULayers, c.CPUMoE = ctx, kv, gpuLayers, cpuMoE
		b, ok := fits(&c)
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
				groupDigits(requested), groupDigits(cfg.ContextSize)))
		} else {
			note("context_size", fmt.Sprintf("%s tokens, as requested. Longer conversations and documents need more context, and more GPU memory.",
				groupDigits(cfg.ContextSize)))
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
	if cfg.GPULayers != base.GPULayers {
		if cfg.GPULayers > 0 && cfg.GPULayers < m.NLayers {
			note("gpu_layers", fmt.Sprintf("Only %d of the model's %d layers fit on the GPU; the rest run on the CPU, which is slower.", cfg.GPULayers, m.NLayers))
		} else {
			note("gpu_layers", "Every layer runs on the GPU.")
		}
	}
	if cfg.GPUAssign != base.GPUAssign {
		note("gpu_assign", "Spreads the model over every dedicated GPU in this machine. An integrated GPU is left out when a dedicated one is present, because it shares system memory and is slower.")
	}
	if cfg.FlashAttention != base.FlashAttention {
		note("flash_attention", "Flash attention uses less memory and is faster on most GPUs.")
	}
	if cfg.BatchSize != base.BatchSize || cfg.UBatchSize != base.UBatchSize {
		note("ubatch_size", "Batch sizes are left at llama.cpp's defaults. Autotune can measure better values for this machine.")
	}
	if cfg.Parallel != base.Parallel {
		note("parallel", "One conversation at a time, so that conversation gets the whole context.")
	}
	if cfg.Threads != base.Threads {
		note("threads", fmt.Sprintf("Part of the model runs on the CPU, so it uses %d threads: one per physical core.", cfg.Threads))
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

// groupDigits writes n with thousands separators: 131072 → "131,072".
func groupDigits(n int) string {
	s := fmt.Sprint(n)
	for i := len(s) - 3; i > 0; i -= 3 {
		s = s[:i] + "," + s[i:]
	}
	return s
}
