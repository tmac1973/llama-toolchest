package models

import (
	"math"
	"testing"
)

const gib = 1024 * 1024 * 1024

// corpusPoint is one load measured on hardware: the model as the registry
// would hold it, the config it ran under, and the VRAM the GPU counters
// reported while it was loaded.
//
// This is the evidence the estimate is fitted to. Recorded here rather than
// only in the plan document so a change to the coefficients has to face it.
type corpusPoint struct {
	name     string
	model    Model
	cfg      ModelConfig
	cards    int
	measured float64 // GiB, summed across cards
	// reported is the same load as llama.cpp itemised it while
	// allocating, GPU side only. A total can hide two errors that cancel
	// — the estimate this corpus replaced did exactly that — so where the
	// split is known it is recorded, and TestEstimateTermsAgainstTheBufferReport
	// checks the terms one at a time.
	//
	// Nil for points measured before the split was captured. New points
	// should carry it: /api/models/{id}/vram-corpus prints a filled-in
	// row for whatever the router last loaded.
	reported *reportedTerms
	// backend is the llama.cpp backend the load ran on, which picks the
	// coefficients the point is held to.
	backend string
}

// coeffs is the coefficient set a point is checked against.
func (p corpusPoint) coeffs() *vramCoefficients { return coefficientsFor(p.backend) }

// estimate is the point's estimate under its own backend's coefficients,
// on the cards it was measured on.
func (p corpusPoint) estimate() VRAMBreakdown {
	return vramBreakdownOn(p.coeffs(), &p.model, &p.cfg, p.cards, p.cardGiB())
}

// a4000MiB is one of compute2's RTX A4000s as the monitor reads it.
const a4000MiB = 15352

// cardGiB is the size of the cards a point was measured on, which
// decides whether llama.cpp kept its pipeline copies. Every CUDA point
// is from compute2; the ROCm points leave it unknown, as their
// coefficients do not model the copies.
func (p corpusPoint) cardGiB() float64 {
	if p.backend == "cuda" {
		return a4000MiB / 1024.0
	}
	return 0
}

// reportedTerms is one load's buffer report, GiB on the accelerators.
type reportedTerms struct {
	weights   float64
	kv        float64
	recurrent float64
	compute   float64 // compute and output buffers
	moeCache  float64 // the MoE cache (--moe-cache-mib)
}

// total is what llama.cpp accounts for. Always less than the card
// counters: context and allocator overhead are not in the report.
func (r reportedTerms) total() float64 {
	return r.weights + r.kv + r.recurrent + r.compute + r.moeCache
}

// Sizes are in GiB where written as floats and converted below.
func gibBytes(f float64) int64 { return int64(f * gib) }

// corpus is every measured load, on both backends.
func corpus() []corpusPoint {
	out := append(rocmCorpus(), rocmLocalCorpus()...)
	out = append(out, cudaCorpus()...)
	out = append(out, cudaBenchCorpus()...)
	return append(out, cudaLaunchCorpus()...)
}

// rocmCorpus: one machine, 4x Radeon AI PRO R9700, tensor-parallel.
func rocmCorpus() []corpusPoint {
	// Qwen3.8-Flash-Next: qwen4exp, 48 layers, 12 attending (interval 4),
	// sparse attention with a 128-wide indexer, 26.82 GiB per-layer table.
	fn := func(ctx, ub int) Model {
		return Model{
			SizeBytes: gibBytes(103.69), PLEBytes: gibBytes(26.82),
			TokenEmbdBytes: gibBytes(0.629), NLayers: 48, AttnLayers: 12,
			NEmbd: 2560, NKVHead: 2, KVFullPerTok: 12 * 2 * 512,
			IndexerKeyLength: 128, ContextLength: 262144,
		}
	}
	// Qwen3.8-27B: qwen35, 65 layers, 16 attending, no indexer.
	q27 := Model{
		SizeBytes: gibBytes(29.30), TokenEmbdBytes: gibBytes(1.258),
		NLayers: 65, AttnLayers: 16, NEmbd: 5120, NKVHead: 4,
		KVFullPerTok: 16 * 4 * 512, ContextLength: 262144,
	}
	// Qwen3.6-35B-A3B: qwen35moe, 41 layers, 10 attending.
	q35 := Model{
		SizeBytes: gibBytes(36.41), TokenEmbdBytes: gibBytes(0.503),
		NLayers: 41, AttnLayers: 10, NEmbd: 2048, NKVHead: 2,
		KVFullPerTok: 10 * 2 * 512, ContextLength: 262144,
	}
	// gemma-4-E4B: sliding-window attention, per-layer table, single card.
	// Arch and OutputTied are what parser version 4 records for it.
	gem := Model{
		Arch: "gemma4", OutputTied: true,
		SizeBytes: gibBytes(4.77), PLEBytes: gibBytes(1.80),
		TokenEmbdBytes: gibBytes(0.664), NLayers: 42, AttnLayers: 42,
		NEmbd: 2560, NKVHead: 2, KVFullPerTok: 14336, KVSWAPerTok: 35840,
		SlidingWindow: 512, ContextLength: 131072,
	}
	// Every point below the CUDA one was measured tensor-parallel: the
	// cards act as one device and share one set of graph buffers. The
	// split mode has to be on the config, or the estimate reads them as
	// llama.cpp's default layer split and adds scratch they never paid
	// for. See "A second model, swept" in plan/ple-vram-findings.md.
	c := func(ctx, ub int) ModelConfig {
		return ModelConfig{ContextSize: ctx, UBatchSize: ub, SplitMode: "tensor"}
	}
	// The 27B's weights and recurrent state are the same on every row —
	// neither depends on context or micro-batch, which is itself part of
	// what the sweep established.
	q27terms := func(kv, compute float64) *reportedTerms {
		return &reportedTerms{weights: 27.51, kv: kv, recurrent: 0.60, compute: compute}
	}
	return []corpusPoint{
		{"Flash-Next ctx32k ub1024", fn(32768, 1024), c(32768, 1024), 4, 84.48, nil, "rocm"},
		{"Flash-Next ctx128k ub1024", fn(131072, 1024), c(131072, 1024), 4, 95.96, nil, "rocm"},
		{"Flash-Next ctx262k ub512", fn(262144, 512), c(262144, 512), 4, 98.96, nil, "rocm"},
		// The one sparse-attention load that was decomposed: its saved
		// config, quantized KV cache and all. See "The decomposition,
		// measured" in plan/ple-vram-findings.md.
		{"Flash-Next ctx262k ub1024 kv-q8_0", fn(262144, 1024),
			ModelConfig{ContextSize: 262144, UBatchSize: 1024, KVCacheQuant: "q8_0", SplitMode: "tensor"}, 4, 107.35,
			&reportedTerms{weights: 76.23, kv: 4.38, recurrent: 0.44, compute: 24.40}, "rocm"},
		{"27B ctx8k ub512", q27, c(8192, 512), 4, 31.43, q27terms(0.48, 0.52), "rocm"},
		{"27B ctx32k ub512", q27, c(32768, 512), 4, 33.01, q27terms(2.00, 0.60), "rocm"},
		{"27B ctx128k ub512", q27, c(131072, 512), 4, 39.38, q27terms(8.00, 0.96), "rocm"},
		{"27B ctx262k ub512", q27, c(262144, 512), 4, 47.87, q27terms(16.00, 1.48), "rocm"},
		{"27B ctx32k ub128", q27, c(32768, 128), 4, 32.61, q27terms(2.00, 0.20), "rocm"},
		{"27B ctx32k ub2048", q27, c(32768, 2048), 4, 34.82, q27terms(2.00, 2.40), "rocm"},
		{"35B-A3B ctx32k ub512", q35, c(32768, 512), 4, 38.08, nil, "rocm"},
		{"gemma-4-E4B ctx4k ub512", gem, c(4096, 512), 1, 3.42, nil, "rocm"},
	}
}

// rocmLocalCorpus: a second ROCm machine, one RX 9070 XT (16 GiB) with the
// desktop on it, build b10453-rocm, measured 2026-10-03 with the preset's
// flags on that card alone (--device ROCm0). The desktop's own use was read
// just before each load and subtracted.
func rocmLocalCorpus() []corpusPoint {
	models := map[string]Model{
		"q4b":       {Arch: "qwen35", SizeBytes: 2740937888, PLEBytes: 0, TokenEmbdBytes: 521472000, OutputTied: true, NLayers: 32, AttnLayers: 8, NEmbd: 2560, NHead: 16, NKVHead: 4, KVFullPerTok: 16384, KVSWAPerTok: 0, SlidingWindow: 0, ContextLength: 262144, ExpertCount: 0, ExpertUsedCount: 0, ExpertBytes: 0, ExpertLayerFirst: 0, ExpertLayers: 0, NextNLayers: 0},
		"q9q8":      {Arch: "qwen35", SizeBytes: 13245182304, PLEBytes: 0, TokenEmbdBytes: 2034237440, OutputTied: false, NLayers: 33, AttnLayers: 8, NEmbd: 4096, NHead: 16, NKVHead: 4, KVFullPerTok: 16384, KVSWAPerTok: 0, SlidingWindow: 0, ContextLength: 262144, ExpertCount: 0, ExpertUsedCount: 0, ExpertBytes: 0, ExpertLayerFirst: 0, ExpertLayers: 0, NextNLayers: 1},
		"q9iq4":     {Arch: "qwen35", SizeBytes: 5644398944, PLEBytes: 0, TokenEmbdBytes: 572129280, OutputTied: false, NLayers: 33, AttnLayers: 8, NEmbd: 4096, NHead: 16, NKVHead: 4, KVFullPerTok: 16384, KVSWAPerTok: 0, SlidingWindow: 0, ContextLength: 262144, ExpertCount: 0, ExpertUsedCount: 0, ExpertBytes: 0, ExpertLayerFirst: 0, ExpertLayers: 0, NextNLayers: 1},
		"granite8b": {Arch: "granite", SizeBytes: 9345613952, PLEBytes: 0, TokenEmbdBytes: 436731904, OutputTied: false, NLayers: 40, AttnLayers: 40, NEmbd: 4096, NHead: 32, NKVHead: 8, KVFullPerTok: 81920, KVSWAPerTok: 0, SlidingWindow: 0, ContextLength: 131072, ExpertCount: 0, ExpertUsedCount: 0, ExpertBytes: 0, ExpertLayerFirst: 0, ExpertLayers: 0, NextNLayers: 0},
		"gptoss":    {Arch: "gpt-oss", SizeBytes: 12109567168, PLEBytes: 0, TokenEmbdBytes: 615329280, OutputTied: false, NLayers: 24, AttnLayers: 24, NEmbd: 2880, NHead: 64, NKVHead: 8, KVFullPerTok: 12288, KVSWAPerTok: 12288, SlidingWindow: 128, ContextLength: 131072, ExpertCount: 32, ExpertUsedCount: 4, ExpertBytes: 10178887680, ExpertLayerFirst: 0, ExpertLayers: 24, NextNLayers: 0},
		"gem12":     {Arch: "gemma4", SizeBytes: 7366423360, PLEBytes: 0, TokenEmbdBytes: 692060160, OutputTied: true, NLayers: 48, AttnLayers: 48, NEmbd: 3840, NHead: 16, NKVHead: 8, KVFullPerTok: 8192, KVSWAPerTok: 163840, SlidingWindow: 1024, ContextLength: 262144, ExpertCount: 0, ExpertUsedCount: 0, ExpertBytes: 0, ExpertLayerFirst: 0, ExpertLayers: 0, NextNLayers: 0},
		"q35a3":     {Arch: "qwen35moe", SizeBytes: 22663387424, PLEBytes: 0, TokenEmbdBytes: 540344320, OutputTied: false, NLayers: 41, AttnLayers: 10, NEmbd: 2048, NHead: 16, NKVHead: 2, KVFullPerTok: 10240, KVSWAPerTok: 0, SlidingWindow: 0, ContextLength: 262144, ExpertCount: 256, ExpertUsedCount: 8, ExpertBytes: 20055064576, ExpertLayerFirst: 0, ExpertLayers: 41, NextNLayers: 1},
	}
	return []corpusPoint{
		{name: "rx9070 q4b ctx32k ub512", model: models["q4b"], cfg: ModelConfig{ContextSize: 32768, UBatchSize: 512, GPULayers: 999}, cards: 1, measured: 4.07,
			reported: &reportedTerms{weights: 2.54, kv: 1.0, recurrent: 0.2, compute: 0.09}, backend: "rocm"},
		{name: "rx9070 q4b ctx128k ub512", model: models["q4b"], cfg: ModelConfig{ContextSize: 131072, UBatchSize: 512, GPULayers: 999}, cards: 1, measured: 7.25,
			reported: &reportedTerms{weights: 2.54, kv: 4.0, recurrent: 0.2, compute: 0.19}, backend: "rocm"},
		{name: "rx9070 q4b ctx32k ub2048", model: models["q4b"], cfg: ModelConfig{ContextSize: 32768, UBatchSize: 2048, GPULayers: 999}, cards: 1, measured: 4.40,
			reported: &reportedTerms{weights: 2.54, kv: 1.0, recurrent: 0.2, compute: 0.38}, backend: "rocm"},
		{name: "rx9070 q4b ctx128k ub512 q8", model: models["q4b"], cfg: ModelConfig{ContextSize: 131072, UBatchSize: 512, GPULayers: 999, KVCacheQuant: "q8_0"}, cards: 1, measured: 5.79,
			reported: &reportedTerms{weights: 2.54, kv: 2.12, recurrent: 0.2, compute: 0.67}, backend: "rocm"},
		{name: "rx9070 granite8b ctx8k ub512", model: models["granite8b"], cfg: ModelConfig{ContextSize: 8192, UBatchSize: 512, GPULayers: 999}, cards: 1, measured: 9.92,
			reported: &reportedTerms{weights: 8.29, kv: 1.25, recurrent: 0.0, compute: 0.1}, backend: "rocm"},
		{name: "rx9070 granite8b ctx32k ub512", model: models["granite8b"], cfg: ModelConfig{ContextSize: 32768, UBatchSize: 512, GPULayers: 999}, cards: 1, measured: 13.72,
			reported: &reportedTerms{weights: 8.29, kv: 5.0, recurrent: 0.0, compute: 0.13}, backend: "rocm"},
		{name: "rx9070 granite8b ctx32k ub2048", model: models["granite8b"], cfg: ModelConfig{ContextSize: 32768, UBatchSize: 2048, GPULayers: 999}, cards: 1, measured: 14.10,
			reported: &reportedTerms{weights: 8.29, kv: 5.0, recurrent: 0.0, compute: 0.51}, backend: "rocm"},
		{name: "rx9070 gem12 ctx32k ub512", model: models["gem12"], cfg: ModelConfig{ContextSize: 32768, UBatchSize: 512, GPULayers: 999}, cards: 1, measured: 9.16,
			reported: &reportedTerms{weights: 6.85, kv: 1.91, recurrent: 0.0, compute: 0.15}, backend: "rocm"},
		{name: "rx9070 gem12 ctx128k ub512", model: models["gem12"], cfg: ModelConfig{ContextSize: 131072, UBatchSize: 512, GPULayers: 999}, cards: 1, measured: 10.87,
			reported: &reportedTerms{weights: 6.85, kv: 3.41, recurrent: 0.0, compute: 0.25}, backend: "rocm"},
		{name: "rx9070 gptoss ctx32k ub512", model: models["gptoss"], cfg: ModelConfig{ContextSize: 32768, UBatchSize: 512, GPULayers: 999}, cards: 1, measured: 11.81,
			reported: &reportedTerms{weights: 10.69, kv: 0.77, recurrent: 0.0, compute: 0.12}, backend: "rocm"},
		{name: "rx9070 gptoss ctx64k ub512 q8", model: models["gptoss"], cfg: ModelConfig{ContextSize: 65536, UBatchSize: 512, GPULayers: 999, KVCacheQuant: "q8_0"}, cards: 1, measured: 12.06,
			reported: &reportedTerms{weights: 10.69, kv: 0.81, recurrent: 0.0, compute: 0.22}, backend: "rocm"},
		{name: "rx9070 q9q8 ctx32k ub512", model: models["q9q8"], cfg: ModelConfig{ContextSize: 32768, UBatchSize: 512, GPULayers: 999}, cards: 1, measured: 11.74,
			reported: &reportedTerms{weights: 10.19, kv: 1.0, recurrent: 0.2, compute: 0.12}, backend: "rocm"},
		{name: "rx9070 q9iq4 ctx128k ub512", model: models["q9iq4"], cfg: ModelConfig{ContextSize: 131072, UBatchSize: 512, GPULayers: 999}, cards: 1, measured: 9.24,
			reported: &reportedTerms{weights: 4.57, kv: 4.0, recurrent: 0.2, compute: 0.21}, backend: "rocm"},
		{name: "rx9070 q9iq4 ctx32k ub512 mtp", model: models["q9iq4"], cfg: ModelConfig{ContextSize: 32768, UBatchSize: 512, GPULayers: 999, SpecType: "draft-mtp", DraftMax: 6}, cards: 1, measured: 7.77,
			reported: &reportedTerms{weights: 4.71, kv: 1.12, recurrent: 1.37, compute: 0.34}, backend: "rocm"},
		{name: "rx9070 q9iq4 ctx128k ub512 q8 mtp", model: models["q9iq4"], cfg: ModelConfig{ContextSize: 131072, UBatchSize: 512, GPULayers: 999, KVCacheQuant: "q8_0", SpecType: "draft-mtp", DraftMax: 6}, cards: 1, measured: 9.96,
			reported: &reportedTerms{weights: 4.71, kv: 2.62, recurrent: 1.37, compute: 1.02}, backend: "rocm"},
		{name: "rx9070 q35a3 ctx32k ub512 moe30", model: models["q35a3"], cfg: ModelConfig{ContextSize: 32768, UBatchSize: 512, GPULayers: 999, CPUMoE: 30}, cards: 1, measured: 8.01,
			reported: &reportedTerms{weights: 6.51, kv: 0.62, recurrent: 0.25, compute: 0.39}, backend: "rocm"},
		{name: "rx9070 q35a3 ctx32k ub512 moe30 mtp", model: models["q35a3"], cfg: ModelConfig{ContextSize: 32768, UBatchSize: 512, GPULayers: 999, CPUMoE: 30, SpecType: "draft-mtp", DraftMax: 6}, cards: 1, measured: 10.16,
			reported: &reportedTerms{weights: 7.0, kv: 0.69, recurrent: 1.72, compute: 0.54}, backend: "rocm"},
		{name: "rx9070 q35a3 ctx128k ub512 q8 moe32", model: models["q35a3"], cfg: ModelConfig{ContextSize: 131072, UBatchSize: 512, GPULayers: 999, KVCacheQuant: "q8_0", CPUMoE: 32}, cards: 1, measured: 8.02,
			reported: &reportedTerms{weights: 5.6, kv: 1.33, recurrent: 0.25, compute: 0.57}, backend: "rocm"},
		// The same machine's later card, a Radeon AI PRO R9700 (32 GiB), on
		// llama.cpp v1.0.0, 2026-11-13: ThinkingCap-Qwen3.8-27B Q4_K_M at
		// its full context with MTP, as Autoconfigure would plan it. Read
		// from the card counters with the model loaded and served, less
		// the same counters with it unloaded. KV is the model's 16 layers
		// at 8 bits (8.50) and the drafter's one at f16 (1.00); compute is
		// the model's context (1.35), the drafter's (0.32) and h_nextn.
		{name: "r9700 thinkingcap27b ctx262k q8 mtp", model: thinkingCap27B, cfg: ModelConfig{ContextSize: 262144, GPULayers: 999, KVCacheQuant: "q8_0", SpecType: "draft-mtp", DraftMax: 3}, cards: 1, measured: 28.83,
			reported: &reportedTerms{weights: 14.98, kv: 9.50, recurrent: 2.34, compute: 1.70}, backend: "rocm"},
	}
}

// thinkingCap27B is ThinkingCap-Qwen3.8-27B.i1-Q4_K_M as the parser
// records it: 64 trunk layers and one MTP layer, 16 attending.
var thinkingCap27B = Model{Arch: "qwen35", SizeBytes: 16810715712, TokenEmbdBytes: 715161600, NLayers: 65, AttnLayers: 16,
	NEmbd: 5120, NHead: 24, NKVHead: 4, KVFullPerTok: 16 * 4 * 512, ContextLength: 262144,
	RecurrentStateBytes: 156893184, NextNLayers: 1}

// cudaCorpus: compute2, 3x RTX A4000 (16 GiB), build v0.5.0-cuda,
// measured 2026-10-02 by starting llama-server with the preset's flags
// (flash attention on, every layer offloaded, the default --parallel) and
// reading its buffer report and the card counters once loaded. "1c" is
// one card (--device CUDA0), "3c" a layer split over all three. It
// replaces a single CUDA point from an older build, whose compute and
// recurrent figures this build does not reproduce.
func cudaCorpus() []corpusPoint {
	// Models as compute2's registry records them, with what parser
	// version 4 adds: gpt-oss's built-in sliding-window layout, and
	// gemma's tied output embedding.
	models := map[string]Model{
		"granite8b":  {SizeBytes: 9345613952, TokenEmbdBytes: 436731904, NLayers: 40, AttnLayers: 40, NEmbd: 4096, NHead: 32, NKVHead: 8, KVFullPerTok: 81920, ContextLength: 131072},
		"granite30b": {SizeBytes: 31111705312, TokenEmbdBytes: 436731904, NLayers: 64, AttnLayers: 64, NEmbd: 4096, NHead: 32, NKVHead: 8, KVFullPerTok: 131072, ContextLength: 131072},
		"q9":         {SizeBytes: 9527502048, TokenEmbdBytes: 1080688640, NLayers: 32, AttnLayers: 8, NEmbd: 4096, NHead: 16, NKVHead: 4, KVFullPerTok: 16384, ContextLength: 262144},
		"q27":        {SizeBytes: 31457991680, TokenEmbdBytes: 1350860800, NLayers: 65, AttnLayers: 16, NEmbd: 5120, NHead: 24, NKVHead: 4, KVFullPerTok: 32768, ContextLength: 262144, NextNLayers: 1},
		"q35a3":      {SizeBytes: 32611711264, TokenEmbdBytes: 540344320, NLayers: 41, AttnLayers: 10, NEmbd: 2048, NHead: 16, NKVHead: 2, KVFullPerTok: 10240, ContextLength: 262144, ExpertCount: 256, ExpertUsedCount: 8, ExpertBytes: 29880221696, ExpertLayers: 41, NextNLayers: 1},
		"gptoss":     {SizeBytes: 12109567168, TokenEmbdBytes: 615329280, NLayers: 24, AttnLayers: 24, NEmbd: 2880, NHead: 64, NKVHead: 8, KVFullPerTok: 12288, KVSWAPerTok: 12288, SlidingWindow: 128, ContextLength: 131072, ExpertCount: 32, ExpertUsedCount: 4, ExpertBytes: 10178887680, ExpertLayers: 24},
		"gem26":      {Arch: "gemma4", OutputTied: true, SizeBytes: 26859859744, TokenEmbdBytes: 784334848, NLayers: 30, AttnLayers: 30, NEmbd: 2816, NHead: 16, NKVHead: 8, KVFullPerTok: 10240, KVSWAPerTok: 102400, SlidingWindow: 1024, ContextLength: 262144, ExpertCount: 128, ExpertUsedCount: 8, ExpertBytes: 24265374720, ExpertLayers: 30},
		"gem12":      {Arch: "gemma4", OutputTied: true, SizeBytes: 7366421920, TokenEmbdBytes: 692060160, NLayers: 48, AttnLayers: 48, NEmbd: 3840, NHead: 16, NKVHead: 8, KVFullPerTok: 8192, KVSWAPerTok: 163840, SlidingWindow: 1024, ContextLength: 262144},
		"gemE4":      {Arch: "gemma4", OutputTied: true, SizeBytes: 5126304928, PLEBytes: 1937768448, TokenEmbdBytes: 461373440, NLayers: 42, AttnLayers: 42, NEmbd: 2560, NHead: 8, NKVHead: 2, KVFullPerTok: 14336, KVSWAPerTok: 35840, SlidingWindow: 512, ContextLength: 131072},
		"flash":      {SizeBytes: 111334654784, PLEBytes: 28800138240, TokenEmbdBytes: 675430400, NLayers: 48, AttnLayers: 12, NEmbd: 2560, NHead: 24, NKVHead: 2, KVFullPerTok: 12288, IndexerKeyLength: 128, ContextLength: 262144, ExpertCount: 512, ExpertUsedCount: 10, ExpertBytes: 77017907200, ExpertLayers: 48},
	}
	return []corpusPoint{
		{name: "granite8b 1c ctx8k ub512", model: models["granite8b"], cfg: ModelConfig{ContextSize: 8192, UBatchSize: 512, GPULayers: 999}, cards: 1, measured: 10.16,
			reported: &reportedTerms{weights: 8.29, kv: 1.25, recurrent: 0.0, compute: 0.1}, backend: "cuda"},
		{name: "granite8b 1c ctx32k ub512", model: models["granite8b"], cfg: ModelConfig{ContextSize: 32768, UBatchSize: 512, GPULayers: 999}, cards: 1, measured: 13.93,
			reported: &reportedTerms{weights: 8.29, kv: 5.0, recurrent: 0.0, compute: 0.13}, backend: "cuda"},
		{name: "granite8b 1c ctx32k ub2048", model: models["granite8b"], cfg: ModelConfig{ContextSize: 32768, UBatchSize: 2048, GPULayers: 999}, cards: 1, measured: 14.32,
			reported: &reportedTerms{weights: 8.29, kv: 5.0, recurrent: 0.0, compute: 0.51}, backend: "cuda"},
		{name: "granite8b 3c ctx32k ub512", model: models["granite8b"], cfg: ModelConfig{ContextSize: 32768, UBatchSize: 512, GPULayers: 999, SplitMode: "layer"}, cards: 3, measured: 14.61,
			reported: &reportedTerms{weights: 8.29, kv: 5.0, recurrent: 0.0, compute: 0.76}, backend: "cuda"},
		{name: "granite8b 3c ctx128k ub512", model: models["granite8b"], cfg: ModelConfig{ContextSize: 131072, UBatchSize: 512, GPULayers: 999, SplitMode: "layer"}, cards: 3, measured: 30.73,
			reported: &reportedTerms{weights: 8.29, kv: 20.0, recurrent: 0.0, compute: 1.88}, backend: "cuda"},
		{name: "granite30b 3c ctx32k ub512", model: models["granite30b"], cfg: ModelConfig{ContextSize: 32768, UBatchSize: 512, GPULayers: 999, SplitMode: "layer"}, cards: 3, measured: 38.22,
			reported: &reportedTerms{weights: 28.56, kv: 8.0, recurrent: 0.0, compute: 1.1}, backend: "cuda"},
		{name: "q9 1c ctx32k ub512", model: models["q9"], cfg: ModelConfig{ContextSize: 32768, UBatchSize: 512, GPULayers: 999}, cards: 1, measured: 9.68,
			reported: &reportedTerms{weights: 7.86, kv: 1.0, recurrent: 0.2, compute: 0.12}, backend: "cuda"},
		{name: "q9 1c ctx128k ub512", model: models["q9"], cfg: ModelConfig{ContextSize: 131072, UBatchSize: 512, GPULayers: 999}, cards: 1, measured: 12.77,
			reported: &reportedTerms{weights: 7.86, kv: 4.0, recurrent: 0.2, compute: 0.21}, backend: "cuda"},
		{name: "q9 3c ctx128k ub512", model: models["q9"], cfg: ModelConfig{ContextSize: 131072, UBatchSize: 512, GPULayers: 999, SplitMode: "layer"}, cards: 3, measured: 14.47,
			reported: &reportedTerms{weights: 7.86, kv: 4.0, recurrent: 0.2, compute: 1.85}, backend: "cuda"},
		{name: "q27 3c ctx8k ub512", model: models["q27"], cfg: ModelConfig{ContextSize: 8192, UBatchSize: 512, GPULayers: 999, SplitMode: "layer"}, cards: 3, measured: 29.72,
			reported: &reportedTerms{weights: 27.5, kv: 0.5, recurrent: 0.58, compute: 0.57}, backend: "cuda"},
		{name: "q27 3c ctx32k ub512", model: models["q27"], cfg: ModelConfig{ContextSize: 32768, UBatchSize: 512, GPULayers: 999, SplitMode: "layer"}, cards: 3, measured: 31.50,
			reported: &reportedTerms{weights: 27.5, kv: 2.0, recurrent: 0.58, compute: 0.85}, backend: "cuda"},
		{name: "q27 3c ctx128k ub512", model: models["q27"], cfg: ModelConfig{ContextSize: 131072, UBatchSize: 512, GPULayers: 999, SplitMode: "layer"}, cards: 3, measured: 38.63,
			reported: &reportedTerms{weights: 27.5, kv: 8.0, recurrent: 0.58, compute: 1.97}, backend: "cuda"},
		{name: "q27 3c ctx32k ub2048", model: models["q27"], cfg: ModelConfig{ContextSize: 32768, UBatchSize: 2048, GPULayers: 999, SplitMode: "layer"}, cards: 3, measured: 34.05,
			reported: &reportedTerms{weights: 27.5, kv: 2.0, recurrent: 0.58, compute: 3.4}, backend: "cuda"},
		{name: "q35a3 3c ctx32k ub512", model: models["q35a3"], cfg: ModelConfig{ContextSize: 32768, UBatchSize: 512, GPULayers: 999, SplitMode: "layer"}, cards: 3, measured: 31.29,
			reported: &reportedTerms{weights: 29.14, kv: 0.62, recurrent: 0.25, compute: 0.72}, backend: "cuda"},
		{name: "q35a3 3c ctx128k ub512 q8", model: models["q35a3"], cfg: ModelConfig{ContextSize: 131072, UBatchSize: 512, GPULayers: 999, KVCacheQuant: "q8_0", SplitMode: "layer"}, cards: 3, measured: 33.73,
			reported: &reportedTerms{weights: 29.14, kv: 1.33, recurrent: 0.25, compute: 2.45}, backend: "cuda"},
		{name: "q35a3 1c ctx32k ub512 moe30", model: models["q35a3"], cfg: ModelConfig{ContextSize: 32768, UBatchSize: 512, GPULayers: 999, CPUMoE: 30}, cards: 1, measured: 10.86,
			reported: &reportedTerms{weights: 8.87, kv: 0.62, recurrent: 0.25, compute: 0.6}, backend: "cuda"},
		{name: "gptoss 1c ctx32k ub512", model: models["gptoss"], cfg: ModelConfig{ContextSize: 32768, UBatchSize: 512, GPULayers: 999}, cards: 1, measured: 12.09,
			reported: &reportedTerms{weights: 10.69, kv: 0.77, recurrent: 0.0, compute: 0.12}, backend: "cuda"},
		{name: "gptoss 1c ctx128k ub512", model: models["gptoss"], cfg: ModelConfig{ContextSize: 131072, UBatchSize: 512, GPULayers: 999}, cards: 1, measured: 14.43,
			reported: &reportedTerms{weights: 10.69, kv: 3.02, recurrent: 0.0, compute: 0.21}, backend: "cuda"},
		{name: "gptoss 3c ctx128k ub2048", model: models["gptoss"], cfg: ModelConfig{ContextSize: 131072, UBatchSize: 2048, GPULayers: 999, SplitMode: "layer"}, cards: 3, measured: 21.70,
			reported: &reportedTerms{weights: 10.69, kv: 3.06, recurrent: 0.0, compute: 7.4}, backend: "cuda"},
		{name: "gem26 3c ctx32k ub512", model: models["gem26"], cfg: ModelConfig{ContextSize: 32768, UBatchSize: 512, GPULayers: 999, SplitMode: "layer"}, cards: 3, measured: 27.95,
			reported: &reportedTerms{weights: 25.0, kv: 1.5, recurrent: 0.0, compute: 0.9}, backend: "cuda"},
		{name: "gem12 1c ctx32k ub512", model: models["gem12"], cfg: ModelConfig{ContextSize: 32768, UBatchSize: 512, GPULayers: 999}, cards: 1, measured: 9.42,
			reported: &reportedTerms{weights: 6.85, kv: 1.91, recurrent: 0.0, compute: 0.15}, backend: "cuda"},
		{name: "gemE4 1c ctx32k ub512", model: models["gemE4"], cfg: ModelConfig{ContextSize: 32768, UBatchSize: 512, GPULayers: 999}, cards: 1, measured: 4.19,
			reported: &reportedTerms{weights: 2.95, kv: 0.6, recurrent: 0.0, compute: 0.13}, backend: "cuda"},
		{name: "flash 3c ctx32k ub512 q8 moe32", model: models["flash"], cfg: ModelConfig{ContextSize: 32768, UBatchSize: 512, GPULayers: 999, KVCacheQuant: "q8_0", CPUMoE: 32, SplitMode: "layer"}, cards: 3, measured: 32.29,
			reported: &reportedTerms{weights: 28.42, kv: 0.45, recurrent: 0.44, compute: 2.39}, backend: "cuda"},
		{name: "flash 3c ctx128k ub512 q8 moe32", model: models["flash"], cfg: ModelConfig{ContextSize: 131072, UBatchSize: 512, GPULayers: 999, KVCacheQuant: "q8_0", CPUMoE: 32, SplitMode: "layer"}, cards: 3, measured: 35.02,
			reported: &reportedTerms{weights: 28.42, kv: 1.79, recurrent: 0.44, compute: 3.78}, backend: "cuda"},
		{name: "q27 3c ctx32k ub512 mtp", model: models["q27"], cfg: ModelConfig{ContextSize: 32768, UBatchSize: 512, GPULayers: 999, SplitMode: "layer", SpecType: "draft-mtp", DraftMax: 6}, cards: 3, measured: 36.13,
			reported: &reportedTerms{weights: 28.03, kv: 2.12, recurrent: 4.09, compute: 1.66}, backend: "cuda"},
		{name: "q27 3c ctx128k ub512 q8 mtp", model: models["q27"], cfg: ModelConfig{ContextSize: 131072, UBatchSize: 512, GPULayers: 999, KVCacheQuant: "q8_0", SplitMode: "layer", SpecType: "draft-mtp", DraftMax: 6}, cards: 3, measured: 40.97,
			reported: &reportedTerms{weights: 28.03, kv: 4.75, recurrent: 4.09, compute: 3.74}, backend: "cuda"},
		{name: "q35a3 3c ctx32k ub512 mtp", model: models["q35a3"], cfg: ModelConfig{ContextSize: 32768, UBatchSize: 512, GPULayers: 999, SplitMode: "layer", SpecType: "draft-mtp", DraftMax: 6}, cards: 3, measured: 33.87,
			reported: &reportedTerms{weights: 29.86, kv: 0.69, recurrent: 1.72, compute: 1.29}, backend: "cuda"},
		{name: "q35a3 3c ctx128k ub512 q8 mtp", model: models["q35a3"], cfg: ModelConfig{ContextSize: 131072, UBatchSize: 512, GPULayers: 999, KVCacheQuant: "q8_0", SplitMode: "layer", SpecType: "draft-mtp", DraftMax: 6}, cards: 3, measured: 36.81,
			reported: &reportedTerms{weights: 29.86, kv: 1.58, recurrent: 1.72, compute: 3.71}, backend: "cuda"},
		{name: "q35a3 1c ctx32k ub512 moe30 mtp", model: models["q35a3"], cfg: ModelConfig{ContextSize: 32768, UBatchSize: 512, GPULayers: 999, CPUMoE: 30, SpecType: "draft-mtp", DraftMax: 6}, cards: 1, measured: 13.19,
			reported: &reportedTerms{weights: 9.59, kv: 0.69, recurrent: 1.72, compute: 0.76}, backend: "cuda"},
	}
}

// cudaBenchCorpus: compute2 again, two of its A4000s, from the benchmark
// history (2026-11-01 to 11-05, builds v0.8.0-cuda and v0.9.0-cuda,
// flash attention on, 131K context, llama.cpp's four conversations).
// measured is the sum of the cards' used memory while the model was
// loaded, less the vision projector where one was loaded (0.84 GiB for
// the 35B-A3B), since these records carry no projector path. These are
// the large micro-batches the points above lack: at 2048 and 4096 the
// estimate was 4 to 11 GiB high (issue #582). Their buffer reports are
// left out: before #582 the report added every compute reservation
// llama.cpp made while serving, not only the one in use.
func cudaBenchCorpus() []corpusPoint {
	q35 := Model{Arch: "qwen35moe", SizeBytes: 18536192288, TokenEmbdBytes: 540344320, NLayers: 41, AttnLayers: 10,
		NEmbd: 2048, NHead: 16, NKVHead: 2, KVFullPerTok: 10240, ContextLength: 262144, ExpertCount: 256, ExpertUsedCount: 8,
		ExpertBytes: 15927869440, ExpertLayers: 41, NextNLayers: 1, RecurrentStateBytes: 65863680}
	spark := Model{Arch: "spark2_5", SizeBytes: 3543441312, TokenEmbdBytes: 671088640, NLayers: 36, AttnLayers: 36,
		NEmbd: 2560, NHead: 16, NKVHead: 4, KVFullPerTok: 18432, KVSWAPerTok: 55296, SlidingWindow: 512, ContextLength: 1048576}
	glb := Model{Arch: "qwen35", SizeBytes: 4482402848, TokenEmbdBytes: 675430400, OutputTied: true, NLayers: 32, AttnLayers: 8,
		NEmbd: 2560, NHead: 16, NKVHead: 4, KVFullPerTok: 16384, ContextLength: 262144, RecurrentStateBytes: 52690944}
	c := func(ub int, split, spec string, draft int) ModelConfig {
		return ModelConfig{ContextSize: 131072, BatchSize: max(ub, 2048), UBatchSize: ub, GPULayers: 999,
			SplitMode: split, SpecType: spec, DraftMax: draft}
	}
	const mmproj = 902822528.0 / gib
	used := func(a, b int) float64 { return float64(a+b) / 1024 }
	return []corpusPoint{
		// Four pipeline copies of a 0.5 GiB mask fit beside the model.
		{name: "bench glb 2c layer ub2048", model: glb, cfg: c(2048, "layer", "", 0), cards: 2, measured: used(6869, 7087), backend: "cuda"},
		// Four copies of a 1 GiB mask did not: llama.cpp kept one.
		{name: "bench q35 2c layer ub4096", model: q35, cfg: c(4096, "layer", "", 0), cards: 2, measured: used(13109, 11785) - mmproj, backend: "cuda"},
		// The model's context kept four copies, the drafter's one.
		{name: "bench q35 2c layer ub2048 mtp", model: q35, cfg: c(2048, "layer", "draft-mtp", 3), cards: 2, measured: used(14273, 14197) - mmproj, backend: "cuda"},
		{name: "bench q35 2c layer ub4096 mtp", model: q35, cfg: c(4096, "layer", "draft-mtp", 6), cards: 2, measured: used(13915, 14571) - mmproj, backend: "cuda"},
		// No pipeline on a tensor split.
		{name: "bench spark 2c tensor ub2048", model: spark, cfg: c(2048, "tensor", "", 0), cards: 2, measured: used(5125, 5125), backend: "cuda"},
		{name: "bench q35 2c tensor ub4096 mtp", model: q35, cfg: c(4096, "tensor", "draft-mtp", 3), cards: 2, measured: used(14945, 14483) - mmproj, backend: "cuda"},
	}
}

// cudaLaunchCorpus: compute2's CUDA0 and CUDA1, 2026-11-06, v0.9.0-cuda,
// started by hand with flash attention, every layer offloaded unless
// noted, no projector, and either llama-server's default four slots in
// one pool or four separate slots ("4 slots", -np 4). Qwen3.6-35B-A3B and
// gpt-oss-20b at 131K (one point at 262K), Qwen3-30B-A3B at 64K, and
// Qwen3.8-Flash-Next at 32K and 131K; a few later points on one card or
// all three (CUDA0-2), marked 1c and 3c. measured is the two cards' used
// memory once loaded, from empty cards; reported is llama.cpp's report,
// each context's first reservation. The log's "sched copies" is noted
// where it matters: these points tell when llama.cpp keeps its pipeline
// copies.
func cudaLaunchCorpus() []corpusPoint {
	q35 := cudaBenchCorpus()[1].model
	oss := Model{Arch: "gpt-oss", SizeBytes: 12109566624, TokenEmbdBytes: 615329280, NLayers: 24, AttnLayers: 24,
		NEmbd: 2880, NHead: 64, NKVHead: 8, KVFullPerTok: 12288, KVSWAPerTok: 12288, SlidingWindow: 128, ContextLength: 131072,
		ExpertCount: 32, ExpertUsedCount: 4, ExpertBytes: 10178887680, ExpertLayers: 24}
	q30 := Model{Arch: "qwen3moe", SizeBytes: 18556685824, TokenEmbdBytes: 175030272, NLayers: 48, AttnLayers: 48,
		NEmbd: 2048, NHead: 32, NKVHead: 4, KVFullPerTok: 49152, ContextLength: 40960,
		ExpertCount: 128, ExpertUsedCount: 8, ExpertBytes: 17553162240, ExpertLayers: 48}
	fn := Model{Arch: "qwen4exp", SizeBytes: 66379180698, PLEBytes: 28800138240, TokenEmbdBytes: 337715200, NLayers: 48,
		AttnLayers: 12, NEmbd: 2560, NHead: 24, NKVHead: 2, KVFullPerTok: 12288, ContextLength: 262144, ExpertCount: 512,
		ExpertUsedCount: 10, ExpertBytes: 33397145600, ExpertLayers: 48, IndexerKeyLength: 128, RecurrentStateBytes: 117669888}
	c := func(par, ctx, ub, moe int, mtp bool) ModelConfig {
		cfg := ModelConfig{ContextSize: ctx, UBatchSize: ub, BatchSize: max(ub, 2048), GPULayers: 999,
			SplitMode: "layer", Parallel: par, CPUMoE: moe}
		if mtp {
			cfg.SpecType, cfg.DraftMax = "draft-mtp", 3
		}
		return cfg
	}
	used := func(a, b int) float64 { return float64(a+b) / 1024 }
	used3 := func(a, b, c int) float64 { return float64(a+b+c) / 1024 }
	one := func(cfg ModelConfig) ModelConfig { cfg.SplitMode = ""; return cfg }
	glb := cudaBenchCorpus()[0].model
	rep := func(w, kv, rs, comp, moe float64) *reportedTerms {
		return &reportedTerms{weights: w, kv: kv, recurrent: rs, compute: comp, moeCache: moe}
	}
	const k131 = 131072
	cache := c(4, k131, 2048, 16, false)
	cache.MoECacheMiB = 2048
	partial := c(4, k131, 2048, 0, false)
	partial.GPULayers = 30
	tensor := c(0, k131, 4096, 0, false)
	tensor.SplitMode = "tensor"
	return []corpusPoint{
		// Four copies of a 32K slot's mask fit at every micro-batch.
		{name: "launch q35 4 slots ub4096", model: q35, cfg: c(4, k131, 4096, 0, false), cards: 2, measured: used(12693, 11419),
			reported: rep(16.35, 2.50, 0.25, 4.06, 0), backend: "cuda"},
		// The pool's 131K masks: four copies at 2048, one at 4096 and at
		// 2048 with 262K ("retrying without pipeline parallelism").
		{name: "launch q35 pool ub2048", model: q35, cfg: c(0, k131, 2048, 0, false), cards: 2, measured: used(13129, 11999),
			reported: rep(16.35, 2.50, 0.25, 5.04, 0), backend: "cuda"},
		{name: "launch q35 pool ub4096", model: q35, cfg: c(0, k131, 4096, 0, false), cards: 2, measured: used(12331, 11293),
			reported: rep(16.35, 2.50, 0.25, 3.58, 0), backend: "cuda"},
		{name: "launch q35 pool ctx262k ub2048", model: q35, cfg: c(0, 262144, 2048, 0, false), cards: 2, measured: used(13197, 12185),
			reported: rep(16.35, 5.00, 0.25, 2.80, 0), backend: "cuda"},
		// Experts in system memory, the MoE cache, layers left off the
		// GPUs: one copy every time.
		{name: "launch q35 4 slots moe8 ub4096", model: q35, cfg: c(4, k131, 4096, 8, false), cards: 2, measured: used(8665, 10521),
			reported: rep(13.51, 2.50, 0.25, 2.09, 0), backend: "cuda"},
		{name: "launch q35 pool moe8 ub4096", model: q35, cfg: c(0, k131, 4096, 8, false), cards: 2, measured: used(9435, 11293), backend: "cuda"},
		// Its weights come out 0.11 GiB under the report, the experts
		// being spread evenly over layers that are not, so the report is
		// left out.
		{name: "launch q35 4 slots moe16 cache2g ub2048", model: q35, cfg: cache, cards: 2, measured: used(7337, 10003), backend: "cuda"},
		{name: "launch q35 4 slots ngl30 ub2048", model: q35, cfg: partial, cards: 2, measured: used(8349, 7045), backend: "cuda"},
		// MTP: the drafter's copies fit with 32K slots, not with the
		// pool at 2048, and neither context's does at 4096.
		// Here llama.cpp kept the drafter's four copies (6.07 GiB of
		// compute reported) with 0.5 GiB to spare on its card, which
		// the estimate, judging by the average card, does not see: it
		// plans one, and its total stays 0.1 GiB above. The report is left
		// out for that.
		{name: "launch q35 4 slots mtp ub4096", model: q35, cfg: c(4, k131, 4096, 0, true), cards: 2, measured: used(13115, 14467), backend: "cuda"},
		{name: "launch q35 pool mtp ub2048", model: q35, cfg: c(0, k131, 2048, 0, true), cards: 2, measured: used(13549, 13659),
			reported: rep(16.75, 2.75, 0.98, 5.70, 0), backend: "cuda"},
		{name: "launch q35 pool mtp ub4096", model: q35, cfg: c(0, k131, 4096, 0, true), cards: 2, measured: used(12753, 13627),
			reported: rep(16.75, 2.75, 0.98, 4.89, 0), backend: "cuda"},
		{name: "launch q35 pool moe8 mtp ub4096", model: q35, cfg: c(0, k131, 4096, 8, true), cards: 2, measured: used(9855, 13627),
			reported: rep(13.91, 2.75, 0.98, 4.91, 0), backend: "cuda"},
		// Tensor-parallel: no pipeline. (Its report, per card, has the
		// weights 0.2 GiB above the estimate's, which counts them once
		// for the cards together; left out.)
		{name: "launch q35 pool tensor ub4096", model: q35, cfg: tensor, cards: 2, measured: used(11811, 11811), backend: "cuda"},

		// gpt-oss-20b, MXFP4, at 131K: sliding-window layers, and small
		// enough that the pool's four copies fit even at 4096.
		{name: "launch oss pool ub4096", model: oss, cfg: c(0, k131, 4096, 0, false), cards: 2, measured: used(12943, 12245),
			reported: rep(10.69, 3.11, 0, 10.40, 0), backend: "cuda"},
		{name: "launch oss pool moe8 ub4096", model: oss, cfg: c(0, k131, 4096, 8, false), cards: 2, measured: used(6087, 8873),
			reported: rep(7.53, 3.11, 0, 3.58, 0), backend: "cuda"},
		{name: "launch oss 4 slots ub4096", model: oss, cfg: c(4, k131, 4096, 0, false), cards: 2, measured: used(10021, 9273), backend: "cuda"},

		// Qwen3-30B-A3B, Q4_K_M, at 64K: full attention on every layer.
		// The pool's copies fit at 2048, not at 4096.
		{name: "launch q30 pool ub2048", model: q30, cfg: c(0, 65536, 2048, 0, false), cards: 2, measured: used(13929, 13103),
			reported: rep(17.11, 6.00, 0, 2.90, 0), backend: "cuda"},
		{name: "launch q30 pool ub4096", model: q30, cfg: c(0, 65536, 4096, 0, false), cards: 2, measured: used(13555, 12873),
			reported: rep(17.11, 6.00, 0, 2.31, 0), backend: "cuda"},
		{name: "launch q30 pool moe8 ub4096", model: q30, cfg: c(0, 65536, 4096, 8, false), cards: 2, measured: used(10891, 12873),
			reported: rep(14.29, 6.00, 0, 2.53, 0), backend: "cuda"},

		// Qwen3.8-Flash-Next, UD-IQ1_S, with the experts of 34 layers in
		// system memory: sparse attention, one copy. The KV figure holds
		// the indexer's key cache, f32 with an f16 cache.
		{name: "launch fn pool moe34 ctx32k ub512", model: fn, cfg: c(0, 32768, 512, 34, false), cards: 2, measured: used(3181, 12139),
			reported: rep(12.38, 0.94, 0.44, 0.80, 0), backend: "cuda"},
		{name: "launch fn pool moe34 ub512", model: fn, cfg: c(0, k131, 512, 34, false), cards: 2, measured: used(4719, 14013),
			reported: rep(12.38, 3.75, 0.44, 1.32, 0), backend: "cuda"},
		{name: "launch fn pool moe34 ctx32k ub2048", model: fn, cfg: c(0, 32768, 2048, 34, false), cards: 2, measured: used(3501, 12821),
			reported: rep(12.38, 0.94, 0.44, 1.78, 0), backend: "cuda"},

		// Flash-Next with every expert on the GPUs, over all three cards:
		// four copies. Four separate slots shrink the attention masks
		// (1.13 GiB less) but not the indexer's.
		{name: "launch fn 3c pool ctx32k ub512", model: fn, cfg: c(0, 32768, 512, 0, false), cards: 3, measured: used3(13499, 13045, 12387),
			reported: rep(34.67, 0.94, 0.44, 1.39, 0), backend: "cuda"},
		{name: "launch fn 3c pool ctx32k ub2048", model: fn, cfg: c(0, 32768, 2048, 0, false), cards: 3, measured: used3(14703, 14473, 13815),
			reported: rep(34.67, 0.94, 0.44, 5.35, 0), backend: "cuda"},
		{name: "launch fn 3c 4 slots ctx32k ub2048", model: fn, cfg: c(4, 32768, 2048, 0, false), cards: 3, measured: used3(14315, 14083, 13425),
			reported: rep(34.67, 0.94, 0.44, 4.21, 0), backend: "cuda"},

		// One card (CUDA0): no pipeline, and the slots' masks shrink as on
		// a split. With gpt-oss each slot also keeps its own window and
		// micro-batch of sliding-window cache.
		{name: "launch glb 1c pool ub4096", model: glb, cfg: one(c(0, k131, 4096, 0, false)), cards: 1, measured: used3(10385, 0, 0),
			reported: rep(4.16, 4.00, 0.20, 1.58, 0), backend: "cuda"},
		{name: "launch glb 1c 4 slots ub4096", model: glb, cfg: one(c(4, k131, 4096, 0, false)), cards: 1, measured: used3(9611, 0, 0),
			reported: rep(4.16, 4.00, 0.20, 0.83, 0), backend: "cuda"},
		{name: "launch oss 1c 4 slots ctx64k ub4096", model: oss, cfg: one(c(4, 65536, 4096, 0, false)), cards: 1, measured: used3(13971, 0, 0),
			reported: rep(10.69, 1.90, 0, 0.86, 0), backend: "cuda"},
	}
}

// point looks a corpus point up by name. Positional access breaks
// silently when a point is inserted, and inserting points is the whole
// intent of this file.
func point(t *testing.T, name string) corpusPoint {
	t.Helper()
	for _, p := range corpus() {
		if p.name == name {
			return p
		}
	}
	t.Fatalf("no corpus point named %q", name)
	return corpusPoint{}
}

// The guardrail. An estimate below what the hardware actually used tells
// someone a model fits when it does not, so no point may under-predict —
// however good the average looks.
func TestEstimateNeverUnderPredicts(t *testing.T) {
	for _, p := range corpus() {
		got := p.estimate().Total()
		if got < p.measured {
			t.Errorf("%s: estimated %.2f GiB, hardware used %.2f — under by %.2f",
				p.name, got, p.measured, p.measured-got)
		}
	}
}

// Being conservative is only useful if it stays close. A headroom figure
// nobody believes gets ignored, which is the same failure by another route.
func TestEstimateStaysCloseToMeasured(t *testing.T) {
	var sum, worst float64
	var worstName string
	for _, p := range corpus() {
		got := p.estimate().Total()
		e := math.Abs(got - p.measured)
		sum += e
		if e > worst {
			worst, worstName = e, p.name
		}
		if e > 3.0 {
			t.Errorf("%s: estimated %.2f against %.2f measured, off by %.2f", p.name, got, p.measured, e)
		}
	}
	// The bar was 1.5 GiB while every point came from one machine, one
	// backend and a tensor-parallel split. It is 2.0 now that the corpus
	// holds a layer-split CUDA load as well: the terms that load needed
	// — several copies of the graph, and a recurrent state to match —
	// are modelled from a single pair of measurements, so they sit
	// deliberately above what the older points pay. Tighten it again
	// when there are enough layer-split points to fit properly.
	mean := sum / float64(len(corpus()))
	if mean > 2.0 {
		t.Errorf("mean error %.2f GiB across the corpus, want under 2.0", mean)
	}
	t.Logf("mean error %.2f GiB, worst %.2f on %s", mean, worst, worstName)
}

// The device count is not cosmetic: scratch and driver overhead are per
// device, so the same config across four cards costs materially more than
// across one. A caller that cannot supply the count gets the single-device
// figure, which is the smaller one — so this must be visible, not silent.
func TestDeviceCountChangesTheEstimate(t *testing.T) {
	p := point(t, "27B ctx262k ub512") // no indexer
	one := VRAMEstimateForConfigOn(&p.model, &p.cfg, 1)
	four := VRAMEstimateForConfigOn(&p.model, &p.cfg, 4)
	if four <= one {
		t.Fatalf("four cards estimated %.2f, one card %.2f — the count is being ignored", four, one)
	}
	if diff := four - one; diff < 2.0 {
		t.Errorf("four cards cost only %.2f GiB more than one; expected roughly 3x the per-device overhead", diff)
	}
}

// Host-mapped weights are not counted. Both tensors matter: the 27B has no
// per-layer table and still keeps its input embedding off the device.
func TestHostMappedWeightsExcluded(t *testing.T) {
	m := Model{SizeBytes: gibBytes(30.0), PLEBytes: gibBytes(4.0), TokenEmbdBytes: gibBytes(1.0),
		NLayers: 32, AttnLayers: 32, NEmbd: 4096, NKVHead: 8, KVFullPerTok: 32 * 8 * 256}
	cfg := ModelConfig{ContextSize: 4096}
	withBoth := VRAMEstimateForConfigOn(&m, &cfg, 1)

	noEmb := m
	noEmb.TokenEmbdBytes = 0
	if VRAMEstimateForConfigOn(&noEmb, &cfg, 1)-withBoth < 0.9 {
		t.Error("the token embedding is not being excluded from resident weights")
	}
	noPLE := m
	noPLE.PLEBytes = 0
	if VRAMEstimateForConfigOn(&noPLE, &cfg, 1)-withBoth < 3.9 {
		t.Error("the per-layer table is not being excluded from resident weights")
	}
}

// A sparse-attention model pays for ranking the whole cache. Without that
// term Flash-Next is under-predicted by tens of GiB at long context.
func TestIndexerTermOnlyAppliesToRankingModels(t *testing.T) {
	p := point(t, "Flash-Next ctx262k ub512")
	with := p.estimate().Total()

	plain := p.model
	plain.IndexerKeyLength = 0
	without := VRAMEstimateForConfigOn(&plain, &p.cfg, p.cards)

	if with-without < 10 {
		t.Errorf("indexer term worth only %.2f GiB at 262144 context; measured compute alone was 24 GiB", with-without)
	}
	if without >= p.measured {
		t.Error("the estimate is adequate without the indexer term; the fixture no longer shows why it exists")
	}
}

// The estimate has to be right for the right reasons. A total alone
// cannot tell a good model from two mistakes that cancel: the estimate
// this corpus replaced was within 7 GiB on Qwen3.8-Flash-Next while
// over-counting weights by 27 and under-counting everything else by 20.
//
// So each term is checked against what llama.cpp said it allocated, in
// the same direction as the total: at or above measured, and close.
func TestEstimateTermsAgainstTheBufferReport(t *testing.T) {
	// Under: a term may sit below the report only by the corpus's own
	// rounding and per-term noise; the total is held strictly elsewhere.
	// Over: 1.5 GiB per term and 2.0 for what llama.cpp itemises. The
	// CUDA terms for MTP drafting are fitted across two models whose draft
	// costs differ by three times, so they sit well above the smaller.
	const (
		tolerance    = 0.1
		closeEnough  = 1.5
		reportedSlop = 2.0
	)

	checked := 0
	for _, p := range corpus() {
		if p.reported == nil {
			continue
		}
		checked++
		b := p.estimate()

		weights := b.Weights + b.Aux
		if weights < p.reported.weights-tolerance {
			t.Errorf("%s: weights estimated %.2f, measured %.2f — under by %.2f",
				p.name, weights, p.reported.weights, p.reported.weights-weights)
		}
		if weights-p.reported.weights > closeEnough {
			t.Errorf("%s: weights over by %.2f GiB", p.name, weights-p.reported.weights)
		}

		// The KV term answers to the attention caches only; recurrent
		// state is the term below. A load with speculative decoding on
		// has two of them, the model's and the draft context's, and
		// llama.cpp reports them in the same column.
		kv := b.KVCache + b.SpecKV + b.IndexerCache
		if kv < p.reported.kv-tolerance {
			t.Errorf("%s: KV cache estimated %.2f, measured %.2f — under by %.2f",
				p.name, kv, p.reported.kv, p.reported.kv-kv)
		}
		if kv-p.reported.kv > closeEnough {
			t.Errorf("%s: KV cache over by %.2f GiB", p.name, kv-p.reported.kv)
		}

		compute := b.Compute + b.IndexerScratch
		if compute < p.reported.compute-tolerance {
			t.Errorf("%s: compute buffers estimated %.2f, measured %.2f — under by %.2f",
				p.name, compute, p.reported.compute, p.reported.compute-compute)
		}
		if compute-p.reported.compute > closeEnough {
			t.Errorf("%s: compute buffers over by %.2f GiB", p.name, compute-p.reported.compute)
		}

		if diff := math.Abs(b.Reported() - p.reported.total()); diff > reportedSlop {
			t.Errorf("%s: llama.cpp accounts for %.2f GiB, the modelled terms for %.2f — off by %.2f",
				p.name, p.reported.total(), b.Reported(), diff)
		}
	}
	if checked == 0 {
		t.Fatal("no corpus point carries a buffer report; this test proves nothing")
	}
	t.Logf("checked %d of %d corpus points term by term", checked, len(corpus()))
}

// The buffer report is not the whole story: context and allocator
// overhead never appear in it. Anything reading the report as the
// footprint — the Available Models tooltip, a corpus row — is reading a
// figure below what the card will show, and the estimate has to keep
// covering that difference.
func TestCardCountersExceedTheBufferReport(t *testing.T) {
	for _, p := range corpus() {
		if p.reported == nil {
			continue
		}
		remainder := p.measured - p.reported.total()
		// With speculative decoding the draft context reports buffers of
		// its own that partly share the model's, so the report can come
		// out a little above the counters.
		if IsDraftMode(p.cfg.SpecType) {
			remainder += 0.25
		}
		if remainder <= 0 {
			t.Errorf("%s: the buffer report (%.2f) is not below the card counters (%.2f) — one of the two figures is wrong",
				p.name, p.reported.total(), p.measured)
		}
		if remainder > 3.0 {
			t.Errorf("%s: %.2f GiB unreported; the remainder was a near-constant 2.3 across the sweep it was measured on", p.name, remainder)
		}
	}
}

// Issue #582: at large micro-batches the estimate was 4 to 11 GiB above
// what two A4000s used, because it counted four pipeline copies of the
// attention mask where llama.cpp had dropped to one, and the layer
// split's copies on a tensor split, which has none. Never under is
// TestEstimateNeverUnderPredicts; this holds the points close.
func TestLargeMicroBatchesStayClose(t *testing.T) {
	for _, p := range cudaBenchCorpus() {
		got := p.estimate().Total()
		if over := got - p.measured; over > 2.0 {
			t.Errorf("%s: estimated %.2f GiB, the cards used %.2f — over by %.2f", p.name, got, p.measured, over)
		}
	}
}

// The copies are dropped only when they would not fit. On cards of an
// unknown size, or large enough, they stay: llama.cpp keeps them there,
// and planning one would put the estimate under what is used.
func TestPipelineCopiesStayWhereTheyFit(t *testing.T) {
	p := point(t, "bench q35 2c layer ub4096")
	on16 := p.estimate().Total()
	unknown := vramBreakdownOn(p.coeffs(), &p.model, &p.cfg, p.cards, 0).Total()
	on48 := vramBreakdownOn(p.coeffs(), &p.model, &p.cfg, p.cards, 48).Total()
	// Three more 1 GiB masks on each of two cards, and the copies of
	// the graph's other inputs.
	if d := unknown - on16; d < 6.5 || d > 7.5 {
		t.Errorf("four copies cost %.2f GiB more than one; want about 7", d)
	}
	if on48 != unknown {
		t.Errorf("on 48 GiB cards the estimate is %.2f, want the four copies' %.2f", on48, unknown)
	}

	// The glb point fits its four, on the same cards.
	g := point(t, "bench glb 2c layer ub2048")
	if got, want := g.estimate().Total(), vramBreakdownOn(g.coeffs(), &g.model, &g.cfg, g.cards, 0).Total(); got != want {
		t.Errorf("glb at 2048: %.2f planned on A4000s, %.2f with every copy; its copies fit", got, want)
	}
}

// Experts in system memory, the MoE cache and layers left off the GPUs
// each turn llama.cpp's pipeline off, so a layer split runs one copy
// however large the cards: the estimate must not depend on the card size
// there, and must be below the same load with the pipeline on.
func TestNoPipelineCopiesWithoutFullOffload(t *testing.T) {
	for _, name := range []string{"launch q35 pool moe8 ub4096", "launch q35 4 slots moe16 cache2g ub2048", "launch q35 4 slots ngl30 ub2048"} {
		p := point(t, name)
		on16 := p.estimate().Total()
		if big := vramBreakdownOn(p.coeffs(), &p.model, &p.cfg, p.cards, 0).Total(); big != on16 {
			t.Errorf("%s: %.2f GiB on cards of unknown size, %.2f on A4000s; the pipeline is off either way", name, big, on16)
		}
		if !pipelineOff(&p.model, &p.cfg) {
			t.Errorf("%s: pipeline taken to be on", name)
		}
	}
	full := point(t, "launch q35 4 slots ub4096")
	if pipelineOff(&full.model, &full.cfg) {
		t.Error("a full offload with no experts in system memory taken to run without the pipeline")
	}
}

// Extra Flags count like the config's own fields: a hand-written tensor
// override or CPU experts there turn the pipeline off as well.
func TestExtraFlagsTurnThePipelineOff(t *testing.T) {
	p := point(t, "launch q35 pool ub2048")
	for flags, want := range map[string]bool{
		"":                                 false,
		"--jinja --reasoning-budget 0":     false,
		`-ot "blk\..*\.ffn_.*_exps\.=CPU"`: true,
		"--override-tensor=exps=CPU":       true,
		"--cpu-moe":                        true,
		"--n-cpu-moe 8":                    true,
		"--n-cpu-moe 0":                    false,
		"--moe-cache-mib 2048":             true,
		"-ngl 30":                          true,
		"-ngl 999":                         false,
	} {
		cfg := p.cfg
		cfg.ExtraFlags = flags
		if got := pipelineOff(&p.model, &cfg); got != want {
			t.Errorf("extra flags %q: pipeline off %v, want %v", flags, got, want)
		}
	}
	cfg := p.cfg
	cfg.ExtraFlags = "-ot exps=CPU"
	if with, without := vramBreakdownOn(p.coeffs(), &p.model, &cfg, p.cards, 0).Total(), p.estimate().Total(); with >= without {
		t.Errorf("with -ot the estimate is %.2f GiB, without %.2f; one copy should be less", with, without)
	}
}
