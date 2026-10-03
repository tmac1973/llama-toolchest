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

// estimate is the point's estimate under its own backend's coefficients.
func (p corpusPoint) estimate() VRAMBreakdown {
	return vramBreakdownWith(p.coeffs(), &p.model, &p.cfg, p.cards)
}

// reportedTerms is one load's buffer report, GiB on the accelerators.
type reportedTerms struct {
	weights   float64
	kv        float64
	recurrent float64
	compute   float64 // compute and output buffers
}

// total is what llama.cpp accounts for. Always less than the card
// counters: context and allocator overhead are not in the report.
func (r reportedTerms) total() float64 {
	return r.weights + r.kv + r.recurrent + r.compute
}

// Sizes are in GiB where written as floats and converted below.
func gibBytes(f float64) int64 { return int64(f * gib) }

// corpus is every measured load, on both backends.
func corpus() []corpusPoint {
	out := append(rocmCorpus(), rocmLocalCorpus()...)
	return append(out, cudaCorpus()...)
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
	}
}

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
