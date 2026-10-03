package models

import (
	"fmt"
	"math"
	"os"
	"strings"
	"sync/atomic"
)

const vramOverheadGB = 0.2 // fixed overhead for compute buffers, scratch space, etc.

// BytesToGiB converts bytes to gibibytes (binary, 1024³).
func BytesToGiB(b int64) float64 {
	return float64(b) / (1024 * 1024 * 1024)
}

// EstimateVRAM returns a rough VRAM estimate in GB based on file size alone.
// Used as a fallback when GGUF metadata isn't available.
func EstimateVRAM(sizeBytes int64) float64 {
	return float64(sizeBytes)*1.1/(1024*1024*1024) + vramOverheadGB
}

// kvBytesPerElem returns the bytes consumed per KV-cache element for a given
// cache quantization (quantized caches carry a small per-block scale).
func kvBytesPerElem(kvCacheQuant string) float64 {
	switch kvCacheQuant {
	case "q4_0":
		return 0.5625 // 4.5 bits = 0.5625 bytes (4 bits + 0.5 bit block scale)
	case "q8_0":
		return 1.0625 // 8.5 bits (8 bits + 0.5 bit block scale)
	default:
		return 2.0 // f16
	}
}

// KVCacheGB returns the estimated KV cache size in GB at the given context size.
//
// When the model carries precomputed per-token KV factors (KVFullPerTok /
// KVSWAPerTok), it uses them — these capture per-layer grouped-query attention
// and sliding-window attention, which matter enormously for architectures like
// gemma-4 (most layers cache only the sliding window, and global layers use far
// fewer KV heads). Otherwise it falls back to the uniform full-attention
// estimate.
func (m *Model) KVCacheGB(ctx int, kvCacheQuant string) float64 {
	if ctx == 0 {
		ctx = m.ContextLength
	}
	if ctx == 0 {
		ctx = 2048
	}
	swa := ctx
	if m.SlidingWindow > 0 && m.SlidingWindow < ctx {
		swa = m.SlidingWindow
	}
	return m.kvCacheGiBFor(ctx, kvCacheQuant, swa)
}

// defaultServerSlots is llama-server's own --parallel when the config sets
// none: four sequences sharing the cache.
const defaultServerSlots = 4

// swaCells is how many positions llama.cpp's sliding-window cache holds:
// the window for every sequence, plus a micro-batch, padded to 256 — not
// the window alone. gemma-4-12B at 32K held 1.91 GiB against 0.81
// estimated from the window, and 1024 x 4 + 512 cells predicts it exactly.
func swaCells(m *Model, cfg *ModelConfig, ctx, ub int) int {
	if m.SlidingWindow <= 0 {
		return ctx
	}
	seqs := cfg.Parallel
	if seqs <= 0 {
		seqs = defaultServerSlots
	}
	n := m.SlidingWindow*seqs + ub
	n = (n + 255) / 256 * 256
	if ctx > 0 && n > ctx {
		n = ctx
	}
	return n
}

// EmbeddingsTied reports whether the model's output layer reuses its input
// embedding: recorded from the tensor table where it was read, and assumed
// for the gemma family otherwise, which ties throughout.
func (m *Model) EmbeddingsTied() bool {
	return m.OutputTied || strings.HasPrefix(m.Arch, "gemma")
}

// kvCacheGiBFor is the attention cache with swaTokens positions cached on
// sliding-window layers.
func (m *Model) kvCacheGiBFor(ctx int, kvCacheQuant string, swaTokens int) float64 {
	if ctx == 0 {
		ctx = m.ContextLength
	}
	if ctx == 0 {
		ctx = 2048
	}
	bpe := kvBytesPerElem(kvCacheQuant)

	if m.KVFullPerTok > 0 || m.KVSWAPerTok > 0 {
		if swaTokens <= 0 || swaTokens > ctx {
			swaTokens = ctx
		}
		// PerTok sums already include both K and V across their layers.
		elems := float64(ctx)*float64(m.KVFullPerTok) + float64(swaTokens)*float64(m.KVSWAPerTok)
		return elems * bpe / (1024 * 1024 * 1024)
	}

	// Legacy fallback for records parsed before per-token factors existed.
	return EstimateKVCacheGB(m.NLayers, m.NKVHead, m.NHead, m.NEmbd, ctx, kvCacheQuant)
}

// EstimateKVCacheGB returns the estimated KV cache size in GB assuming uniform
// full attention. Used as a fallback by Model.KVCacheGB when richer per-layer
// metadata isn't available.
//
// Formula: 2 (K+V) × n_layers × n_kv_head × head_dim × ctx × bytes_per_element
func EstimateKVCacheGB(nLayers, nKVHead, nHead, nEmbd, contextSize int, kvCacheQuant string) float64 {
	if nLayers == 0 || nEmbd == 0 {
		return 0
	}

	// If KV heads not specified, fall back to full attention (n_kv_head = n_head)
	kvHeads := nKVHead
	if kvHeads == 0 {
		kvHeads = nHead
	}
	if kvHeads == 0 {
		return 0
	}

	// Head dimension
	headDim := nEmbd
	if nHead > 0 {
		headDim = nEmbd / nHead
	}

	// Default context if not set
	ctx := contextSize
	if ctx == 0 {
		ctx = 2048
	}

	// 2 (K + V) × layers × kv_heads × head_dim × context × bytes
	totalBytes := 2.0 * float64(nLayers) * float64(kvHeads) * float64(headDim) * float64(ctx) * kvBytesPerElem(kvCacheQuant)

	return totalBytes / (1024 * 1024 * 1024)
}

// VRAMEstimateForConfig returns the total estimated VRAM for a model with
// the given configuration. This is the primary function used by the UI.
func VRAMEstimateForConfig(m *Model, cfg *ModelConfig) float64 {
	if m.NLayers == 0 || m.NEmbd == 0 {
		// No GGUF metadata — fall back to rough estimate
		return EstimateVRAM(m.SizeBytes)
	}
	// ContextSize 0 means "Model Default" in the UI; llama-server resolves
	// that to the model's trained context length at load time, so estimate
	// against that, not the bare 2048 fallback in KVCacheGB.
	ctx := cfg.ContextSize
	if ctx == 0 {
		ctx = m.ContextLength
	}
	return VRAMEstimateForConfigOn(m, cfg, DeviceCountForConfig(cfg, 0))
}

// The estimate's coefficients: what a llama.cpp backend allocates beyond
// the weights and caches, which the file cannot say. They are empirical,
// fitted to llama.cpp's own buffer report and the card counters on
// measured loads (vram_corpus_test.go holds the points; see
// plan/ple-vram-findings.md for the method), and each set is held to the
// same rule: on every point of its backend the estimate lands at or above
// what the hardware used. Telling someone a model fits when it does not
// is the failure worth avoiding.
//
// The backends differ enough to need separate sets. On the same model,
// context and micro-batch, ROCm's graph scratch and per-card overhead ran
// several times CUDA's, and Qwen3.8-Flash-Next's sparse-attention scratch,
// 24 GiB on ROCm, did not appear on CUDA at all.

// computeCoeffs is graph scratch on one device, in GiB:
// base + perUB·ubatch + perCtx·context + perUBCtx·ubatch·context.
type computeCoeffs struct {
	base, perUB, perCtx, perUBCtx float64
}

func (c computeCoeffs) at(ub, ctx int) float64 {
	u, x := float64(ub), float64(ctx)
	return c.base + c.perUB*u + c.perCtx*x + c.perUBCtx*u*x
}

type vramCoefficients struct {
	// single is the graph scratch of a model on one device; tensor is per
	// device of a tensor-parallel split, where the cards act as one; layer
	// is per device of a layer split, where llama.cpp runs the cards as a
	// pipeline with several copies of the graph in flight.
	single, tensor, layer computeCoeffs
	// indexerScratchCopies scales a sparse-attention model's ranking
	// scratch (context x micro-batch, f32) on every device.
	indexerScratchCopies float64
	// indexerCacheAtKType stores the sparse-attention key cache at the KV
	// cache's K type rather than f32.
	indexerCacheAtKType bool
	// Overhead llama.cpp does not itemise: on the first device and on
	// each further one.
	overheadFirst, overheadExtra float64
	// A hybrid model's linear-attention state, per recurrent layer, and
	// how many copies a layer split keeps.
	recurrentPerLayer, recurrentLayerSplitCopies float64
	// recurrentPerDraftToken: with speculative decoding the state is kept
	// once per drafted token as well, so a rejected draft can be undone.
	recurrentPerDraftToken bool
	// quantKVScratch: with a quantized KV cache, each device holds one
	// attention layer's K and V converted to f16 at the full context.
	quantKVScratch bool
	// expertOffloadScratch: with experts in system memory, the first
	// device holds one layer's experts while it works on them.
	expertOffloadScratch bool
	// specCompute is the draft context's graph scratch, on top of the
	// model's own, when speculative decoding is on.
	specCompute computeCoeffs
}

// rocmCoefficients come from two ROCm machines: a 4x Radeon AI PRO R9700
// box split tensor-parallel (eleven loads, four architectures), and one
// RX 9070 XT (eighteen loads, seven models, 2026-10-03). They are also
// the set used when the backend is not known, being the more cautious one.
var rocmCoefficients = vramCoefficients{
	// One card, the RX 9070 XT: the same as CUDA's to within noise — 0.10
	// GiB at 8K and 0.19-0.25 at 128K with a 512 micro-batch, 0.38-0.51 at
	// 32K with 2048.
	single: computeCoeffs{base: 0.04, perUB: 1.9e-4, perUBCtx: 1.75e-9},
	// Tensor-parallel, per card: linear in micro-batch, weakly in context,
	// 0.2911 MiB per micro-batch token and 0.0009 MiB per context token.
	tensor: computeCoeffs{perUB: 0.2911 / 1024, perCtx: 0.00090 / 1024},
	// A layer split at 6.5 times that: the measured ratio of one pair (the
	// same 27B split by layer over three NVIDIA cards against
	// tensor-parallel over four AMD ones), kept although the backend
	// differed as well as the split, because promising a fit that fails is
	// the error worth avoiding.
	layer: computeCoeffs{perUB: 6.5 * 0.2911 / 1024, perCtx: 6.5 * 0.00090 / 1024},
	// About six tensors of context x micro-batch at once while ranking.
	indexerScratchCopies: 5.69,
	// 0.21-0.37 GiB on the RX 9070 XT alone; about 0.6 a card more across
	// the R9700s, which the further cards' figure still covers.
	overheadFirst: 0.40,
	overheadExtra: 0.85,
	// Flat with context: the 27B held 0.60 GiB from 8,192 to 262,144
	// tokens. 1.75 GiB split by layer, hence the copies.
	recurrentPerLayer:         0.0125,
	recurrentLayerSplitCopies: 2.9,
	// The costs found on CUDA, seen on the RX 9070 XT: an 8-bit cache
	// added 0.48 GiB on Qwen3.5-4B at 128K (0.50 predicted), expert
	// offload 0.26 GiB on the 35B-A3B, and MTP the recurrent state seven
	// times over. MTP's draft scratch is smaller than CUDA's: +0.15 to
	// +0.22 GiB at 32K and +0.31 at 128K.
	recurrentPerDraftToken: true,
	quantKVScratch:         true,
	expertOffloadScratch:   true,
	specCompute:            computeCoeffs{base: 0.30, perUBCtx: 3.0e-9},
}

// cudaCoefficients come from compute2 (3x RTX A4000, build v0.5.0-cuda):
// 29 loads of nine models (dense, hybrid, mixture-of-experts, sliding
// window, sparse attention), on one card and split by layer over three,
// at 8K to 128K context and micro-batches of 512 and 2048, with and
// without an 8-bit KV cache, experts in system memory and MTP.
var cudaCoefficients = vramCoefficients{
	// One card: 0.10 GiB at 8K and 0.21 at 128K with a 512 micro-batch,
	// 0.51 at 32K with 2048 — the context term grows with the
	// micro-batch, so it is a product.
	single: computeCoeffs{base: 0.03, perUB: 1.9e-4, perUBCtx: 1.75e-9},
	// Not measured tensor-parallel on CUDA: the layer split's figure, the
	// larger of the two kinds measured, stands in.
	tensor: computeCoeffs{base: 0.10, perUB: 2.95e-4, perUBCtx: 7.95e-9},
	// Split by layer, per card: 0.19 GiB at 8K to 0.66 at 128K with a 512
	// micro-batch, 1.13 at 32K and 2.47 at 128K with 2048.
	// The base covers granite-30B, whose cards held 0.37 GiB at 32K where
	// the 8B's held 0.25: wider feed-forward layers take more.
	layer: computeCoeffs{base: 0.10, perUB: 2.95e-4, perUBCtx: 7.95e-9},
	// Flash-Next's cards with no other scratch held 0.30 GiB at 32K and
	// 0.92 at 128K, all of it accounted for by the terms above.
	indexerScratchCopies: 0,
	indexerCacheAtKType:  true,
	// 0.50-0.51 GiB on one card; 0.55-0.59 in all on three.
	overheadFirst: 0.53,
	overheadExtra: 0.04,
	// 0.0118 GiB per layer on the 27B, 0.008 on the 35B-A3B, the same on
	// one card as split by layer; seven times that with MTP drafting six
	// tokens.
	recurrentPerLayer:         0.0125,
	recurrentLayerSplitCopies: 1,
	recurrentPerDraftToken:    true,
	// +0.16 to +0.24 GiB per card at 128K with an 8-bit cache, about one
	// attention layer's K and V at f16.
	quantKVScratch: true,
	// +1.5 GiB on Flash-Next's first card (one layer's experts are 1.49)
	// and +0.47 on the 35B-A3B's (0.68).
	expertOffloadScratch: true,
	// MTP's draft context: +0.4 to +0.64 GiB at 32K, and +0.38 (27B) to
	// +1.25 (35B-A3B) at 128K. With the recurrent copies below, every MTP
	// point stays at least 0.8 GiB above what it used.
	specCompute: computeCoeffs{base: 0.45, perUBCtx: 1.0e-8},
}

var activeVRAMBackend atomic.Value // string

// SetVRAMBackend chooses the coefficients the estimate uses: the llama.cpp
// backend of the build that will run ("cuda", "rocm", ...). Anything
// without a fitted set of its own uses ROCm's, the more cautious.
func SetVRAMBackend(backend string) {
	activeVRAMBackend.Store(backend)
}

func coefficientsFor(backend string) *vramCoefficients {
	if backend == "cuda" {
		return &cudaCoefficients
	}
	return &rocmCoefficients
}

func activeCoefficients() *vramCoefficients {
	b, _ := activeVRAMBackend.Load().(string)
	return coefficientsFor(b)
}

// specDraftTokens is how many tokens a draft holds when the config does
// not say: llama.cpp's default --spec-draft-n-max.
const specDraftTokens = 16

// VRAMBreakdown is the estimate term by term, in GiB. The terms are named
// for what llama.cpp reports, so an estimate can be checked against a
// measured load piece by piece rather than only in total — which is what
// a total alone cannot do: the estimate this replaced was accurate on one
// model purely because a 27 GiB over-count of weights cancelled a 20 GiB
// under-count of everything else.
//
// Which measured figure each term answers to:
//
//	Weights, Aux    model buffers on a device
//	KVCache, SpecKV  the attention caches: the model's, and the draft
//	                 context's when speculative decoding is on
//	Recurrent       the linear-attention state buffers of a hybrid model
//	IndexerCache    the sparse-attention key cache
//	Compute         compute and output buffers
//	IndexerScratch  the rest of the compute buffers on a sparse model
//	Overhead        nothing — it is the remainder llama.cpp never itemises,
//	                the gap between its own accounting and the card counters
type VRAMBreakdown struct {
	Weights float64
	Aux     float64
	KVCache float64
	// SpecKV is the KV cache of the speculative draft context, which is
	// separate from the model's own and is not quantized.
	SpecKV float64
	// Recurrent is the state buffer of a hybrid model's linear-attention
	// layers. Unlike a KV cache it does not grow with the context.
	Recurrent      float64
	IndexerCache   float64
	Compute        float64
	IndexerScratch float64
	Overhead       float64

	// CPURAM is not part of the GPU total: it is what the config keeps in
	// system memory instead — the weights and KV cache of layers left off
	// the GPU (gpu_layers below the layer count), and expert layers kept
	// on the CPU (cpu_moe).
	CPURAM float64
}

// Total is the figure the UI shows.
func (b VRAMBreakdown) Total() float64 {
	return b.Weights + b.Aux + b.KVCache + b.SpecKV + b.Recurrent + b.IndexerCache + b.Compute + b.IndexerScratch + b.Overhead
}

// Reported is the part of the estimate llama.cpp itemises while loading,
// which is what a measured buffer report can be compared against.
func (b VRAMBreakdown) Reported() float64 { return b.Total() - b.Overhead }

// VRAMEstimateForConfigOn estimates VRAM for a model spread over devices
// cards. The count matters because graph scratch and driver overhead are
// per device, not per model: the same config across four cards costs four
// times the scratch of one.
func VRAMEstimateForConfigOn(m *Model, cfg *ModelConfig, cards int) float64 {
	return VRAMBreakdownForConfigOn(m, cfg, cards).Total()
}

// VRAMBreakdownForConfigOn is VRAMEstimateForConfigOn with its terms kept
// apart. Same arithmetic; the split exists so each term can be measured.
func VRAMBreakdownForConfigOn(m *Model, cfg *ModelConfig, cards int) VRAMBreakdown {
	return vramBreakdownWith(activeCoefficients(), m, cfg, cards)
}

func vramBreakdownWith(co *vramCoefficients, m *Model, cfg *ModelConfig, cards int) VRAMBreakdown {
	if cards < 1 {
		cards = 1
	}
	ctx := cfg.ContextSize
	if ctx == 0 {
		ctx = m.ContextLength
	}

	var b VRAMBreakdown

	// Weights, less the parts llama.cpp holds host-mapped rather than on a
	// device. Two tensors do that on every model measured: the per-layer
	// embedding table where one exists, and the input embedding table,
	// which is mapped even on models with no per-layer table at all.
	//
	// A model with tied embeddings is the exception for the second: its
	// output layer reuses the input table, and llama.cpp keeps a copy of
	// it on the GPU for that (gemma-4: 0.4-0.7 GiB more than estimated).
	resident := m.SizeBytes - m.PLEBytes
	if !m.EmbeddingsTied() {
		resident -= m.TokenEmbdBytes
	}
	if resident < 0 {
		resident = m.SizeBytes
	}
	// A built-in MTP layer is loaded only when MTP drafting is on; without
	// it llama.cpp leaves that layer off the GPU (Qwen3.6-35B-A3B: 0.72
	// GiB, one layer in 41).
	if m.NextNLayers > 0 && m.NLayers > m.NextNLayers && cfg.SpecType != "draft-mtp" {
		resident -= resident * int64(m.NextNLayers) / int64(m.NLayers)
	}
	onCPU := CPUWeightBytes(m, cfg)
	if onCPU > resident {
		onCPU = resident
	}
	b.Weights = BytesToGiB(resident - onCPU)
	b.CPURAM = BytesToGiB(onCPU)

	ub := cfg.EffectiveUBatchSize()
	b.KVCache = m.kvCacheGiBFor(ctx, cfg.KVCacheQuant, swaCells(m, cfg, ctx, ub))
	// llama.cpp keeps each layer's KV cache on the device that runs the
	// layer, so layers left on the CPU take their share of it to system
	// memory. Same "zero is not set" rule as the weights.
	if cfg.GPULayers > 0 && cfg.GPULayers < m.NLayers && m.NLayers > 0 {
		onGPU := b.KVCache * float64(cfg.GPULayers) / float64(m.NLayers)
		b.CPURAM += b.KVCache - onGPU
		b.KVCache = onGPU
	}

	// Graph scratch, per device.
	layerSplit := isLayerSplit(cfg, cards)
	perCard := co.single.at(ub, ctx)
	switch {
	case layerSplit:
		perCard = co.layer.at(ub, ctx)
	case cards >= 2:
		perCard = co.tensor.at(ub, ctx)
	}
	// Not on a tensor-parallel split: the only such quantized-cache point
	// (ROCm, Flash-Next) is inside the fit of the tensor coefficients.
	tensorSplit := cards >= 2 && !layerSplit
	if co.quantKVScratch && !tensorSplit && cfg.KVCacheQuant != "" && cfg.KVCacheQuant != "f16" && m.KVFullPerTok > 0 && m.AttnLayers > 0 {
		perCard += float64(m.KVFullPerTok) / float64(m.AttnLayers) * float64(ctx) * 2 / (1024 * 1024 * 1024)
	}
	b.Compute = float64(cards) * perCard
	if co.expertOffloadScratch && cfg.CPUMoE > 0 && m.ExpertLayers > 0 {
		b.Compute += BytesToGiB(m.ExpertBytes / int64(m.ExpertLayers))
	}
	if IsDraftMode(cfg.SpecType) {
		b.Compute += co.specCompute.at(ub, ctx)
	}

	// Sparse attention: a key cache of its own, plus scratch that scales
	// with context times micro-batch on every device.
	if m.IndexerKeyLength > 0 {
		layers := m.AttnLayers
		if layers == 0 {
			layers = m.NLayers
		}
		bpe := 4.0
		if co.indexerCacheAtKType {
			bpe = kvBytesPerElem(cfg.KVCacheQuant)
		}
		b.IndexerCache = float64(layers) * float64(ctx) * float64(m.IndexerKeyLength) * bpe / (1024 * 1024 * 1024)
		b.IndexerScratch = float64(cards) * co.indexerScratchCopies * float64(ctx) * float64(ub) * 4 / (1024 * 1024 * 1024)
	}

	b.SpecKV = SpecKVCacheGB(m, cfg, ctx)

	// Hybrid models: the layers that are not attention layers keep a
	// recurrent state buffer instead of a KV cache.
	if m.AttnLayers > 0 && m.AttnLayers < m.NLayers {
		recurrent := float64(m.NLayers - m.AttnLayers)
		b.Recurrent = recurrent * co.recurrentPerLayer
		if layerSplit {
			b.Recurrent *= co.recurrentLayerSplitCopies
		}
		if co.recurrentPerDraftToken && IsDraftMode(cfg.SpecType) {
			n := cfg.DraftMax
			if n <= 0 {
				n = specDraftTokens
			}
			b.Recurrent *= float64(n + 1)
		}
	}

	b.Aux = AuxFilesVRAMGB(cfg)
	b.Overhead = co.overheadFirst + float64(cards-1)*co.overheadExtra
	return b
}

// specDraftLayers is how many layers of KV cache the draft context holds
// when the count is not known from the file: the drafter's own layer,
// and the one it predicts from.
const specDraftLayers = 2

// SpecKVCacheGB estimates the KV cache of the speculative draft context,
// which llama.cpp allocates separately from the model's own.
//
// It is the term that was missing when a 27B model with built-in MTP was
// planned at its full 262,144-token context: everything else fitted, and
// the load failed at "failed to allocate buffer for kv cache" while
// creating the draft context. The cache is worth nothing at a short
// context and gigabytes at a long one, which is exactly where a planner
// has to get it right.
//
// Two things make it larger than its share of layers suggests. It is not
// quantized — the model's cache-type setting does not reach it, so it is
// f16 whatever the main cache is — and it spans the drafter's layers
// plus the one it predicts from.
func SpecKVCacheGB(m *Model, cfg *ModelConfig, ctx int) float64 {
	if m == nil || cfg == nil || !IsDraftMode(cfg.SpecType) || m.NLayers <= 0 {
		return 0
	}
	if ctx <= 0 {
		ctx = m.ContextLength
	}
	if ctx <= 0 {
		return 0
	}
	layers := specDraftLayers
	if n := m.NextNLayers; n > 0 && n+1 > layers {
		layers = n + 1
	}
	if layers > m.NLayers {
		layers = m.NLayers
	}
	// Full attention at f16, deliberately. The draft context caches
	// every position it drafts over, so a share of the model's own
	// cache would understate it badly on a model whose layers mostly
	// cache a sliding window — which is where the context is longest
	// and the term matters most.
	//
	// The model's own per-token rate is the best measure of a layer
	// available: it counts elements per token across the attention
	// layers, so dividing by them gives one layer's rate directly,
	// without having to guess a head size from the embedding width.
	if m.KVFullPerTok > 0 && m.AttnLayers > 0 {
		perLayer := float64(m.KVFullPerTok) / float64(m.AttnLayers)
		return perLayer * float64(layers) * float64(ctx) * kvBytesPerElem("") / (1024 * 1024 * 1024)
	}
	return EstimateKVCacheGB(layers, m.NKVHead, m.NHead, m.NEmbd, ctx, "")
}

// isLayerSplit reports whether a placement splits the model by layer:
// more than one card, and not tensor-parallel. An empty mode is
// llama.cpp's default, which is a layer split.
func isLayerSplit(cfg *ModelConfig, cards int) bool {
	return cards >= 2 && cfg.SplitMode != "tensor"
}

// CPUWeightBytes estimates the model weights a config keeps in system
// memory rather than on a GPU: the expert tensors of the first CPUMoE
// layers, and the layers gpu_layers leaves off the GPU. Both are
// approximations spread evenly over layers, which matches how uniform
// transformer layers are; the estimate stays conservative because what is
// not moved is still counted on the GPU.
func CPUWeightBytes(m *Model, cfg *ModelConfig) int64 {
	if m.NLayers <= 0 {
		return 0
	}
	var moved int64

	// --n-cpu-moe N counts layers from 0, dense leading layers included,
	// so only the part of that range that carries experts moves.
	var expertMoved int64
	if cfg.CPUMoE > 0 && m.ExpertBytes > 0 && m.ExpertLayers > 0 {
		n := cfg.CPUMoE - m.ExpertLayerFirst
		if n > m.ExpertLayers {
			n = m.ExpertLayers
		}
		if n > 0 {
			expertMoved = m.ExpertBytes / int64(m.ExpertLayers) * int64(n)
		}
	}
	moved += expertMoved

	// gpu_layers counts from the top: llama.cpp offloads the last N
	// layers, so the rest stay on the CPU. 999 (or anything at or above
	// the layer count) offloads all of them. Zero is left out: many
	// callers build a config with only the fields they care about, where
	// a zero means "not set" rather than "CPU only", and counting it as a
	// full move would report a model as fitting that does not.
	if cfg.GPULayers > 0 && cfg.GPULayers < m.NLayers {
		body := m.SizeBytes - m.PLEBytes - m.TokenEmbdBytes - expertMoved
		if body > 0 {
			moved += body / int64(m.NLayers) * int64(m.NLayers-cfg.GPULayers)
		}
	}
	return moved
}

// PLEAutoMinBytes mirrors auto_lazy_min_size in llama.cpp's model loader:
// under the default auto mode, only tables above this size are read on
// demand. Smaller ones stay resident, because the per-row read latency
// costs more than the memory is worth on a small model.
const PLEAutoMinBytes = 4 * 1024 * 1024 * 1024

// AuxFilesVRAMGB sums the on-disk sizes of auxiliary GGUFs that load into VRAM
// alongside the main model: the vision projector (mmproj), a separate MTP
// drafter head, and a speculative draft model. Each is counted only when it is
// both downloaded (file present) and activated (its enable toggle on, or, for a
// draft model, draft mode selected). File size approximates the loaded weights;
// a draft model's own small KV cache is not separately modeled.
func AuxFilesVRAMGB(cfg *ModelConfig) float64 {
	var gb float64
	if cfg.MmprojPath != "" && !cfg.MmprojDisabled {
		gb += fileSizeGB(cfg.MmprojPath)
	}
	if cfg.MtpPath != "" && !cfg.MtpDisabled {
		gb += fileSizeGB(cfg.MtpPath)
	}
	// Every draft method except draft-mtp loads its drafter — a smaller
	// model of the same family, or a converted EAGLE-3 / DFlash / DSpark
	// head — from DraftModelPath. draft-mtp is excluded because its head
	// comes from MtpPath and is already counted just above, so each
	// method contributes exactly once.
	//
	// The n-gram assist is deliberately absent: it matches text already in
	// the context and loads no file, so it adds nothing here.
	if IsDraftMode(cfg.SpecType) && cfg.SpecType != "draft-mtp" && cfg.DraftModelPath != "" {
		gb += fileSizeGB(cfg.DraftModelPath)
	}
	return gb
}

// fileSizeGB returns a file's size in GB, or 0 if it can't be stat'd (e.g. not
// downloaded yet).
func fileSizeGB(path string) float64 {
	if fi, err := os.Stat(path); err == nil {
		return BytesToGiB(fi.Size())
	}
	return 0
}

// VRAMFitLabel returns a human-readable label for how a model fits
// relative to the available VRAM. perGPU is the size of one GPU in GB,
// numGPUs is how many are available. For tensor parallelism, estimatedGB
// is automatically divided by numberProcessors.
// Returns: "fits" (single GPU), "2 GPU", "3 GPU", etc., or "too_large".
func VRAMFitLabel(estimatedGB float64, perGPU float64, numGPUs int, numberProcessors int) string {
	if perGPU <= 0 || numGPUs <= 0 {
		return ""
	}

	// For tensor parallelism, divide estimated VRAM by number of processors
	// since tensors are split across all GPUs
	if numberProcessors > 0 && numberProcessors < numGPUs {
		numGPUs = numberProcessors
	}
	totalVRAM := perGPU * float64(numGPUs)
	needed := int(math.Ceil(estimatedGB / perGPU))
	if needed <= 0 {
		needed = 1
	}

	if estimatedGB > totalVRAM {
		return "too_large"
	}
	if needed == 1 {
		return "fits"
	}
	return fmt.Sprintf("%d GPU", needed)
}

// FormatVRAM formats a VRAM estimate (in GiB) as a human-readable string.
func FormatVRAM(gb float64) string {
	if gb < 1 {
		return formatFloat(gb*1024, 0) + " MiB"
	}
	return formatFloat(gb, 1) + " GiB"
}

func formatFloat(f float64, decimals int) string {
	p := math.Pow(10, float64(decimals))
	return trimTrailingZeros(math.Round(f*p) / p)
}

func trimTrailingZeros(f float64) string {
	s := math.Floor(f)
	if f == s {
		return fmt.Sprintf("%.0f", f)
	}
	return fmt.Sprintf("%.1f", f)
}
