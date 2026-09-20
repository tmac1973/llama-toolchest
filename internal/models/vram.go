package models

import (
	"fmt"
	"math"
	"os"
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
	bpe := kvBytesPerElem(kvCacheQuant)

	if m.KVFullPerTok > 0 || m.KVSWAPerTok > 0 {
		swaTokens := ctx
		if m.SlidingWindow > 0 && m.SlidingWindow < ctx {
			swaTokens = m.SlidingWindow
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

// Per-device coefficients, fitted to llama.cpp's own buffer report across
// eleven measured loads: four architectures, one and four cards, 3.4 to 99
// GiB of VRAM. See plan/ple-vram-findings.md for the corpus and the method,
// and vram_corpus_test.go for the points themselves.
//
// They are empirical, and honestly so: they come from one ROCm machine, and
// a different backend may allocate differently. The guardrail that makes
// that acceptable is the direction of error — every coefficient is set so
// the estimate lands at or above measured on the whole corpus. Telling
// someone a model fits when it does not is the failure worth avoiding.
const (
	// Graph scratch, per device. Linear in micro-batch, weakly in context.
	computeMiBPerUBatchTok = 0.2911
	computeMiBPerCtxTok    = 0.00090
	// A sparse-attention model scores every cached position against every
	// token of the micro-batch before it can take the top-k, and holds
	// about six tensors of that shape at once. This is the term that makes
	// such a model cost multiples of an ordinary one at the same context.
	indexerScratchCopies = 5.69
	// CUDA/HIP context and allocations llama.cpp does not itemise. Constant
	// per device across the corpus.
	vramPerDeviceOverheadGB = 0.85
	// What a layer split costs in graph scratch over a tensor-parallel
	// one, on top of the per-card figure above.
	//
	// Every point the coefficients were fitted on was split
	// tensor-parallel, where the cards act as one device and share one
	// set of buffers. A layer split gives each card its own, and
	// llama.cpp then runs the layers as a pipeline, keeping several
	// copies of the graph in flight so a card is not idle waiting for
	// the one before it. Its default is four copies.
	//
	// Four does not cover what was measured: the same model at the same
	// context and micro-batch took 4.99 GiB of graph scratch split by
	// layer over three NVIDIA cards against 0.96 GiB tensor-parallel
	// over four AMD ones. This is the measured ratio, and it stands on
	// that single pair — the backend differs as well as the split, so
	// part of it may not be the split at all. It is kept at the measured
	// figure rather than the explainable four because the estimate
	// decides whether a model is offered at a context it can load at,
	// and promising a fit that fails is the error worth avoiding.
	layerSplitComputeCopies = 6.5
	// A hybrid model's linear-attention layers keep a state buffer
	// instead of a KV cache. It does not grow with the context: the 27B
	// held the same 0.60 GiB across a sweep from 8,192 to 262,144
	// tokens, which is what makes a flat per-layer figure the right
	// shape. Split by layer it measured 1.75 GiB for the same model,
	// hence the second constant — one pair, like the compute factor
	// above.
	recurrentStateGiBPerLayer = 0.0125
	recurrentLayerSplitCopies = 2.9
)

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
	resident := m.SizeBytes - m.PLEBytes - m.TokenEmbdBytes
	if resident < 0 {
		resident = m.SizeBytes
	}
	onCPU := CPUWeightBytes(m, cfg)
	if onCPU > resident {
		onCPU = resident
	}
	b.Weights = BytesToGiB(resident - onCPU)
	b.CPURAM = BytesToGiB(onCPU)

	b.KVCache = m.KVCacheGB(ctx, cfg.KVCacheQuant)
	// llama.cpp keeps each layer's KV cache on the device that runs the
	// layer, so layers left on the CPU take their share of it to system
	// memory. Same "zero is not set" rule as the weights.
	if cfg.GPULayers > 0 && cfg.GPULayers < m.NLayers && m.NLayers > 0 {
		onGPU := b.KVCache * float64(cfg.GPULayers) / float64(m.NLayers)
		b.CPURAM += b.KVCache - onGPU
		b.KVCache = onGPU
	}

	// Graph scratch.
	ub := cfg.EffectiveUBatchSize()
	perCard := computeMiBPerUBatchTok*float64(ub) + computeMiBPerCtxTok*float64(ctx)
	b.Compute = float64(cards) * perCard / 1024 * layerSplitComputeFactor(cfg, cards)

	// Sparse attention: a key cache of its own, plus scratch that scales
	// with context times micro-batch on every device.
	if m.IndexerKeyLength > 0 {
		layers := m.AttnLayers
		if layers == 0 {
			layers = m.NLayers
		}
		b.IndexerCache = float64(layers) * float64(ctx) * float64(m.IndexerKeyLength) * 4 / (1024 * 1024 * 1024)
		b.IndexerScratch = float64(cards) * indexerScratchCopies * float64(ctx) * float64(ub) * 4 / (1024 * 1024 * 1024)
	}

	b.SpecKV = SpecKVCacheGB(m, cfg, ctx)

	// Hybrid models: the layers that are not attention layers keep a
	// recurrent state buffer instead of a KV cache.
	if m.AttnLayers > 0 && m.AttnLayers < m.NLayers {
		recurrent := float64(m.NLayers - m.AttnLayers)
		b.Recurrent = recurrent * recurrentStateGiBPerLayer
		if layerSplitComputeFactor(cfg, cards) > 1 {
			b.Recurrent *= recurrentLayerSplitCopies
		}
	}

	b.Aux = AuxFilesVRAMGB(cfg)
	b.Overhead = float64(cards) * vramPerDeviceOverheadGB
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

// layerSplitComputeFactor is how much the graph scratch is multiplied by
// on this placement: one for a single card or a tensor-parallel split,
// and layerSplitComputeCopies for a layer split over several cards.
//
// A split mode is only read when there is more than one card to split
// over. An empty mode is llama.cpp's default, which is a layer split.
func layerSplitComputeFactor(cfg *ModelConfig, cards int) float64 {
	if cards < 2 || cfg.SplitMode == "tensor" {
		return 1
	}
	return layerSplitComputeCopies
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
