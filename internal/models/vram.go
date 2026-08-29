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
)

// VRAMEstimateForConfigOn estimates VRAM for a model spread over devices
// cards. The count matters because graph scratch and driver overhead are
// per device, not per model: the same config across four cards costs four
// times the scratch of one.
func VRAMEstimateForConfigOn(m *Model, cfg *ModelConfig, cards int) float64 {
	if cards < 1 {
		cards = 1
	}
	ctx := cfg.ContextSize
	if ctx == 0 {
		ctx = m.ContextLength
	}

	// Weights, less the parts llama.cpp holds host-mapped rather than on a
	// device. Two tensors do that on every model measured: the per-layer
	// embedding table where one exists, and the input embedding table,
	// which is mapped even on models with no per-layer table at all.
	resident := m.SizeBytes - m.PLEBytes - m.TokenEmbdBytes
	if resident < 0 {
		resident = m.SizeBytes
	}
	gb := BytesToGiB(resident)

	gb += m.KVCacheGB(ctx, cfg.KVCacheQuant)

	// Graph scratch.
	ub := cfg.EffectiveUBatchSize()
	perCard := computeMiBPerUBatchTok*float64(ub) + computeMiBPerCtxTok*float64(ctx)
	gb += float64(cards) * perCard / 1024

	// Sparse attention: a key cache of its own, plus scratch that scales
	// with context times micro-batch on every device.
	if m.IndexerKeyLength > 0 {
		layers := m.AttnLayers
		if layers == 0 {
			layers = m.NLayers
		}
		gb += float64(layers) * float64(ctx) * float64(m.IndexerKeyLength) * 4 / (1024 * 1024 * 1024)
		gb += float64(cards) * indexerScratchCopies * float64(ctx) * float64(ub) * 4 / (1024 * 1024 * 1024)
	}

	gb += AuxFilesVRAMGB(cfg)
	gb += float64(cards) * vramPerDeviceOverheadGB
	return gb
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
	if cfg.SpecType == "draft" && cfg.DraftModelPath != "" {
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
