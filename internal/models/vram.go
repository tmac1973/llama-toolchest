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
	// Weight memory: file size is a good proxy for quantized weights in VRAM.
	return BytesToGiB(m.SizeBytes) + m.KVCacheGB(ctx, cfg.KVCacheQuant) + AuxFilesVRAMGB(cfg) + vramOverheadGB
}

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
