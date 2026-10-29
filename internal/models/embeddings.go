package models

import "fmt"

// llama.cpp's pooling types ({arch}.pooling_type in a GGUF file).
const (
	PoolingNone = 0
	PoolingMean = 1
	PoolingCLS  = 2
	PoolingLast = 3
	PoolingRank = 4 // a reranker: one score per query and document pair
)

// encoderArchs are architectures llama.cpp loads for embeddings and
// reranking, not for chat.
var encoderArchs = map[string]bool{
	"bert": true, "modern-bert": true, "nomic-bert": true, "nomic-bert-moe": true,
	"neo-bert": true, "jina-bert-v2": true, "jina-bert-v3": true, "t5encoder": true,
	"eurobert": true, "gemma-embedding": true, "gemma-embedding2": true,
}

// IsEncoderArch reports whether llama.cpp loads arch only for embeddings
// or reranking. Their HuggingFace repos are often tagged as text models.
func IsEncoderArch(arch string) bool { return encoderArchs[arch] }

// IsEmbedding reports whether the file is an embedding model. The pooling
// type decides when the file has one: converted embedding models carry
// it, chat models do not, and a model built on a chat architecture
// (Qwen3-Embedding is a qwen3) has nothing else to tell it apart. A
// reranker has a pooling type too, but is not served as an embedding
// model. Without the key, an encoder architecture is enough.
func (meta *GGUFMeta) IsEmbedding() bool {
	switch meta.PoolingType {
	case PoolingMean, PoolingCLS, PoolingLast:
		return true
	case PoolingRank:
		return false
	}
	return IsEncoderArch(meta.Architecture)
}

// EmbeddingDefaultContext is the context EmbeddingLaunchConfig gives a
// model that reads each input in one micro-batch, when the config leaves
// the context at the model's default. The micro-batch has to be as large,
// and the compute buffer grows faster than both: EmbeddingGemma 2 on one
// RTX A4000 needs 1.1 GiB at 8192, 7.3 GiB at 32768, and 22.5 GiB (no
// load) at 65536. Its own default context of 262144 would ask for 1 TiB.
// A context above the micro-batch gains nothing for these models, as no
// input may be longer than the micro-batch, and still costs memory.
const EmbeddingDefaultContext = 8192

// EmbeddingLaunchConfig returns cfg as llama-server must get it for the
// embedding model m, or cfg itself when nothing changes.
//
// The context is capped at the model's trained context: an encoder has
// no position past it. And unless the model pools from the last token,
// llama-server reads each input in one micro-batch and rejects an input
// longer than that ("input is too large to process"), so the micro-batch
// is raised to the context unless the config sets one. Its default of 512
// would otherwise limit every input to 512 tokens, whatever the context.
// For those models a context left at the model's default is at most
// EmbeddingDefaultContext; a context set on purpose is followed.
//
// Like MoESplitConfig, it is applied each time the flags are written and
// the VRAM is estimated, so the saved config and the form are unchanged.
func EmbeddingLaunchConfig(cfg *ModelConfig, m *Model) *ModelConfig {
	oneBatch := m.PoolingType != PoolingLast
	ctx := cfg.ContextSize
	if ctx <= 0 || (m.ContextLength > 0 && ctx > m.ContextLength) {
		ctx = m.ContextLength
	}
	if cfg.ContextSize <= 0 && oneBatch {
		ctx = min(ctx, EmbeddingDefaultContext)
	}
	raiseUB := oneBatch && cfg.UBatchSize <= 0
	if ctx <= 0 || (ctx == cfg.ContextSize && !raiseUB) {
		return cfg
	}
	out := *cfg
	out.ContextSize = ctx
	if raiseUB {
		out.UBatchSize = ctx
		if out.EffectiveBatchSize() < ctx {
			out.BatchSize = ctx
		}
	}
	return &out
}

// EmbeddingModelPreset defines a curated embedding model available for one-click download.
type EmbeddingModelPreset struct {
	Name     string `json:"name"`
	Repo     string `json:"repo"`     // HuggingFace repo ID
	Filename string `json:"filename"` // GGUF filename in the repo
	SizeMB   int    `json:"size_mb"`  // Approximate size
	Dims     int    `json:"dims"`     // Embedding dimensions
	Context  int    `json:"context"`  // Longest input the model was trained on, in tokens
	License  string `json:"license"`
	Desc     string `json:"desc"` // Short description
}

// SizeLabel is the download size for the card: MB below 1 GB, GB above.
func (p EmbeddingModelPreset) SizeLabel() string {
	if p.SizeMB >= 1000 {
		return fmt.Sprintf("%.1f GB", float64(p.SizeMB)/1000)
	}
	return fmt.Sprintf("%d MB", p.SizeMB)
}

// CuratedEmbeddingModels returns the embedding models offered on the
// Download Models tab, smallest first within each family. Every entry
// was checked against the repo on HuggingFace (file name, size, and the
// dimensions and context in the file's header); see issue #421 for the
// models left out and why.
func CuratedEmbeddingModels() []EmbeddingModelPreset {
	return []EmbeddingModelPreset{
		{
			Name:     "EmbeddingGemma 2",
			Repo:     "ggml-org/embeddinggemma-2-GGUF",
			Filename: "embeddinggemma-2-Q8_0.gguf",
			SizeMB:   310,
			Dims:     768,
			Context:  262144,
			License:  "Apache-2.0",
			Desc:     "Google, 2026. Many languages, long inputs, 271M params",
		},
		{
			Name:     "EmbeddingGemma 300M",
			Repo:     "ggml-org/embeddinggemma-300M-GGUF",
			Filename: "embeddinggemma-300M-Q8_0.gguf",
			SizeMB:   334,
			Dims:     768,
			Context:  2048,
			License:  "Gemma",
			Desc:     "Google, 2025. Many languages, widely used, 300M params",
		},
		{
			Name:     "Qwen3-Embedding-0.6B",
			Repo:     "Qwen/Qwen3-Embedding-0.6B-GGUF",
			Filename: "Qwen3-Embedding-0.6B-Q8_0.gguf",
			SizeMB:   639,
			Dims:     1024,
			Context:  32768,
			License:  "Apache-2.0",
			Desc:     "Strong retrieval in many languages and code, 0.6B params",
		},
		{
			Name:     "Qwen3-Embedding-4B",
			Repo:     "Qwen/Qwen3-Embedding-4B-GGUF",
			Filename: "Qwen3-Embedding-4B-Q4_K_M.gguf",
			SizeMB:   2497,
			Dims:     2560,
			Context:  40960,
			License:  "Apache-2.0",
			Desc:     "Higher quality, needs a GPU with room, 4B params",
		},
		{
			Name:     "Qwen3-Embedding-8B",
			Repo:     "Qwen/Qwen3-Embedding-8B-GGUF",
			Filename: "Qwen3-Embedding-8B-Q4_K_M.gguf",
			SizeMB:   4677,
			Dims:     4096,
			Context:  40960,
			License:  "Apache-2.0",
			Desc:     "Highest quality of the open models, 8B params",
		},
		{
			Name:     "nomic-embed-text-v2-moe",
			Repo:     "nomic-ai/nomic-embed-text-v2-moe-GGUF",
			Filename: "nomic-embed-text-v2-moe.Q8_0.gguf",
			SizeMB:   512,
			Dims:     768,
			Context:  512,
			License:  "Apache-2.0",
			Desc:     "Many languages, mixture of experts, 475M params",
		},
		{
			Name:     "nomic-embed-text-v1.5",
			Repo:     "nomic-ai/nomic-embed-text-v1.5-GGUF",
			Filename: "nomic-embed-text-v1.5.Q8_0.gguf",
			SizeMB:   142,
			Dims:     768,
			Context:  2048,
			License:  "Apache-2.0",
			Desc:     "English, small and fast, named in many guides, 137M params",
		},
		{
			Name:     "bge-m3",
			Repo:     "gpustack/bge-m3-GGUF",
			Filename: "bge-m3-Q8_0.gguf",
			SizeMB:   635,
			Dims:     1024,
			Context:  8192,
			License:  "MIT",
			Desc:     "BAAI. Many languages, well tested, 567M params",
		},
	}
}
