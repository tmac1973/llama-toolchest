package models

// RerankModelPreset defines a curated reranking model available for
// one-click download.
type RerankModelPreset struct {
	Name     string `json:"name"`
	Repo     string `json:"repo"`     // HuggingFace repo ID
	Filename string `json:"filename"` // GGUF filename in the repo
	SizeMB   int    `json:"size_mb"`  // Approximate size
	Context  int    `json:"context"`  // Longest query plus document the model was trained on, in tokens
	License  string `json:"license"`
	Desc     string `json:"desc"` // Short description
}

// SizeLabel is the download size for the card: MB below 1 GB, GB above.
func (p RerankModelPreset) SizeLabel() string { return sizeLabelMB(p.SizeMB) }

// CuratedRerankModels returns the reranking models offered on the
// Download Models tab, smallest first within each family. Every entry was
// checked against its file: the size, the context in the header, a
// classification head in the tensor table (see Model.RerankHeadMissing),
// and a correct ranking from scripts/test-rerank.sh on llama.cpp b11868.
//
// Left out: Qwen publishes no GGUF of its rerankers, and ggml-org only the
// 0.6B. Of the Qwen3-Reranker-4B conversions, mradermacher's and
// dean2155's (June 2025) predate llama.cpp's reranker conversion and have
// no classification head, so their scores are meaningless.
func CuratedRerankModels() []RerankModelPreset {
	return []RerankModelPreset{
		{
			Name:     "bge-reranker-v2-m3",
			Repo:     "gpustack/bge-reranker-v2-m3-GGUF",
			Filename: "bge-reranker-v2-m3-Q8_0.gguf",
			SizeMB:   636,
			Context:  8192,
			License:  "Apache-2.0",
			Desc:     "BAAI. Many languages, the usual RAG reranker, 568M params",
		},
		{
			Name:     "jina-reranker-v2-base-multilingual",
			Repo:     "gpustack/jina-reranker-v2-base-multilingual-GGUF",
			Filename: "jina-reranker-v2-base-multilingual-Q8_0.gguf",
			SizeMB:   305,
			Context:  1024,
			License:  "CC-BY-NC-4.0",
			Desc:     "Jina AI. Many languages, small and fast, non-commercial use only, 278M params",
		},
		{
			Name:     "Qwen3-Reranker-0.6B",
			Repo:     "ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF",
			Filename: "qwen3-reranker-0.6b-q8_0.gguf",
			SizeMB:   639,
			Context:  40960,
			License:  "Apache-2.0",
			Desc:     "Many languages and code, long inputs, 0.6B params",
		},
		{
			Name:     "Qwen3-Reranker-4B",
			Repo:     "Voodisss/Qwen3-Reranker-4B-GGUF-llama_cpp",
			Filename: "Qwen3-Reranker-4B.Q8_0.gguf",
			SizeMB:   4280,
			Context:  40960,
			License:  "Apache-2.0",
			Desc:     "Higher quality, needs a GPU with room, 4B params",
		},
	}
}
