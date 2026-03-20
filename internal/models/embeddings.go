package models

// EmbeddingModelPreset defines a curated embedding model available for one-click download.
type EmbeddingModelPreset struct {
	Name     string `json:"name"`
	Repo     string `json:"repo"`     // HuggingFace repo ID
	Filename string `json:"filename"` // GGUF filename in the repo
	SizeMB   int    `json:"size_mb"`  // Approximate size
	Dims     int    `json:"dims"`     // Embedding dimensions
	Desc     string `json:"desc"`     // Short description
}

// CuratedEmbeddingModels returns a list of popular, high-quality embedding models.
func CuratedEmbeddingModels() []EmbeddingModelPreset {
	return []EmbeddingModelPreset{
		{
			Name:     "nomic-embed-text-v1.5",
			Repo:     "nomic-ai/nomic-embed-text-v1.5-GGUF",
			Filename: "nomic-embed-text-v1.5.Q8_0.gguf",
			SizeMB:   142,
			Dims:     768,
			Desc:     "Strong general-purpose, 137M params",
		},
		{
			Name:     "bge-large-en-v1.5",
			Repo:     "CompendiumLabs/bge-large-en-v1.5-gguf",
			Filename: "bge-large-en-v1.5-q8_0.gguf",
			SizeMB:   355,
			Dims:     1024,
			Desc:     "High quality English, 335M params",
		},
		{
			Name:     "mxbai-embed-large-v1",
			Repo:     "ChristianAzinn/mxbai-embed-large-v1-gguf",
			Filename: "mxbai-embed-large-v1.Q8_0.gguf",
			SizeMB:   358,
			Dims:     1024,
			Desc:     "Top-tier retrieval, 335M params",
		},
		{
			Name:     "snowflake-arctic-embed-l",
			Repo:     "ChristianAzinn/snowflake-arctic-embed-l-gguf",
			Filename: "snowflake-arctic-embed-l-Q8_0.GGUF",
			SizeMB:   357,
			Dims:     1024,
			Desc:     "Snowflake multilingual, 335M params",
		},
	}
}
