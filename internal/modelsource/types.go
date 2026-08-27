// Package modelsource holds the types shared by the model repositories
// llama-toolchest can search and download from (HuggingFace, ModelScope).
//
// They live here rather than in one provider's package so a second
// provider does not have to import the first just to describe a search
// result. The provider packages alias these names, so existing references
// like huggingface.ModelFile keep working and mean exactly this type.
package modelsource

// SearchResult is one model repository returned by a search.
type SearchResult struct {
	ID        string   `json:"id"`
	Author    string   `json:"author"`
	Downloads int      `json:"downloads"`
	Likes     int      `json:"likes"`
	Tags      []string `json:"tags"`
	License   string   `json:"license,omitempty"`
}

// File is a single GGUF file (or a grouped shard set) in a repository.
type File struct {
	Filename  string   `json:"filename"`
	Size      int64    `json:"size"`
	Quant     string   `json:"quant"`
	VRAMEstGB float64  `json:"vram_est_gb"`
	Shards    []string `json:"shards,omitempty"`    // all shard filenames if split; nil for single files
	IsMMProj  bool     `json:"is_mmproj,omitempty"` // true for vision projector files
}

// Detail holds one repository's GGUF files.
type Detail struct {
	ID    string `json:"id"`
	Files []File `json:"files"`
}
