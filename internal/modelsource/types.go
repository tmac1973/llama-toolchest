// Package modelsource holds the types shared by the model repositories
// llama-toolchest can search and download from (HuggingFace, ModelScope).
//
// They live here rather than in one provider's package so a second
// provider does not have to import the first just to describe a search
// result. The provider packages alias these names, so existing references
// like huggingface.ModelFile keep working and mean exactly this type.
package modelsource

import "context"

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
	Shards    []string `json:"shards,omitempty"` // all shard filenames if split; nil for single files
	// ShardSizes parallels Shards. Grouping sums the shards into Size,
	// but the individual lengths are what a header probe needs, so keep
	// them rather than paying a HEAD per shard to ask again.
	ShardSizes []int64 `json:"shard_sizes,omitempty"`
	// StreamedBytes is the part of this download that llama.cpp reads
	// from disk on demand and never makes resident (the per-layer
	// embedding table). Zero when there is none, or when it was not
	// measured — see StreamProbed.
	StreamedBytes int64 `json:"streamed_bytes,omitempty"`
	StreamProbed  bool  `json:"stream_probed,omitempty"`
	IsMMProj      bool  `json:"is_mmproj,omitempty"` // true for vision projector files
}

// Detail holds one repository's GGUF files.
type Detail struct {
	ID    string `json:"id"`
	Files []File `json:"files"`
}

// Source ids for the model repositories llama-toolchest can use. They are
// persisted on download records and model registry entries, so the string
// values are part of the on-disk format and must not change.
const (
	SourceHuggingFace = "hf"
	SourceModelScope  = "modelscope"
)

// KnownSource reports whether id names a source this build understands.
// The empty string is HuggingFace: every record written before a second
// source existed came from there.
func KnownSource(id string) bool {
	return id == "" || id == SourceHuggingFace || id == SourceModelScope
}

// NormalizeSource maps an empty or unrecognized source id onto
// HuggingFace, the default.
func NormalizeSource(id string) string {
	if id == SourceModelScope {
		return SourceModelScope
	}
	return SourceHuggingFace
}

// Client is what a model source must provide for the browse-and-download
// flow. Both the HuggingFace and ModelScope clients implement it, which
// is what lets the API layer pick one by source id and leave every
// downstream handler, template and struct untouched.
type Client interface {
	Search(ctx context.Context, query string) ([]SearchResult, error)
	GetModel(ctx context.Context, modelID string) (*Detail, error)
	// ModelURL is the human-facing repository page, or empty when the id
	// is not a linkable owner/name pair.
	ModelURL(modelID string) string
	// DownloadURL locates one file within a repository.
	DownloadURL(modelID, filename string) string
}
