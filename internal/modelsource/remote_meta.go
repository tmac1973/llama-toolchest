package modelsource

import (
	"context"
	"errors"
	"net/http"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/models"
)

// metaProbeBudget bounds one metadata probe. Nearly every GGUF reaches its
// tokenizer within the first few kilobytes, so the first 64 KiB request
// is normally the only one. The budget is for a file whose writer
// scattered its model keys through the tokenizer (seen on a popular
// repository, with block_count behind a 4.8 MB token list): it is read
// on rather than misdescribed. 16 MiB covers the largest metadata
// section measured, gemma-4's 15.8 MB.
const metaProbeBudget = 16 * 1024 * 1024

// ErrNoModelMeta means a file's header was read but does not describe a
// model the estimate can use: no architecture, or no layer count.
var ErrNoModelMeta = errors.New("the file's header does not describe a model")

// ProbeMeta reads the model description of a GGUF on a model host,
// stopping at the tokenizer (see models.ParseGGUFMetaOnly). For a split
// file, pass the first shard: it holds all of the metadata.
//
// The result has no tensor-table sizes. DerivedFor works them out for
// each file of the repository from this one read.
func ProbeMeta(ctx context.Context, client *http.Client, token, url string, size int64) (*models.GGUFMeta, error) {
	if client == nil {
		client = &http.Client{Timeout: 30 * time.Second}
	}
	r := newRangeReader(ctx, client, url, token, size, probeChunk, metaProbeBudget)
	meta, err := models.ParseGGUFMetaOnly(r)
	if err != nil {
		return nil, err
	}
	// The parse stops quietly at a read error, so an unreachable or
	// budget-exhausted file shows up here as a description with no model.
	if meta.Architecture == "" || meta.NLayers == 0 {
		return nil, ErrNoModelMeta
	}
	return meta, nil
}
