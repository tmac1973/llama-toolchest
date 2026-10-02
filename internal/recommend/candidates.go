package recommend

import (
	"context"
	"errors"
	"strings"
	"sync"

	"github.com/tmac1973/llama-toolchest/internal/huggingface"
	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/modelsource"
)

// Hub is what the engine needs from HuggingFace. The server provides it
// over its HuggingFace client and header cache; tests provide a fake.
type Hub interface {
	ListGGUF(ctx context.Context, q huggingface.ListQuery) ([]huggingface.ListedModel, error)
	// Files lists a repository's GGUF files, shards grouped.
	Files(ctx context.Context, repo string) ([]modelsource.File, error)
	// Meta reads the model description from one file's header.
	Meta(ctx context.Context, repo string, f modelsource.File) (*models.GGUFMeta, error)
}

const (
	// broadLimit is the page size of the popularity queries; publisherLimit
	// that of each trusted publisher's newest uploads.
	broadLimit     = 100
	publisherLimit = 50
	// listWorkers bounds the list queries running at once.
	listWorkers = 4
)

// listQueries are the candidate pool: the most downloaded and the
// trending text and image-and-text models, and each trusted publisher's
// newest uploads. Trusted publishers release GGUFs of a new model within
// days, so their newest uploads are where "Newest" finds its models
// before the downloads catch up.
func listQueries() []huggingface.ListQuery {
	var qs []huggingface.ListQuery
	for _, tag := range []string{"text-generation", "image-text-to-text"} {
		for _, sort := range []string{"downloads", "trendingScore"} {
			qs = append(qs, huggingface.ListQuery{PipelineTag: tag, Sort: sort, Limit: broadLimit})
		}
	}
	for _, p := range trustedPublishers {
		qs = append(qs, huggingface.ListQuery{Author: p, Sort: "createdAt", Limit: publisherLimit})
	}
	return qs
}

// fetchCandidates runs every list query and merges the results by repo,
// the first occurrence winning. A query that fails is skipped; only when
// every one fails is the result an error.
func fetchCandidates(ctx context.Context, hub Hub) ([]huggingface.ListedModel, error) {
	qs := listQueries()
	results := make([][]huggingface.ListedModel, len(qs))
	errs := make([]error, len(qs))
	sem := make(chan struct{}, listWorkers)
	var wg sync.WaitGroup
	for i, q := range qs {
		wg.Add(1)
		go func() {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			results[i], errs[i] = hub.ListGGUF(ctx, q)
		}()
	}
	wg.Wait()

	var out []huggingface.ListedModel
	seen := map[string]bool{}
	failed := 0
	for i := range qs {
		if errs[i] != nil {
			failed++
			continue
		}
		for _, r := range results[i] {
			if !seen[r.ID] {
				seen[r.ID] = true
				out = append(out, r)
			}
		}
	}
	if failed == len(qs) {
		return nil, errors.Join(errs...)
	}
	return out, nil
}

const (
	// minParams keeps very small models out. They are real models, but on
	// any machine with a GPU they would fill Fastest; anyone who wants one
	// can search for it.
	minParams = 1_000_000_000
	// minDownloads keeps out test uploads and abandoned experiments,
	// except from trusted publishers, whose newest uploads start at zero.
	minDownloads = 50
)

// generativeTags are the pipeline tags of models that write text. A repo
// without a tag is kept: many quantizers leave it empty.
var generativeTags = map[string]bool{"": true, "text-generation": true, "image-text-to-text": true}

// encoderArchs are architectures llama.cpp loads for embeddings and
// reranking, not for chat. Their repos are often tagged as text models.
var encoderArchs = map[string]bool{
	"bert": true, "modern-bert": true, "nomic-bert": true, "nomic-bert-moe": true,
	"neo-bert": true, "jina-bert-v2": true, "jina-bert-v3": true, "t5encoder": true,
	"eurobert": true,
}

// clipArch is what HuggingFace's GGUF summary says when it happened to
// describe the repo's image reader rather than the model. The summary's
// parameter count is then the image reader's too.
const clipArch = "clip"

// draftNames mark repos holding draft models for speculative decoding,
// which are not models to chat with. "MTP" is not among them: an -MTP-
// repo is usually a full model with its own built-in draft layers.
var draftNames = []string{"draft", "eagle", "dflash"}

// candidateVerdict sorts a listed repo.
type candidateVerdict int

const (
	candidateKeep        candidateVerdict = iota
	candidateDrop                         // not a model to recommend; dropped silently
	candidateUnsupported                  // a model the active build cannot load
)

func judgeCandidate(r huggingface.ListedModel, p Profile) candidateVerdict {
	if r.Private || !generativeTags[r.PipelineTag] || r.GGUF == nil || r.GGUF.Architecture == "" {
		return candidateDrop
	}
	arch := r.GGUF.Architecture
	if encoderArchs[arch] || models.IsMTPHeadArch(arch) {
		return candidateDrop
	}
	lower := strings.ToLower(r.ID)
	for _, d := range draftNames {
		if strings.Contains(lower, d) {
			return candidateDrop
		}
	}
	if arch != clipArch && r.GGUF.Total > 0 && r.GGUF.Total < minParams {
		return candidateDrop
	}
	if r.Downloads < minDownloads && publisherRank(r.Author) < 0 {
		return candidateDrop
	}
	if arch != clipArch && !p.supports(arch) {
		return candidateUnsupported
	}
	return candidateKeep
}
