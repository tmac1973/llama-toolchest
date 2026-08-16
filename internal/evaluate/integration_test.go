package evaluate

import (
	"context"
	"os"
	"testing"
	"time"
)

// An opt-in test that runs the REAL llama-perplexity against a real
// model and parses its real output.
//
// The unit tests above parse canned transcripts, which is why they all
// passed while every perplexity and Winogrande run failed in
// production: the canned blocks had no log prefix and the tool's actual
// output does. Anything pinned to another program's output format
// needs one test that reads that program's output rather than our idea
// of it.
//
// Skipped unless both paths are supplied, since it needs a GPU, a
// model, and a built binary:
//
//	LLAMA_TOOLCHEST_EVAL_BINARY=~/.local/share/llama-toolchest/builds/<id>/llama-perplexity \
//	LLAMA_TOOLCHEST_EVAL_MODEL=/path/to/model.gguf \
//	go test ./internal/evaluate/ -run TestRunAgainstRealBinary -v
func TestRunAgainstRealBinary(t *testing.T) {
	binary := os.Getenv("LLAMA_TOOLCHEST_EVAL_BINARY")
	model := os.Getenv("LLAMA_TOOLCHEST_EVAL_MODEL")
	if binary == "" || model == "" {
		t.Skip("set LLAMA_TOOLCHEST_EVAL_BINARY and LLAMA_TOOLCHEST_EVAL_MODEL to run this")
	}
	dataset := os.Getenv("LLAMA_TOOLCHEST_EVAL_DATASET")
	if dataset == "" {
		t.Skip("set LLAMA_TOOLCHEST_EVAL_DATASET to the wikitext-2 test file")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	// Two chunks: enough to produce a final estimate, cheap enough to
	// run on any machine that can load the model at all.
	res, err := Run(ctx, Spec{
		Binary:      binary,
		ModelPath:   model,
		Mode:        ModePerplexity,
		DatasetPath: dataset,
		Chunks:      2,
		Flags:       []string{"--n-gpu-layers", "999", "--threads", "8"},
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if res.Perplexity <= 0 {
		t.Errorf("Perplexity = %v, want a positive value", res.Perplexity)
	}
	if res.PerplexityErr <= 0 {
		t.Errorf("PerplexityErr = %v, want a positive value", res.PerplexityErr)
	}
	if res.Chunks != 2 {
		t.Errorf("Chunks = %d, want 2", res.Chunks)
	}
	t.Logf("PPL = %.4f ± %.5f over %d chunks", res.Perplexity, res.PerplexityErr, res.Chunks)
}
