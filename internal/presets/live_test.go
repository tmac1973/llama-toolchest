package presets

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/models"
)

// TestLiveFetchQwen38 exercises the real Unsloth docs and HuggingFace
// endpoints. Skipped unless PRESETS_LIVE_TEST=1 — it validates the format
// assumptions (llms.txt shape, docs table layout) against the live services,
// not code correctness.
func TestLiveFetchQwen38(t *testing.T) {
	if os.Getenv("PRESETS_LIVE_TEST") != "1" {
		t.Skip("set PRESETS_LIVE_TEST=1 to run against live services")
	}
	f := NewFetcher(t.TempDir(), os.Getenv("HF_TOKEN"))
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	m := &models.Model{
		ModelID:       "unsloth/Qwen3.8-27B-GGUF",
		BaseModelRepo: "Qwen/Qwen3.8-27B",
		SamplingPresets: []models.SamplingPreset{{
			Name: "default", Label: "Model-embedded default", Source: "gguf",
		}},
	}
	out := f.Fetch(ctx, m)
	t.Logf("presets: %d", len(out))
	for _, p := range out {
		t.Logf("  %s (%s) temp=%v top_p=%v top_k=%v min_p=%v presence=%v repeat=%v [%s]",
			p.Name, p.Source, deref(p.Temperature), deref(p.TopP), derefI(p.TopK),
			deref(p.MinP), deref(p.PresencePenalty), deref(p.RepeatPenalty), p.SourceURL)
	}
	if len(out) < 2 {
		t.Errorf("expected docs variants beyond the embedded default, got %d presets", len(out))
	}
}

func deref(v *float64) any {
	if v == nil {
		return nil
	}
	return *v
}

func derefI(v *int) any {
	if v == nil {
		return nil
	}
	return *v
}
