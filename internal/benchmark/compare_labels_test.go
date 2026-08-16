package benchmark

import (
	"strings"
	"testing"
)

// sweepRun builds a completed run with a summary and sweep values.
func sweepRun(id string, gen float64, sweep map[string]string) BenchmarkRun {
	return BenchmarkRun{
		ID: id, ModelName: "Qwen3.5-9B", Quant: "IQ4_NL", Preset: "internal-quick",
		BuildProfile: "vulkan", SweepValues: sweep,
		Summary: &BenchmarkSummary{AvgGenTokPerSec: gen, AvgPromptTokPerSec: gen * 10},
	}
}

// The case the compare view exists for: one model, one build, one
// preset, swept across a grid. Model, quant and build are identical on
// every run, so a label built from them names the winner and the loser
// the same thing — which is what it used to do.
func TestLabelsNameTheSweptParameters(t *testing.T) {
	runs := []BenchmarkRun{
		sweepRun("a", 85.6, map[string]string{"batch_size": "1024", "ubatch_size": "64"}),
		sweepRun("b", 100.5, map[string]string{"batch_size": "1024", "ubatch_size": "256"}),
		sweepRun("c", 77.6, map[string]string{"batch_size": "4096", "ubatch_size": "512"}),
	}
	labels := BuildRunLabels(runs)

	seen := map[string]bool{}
	for _, r := range runs {
		l := labels[r.ID].Short
		if seen[l] {
			t.Errorf("two runs share the label %q — the winner cannot be told from the loser", l)
		}
		seen[l] = true
		if strings.Contains(l, "Qwen3.5-9B") {
			t.Errorf("label %q repeats the model name, which is the same on every run", l)
		}
	}
	// The swept parameters are named, using the same words the rest of
	// the interface uses rather than the raw field names.
	if got := labels["b"].Short; !strings.Contains(got, "batch 1024") || !strings.Contains(got, "micro-batch 256") {
		t.Errorf("winner label = %q, want it to name both swept values", got)
	}
}

// Comparing different models at one setting should name the models, not
// the settings they share.
func TestLabelsNameWhateverActuallyVaries(t *testing.T) {
	a := sweepRun("a", 50, nil)
	a.ModelName, a.Quant = "Qwen3.5-4B", "Q4_K_XL"
	b := sweepRun("b", 30, nil)
	b.ModelName, b.Quant = "Qwen3.5-9B", "IQ4_NL"

	labels := BuildRunLabels([]BenchmarkRun{a, b})
	if !strings.Contains(labels["a"].Short, "Qwen3.5-4B") {
		t.Errorf("label = %q, want the model name", labels["a"].Short)
	}
	if strings.Contains(labels["a"].Short, "internal-quick") {
		t.Errorf("label = %q names the preset, which both runs share", labels["a"].Short)
	}
}

// When the runs really are identical repeats there is nothing to choose
// between them, so the label falls back to the identity instead of
// being blank.
func TestLabelsFallBackWhenNothingVaries(t *testing.T) {
	runs := []BenchmarkRun{sweepRun("a", 90, nil), sweepRun("b", 91, nil)}
	labels := BuildRunLabels(runs)
	for _, r := range runs {
		if labels[r.ID].Short == "" {
			t.Fatal("empty bar label")
		}
		if !strings.Contains(labels[r.ID].Short, "Qwen3.5-9B") {
			t.Errorf("fallback label = %q, want the model name", labels[r.ID].Short)
		}
	}
}

// Every run must end up with its own name even when that takes more
// dimensions than a label would normally carry.
func TestLabelsAreAlwaysUnique(t *testing.T) {
	var runs []BenchmarkRun
	for _, kv := range []string{"", "q8_0"} {
		for _, ub := range []string{"64", "128"} {
			for _, fa := range []bool{true, false} {
				r := sweepRun("r"+kv+ub+onOff(fa), 50, map[string]string{"ubatch_size": ub})
				r.Config.KVCacheQuant = kv
				r.Config.FlashAttention = fa
				r.Config.ContextSize = 4096
				runs = append(runs, r)
			}
		}
	}
	labels := BuildRunLabels(runs)
	seen := map[string]bool{}
	for _, r := range runs {
		l := labels[r.ID].Short
		if seen[l] {
			t.Errorf("duplicate label %q across %d runs", l, len(runs))
		}
		seen[l] = true
	}
	if len(seen) != len(runs) {
		t.Errorf("%d distinct labels for %d runs", len(seen), len(runs))
	}
}

// The tooltip states the whole configuration, including settings the
// runs agree on — it is where someone checks a value they are unsure
// of, so filtering it to differences would defeat the purpose.
func TestFullLabelStatesEverything(t *testing.T) {
	r := sweepRun("a", 100.5, map[string]string{"batch_size": "1024", "ubatch_size": "256"})
	r.Config.ContextSize = 131072
	r.Config.Threads = 8
	full := BuildRunLabels([]BenchmarkRun{r, sweepRun("b", 90, nil)})["a"].Full

	for _, want := range []string{
		"model: Qwen3.5-9B", "quant: IQ4_NL", "preset: internal-quick",
		"batch: 1024", "micro-batch: 256", "context: 131072", "threads: 8",
		"generation: 100.5 tok/s",
	} {
		if !strings.Contains(full, want) {
			t.Errorf("tooltip is missing %q:\n%s", want, full)
		}
	}
}

// The bar charts average each run over whatever prompt lengths it
// measured, and prompt speed climbs steeply with prompt length, so a
// comparison spanning different size sets needs saying.
func TestMixedPromptSizesDetected(t *testing.T) {
	one := sweepRun("a", 90, nil)
	one.Summary.PerSize = []SizeSummary{{PromptTokens: 512, TGMean: 90}}
	same := sweepRun("b", 91, nil)
	same.Summary.PerSize = []SizeSummary{{PromptTokens: 512, TGMean: 91}}
	many := sweepRun("c", 80, nil)
	many.Summary.PerSize = []SizeSummary{{PromptTokens: 128}, {PromptTokens: 512}, {PromptTokens: 2048}}

	if c := BuildComparison([]BenchmarkRun{one, same}); c.MixedPromptSizes {
		t.Error("runs measured at the same prompt length flagged as mixed")
	}
	if c := BuildComparison([]BenchmarkRun{one, many}); !c.MixedPromptSizes {
		t.Error("runs measured at different prompt lengths not flagged")
	}
}

// A selected run that produced nothing used to vanish from both the
// chart and the table, so a comparison could quietly hold fewer runs
// than the user picked.
func TestNoResultRunsAreCollected(t *testing.T) {
	ok := sweepRun("a", 90, nil)
	failed := BenchmarkRun{ID: "b", ModelName: "Qwen3.5-9B", Preset: "perplexity-quick",
		Status: StatusFailed, Error: "binary missing"}
	scored := BenchmarkRun{ID: "c", ModelName: "Qwen3.5-9B", Preset: "hellaswag-quick",
		Eval: &EvalScores{Mode: "hellaswag", Accuracy: 78}}

	c := BuildComparison([]BenchmarkRun{ok, failed, scored})
	if len(c.NoResultRuns) != 1 || c.NoResultRuns[0].ID != "b" {
		t.Errorf("NoResultRuns = %+v, want just the failed run", c.NoResultRuns)
	}
	if c.BestGenRunID != "a" {
		t.Errorf("BestGenRunID = %q, want a", c.BestGenRunID)
	}
}
