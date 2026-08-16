package benchmark

import (
	"reflect"
	"testing"
)

func vizRun(id string, gen, prompt, ttft float64, sweep map[string]string) BenchmarkRun {
	return BenchmarkRun{
		ID: id, ModelName: "Qwen3.5-9B", Quant: "IQ4_NL", Preset: "internal-quick",
		BuildProfile: "vulkan", SweepValues: sweep,
		Config:  ConfigSnapshot{GPULayers: 999, ContextSize: 8192, Threads: 8},
		Summary: &BenchmarkSummary{AvgGenTokPerSec: gen, AvgPromptTokPerSec: prompt, AvgTTFTMs: ttft},
	}
}

// The axes come from what the runs vary on, and a numeric parameter has
// to be ordered by value. Sorted as text, "1024, 128, 2048, 512" turns a
// smooth curve into what looks like noise.
func TestVisualizationOffersSortedNumericAxes(t *testing.T) {
	var runs []BenchmarkRun
	for _, b := range []string{"1024", "8192", "2048", "512"} {
		runs = append(runs, vizRun("r"+b, 90, 900, 100, map[string]string{"batch_size": b}))
	}
	v := BuildVisualization(runs)

	if len(v.Dimensions) != 1 {
		t.Fatalf("dimensions = %+v, want just the swept one", v.Dimensions)
	}
	d := v.Dimensions[0]
	if d.Name != "batch" {
		t.Errorf("dimension name = %q, want the display name", d.Name)
	}
	if !d.Numeric {
		t.Error("batch sizes not recognised as numbers")
	}
	if want := []string{"512", "1024", "2048", "8192"}; !reflect.DeepEqual(d.Values, want) {
		t.Errorf("values = %v, want %v (ascending by value, not by text)", d.Values, want)
	}
}

// A dimension every run agrees on is not an axis — plotting against it
// draws every point in one line.
func TestVisualizationDropsConstantDimensions(t *testing.T) {
	runs := []BenchmarkRun{
		vizRun("a", 90, 900, 100, map[string]string{"ubatch_size": "64"}),
		vizRun("b", 95, 950, 110, map[string]string{"ubatch_size": "128"}),
	}
	v := BuildVisualization(runs)
	for _, d := range v.Dimensions {
		if d.Name == "model" || d.Name == "quant" || d.Name == "threads" {
			t.Errorf("offered %q as an axis, but every run has the same value", d.Name)
		}
	}
	if len(v.Dimensions) != 1 || v.Dimensions[0].Name != "micro-batch" {
		t.Errorf("dimensions = %+v, want only the one that varies", v.Dimensions)
	}
}

// A text parameter must not be treated as numeric, or the page will
// space its values as though they had distances between them.
func TestVisualizationMarksCategoricalAxes(t *testing.T) {
	a := vizRun("a", 90, 900, 100, nil)
	a.Config.KVCacheQuant = "q8_0"
	b := vizRun("b", 95, 950, 110, nil)
	b.Config.KVCacheQuant = "q4_0"

	v := BuildVisualization([]BenchmarkRun{a, b})
	var found bool
	for _, d := range v.Dimensions {
		if d.Name != "KV cache" {
			continue
		}
		found = true
		if d.Numeric {
			t.Error("KV cache types treated as numbers")
		}
	}
	if !found {
		t.Errorf("KV cache not offered as an axis: %+v", v.Dimensions)
	}
}

// Runs with no timings cannot go on a throughput chart, and dropping
// them quietly would plot fewer points than the user selected.
func TestVisualizationCountsRunsItCannotPlot(t *testing.T) {
	runs := []BenchmarkRun{
		vizRun("a", 90, 900, 100, map[string]string{"ubatch_size": "64"}),
		vizRun("b", 95, 950, 110, map[string]string{"ubatch_size": "128"}),
		{ID: "cap", ModelName: "Qwen3.5-9B", Preset: "hellaswag-quick",
			Eval: &EvalScores{Mode: "hellaswag", Accuracy: 78}},
		{ID: "failed", ModelName: "Qwen3.5-9B", Preset: "perplexity-quick", Status: StatusFailed},
	}
	v := BuildVisualization(runs)
	if len(v.Points) != 2 {
		t.Errorf("points = %d, want 2", len(v.Points))
	}
	if v.Skipped != 2 {
		t.Errorf("skipped = %d, want 2 (the capability run and the failure)", v.Skipped)
	}
}

// Each point carries every metric and its position on every dimension,
// plus the same labels the compare view uses — one definition of what
// makes these runs different, serving both views.
func TestVisualizationPointsCarryMetricsAndLabels(t *testing.T) {
	runs := []BenchmarkRun{
		vizRun("a", 85.6, 765, 278, map[string]string{"batch_size": "1024", "ubatch_size": "64"}),
		vizRun("b", 100.5, 1450, 147, map[string]string{"batch_size": "1024", "ubatch_size": "256"}),
	}
	v := BuildVisualization(runs)

	var p VizPoint
	for _, q := range v.Points {
		if q.RunID == "b" {
			p = q
		}
	}
	if p.Metrics["gen"] != 100.5 || p.Metrics["prompt"] != 1450 || p.Metrics["ttft"] != 147 {
		t.Errorf("metrics = %v", p.Metrics)
	}
	if p.Dims["micro-batch"] != "256" || p.Dims["batch"] != "1024" {
		t.Errorf("dims = %v, want the swept values under their display names", p.Dims)
	}
	if p.Label == "" || p.Detail == "" {
		t.Error("point carries no label or tooltip")
	}
	// Every metric the page offers must be present on every point, or a
	// chart will plot undefined.
	for _, m := range v.Metrics {
		if _, ok := p.Metrics[m.Key]; !ok {
			t.Errorf("point has no value for the offered metric %q", m.Key)
		}
	}
}
