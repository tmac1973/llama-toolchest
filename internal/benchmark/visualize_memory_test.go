package benchmark

import "testing"

// vizMemRun is vizRun (visualize_test.go) with a footprint attached.
func vizMemRun(id string, gen float64, mem *MemorySnapshot) BenchmarkRun {
	return BenchmarkRun{
		ID: id, ModelName: "M-4B", Quant: "Q4_K_XL", Preset: "internal-standard",
		SweepValues: map[string]string{"ubatch_size": id},
		Summary: &BenchmarkSummary{
			AvgGenTokPerSec: gen, AvgPromptTokPerSec: 100, AvgTTFTMs: 10,
		},
		Memory: mem,
	}
}

func metricKeys(d VizData) map[string]bool {
	out := map[string]bool{}
	for _, m := range d.Metrics {
		out[m.Key] = true
	}
	return out
}

// The point of plotting memory beside speed: a setting that buys
// throughput usually spends memory, and the chart is where that trade
// becomes visible.
func TestVisualizationOffersMemoryWhenRunsCarryIt(t *testing.T) {
	d := BuildVisualization([]BenchmarkRun{
		vizMemRun("128", 40, &MemorySnapshot{GPUGiB: 23, WeightsGiB: 20, KVGiB: 2, ComputeGiB: 1, HostGiB: 1.5, Cards: 4}),
		vizMemRun("512", 55, &MemorySnapshot{GPUGiB: 25, WeightsGiB: 20, KVGiB: 2, ComputeGiB: 3, HostGiB: 1.5, Cards: 4}),
	})

	keys := metricKeys(d)
	for _, want := range []string{"mem_gpu", "mem_weights", "mem_kv", "mem_compute", "mem_host"} {
		if !keys[want] {
			t.Errorf("metric %q is not offered", want)
		}
	}
	for _, m := range d.Metrics {
		if m.Key == "mem_gpu" && m.HigherIsBetter {
			t.Error("more memory is not better; a heatmap would be coloured backwards")
		}
	}
	if len(d.Points) != 2 {
		t.Fatalf("points = %d; want 2", len(d.Points))
	}
	if got := d.Points[0].Metrics["mem_compute"]; got != 1 {
		t.Errorf("mem_compute = %v; want 1", got)
	}
}

// A run measured before this existed has timings and no footprint. It
// must not be plotted at zero, which would read as "used no memory".
func TestVisualizationLeavesMemoryAbsentRatherThanZero(t *testing.T) {
	d := BuildVisualization([]BenchmarkRun{
		vizMemRun("128", 40, &MemorySnapshot{GPUGiB: 23, Cards: 1}),
		vizMemRun("512", 55, nil),
	})

	if _, ok := d.Points[1].Metrics["mem_gpu"]; ok {
		t.Error("a run with no measurement carries a memory value")
	}
	if _, ok := d.Points[1].Metrics["gen"]; !ok {
		t.Error("the run should still be plottable on the speed metrics")
	}
}

// Offering a metric that plots nothing is worse than not offering it.
func TestVisualizationHidesMemoryWhenNothingWasMeasured(t *testing.T) {
	d := BuildVisualization([]BenchmarkRun{vizMemRun("128", 40, nil), vizMemRun("512", 55, nil)})
	if metricKeys(d)["mem_gpu"] {
		t.Error("memory is offered although no run measured any")
	}
	if !metricKeys(d)["gen"] {
		t.Error("the speed metrics went missing")
	}
}

// The metric list is built per call. Appending to the package-level
// slice would leak one comparison's memory metrics into the next.
func TestVisualizationMetricsDoNotLeakBetweenCalls(t *testing.T) {
	BuildVisualization([]BenchmarkRun{vizMemRun("128", 40, &MemorySnapshot{GPUGiB: 23, Cards: 1})})
	d := BuildVisualization([]BenchmarkRun{vizMemRun("128", 40, nil)})
	if metricKeys(d)["mem_gpu"] {
		t.Error("memory metrics leaked from the previous call")
	}
}
