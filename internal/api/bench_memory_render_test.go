package api

import (
	"bytes"
	"html/template"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/benchmark"
	"github.com/tmac1973/llama-toolchest/web"
)

func benchTemplates(t *testing.T) *template.Template {
	t.Helper()
	base, err := template.New("").Funcs(testFuncMap).ParseFS(web.Templates,
		"templates/layout.html", "templates/partials/*.html")
	if err != nil {
		t.Fatalf("parse templates: %v", err)
	}
	return base
}

func measuredRun(id string) benchmark.BenchmarkRun {
	r := perfRun(id)
	r.Memory = &benchmark.MemorySnapshot{
		GPUGiB: 23.0, WeightsGiB: 20.0, KVGiB: 2.0, ComputeGiB: 1.0,
		HostGiB: 1.0, CardDeltaGiB: 24.5, Cards: 4,
	}
	return r
}

// The compare table's VRAM column is the headline of this feature: the
// memory each configuration in a sweep actually used, beside its speed.
func TestCompareTableShowsMeasuredMemory(t *testing.T) {
	data := benchmark.BuildComparison([]benchmark.BenchmarkRun{measuredRun("r1")})
	var buf bytes.Buffer
	if err := benchTemplates(t).ExecuteTemplate(&buf, "benchmark_compare", data); err != nil {
		t.Fatalf("execute: %v", err)
	}
	out := buf.String()

	if !strings.Contains(out, ">VRAM</th>") {
		t.Errorf("no VRAM column in the compare table\n%s", out)
	}
	if !strings.Contains(out, ">23.0</td>") {
		t.Errorf("the measured figure is missing from the row\n%s", out)
	}
	// The number alone does not say what it counts; the tooltip must.
	for _, want := range []string{
		"20.0 weights", "2.0 KV cache", "1.0 working buffers",
		"cards themselves reported 24.5 GiB",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("tooltip missing %q\n%s", want, out)
		}
	}
}

// A run from before this was measured must read as absent, not as zero.
func TestCompareTableShowsAnEmDashWhenNothingWasMeasured(t *testing.T) {
	data := benchmark.BuildComparison([]benchmark.BenchmarkRun{perfRun("r1")})
	var buf bytes.Buffer
	if err := benchTemplates(t).ExecuteTemplate(&buf, "benchmark_compare", data); err != nil {
		t.Fatalf("execute: %v", err)
	}
	out := buf.String()

	if strings.Contains(out, ">0.0</td>") {
		t.Errorf("an unmeasured run is rendering as 0.0 GiB\n%s", out)
	}
	if !strings.Contains(out, "Model loading detail") {
		t.Errorf("the tooltip should say why there is no figure\n%s", out)
	}
}

func TestRunDetailShowsTheMemoryBreakdown(t *testing.T) {
	run := measuredRun("r1")
	var buf bytes.Buffer
	if err := benchTemplates(t).ExecuteTemplate(&buf, "benchmark_detail", &run); err != nil {
		t.Fatalf("execute: %v", err)
	}
	out := buf.String()

	for _, want := range []string{
		"Memory used", "23.0 GiB on the GPUs", "20.0 weights",
		"1.0 GiB in system memory", "Cards reported 24.5 GiB",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("detail view missing %q\n%s", want, out)
		}
	}
}

// A load that overlapped another keeps its own buffer figures but has no
// attributable card total, and the view has to say which is which.
func TestRunDetailFlagsAContendedLoad(t *testing.T) {
	run := measuredRun("r1")
	run.Memory.Contended = true
	run.Memory.CardDeltaGiB = 0
	var buf bytes.Buffer
	if err := benchTemplates(t).ExecuteTemplate(&buf, "benchmark_detail", &run); err != nil {
		t.Fatalf("execute: %v", err)
	}
	out := buf.String()

	if !strings.Contains(out, "23.0 GiB on the GPUs") {
		t.Errorf("the per-instance figures survive contention\n%s", out)
	}
	if !strings.Contains(out, "could not be attributed") {
		t.Errorf("the detail view does not say the card total is unusable\n%s", out)
	}
	if strings.Contains(out, "Cards reported") {
		t.Errorf("a contended run must not show a card total\n%s", out)
	}
}

func TestRunDetailWithoutAMeasurement(t *testing.T) {
	run := perfRun("r1")
	var buf bytes.Buffer
	if err := benchTemplates(t).ExecuteTemplate(&buf, "benchmark_detail", &run); err != nil {
		t.Fatalf("execute: %v", err)
	}
	if !strings.Contains(buf.String(), "Not measured") {
		t.Errorf("detail view should say nothing was measured\n%s", buf.String())
	}
}

// sweepRun is one cell of a single-parameter sweep: everything about it
// is the same as its siblings except the swept value.
func sweepRun(id, ubatch string, gen float64) benchmark.BenchmarkRun {
	r := measuredRun(id)
	r.SweepValues = map[string]string{"ubatch_size": ubatch}
	r.Summary.AvgGenTokPerSec = gen
	r.Config = benchmark.ConfigSnapshot{
		ContextSize: 32768, BatchSize: 2048, UBatchSize: 512,
		GPUAssign: "all", FlashAttention: true,
	}
	return r
}

// A one-parameter sweep leaves most of the table repeating itself, and
// those columns are what push the compared numbers off the side of the
// page. They are hidden, and stated once above the table.
func TestCompareTableHidesColumnsEveryRunShares(t *testing.T) {
	data := benchmark.BuildComparison([]benchmark.BenchmarkRun{
		sweepRun("r1", "128", 40), sweepRun("r2", "512", 55),
	})
	var buf bytes.Buffer
	if err := benchTemplates(t).ExecuteTemplate(&buf, "benchmark_compare", data); err != nil {
		t.Fatalf("execute: %v", err)
	}
	out := buf.String()

	if !strings.Contains(out, `<th class="bc-const">Quant</th>`) {
		t.Errorf("Quant is identical in both runs and should be marked constant\n%s", out)
	}
	if !strings.Contains(out, `<th class="" title="Which point of a parameter sweep`) {
		t.Errorf("the swept column must not be marked constant\n%s", out)
	}
	if !strings.Contains(out, "The same for every run:") {
		t.Errorf("the collapsed columns are not stated above the table\n%s", out)
	}
	if !strings.Contains(out, "flash attention <kbd>yes</kbd>") {
		t.Errorf("a collapsed column's shared value is missing from the summary\n%s", out)
	}
	// Hidden, not dropped: the toggle puts them back with no round trip.
	if !strings.Contains(out, "hide-constant") || !strings.Contains(out, "toggleConstantColumns") {
		t.Errorf("the reveal control is missing\n%s", out)
	}
}

// With one run there is nothing to compare, so nothing is redundant.
func TestCompareTableKeepsEveryColumnForASingleRun(t *testing.T) {
	data := benchmark.BuildComparison([]benchmark.BenchmarkRun{sweepRun("r1", "128", 40)})
	var buf bytes.Buffer
	if err := benchTemplates(t).ExecuteTemplate(&buf, "benchmark_compare", data); err != nil {
		t.Fatalf("execute: %v", err)
	}
	out := buf.String()
	// The class appears once in the stylesheet regardless; what matters
	// is that no cell carries it.
	if strings.Contains(out, `class="bc-const"`) {
		t.Errorf("a single run had columns collapsed\n%s", out)
	}
	if strings.Contains(out, "The same for every run:") {
		t.Errorf("nothing should be summarised away for a single run\n%s", out)
	}
}

// The capability row used to span the six columns after VRAM. A span
// cannot carry a per-column class, so hiding one column would leave that
// row a different width from the header.
func TestCapabilityRowHasOneCellPerColumn(t *testing.T) {
	ev := perfRun("r-ev")
	ev.Summary = nil
	ev.Eval = &benchmark.EvalScores{
		Mode: "perplexity", Dataset: "wikitext-2", ContextSize: 512,
		Chunks: 100, Perplexity: 6.234, PerplexityErr: 0.04,
	}
	data := benchmark.BuildComparison([]benchmark.BenchmarkRun{ev, measuredRun("r-perf")})
	var buf bytes.Buffer
	if err := benchTemplates(t).ExecuteTemplate(&buf, "benchmark_compare", data); err != nil {
		t.Fatalf("execute: %v", err)
	}
	if strings.Contains(buf.String(), "colspan=") {
		t.Errorf("the capability row still spans columns\n%s", buf.String())
	}
}
