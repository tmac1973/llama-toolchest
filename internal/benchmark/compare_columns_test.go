package benchmark

import "testing"

func colRun(id string) BenchmarkRun {
	return BenchmarkRun{
		ID: id, ModelName: "Qwen3.8-27B", Quant: "Q4_K_XL", SizeGiB: 29.3,
		Preset: "internal-standard",
		Config: ConfigSnapshot{
			ContextSize: 32768, BatchSize: 2048, UBatchSize: 512,
			GPUAssign: "all", FlashAttention: true,
		},
		Build:   BuildSnapshot{ID: "b1", Profile: "rocm", GitRef: "b10679"},
		Summary: &BenchmarkSummary{AvgGenTokPerSec: 40, AvgPromptTokPerSec: 100},
	}
}

func has(common []CommonColumn, name string) (string, bool) {
	for _, c := range common {
		if c.Name == name {
			return c.Value, true
		}
	}
	return "", false
}

// The case this exists for: a sweep of one parameter on one model and
// one build. Everything except the swept value is the same in every row,
// and those columns push the ones that matter off the side of the page.
func TestOneParameterSweepCollapsesEverythingElse(t *testing.T) {
	var runs []BenchmarkRun
	for _, ub := range []string{"128", "512", "2048"} {
		r := colRun("r" + ub)
		r.SweepValues = map[string]string{"ubatch_size": ub}
		r.Config.UBatchSize = 512 // config is the saved one; the sweep is the axis
		runs = append(runs, r)
	}

	c := BuildComparison(runs)

	if !c.Varies["sweep"] {
		t.Error("the swept column must stay: it is the whole comparison")
	}
	for _, name := range []string{"quant", "size", "context", "batch", "KV cache", "flash attention", "build", "GPUs"} {
		if c.Varies[name] {
			t.Errorf("%q is identical in every run but was kept", name)
		}
		if _, ok := has(c.Common, name); !ok {
			t.Errorf("%q was collapsed without being stated in the summary", name)
		}
	}
}

// Nothing is collapsed silently: every column taken out of the table has
// to appear above it, with the value the runs share.
func TestCollapsedColumnsKeepTheTemplateWording(t *testing.T) {
	a, b := colRun("a"), colRun("b")
	a.Config.KVCacheQuant, b.Config.KVCacheQuant = "", ""
	a.Config.GPUAssign, b.Config.GPUAssign = "", ""
	a.Config.BatchSize, b.Config.BatchSize = 0, 0
	a.Config.UBatchSize, b.Config.UBatchSize = 0, 0
	a.Summary.AvgGenTokPerSec = 55

	c := BuildComparison([]BenchmarkRun{a, b})

	// Each of these mirrors a fallback the table cell prints, and the
	// two have to agree — the toggle shows the cell beside the summary.
	for name, want := range map[string]string{
		"KV cache":        "f16",
		"GPUs":            "all",
		"size":            "29.3 GiB",
		"flash attention": "yes",
		"build":           "rocm · b10679",
	} {
		got, ok := has(c.Common, name)
		if !ok {
			t.Errorf("%q missing from the summary", name)
			continue
		}
		if got != want {
			t.Errorf("%s = %q; the table cell shows %q", name, got, want)
		}
	}

	// Batch and sweep are em-dashes in both rows here. They collapse
	// like any other constant, but there is nothing to state about them
	// — see TestBlankColumnsAreHiddenWithoutBeingStated.
	for _, name := range []string{"batch", "sweep"} {
		if c.Varies[name] {
			t.Errorf("%q is the same in both runs and should be collapsed", name)
		}
		if v, ok := has(c.Common, name); ok {
			t.Errorf("%q appears in the summary as %q", name, v)
		}
	}
}

// A column that differs stays in the table and out of the summary, or
// the comparison loses the thing being compared.
func TestVaryingColumnsAreKept(t *testing.T) {
	a, b := colRun("a"), colRun("b")
	b.Quant = "Q8_0"
	b.SizeGiB = 51.2
	b.Config.FlashAttention = false

	c := BuildComparison([]BenchmarkRun{a, b})

	for _, name := range []string{"quant", "size", "flash attention"} {
		if !c.Varies[name] {
			t.Errorf("%q differs between the runs but was collapsed", name)
		}
		if v, ok := has(c.Common, name); ok {
			t.Errorf("%q is in the summary as %q although the runs disagree", name, v)
		}
	}
	if c.Varies["build"] {
		t.Error("build is identical here and should have been collapsed")
	}
}

// One run is not a comparison. Its table is a statement of what it was,
// and every column of it is worth reading.
func TestASingleRunCollapsesNothing(t *testing.T) {
	c := BuildComparison([]BenchmarkRun{colRun("a")})
	if c.Common != nil {
		t.Errorf("common = %+v; want nothing collapsed for a single run", c.Common)
	}
	if c.Varies != nil {
		t.Errorf("varies = %+v; want nil, which the view reads as show everything", c.Varies)
	}
}

// A column blank in every row is hidden like any other constant, but
// there is nothing to state about it: "build —" in the summary reads as
// a shared setting rather than as missing data.
func TestBlankColumnsAreHiddenWithoutBeingStated(t *testing.T) {
	a, b := colRun("a"), colRun("b")
	a.Build, b.Build = BuildSnapshot{}, BuildSnapshot{}
	a.BuildProfile, b.BuildProfile = "", ""
	a.BuildRef, b.BuildRef = "", ""
	a.Summary.AvgGenTokPerSec = 55

	c := BuildComparison([]BenchmarkRun{a, b})

	if c.Varies["build"] {
		t.Error("an empty build column should still be collapsed")
	}
	if v, ok := has(c.Common, "build"); ok {
		t.Errorf("build is in the summary as %q; an absence is not a shared value", v)
	}
	if v, ok := has(c.Common, "sweep"); ok {
		t.Errorf("sweep is in the summary as %q although neither run swept anything", v)
	}
	// The columns that do hold something are unaffected.
	if _, ok := has(c.Common, "quant"); !ok {
		t.Error("quant should still be stated")
	}
}
