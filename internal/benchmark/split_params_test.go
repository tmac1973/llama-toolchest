package benchmark

import (
	"strings"
	"testing"
)

// One value fixes a parameter for the whole job; two or more sweep it.
// This is the rule that replaced the separate override and sweep
// controls, where setting both left the override silently ignored.
func TestSplitParamsSingleValueBecomesOverride(t *testing.T) {
	ov, sweeps, err := SplitParams(map[string][]string{"gpu_layers": {"40"}})
	if err != nil {
		t.Fatalf("SplitParams: %v", err)
	}
	if len(sweeps) != 0 {
		t.Errorf("got %d sweeps, want 0 for a single value", len(sweeps))
	}
	if ov == nil || ov.GPULayers == nil || *ov.GPULayers != 40 {
		t.Errorf("override not set: %+v", ov)
	}
}

func TestSplitParamsMultipleValuesBecomeSweep(t *testing.T) {
	ov, sweeps, err := SplitParams(map[string][]string{"ubatch_size": {"512", "1024"}})
	if err != nil {
		t.Fatalf("SplitParams: %v", err)
	}
	if len(sweeps) != 1 || sweeps[0].Field != "ubatch_size" {
		t.Fatalf("got %+v, want one ubatch_size axis", sweeps)
	}
	if len(sweeps[0].Values) != 2 {
		t.Errorf("got %v, want both values", sweeps[0].Values)
	}
	if ov != nil && ov.UBatchSize != nil {
		t.Error("a swept parameter must not also be a fixed override")
	}
}

// The mixed case is the whole point: fix some parameters, sweep others,
// in one pass with no ambiguity about which wins.
func TestSplitParamsMixesFixedAndSwept(t *testing.T) {
	ov, sweeps, err := SplitParams(map[string][]string{
		"gpu_layers":  {"999"},
		"ubatch_size": {"256", "512", "1024"},
		"threads":     {"8"},
	})
	if err != nil {
		t.Fatalf("SplitParams: %v", err)
	}
	if ov == nil || ov.GPULayers == nil || *ov.GPULayers != 999 {
		t.Errorf("gpu_layers should be fixed: %+v", ov)
	}
	if ov.Threads == nil || *ov.Threads != 8 {
		t.Errorf("threads should be fixed: %+v", ov)
	}
	if len(sweeps) != 1 || sweeps[0].Field != "ubatch_size" {
		t.Errorf("only ubatch_size should be swept, got %+v", sweeps)
	}
}

// No selection means "use each model's saved value", which is the
// absence of an entry rather than an empty override.
func TestSplitParamsEmptyMeansInherit(t *testing.T) {
	ov, sweeps, err := SplitParams(map[string][]string{
		"gpu_layers":  {},
		"ubatch_size": {"", "  "},
	})
	if err != nil {
		t.Fatalf("SplitParams: %v", err)
	}
	if ov != nil {
		t.Errorf("got %+v, want no overrides", ov)
	}
	if len(sweeps) != 0 {
		t.Errorf("got %+v, want no sweeps", sweeps)
	}
}

// The custom-entry box can produce a value already offered as a
// checkbox, so duplicates collapse rather than erroring.
func TestSplitParamsDeduplicates(t *testing.T) {
	_, sweeps, err := SplitParams(map[string][]string{"ubatch_size": {"512", "512"}})
	if err != nil {
		t.Fatalf("SplitParams: %v", err)
	}
	if len(sweeps) != 0 {
		t.Errorf("duplicates should collapse to a single fixed value, got %+v", sweeps)
	}
}

func TestSplitParamsRejectsBadValue(t *testing.T) {
	if _, _, err := SplitParams(map[string][]string{"gpu_layers": {"40", "abc"}}); err == nil {
		t.Error("expected an error for an unparseable value")
	}
}

func TestSplitParamsRejectsUnknownParameter(t *testing.T) {
	if _, _, err := SplitParams(map[string][]string{"nonsense": {"1"}}); err == nil {
		t.Error("expected an error for an unknown parameter")
	}
}

// Every curated choice must survive its own parser, so a dropdown can't
// offer a value the server will reject.
func TestEveryCuratedChoiceParses(t *testing.T) {
	for _, f := range SweepFields() {
		for _, c := range f.Choices {
			if _, _, err := SplitParams(map[string][]string{f.Name: {c.Value}}); err != nil {
				t.Errorf("%s: offered choice %q does not parse: %v", f.Name, c.Value, err)
			}
		}
	}
}

// A blank choice would be dropped as "inherit", making it unreachable.
func TestNoCuratedChoiceIsBlank(t *testing.T) {
	for _, f := range SweepFields() {
		for _, c := range f.Choices {
			if c.Value == "" {
				t.Errorf("%s offers a blank choice (%q), which is indistinguishable from inherit", f.Name, c.Label)
			}
		}
	}
}

// Free-text parameters have no curated values by definition; everything
// else should offer some, or its dropdown renders empty.
func TestEnumerableParamsHaveChoices(t *testing.T) {
	for _, f := range SweepFields() {
		if f.FreeText || f.DynamicChoices != "" {
			continue
		}
		if len(f.Choices) == 0 {
			t.Errorf("%s has no choices and would render an empty dropdown", f.Name)
		}
	}
}

// Ordering is a UX contract, not incidental: reload-affecting parameters
// first (those are the ones that move throughput), sampling next, and
// free-text controls last so they don't interleave with the dropdowns
// and read as a rendering fault.
func TestSweepFieldOrdering(t *testing.T) {
	fields := SweepFields()

	rank := func(f SweepField) int {
		switch {
		case f.FreeText:
			return 2
		case !f.RestartsRouter:
			return 1
		default:
			return 0
		}
	}
	for i := 1; i < len(fields); i++ {
		prev, cur := fields[i-1], fields[i]
		if rank(prev) > rank(cur) {
			t.Errorf("%s (tier %d) sorts before %s (tier %d)",
				prev.Name, rank(prev), cur.Name, rank(cur))
		}
		if rank(prev) == rank(cur) && prev.Label > cur.Label {
			t.Errorf("within a tier, %q should follow %q", prev.Label, cur.Label)
		}
	}
	if len(fields) == 0 || !fields[len(fields)-1].FreeText {
		t.Error("the last parameter should be a free-text one")
	}
}

// A draft model path is a thing to select, not a knob to tune, and it
// was the only parameter whose values were filesystem paths.
func TestDraftModelPathIsNotSweepable(t *testing.T) {
	if _, ok := LookupSweepField("draft_model_path"); ok {
		t.Error("draft_model_path should not be offered as a tunable parameter")
	}
}

// Editing a job's fixed override must invalidate completed cells. Keying
// only on (model, build, preset, sweep) let a cell measured at ubatch 512
// satisfy the same cell at 1024, re-attributing the old measurement.
func TestCellIdentityIncludesFixedOverrides(t *testing.T) {
	cell := JobCell{ModelID: "m", BuildID: "b", Preset: "p"}
	a, b := 512, 1024

	if identifyIn(cell, &ConfigOverrides{UBatchSize: &a}) ==
		identifyIn(cell, &ConfigOverrides{UBatchSize: &b}) {
		t.Error("cells under different fixed overrides must not share an identity")
	}
	if identifyIn(cell, &ConfigOverrides{UBatchSize: &a}) ==
		identifyIn(cell, nil) {
		t.Error("an override and no override must not share an identity")
	}
	if identifyIn(cell, &ConfigOverrides{UBatchSize: &a}) !=
		identifyIn(cell, &ConfigOverrides{UBatchSize: &a}) {
		t.Error("identical overrides must produce a stable identity")
	}
}

// Duplicate values expand into byte-identical cells: double the runtime,
// and one measurement reported twice as if it were two data points.
// ParseSweepValues rejected them; the JSON API path did not.
func TestValidateSweepsRejectsDuplicateValues(t *testing.T) {
	err := ValidateSweeps([]SweepAxis{{Field: "threads", Values: []string{"8", "8"}}})
	if err == nil {
		t.Fatal("expected duplicate values to be rejected")
	}
	if !strings.Contains(err.Error(), "duplicate") {
		t.Errorf("error should name the problem, got: %v", err)
	}
}

// An override the form cannot display must survive an edit rather than
// being deleted by the params round trip.
func TestMergeOverridesKeepsUnmappedFields(t *testing.T) {
	path := "/models/draft.gguf"
	ngl := 40
	base := &ConfigOverrides{DraftModelPath: &path}
	derived := &ConfigOverrides{GPULayers: &ngl}

	got := MergeOverrides(base, derived)
	if got.DraftModelPath == nil || *got.DraftModelPath != path {
		t.Error("unmapped override was dropped")
	}
	if got.GPULayers == nil || *got.GPULayers != 40 {
		t.Error("derived override was lost")
	}
}

// What the user just edited wins over what was carried through.
func TestMergeOverridesDerivedWins(t *testing.T) {
	oldNgl, newNgl := 20, 40
	got := MergeOverrides(&ConfigOverrides{GPULayers: &oldNgl}, &ConfigOverrides{GPULayers: &newNgl})
	if got.GPULayers == nil || *got.GPULayers != 40 {
		t.Errorf("got %v, want the derived 40", got.GPULayers)
	}
}

func TestMergeOverridesNils(t *testing.T) {
	ngl := 40
	only := &ConfigOverrides{GPULayers: &ngl}
	if MergeOverrides(nil, only) != only {
		t.Error("nil base should return derived")
	}
	if MergeOverrides(only, nil) != only {
		t.Error("nil derived should return base")
	}
}
