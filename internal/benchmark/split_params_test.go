package benchmark

import "testing"

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
