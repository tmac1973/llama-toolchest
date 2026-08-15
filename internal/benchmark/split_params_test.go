package benchmark

import (
	"reflect"
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
// ungrouped free-text controls last so they don't interleave with the
// dropdowns and read as a rendering fault. Grouped fields are the
// exception to both rules: they take their parent's tier and sort
// directly beneath it, so someone selecting a speculative decoding mode
// finds its parameters right below the selector.
func TestSweepFieldOrdering(t *testing.T) {
	fields := SweepFields()

	rank := func(f SweepField) int {
		if f.Group != "" {
			p, _ := LookupSweepField(f.Group)
			f = p
		}
		switch {
		case f.FreeText:
			return 2
		case !f.RestartsRouter:
			return 1
		default:
			return 0
		}
	}
	key := func(f SweepField) string {
		if f.Group != "" {
			p, _ := LookupSweepField(f.Group)
			return p.Label + "\x00" + f.Label
		}
		return f.Label
	}
	for i := 1; i < len(fields); i++ {
		prev, cur := fields[i-1], fields[i]
		if rank(prev) > rank(cur) {
			t.Errorf("%s (tier %d) sorts before %s (tier %d)",
				prev.Name, rank(prev), cur.Name, rank(cur))
		}
		if rank(prev) == rank(cur) && key(prev) > key(cur) {
			t.Errorf("within a tier, %q should follow %q", prev.Label, cur.Label)
		}
	}
	if len(fields) == 0 || !fields[len(fields)-1].FreeText {
		t.Error("the last parameter should be a free-text one")
	}

	// The group contract itself: every spec parameter sits in a
	// contiguous run immediately after the Speculative Decoding row.
	idx := map[string]int{}
	for i, f := range fields {
		idx[f.Name] = i
	}
	specIdx, ok := idx["spec_type"]
	if !ok {
		t.Fatal("spec_type missing")
	}
	members := []string{"draft_max", "draft_min", "draft_p_min", "draft_model_path", "ngram_size_m", "ngram_size_n"}
	for _, m := range members {
		i, ok := idx[m]
		if !ok {
			t.Fatalf("%s missing", m)
		}
		if i <= specIdx || i > specIdx+len(members) {
			t.Errorf("%s (index %d) is not grouped under spec_type (index %d)", m, i, specIdx)
		}
	}
}

// draft_model_path is sweepable as a free-text parameter: without a
// form control for it, the draft speculative mode could only run
// against whatever path the model's saved config happened to hold, so
// selecting the mode in a job was not enough to use it. Free-text
// rendering keeps filesystem paths out of the choice dropdowns.
func TestDraftModelPathIsFreeTextSweepable(t *testing.T) {
	f, ok := LookupSweepField("draft_model_path")
	if !ok {
		t.Fatal("draft_model_path should be offered as a parameter")
	}
	if !f.FreeText {
		t.Error("a filesystem path must render as free text, not choices")
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

// Dedup must compare canonical values, not raw text: "1.0" and "1"
// resolve to the same ConfigOverrides and would otherwise expand into
// byte-identical cells — one measurement shown twice.
func TestValidateSweepsRejectsNumericallyEqualValues(t *testing.T) {
	cases := []struct {
		field  string
		values []string
	}{
		{"temperature", []string{"1.0", "1"}},
		{"threads", []string{"8", "08"}},
		{"ubatch_size", []string{"512", "0512"}},
	}
	for _, c := range cases {
		if err := ValidateSweeps([]SweepAxis{{Field: c.field, Values: c.values}}); err == nil {
			t.Errorf("%s %v: expected numerically identical values to be rejected", c.field, c.values)
		}
	}
}

// Genuinely distinct values must still be allowed.
func TestValidateSweepsAllowsDistinctValues(t *testing.T) {
	if err := ValidateSweeps([]SweepAxis{
		{Field: "temperature", Values: []string{"0", "0.7", "1.0"}},
	}); err != nil {
		t.Errorf("distinct values rejected: %v", err)
	}
}

func TestSplitParamsCollapsesNumericallyEqualValues(t *testing.T) {
	_, sweeps, err := SplitParams(map[string][]string{"temperature": {"1.0", "1"}})
	if err != nil {
		t.Fatalf("SplitParams: %v", err)
	}
	if len(sweeps) != 0 {
		t.Errorf("got %+v, want them collapsed to a single fixed value", sweeps)
	}
}

// A nil and an all-nil override struct mean the same thing. Keying them
// differently would drop every completed cell to pending on an unrelated
// edit and re-parent its runs to Ad-Hoc.
func TestOverrideKeyTreatsEmptyAsNil(t *testing.T) {
	if overrideKey(nil) != overrideKey(&ConfigOverrides{}) {
		t.Errorf("nil (%q) and empty (%q) must key identically",
			overrideKey(nil), overrideKey(&ConfigOverrides{}))
	}
	cell := JobCell{ModelID: "m", BuildID: "b", Preset: "p"}
	if identifyIn(cell, nil) != identifyIn(cell, &ConfigOverrides{}) {
		t.Error("an unchanged job must not invalidate its completed cells")
	}
}

// Params own every parameter they can express, including by omission —
// otherwise an override supplied alongside them can never be cleared.
// Every ConfigOverrides field now has a form control (draft_model_path
// gained a free-text one), so nothing carries through; the mechanism
// stays for any future field the form cannot express.
func TestKeepUnsweepableDropsExpressibleFields(t *testing.T) {
	ngl := 40
	path := "/models/draft.gguf"
	got := KeepUnsweepable(&ConfigOverrides{GPULayers: &ngl, DraftModelPath: &path})
	if got != nil {
		t.Errorf("every field is expressible via params now; nothing should carry through, got %+v", got)
	}

	// The reflection walk must cover every field: any override field
	// whose JSON tag is missing from the registry would silently carry
	// through edits, so require full coverage explicitly.
	tp := reflect.TypeOf(ConfigOverrides{})
	for i := 0; i < tp.NumField(); i++ {
		tag := strings.Split(tp.Field(i).Tag.Get("json"), ",")[0]
		if !IsSweepable(tag) {
			t.Errorf("ConfigOverrides field %s (%s) has no sweep registry entry", tp.Field(i).Name, tag)
		}
	}
}

func TestKeepUnsweepableNilWhenNothingRemains(t *testing.T) {
	ngl := 40
	if got := KeepUnsweepable(&ConfigOverrides{GPULayers: &ngl}); got != nil {
		t.Errorf("got %+v, want nil when every field is expressible", got)
	}
	if KeepUnsweepable(nil) != nil {
		t.Error("nil in, nil out")
	}
}

// Reload-affecting axes must vary slowest so cheap axes cycle inside
// them. Alphabetical ordering put temperature before ubatch_size, making
// ubatch alternate every cell — one llama-server reload per cell instead
// of one per ubatch value.
func TestSweepAxisOrderMinimizesReloads(t *testing.T) {
	combos := sweepCombinations([]SweepAxis{
		{Field: "temperature", Values: []string{"0", "0.7", "1.0"}},
		{Field: "ubatch_size", Values: []string{"512", "1024"}},
	})
	if len(combos) != 6 {
		t.Fatalf("got %d combinations, want 6", len(combos))
	}

	changes := 0
	for i := 1; i < len(combos); i++ {
		if combos[i]["ubatch_size"] != combos[i-1]["ubatch_size"] {
			changes++
		}
	}
	// Two ubatch values means one transition when it varies slowest.
	if changes != 1 {
		t.Errorf("ubatch_size changed %d times across the matrix, want 1 — it is the reload-expensive axis and must vary slowest", changes)
	}
}

// llama-benchy shells out with a fixed argument list and never sees
// sampling settings, so those cells would measure the same thing under
// different labels.
func TestValidateSamplingSupportRejectsBenchyCombination(t *testing.T) {
	err := ValidateSamplingSupport(
		[]string{"benchy-quick"}, nil,
		[]SweepAxis{{Field: "temperature", Values: []string{"0", "1"}}})
	if err == nil {
		t.Fatal("expected a sampling sweep against a benchy preset to be rejected")
	}
	if !strings.Contains(err.Error(), "Temperature") {
		t.Errorf("error should name the offending parameter, got: %v", err)
	}
}

func TestValidateSamplingSupportRejectsFixedSamplingOverride(t *testing.T) {
	temp := 0.7
	err := ValidateSamplingSupport(
		[]string{"benchy-standard"}, &ConfigOverrides{Temperature: &temp}, nil)
	if err == nil {
		t.Error("a fixed sampling override is equally ignored by benchy and must be rejected")
	}
}

func TestValidateSamplingSupportAllowsInternalPresets(t *testing.T) {
	if err := ValidateSamplingSupport(
		[]string{"internal-quick", "internal-long-ctx"}, nil,
		[]SweepAxis{{Field: "temperature", Values: []string{"0", "1"}}}); err != nil {
		t.Errorf("internal presets apply sampling per request: %v", err)
	}
}

// A benchy preset with only router-config parameters is fine — those
// reach llama-server through the preset, which benchy runs against.
func TestValidateSamplingSupportAllowsConfigSweepWithBenchy(t *testing.T) {
	if err := ValidateSamplingSupport(
		[]string{"benchy-quick"}, nil,
		[]SweepAxis{{Field: "ubatch_size", Values: []string{"512", "1024"}}}); err != nil {
		t.Errorf("config parameters do reach benchy runs: %v", err)
	}
}

// Two cells of one job that share a model and preset must not send
// byte-identical prompts. When the router isn't restarted between them —
// which a sampling sweep never does — llama.cpp serves the second from
// its prompt cache, prompt_n collapses, and the recorded
// prompt-processing rate is cache-lookup overhead reported as a
// measurement.
func TestPromptsDifferPerCell(t *testing.T) {
	a := buildPromptFor("bench-1-1-1", 256, 1)
	b := buildPromptFor("bench-2-2-1", 256, 1)
	if a == b {
		t.Error("two cells produced identical prompts; the second would hit the prompt cache")
	}
	// Same nonce and repetition must still be reproducible.
	if buildPromptFor("bench-1-1-1", 256, 1) != a {
		t.Error("prompt generation must be deterministic for a given cell")
	}
}

// The existing per-repetition variation must survive.
func TestPromptsDifferPerRepetition(t *testing.T) {
	if buildPromptFor("x", 256, 1) == buildPromptFor("x", 256, 2) {
		t.Error("repetitions must differ")
	}
}

// Sizing is approximate but must not be thrown off by the nonce.
func TestPromptSizeUnaffectedByNonce(t *testing.T) {
	want := 256 * BenchPromptCharsPerToken
	for _, n := range []string{"", "bench-1784407486983-1-1"} {
		if got := len(buildPromptFor(n, 256, 1)); got != want {
			t.Errorf("nonce %q: prompt is %d chars, want %d", n, got, want)
		}
	}
}

// Prompts of different sizes must not share a prefix. They previously
// differed only in total length, so llama.cpp cached the common part and
// every size after the first measured incremental prefill while
// reporting only the newly-processed token count.
func TestPromptsOfDifferentSizesDoNotSharePrefix(t *testing.T) {
	short := buildPromptFor("cell-1", 256, 1)
	long := buildPromptFor("cell-1", 512, 1)

	n := len(short)
	if len(long) < n {
		t.Fatal("longer target produced a shorter prompt")
	}
	if long[:n] == short {
		t.Error("the longer prompt starts with the shorter one verbatim; llama.cpp would serve the overlap from cache and report only the tail")
	}
	// They must diverge within the first few tokens, so the cacheable
	// overlap is negligible rather than most of the shorter prompt.
	div := 0
	for div < len(short) && short[div] == long[div] {
		div++
	}
	if div > 64 {
		t.Errorf("prompts share their first %d chars; the overlap should be a few tokens at most", div)
	}
}

// "none" is the explicit speculative-decoding-off value: it must survive
// SplitParams (a literal empty string would be dropped as unset) and map
// to an explicit empty SpecType override.
func TestSpecTypeNoneMapsToExplicitOff(t *testing.T) {
	// Single value: fixed override.
	o, axes, err := SplitParams(map[string][]string{"spec_type": {"none"}})
	if err != nil {
		t.Fatal(err)
	}
	if len(axes) != 0 {
		t.Fatalf("single value should fix, not sweep: %+v", axes)
	}
	if o == nil || o.SpecType == nil || *o.SpecType != "" {
		t.Fatalf("none should become an explicit empty SpecType override, got %+v", o)
	}

	// Two values: a sweep comparing off against a mode in one job.
	_, axes, err = SplitParams(map[string][]string{"spec_type": {"none", "draft-mtp"}})
	if err != nil {
		t.Fatal(err)
	}
	if len(axes) != 1 || len(axes[0].Values) != 2 {
		t.Fatalf("none+mode should sweep with both values kept: %+v", axes)
	}

	// And the axis value applies per cell the same way.
	var cell ConfigOverrides
	f, _ := LookupSweepField("spec_type")
	if err := f.set(&cell, "none"); err != nil {
		t.Fatal(err)
	}
	if cell.SpecType == nil || *cell.SpecType != "" {
		t.Fatalf("axis value none should apply as explicit off, got %+v", cell.SpecType)
	}
}

// The speculative decoding parameters are sweepable and reach the
// snapshot, so a job can tune a mode, not just select it.
func TestSpecParamsFlowToSnapshot(t *testing.T) {
	o, _, err := SplitParams(map[string][]string{
		"spec_type":   {"draft-mtp"},
		"draft_max":   {"6"},
		"draft_p_min": {"0.75"},
	})
	if err != nil {
		t.Fatal(err)
	}
	snap := applyOverrides(ConfigSnapshot{DraftMax: 16}, o)
	if snap.SpecType != "draft-mtp" || snap.DraftMax != 6 || snap.DraftPMin != "0.75" {
		t.Fatalf("spec params did not reach the snapshot: %+v", snap)
	}
}
