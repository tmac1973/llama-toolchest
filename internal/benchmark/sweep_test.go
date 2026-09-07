package benchmark

import (
	"reflect"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/models"
)

func TestSweepCombinationsCartesianProduct(t *testing.T) {
	combos := sweepCombinations([]SweepAxis{
		{Field: "gpu_layers", Values: []string{"20", "40"}},
		{Field: "threads", Values: []string{"4", "8", "16"}},
	})
	if len(combos) != 6 {
		t.Fatalf("got %d combinations, want 6", len(combos))
	}
	seen := map[string]bool{}
	for _, c := range combos {
		if len(c) != 2 {
			t.Errorf("combination %v should set both fields", c)
		}
		seen[c["gpu_layers"]+"/"+c["threads"]] = true
	}
	for _, want := range []string{"20/4", "20/8", "20/16", "40/4", "40/8", "40/16"} {
		if !seen[want] {
			t.Errorf("missing combination %s", want)
		}
	}
}

// No sweeps must yield exactly one empty combination so unswept jobs
// expand exactly as they did before sweeps existed.
func TestSweepCombinationsEmpty(t *testing.T) {
	combos := sweepCombinations(nil)
	if len(combos) != 1 || len(combos[0]) != 0 {
		t.Fatalf("got %v, want a single empty combination", combos)
	}
}

// Axis order in the job must not change cell order, so a job edited in
// the UI doesn't silently reshuffle its matrix.
func TestSweepCombinationsDeterministicRegardlessOfAxisOrder(t *testing.T) {
	a := sweepCombinations([]SweepAxis{
		{Field: "threads", Values: []string{"4", "8"}},
		{Field: "gpu_layers", Values: []string{"20", "40"}},
	})
	b := sweepCombinations([]SweepAxis{
		{Field: "gpu_layers", Values: []string{"20", "40"}},
		{Field: "threads", Values: []string{"4", "8"}},
	})
	if !reflect.DeepEqual(a, b) {
		t.Errorf("axis order changed the expansion:\n%v\n%v", a, b)
	}
}

func TestExpandCellsUnsweptMatchesLegacyShape(t *testing.T) {
	legacy := ExpandCells([]string{"m1", "m2"}, []string{"b1"}, []string{"p1", "p2"})
	if len(legacy) != 4 {
		t.Fatalf("got %d cells, want 4", len(legacy))
	}
	for _, c := range legacy {
		if c.SweepValues != nil {
			t.Errorf("unswept cell carries SweepValues: %v", c.SweepValues)
		}
	}
}

func TestExpandCellsWithSweepsCellCount(t *testing.T) {
	cells := ExpandCellsWithSweeps(
		[]string{"m1", "m2"}, []string{"b1"}, []string{"p1"},
		[]SweepAxis{{Field: "gpu_layers", Values: []string{"20", "40", "99"}}},
	)
	if len(cells) != 6 { // 2 models × 1 build × 3 sweep points × 1 preset
		t.Fatalf("got %d cells, want 6", len(cells))
	}
}

// Presets must sit inside sweep points: changing a swept config forces a
// router reload, changing preset does not. Reversing them would multiply
// reloads by the preset count.
func TestExpandCellsOrdersPresetsInsideSweepPoints(t *testing.T) {
	cells := ExpandCellsWithSweeps(
		[]string{"m"}, []string{"b"}, []string{"p1", "p2"},
		[]SweepAxis{{Field: "gpu_layers", Values: []string{"20", "40"}}},
	)
	if len(cells) != 4 {
		t.Fatalf("got %d cells, want 4", len(cells))
	}
	// Expect: (20,p1) (20,p2) (40,p1) (40,p2)
	want := []struct{ ngl, preset string }{
		{"20", "p1"}, {"20", "p2"}, {"40", "p1"}, {"40", "p2"},
	}
	for i, w := range want {
		if cells[i].SweepValues["gpu_layers"] != w.ngl || cells[i].Preset != w.preset {
			t.Errorf("cell %d = (ngl %s, %s), want (ngl %s, %s)",
				i, cells[i].SweepValues["gpu_layers"], cells[i].Preset, w.ngl, w.preset)
		}
	}
}

// Builds stay outermost so EnsureBuildActive fires once per build.
func TestExpandCellsKeepsBuildsOutermost(t *testing.T) {
	cells := ExpandCellsWithSweeps(
		[]string{"m"}, []string{"b1", "b2"}, []string{"p"},
		[]SweepAxis{{Field: "threads", Values: []string{"4", "8"}}},
	)
	switches := 0
	for i := 1; i < len(cells); i++ {
		if cells[i].BuildID != cells[i-1].BuildID {
			switches++
		}
	}
	if switches != 1 {
		t.Errorf("build changed %d times, want 1", switches)
	}
}

// Each cell must own its map; a shared map would let one cell's value
// leak into its siblings.
func TestExpandCellsSweepValuesNotAliased(t *testing.T) {
	cells := ExpandCellsWithSweeps(
		[]string{"m"}, []string{"b"}, []string{"p"},
		[]SweepAxis{{Field: "threads", Values: []string{"4", "8"}}},
	)
	cells[0].SweepValues["threads"] = "999"
	if cells[1].SweepValues["threads"] != "8" {
		t.Errorf("cells share a map: cell 1 threads = %s", cells[1].SweepValues["threads"])
	}
}

func TestCellOverridesSweepWinsOverFixed(t *testing.T) {
	ngl, threads := 999, 8
	base := &ConfigOverrides{GPULayers: &ngl, Threads: &threads}

	got, err := CellOverrides(base, map[string]string{"gpu_layers": "40"})
	if err != nil {
		t.Fatalf("CellOverrides: %v", err)
	}
	if got.GPULayers == nil || *got.GPULayers != 40 {
		t.Errorf("GPULayers = %v, want the swept 40", got.GPULayers)
	}
	if got.Threads == nil || *got.Threads != 8 {
		t.Errorf("Threads = %v, want the fixed 8 to carry through", got.Threads)
	}
	// The caller's struct must not be modified.
	if *base.GPULayers != 999 {
		t.Errorf("base overrides mutated: GPULayers = %d", *base.GPULayers)
	}
}

func TestCellOverridesNilWhenNothingSet(t *testing.T) {
	got, err := CellOverrides(nil, nil)
	if err != nil {
		t.Fatalf("CellOverrides: %v", err)
	}
	if got != nil {
		t.Errorf("got %+v, want nil so the cell skips the router restart", got)
	}
}

func TestCellOverridesRejectsUnknownField(t *testing.T) {
	if _, err := CellOverrides(nil, map[string]string{"nonsense": "1"}); err == nil {
		t.Error("expected an error for an unknown sweep field")
	}
}

func TestValidateSweeps(t *testing.T) {
	if err := ValidateSweeps([]SweepAxis{
		{Field: "gpu_layers", Values: []string{"20", "40"}},
		{Field: "threads", Values: []string{"8"}},
	}); err != nil {
		t.Errorf("valid sweeps rejected: %v", err)
	}

	if err := ValidateSweeps([]SweepAxis{
		{Field: "gpu_layers", Values: []string{"20"}},
		{Field: "gpu_layers", Values: []string{"40"}},
	}); err == nil {
		t.Error("expected an error when a field is swept twice")
	}

	if err := ValidateSweeps([]SweepAxis{{Field: "gpu_layers"}}); err == nil {
		t.Error("expected an error for an axis with no values")
	}
}

// Sampling params ride with the request, so sweeping them must not be
// reported as reload-expensive.
func TestSweepRestartsRouterDistinguishesSamplingFromConfig(t *testing.T) {
	if SweepRestartsRouter([]SweepAxis{{Field: "temperature", Values: []string{"0", "1"}}}) {
		t.Error("sweeping temperature should not require router restarts")
	}
	if SweepRestartsRouter([]SweepAxis{{Field: "top_k", Values: []string{"20", "40"}}}) {
		t.Error("sweeping top_k should not require router restarts")
	}
	if !SweepRestartsRouter([]SweepAxis{{Field: "gpu_layers", Values: []string{"20", "40"}}}) {
		t.Error("sweeping gpu_layers requires router restarts")
	}
}

// Run IDs must be unique even when generated in the same millisecond.
// A timestamp-only ID collided and the store silently overwrote the
// earlier run — sweeps make same-millisecond cells common.
func TestNewRunIDIsUnique(t *testing.T) {
	seen := make(map[string]bool, 1000)
	for i := 0; i < 1000; i++ {
		id := newRunID(1)
		if seen[id] {
			t.Fatalf("duplicate run ID %q after %d generations", id, i)
		}
		seen[id] = true
	}
}

// Editing a swept job must match completed cells to the same sweep
// point. Keying only on (model, build, preset) would let a completed
// ngl=20 cell satisfy the ngl=40 cell and report its result under the
// wrong configuration.
func TestCellIdentityDistinguishesSweepPoints(t *testing.T) {
	a := JobCell{ModelID: "m", BuildID: "b", Preset: "p",
		SweepValues: map[string]string{"gpu_layers": "20"}}
	b := JobCell{ModelID: "m", BuildID: "b", Preset: "p",
		SweepValues: map[string]string{"gpu_layers": "40"}}

	if identify(a) == identify(b) {
		t.Error("cells at different sweep points must not share an identity")
	}
}

// Map iteration order must not affect identity, or an edit would fail to
// match cells to themselves.
func TestCellIdentityStableAcrossMapOrder(t *testing.T) {
	a := JobCell{ModelID: "m", BuildID: "b", Preset: "p",
		SweepValues: map[string]string{"gpu_layers": "20", "threads": "8"}}
	b := JobCell{ModelID: "m", BuildID: "b", Preset: "p",
		SweepValues: map[string]string{"threads": "8", "gpu_layers": "20"}}

	if identify(a) != identify(b) {
		t.Errorf("identity depends on map order:\n%+v\n%+v", identify(a), identify(b))
	}
}

func TestCellIdentityUnsweptCellsMatch(t *testing.T) {
	a := JobCell{ModelID: "m", BuildID: "b", Preset: "p"}
	b := JobCell{ModelID: "m", BuildID: "b", Preset: "p"}
	if identify(a) != identify(b) {
		t.Error("identical unswept cells must share an identity")
	}
}

func TestSpecValueRoundTripsBothSlots(t *testing.T) {
	cases := []struct {
		raw          string
		mode, assist string
		params       map[string]string
	}{
		{"none", "", "", map[string]string{}},
		{"draft-mtp", "draft-mtp", "", map[string]string{}},
		{"draft-mtp:draft_max=3", "draft-mtp", "", map[string]string{"draft_max": "3"}},
		{"ngram-mod:assist_n_max=64", "", "ngram-mod", map[string]string{"assist_n_max": "64"}},
		{"draft-mtp+ngram-mod", "draft-mtp", "ngram-mod", map[string]string{}},
		{
			"draft-mtp+ngram-mod:draft_max=3,assist_n_max=64,assist_n_match=24",
			"draft-mtp", "ngram-mod",
			map[string]string{"draft_max": "3", "assist_n_max": "64", "assist_n_match": "24"},
		},
		{
			"draft+ngram-simple:draft_p_min=0.75,assist_size_n=12,assist_min_hits=1",
			"draft", "ngram-simple",
			map[string]string{"draft_p_min": "0.75", "assist_size_n": "12", "assist_min_hits": "1"},
		},
		// ngram-cache takes no settings but is still a real mode.
		{"draft-eagle3+ngram-cache", "draft-eagle3", "ngram-cache", map[string]string{}},
	}
	for _, tc := range cases {
		t.Run(tc.raw, func(t *testing.T) {
			sv, err := parseSpecValue(tc.raw)
			if err != nil {
				t.Fatalf("parse %q: %v", tc.raw, err)
			}
			if sv.mode != tc.mode || sv.assist != tc.assist {
				t.Errorf("slots = %q/%q, want %q/%q", sv.mode, sv.assist, tc.mode, tc.assist)
			}
			if !reflect.DeepEqual(sv.params, tc.params) {
				t.Errorf("params = %v, want %v", sv.params, tc.params)
			}
			// Canonicalising is idempotent, so dedup is stable.
			c1 := canonicalSpecValue(tc.raw)
			if c2 := canonicalSpecValue(c1); c1 != c2 {
				t.Errorf("canonical not idempotent: %q then %q", c1, c2)
			}
		})
	}
}

func TestCanonicalSpecValueDedupsOrderAndSpacing(t *testing.T) {
	// The modes are a set, not an order: common_speculative_init builds a
	// bitmask and walks a fixed priority, so these are the same cell.
	a := canonicalSpecValue("draft-mtp+ngram-mod:assist_n_max=64,draft_max=3")
	b := canonicalSpecValue(" ngram-mod + draft-mtp : draft_max=3 , assist_n_max=64 ")
	if a != b {
		t.Errorf("values differing only in order/spacing did not dedup:\n%q\n%q", a, b)
	}
	if !strings.HasPrefix(a, "draft-mtp+ngram-mod:") {
		t.Errorf("canonical form should put the draft method first, got %q", a)
	}
}

func TestParseSpecValueRejections(t *testing.T) {
	cases := []struct{ raw, wantSubstr string }{
		{"draft+draft-mtp", "both draft methods"},
		{"ngram-mod+ngram-simple", "both n-gram methods"},
		{"draft-mtp+ngram-mod+ngram-simple", "at most one draft method"},
		{"draft-mtp:assist_n_max=64", "is not a setting"},
		{"ngram-mod:draft_p_min=0.5", "is not a setting"},
		{"draft-mtp:draft_max=abc", "is not an integer"},
		{"draft:draft_p_min=nope", "is not a number"},
		{"nonsense", "is not a speculative decoding mode"},
		// "none" is a real llama.cpp value that poisons a list; it is only
		// ever the standalone off entry, never a list member.
		{"none+ngram-mod", "is not a speculative decoding mode"},
	}
	for _, tc := range cases {
		t.Run(tc.raw, func(t *testing.T) {
			_, err := parseSpecValue(tc.raw)
			if err == nil {
				t.Fatalf("parse %q: want an error", tc.raw)
			}
			if !strings.Contains(err.Error(), tc.wantSubstr) {
				t.Errorf("error = %q, want it to mention %q", err, tc.wantSubstr)
			}
		})
	}
}

func TestApplySpecValueWritesBothSlots(t *testing.T) {
	var o ConfigOverrides
	if err := applySpecValue(&o, "draft-mtp+ngram-mod:draft_max=3,assist_n_max=64"); err != nil {
		t.Fatal(err)
	}
	if o.SpecType == nil || *o.SpecType != "draft-mtp" {
		t.Errorf("SpecType = %v, want draft-mtp", o.SpecType)
	}
	if o.SpecAssist == nil || *o.SpecAssist != "ngram-mod" {
		t.Errorf("SpecAssist = %v, want ngram-mod", o.SpecAssist)
	}
	if o.DraftMax == nil || *o.DraftMax != 3 || o.AssistNMax == nil || *o.AssistNMax != 64 {
		t.Errorf("parameters not applied: %+v", o)
	}

	// A value naming one mode must clear the other slot, not inherit it:
	// "MTP alone" has to actually mean alone, even on a model whose saved
	// config has an assist.
	var solo ConfigOverrides
	if err := applySpecValue(&solo, "draft-mtp:draft_max=3"); err != nil {
		t.Fatal(err)
	}
	if solo.SpecAssist == nil || *solo.SpecAssist != "" {
		t.Errorf("SpecAssist = %v, want an explicit empty string", solo.SpecAssist)
	}

	var off ConfigOverrides
	if err := applySpecValue(&off, "none"); err != nil {
		t.Fatal(err)
	}
	if off.SpecType == nil || *off.SpecType != "" || off.SpecAssist == nil || *off.SpecAssist != "" {
		t.Errorf("none should clear both slots, got %+v", off)
	}
}

func TestSpecChoicesCoverEveryMode(t *testing.T) {
	got := map[string]string{}
	for _, c := range sweepFields["spec_type"].Choices {
		got[c.Mode] = c.Slot
	}
	// Eleven entries: five draft, five n-gram, and off (which has no mode).
	if n := len(sweepFields["spec_type"].Choices); n != 11 {
		t.Errorf("want 11 spec_type choices, got %d", n)
	}
	for _, m := range models.DraftModes() {
		if got[m.Name] != "draft" {
			t.Errorf("%s slot = %q, want draft", m.Name, got[m.Name])
		}
	}
	for _, m := range models.AssistModes() {
		if got[m.Name] != "assist" {
			t.Errorf("%s slot = %q, want assist", m.Name, got[m.Name])
		}
	}
	// Only a draft choice carries the assist dropdown, which is what makes
	// two draft methods unreachable from the form.
	for _, c := range sweepFields["spec_type"].Choices {
		if (len(c.AssistModes) > 0) != (c.Slot == "draft") {
			t.Errorf("%q: AssistModes present = %v, slot = %q", c.Value, len(c.AssistModes) > 0, c.Slot)
		}
	}
	// Every encoded choice value must parse.
	for _, c := range sweepFields["spec_type"].Choices {
		if _, err := parseSpecValue(c.Value); err != nil {
			t.Errorf("choice %q does not parse: %v", c.Value, err)
		}
	}
}

// spec_type's value separator is ";" — it already had to be, because an
// encoded value's parameters are comma-separated. "+" appears only inside
// a single value, so the form's live cell-count estimate, which splits on
// the separator, keeps counting correctly.
func TestSpecTypeSeparatorUnchanged(t *testing.T) {
	if sep := sweepFields["spec_type"].Separator; sep != ";" {
		t.Errorf("spec_type separator = %q, want %q", sep, ";")
	}
	// A combined value must survive a split on the separator intact.
	raw := "draft-mtp+ngram-mod:draft_max=3,assist_n_max=64"
	if parts := strings.Split(raw, ";"); len(parts) != 1 {
		t.Errorf("combined value split into %d parts on the separator", len(parts))
	}
	if _, err := parseSpecValue(raw); err != nil {
		t.Errorf("combined value does not parse: %v", err)
	}
}
