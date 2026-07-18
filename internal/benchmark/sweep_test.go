package benchmark

import (
	"reflect"
	"testing"
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

func TestParseSweepValues(t *testing.T) {
	got, err := ParseSweepValues("gpu_layers", " 20, 40 ,99 ")
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if !reflect.DeepEqual(got, []string{"20", "40", "99"}) {
		t.Errorf("got %v, want [20 40 99]", got)
	}
}

func TestParseSweepValuesRejectsBadInput(t *testing.T) {
	cases := map[string]struct{ field, raw string }{
		"non-integer":   {"gpu_layers", "20,abc"},
		"duplicate":     {"gpu_layers", "20,20"},
		"empty":         {"gpu_layers", "  ,  "},
		"bad bool":      {"flash_attention", "yes,no"},
		"unknown field": {"nonsense", "1"},
		"non-float":     {"temperature", "warm"},
	}
	for name, c := range cases {
		if _, err := ParseSweepValues(c.field, c.raw); err == nil {
			t.Errorf("%s: expected an error for %q", name, c.raw)
		}
	}
}

// tensor_split values contain commas, so its list uses "|".
func TestParseSweepValuesTensorSplitUsesPipe(t *testing.T) {
	got, err := ParseSweepValues("tensor_split", "1,1 | 3,1")
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if !reflect.DeepEqual(got, []string{"1,1", "3,1"}) {
		t.Errorf("got %v, want [1,1 3,1]", got)
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

// Every registry entry must round-trip through its own parser, so a
// newly added field can't ship with a broken example.
func TestSweepFieldExamplesParse(t *testing.T) {
	for _, f := range SweepFields() {
		if f.Example == "" {
			continue
		}
		if _, err := ParseSweepValues(f.Name, f.Example); err != nil {
			t.Errorf("%s: example %q does not parse: %v", f.Name, f.Example, err)
		}
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
