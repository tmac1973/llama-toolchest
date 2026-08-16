package benchmark

import (
	"reflect"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/evaluate"
)

// The collapse rule: a capability preset gets ONE cell per distinct
// eval-reaching configuration, while performance presets on the same
// job keep the full fan-out.
func TestCollapseSamplingAxisCapabilityCollapsesPerfFansOut(t *testing.T) {
	cells := ExpandCellsWithSweeps(
		[]string{"m"}, []string{"b"},
		[]string{"hellaswag-quick", "internal-quick"},
		[]SweepAxis{{Field: "temperature", Values: []string{"0.7", "0.8"}}},
	)
	// capability: 1 (collapsed) + performance: 2 (full fan-out).
	if len(cells) != 3 {
		t.Fatalf("got %d cells, want 3 (1 collapsed capability + 2 performance): %+v", len(cells), cells)
	}
	var capCells, perfCells []JobCell
	for _, c := range cells {
		if c.Preset == "hellaswag-quick" {
			capCells = append(capCells, c)
		} else {
			perfCells = append(perfCells, c)
		}
	}
	if len(capCells) != 1 || len(perfCells) != 2 {
		t.Fatalf("capability/performance split = %d/%d, want 1/2", len(capCells), len(perfCells))
	}
	// The collapsed capability cell must not carry the swept sampling
	// value: recording it would mislabel an eval that never saw it.
	if _, ok := capCells[0].SweepValues["temperature"]; ok {
		t.Errorf("collapsed capability cell recorded the swept sampling value: %+v", capCells[0].SweepValues)
	}
	temps := map[string]bool{}
	for _, c := range perfCells {
		temps[c.SweepValues["temperature"]] = true
	}
	if !temps["0.7"] || !temps["0.8"] {
		t.Errorf("performance cells lost their temperature points: %+v", perfCells)
	}
}

// context_size is a load-time axis (RestartsRouter) that never reaches
// the evaluation — the "deliberately not RestartsRouter" case.
func TestCollapseContextSizeAxisCollapsesForCapability(t *testing.T) {
	cells := ExpandCellsWithSweeps(
		[]string{"m"}, []string{"b"},
		[]string{"perplexity-quick", "internal-quick"},
		[]SweepAxis{{Field: "context_size", Values: []string{"4096", "8192"}}},
	)
	if len(cells) != 3 {
		t.Fatalf("got %d cells, want 3 (1 collapsed capability + 2 performance): %+v", len(cells), cells)
	}
	for _, c := range cells {
		if c.Preset == "perplexity-quick" {
			if _, ok := c.SweepValues["context_size"]; ok {
				t.Errorf("collapsed capability cell recorded context_size: %+v", c.SweepValues)
			}
		}
	}
}

// An eval-reaching axis (kv_cache_quant) fans capability cells out
// normally — each point is a different invocation.
func TestCollapseEvalReachingAxisFansOutCapability(t *testing.T) {
	cells := ExpandCellsWithSweeps(
		[]string{"m"}, []string{"b"},
		[]string{"hellaswag-quick"},
		[]SweepAxis{{Field: "kv_cache_quant", Values: []string{"f16", "q8_0"}}},
	)
	if len(cells) != 2 {
		t.Fatalf("got %d cells, want 2: %+v", len(cells), cells)
	}
	qs := map[string]bool{}
	for _, c := range cells {
		qs[c.SweepValues["kv_cache_quant"]] = true
	}
	if !qs["f16"] || !qs["q8_0"] {
		t.Errorf("capability cells lost their kv_cache_quant points: %+v", cells)
	}
}

// The tensor_split axis reaches the evaluation through placement
// resolution, so it fans capability cells out too.
func TestCollapseTensorSplitAxisFansOutCapability(t *testing.T) {
	cells := ExpandCellsWithSweeps(
		[]string{"m"}, []string{"b"},
		[]string{"hellaswag-quick"},
		[]SweepAxis{{Field: "tensor_split", Values: []string{"1", "1,1"}}},
	)
	if len(cells) != 2 {
		t.Fatalf("got %d cells, want 2: %+v", len(cells), cells)
	}
	ts := map[string]bool{}
	for _, c := range cells {
		ts[c.SweepValues["tensor_split"]] = true
	}
	if !ts["1"] || !ts["1,1"] {
		t.Errorf("capability cells lost their tensor_split points: %+v", cells)
	}
}

// Mixed axes on one job: temperature (eval-inert) collapses, gpu_assign
// (eval-reaching) fans out — the capability preset gets one cell per
// distinct gpu_assign value, and the performance preset gets the full
// 2×2 fan-out.
func TestCollapseMixedAxesOnlyEvalReachingVaryCapability(t *testing.T) {
	cells := ExpandCellsWithSweeps(
		[]string{"m"}, []string{"b"},
		[]string{"hellaswag-quick", "internal-quick"},
		[]SweepAxis{
			{Field: "temperature", Values: []string{"0.7", "0.8"}},
			{Field: "gpu_assign", Values: []string{"all", "0"}},
		},
	)
	// capability: 2 (one per gpu_assign value) + performance: 4.
	if len(cells) != 6 {
		t.Fatalf("got %d cells, want 6: %+v", len(cells), cells)
	}
	var capCells []JobCell
	for _, c := range cells {
		if c.Preset == "hellaswag-quick" {
			capCells = append(capCells, c)
		}
	}
	if len(capCells) != 2 {
		t.Fatalf("capability cells = %d, want 2", len(capCells))
	}
	for _, c := range capCells {
		if _, ok := c.SweepValues["temperature"]; ok {
			t.Errorf("capability cell recorded the eval-inert temperature: %+v", c.SweepValues)
		}
		if c.SweepValues["gpu_assign"] == "" {
			t.Errorf("capability cell lost its gpu_assign point: %+v", c.SweepValues)
		}
	}
}

// The collapse is per preset: two capability presets on an inert sweep
// each get their own single cell (not one shared cell), and both keep
// the identity of the config they ran at.
func TestCollapseTwoCapabilityPresetsEachGetTheirOwnCollapsedCell(t *testing.T) {
	cells := ExpandCellsWithSweeps(
		[]string{"m"}, []string{"b"},
		[]string{"hellaswag-quick", "winogrande-quick"},
		[]SweepAxis{{Field: "top_p", Values: []string{"0.9", "1.0"}}},
	)
	if len(cells) != 2 {
		t.Fatalf("got %d cells, want 2: %+v", len(cells), cells)
	}
	presets := map[string]bool{}
	for _, c := range cells {
		presets[c.Preset] = true
		if len(c.SweepValues) != 0 {
			t.Errorf("capability cell recorded eval-inert values: %+v", c.SweepValues)
		}
	}
	if !presets["hellaswag-quick"] || !presets["winogrande-quick"] {
		t.Errorf("presets collapsed into each other: %+v", cells)
	}
}

// Identity stability: the collapsed cells carry only eval-reaching
// sweep values, so identify() keeps distinguishing them from
// performance cells of the same model/build and from each other.
func TestCollapseIdentityStable(t *testing.T) {
	cells := ExpandCellsWithSweeps(
		[]string{"m"}, []string{"b"},
		[]string{"hellaswag-quick", "internal-quick"},
		[]SweepAxis{
			{Field: "temperature", Values: []string{"0.7", "0.8"}},
			{Field: "kv_cache_quant", Values: []string{"f16", "q8_0"}},
		},
	)
	// capability: 2 (kv_cache points) + performance: 4.
	if len(cells) != 6 {
		t.Fatalf("got %d cells, want 6", len(cells))
	}
	ids := map[cellIdentity]int{}
	for _, c := range cells {
		ids[identify(c)]++
	}
	if len(ids) != 6 {
		t.Errorf("identities collide: %v", ids)
	}
}

// The full matrix on a no-sweep job is unchanged by the collapse work:
// every (model, build, preset) point exists exactly once.
func TestNoSweepExpansionUnchanged(t *testing.T) {
	cells := ExpandCellsWithSweeps(
		[]string{"m1", "m2"}, []string{"b"},
		[]string{"perplexity-quick", "kl-divergence-quick", "hellaswag-quick", "internal-quick"},
		nil,
	)
	if len(cells) != 8 {
		t.Fatalf("got %d cells, want 8", len(cells))
	}
	got := map[string]bool{}
	for _, c := range cells {
		got[c.ModelID+"/"+c.Preset] = true
	}
	if len(got) != 8 {
		t.Errorf("cell set changed: %v", got)
	}
}

// Sanity of the registry itself: exactly the fields that reach the eval
// command line are marked AffectsEval — the allow-list is the single
// source, so a regression here silently mislabels capability cells.
func TestSweepFieldAffectsEvalClassification(t *testing.T) {
	affects := map[string]bool{}
	for _, f := range SweepFields() {
		affects[f.Name] = f.AffectsEval
	}
	wantTrue := map[string]bool{
		"gpu_layers": true, "ubatch_size": true, "batch_size": true,
		"threads": true, "flash_attention": true, "direct_io": true,
		"kv_cache_quant": true, "gpu_assign": true, "tensor_split": true,
	}
	wantFalse := map[string]bool{
		"context_size": false, "spec_type": false,
		"temperature": false, "top_p": false, "min_p": false,
		"repeat_penalty": false, "top_k": false,
	}
	for name, want := range wantTrue {
		if affects[name] != want {
			t.Errorf("%s: AffectsEval = %v, want %v", name, affects[name], want)
		}
	}
	for name, want := range wantFalse {
		if affects[name] != want {
			t.Errorf("%s: AffectsEval = %v, want %v", name, affects[name], want)
		}
	}
	if len(affects) != len(wantTrue)+len(wantFalse) {
		t.Errorf("registry size = %d, want %d — a new field was added without an AffectsEval classification",
			len(affects), len(wantTrue)+len(wantFalse))
	}
}

// The presets themselves: eight capability presets, four modes, quick
// and full limits, and the labels follow the existing duration-hint
// style so the preset lists render without special-casing.
func TestCapabilityPresetsDefined(t *testing.T) {
	var caps []Preset
	for _, p := range Presets() {
		if p.EffectiveSource() == PresetSourceCapability {
			caps = append(caps, p)
		}
	}
	if len(caps) != 8 {
		t.Fatalf("got %d capability presets, want 8: %v", len(caps), presetNames(caps))
	}
	byName := map[string]Preset{}
	for _, p := range caps {
		byName[p.Name] = p
	}
	checks := []struct {
		name     string
		mode     string
		tasks    int
		chunks   int
	}{
		{"perplexity-quick", "perplexity", 0, 100},
		{"perplexity-full", "perplexity", 0, 0},
		{"kl-divergence-quick", "kl-divergence", 0, 100},
		{"kl-divergence-full", "kl-divergence", 0, 0},
		{"hellaswag-quick", "hellaswag", 400, 0},
		{"hellaswag-full", "hellaswag", 0, 0},
		{"winogrande-quick", "winogrande", 400, 0},
		{"winogrande-full", "winogrande", 0, 0},
	}
	for _, c := range checks {
		p, ok := byName[c.name]
		if !ok {
			t.Errorf("missing capability preset %q", c.name)
			continue
		}
		if p.EvalMode != evaluate.Mode(c.mode) {
			t.Errorf("%s: EvalMode = %q, want %q", c.name, p.EvalMode, c.mode)
		}
		if p.EvalTasks != c.tasks || p.EvalChunks != c.chunks {
			t.Errorf("%s: limits = tasks %d / chunks %d, want %d / %d", c.name, p.EvalTasks, p.EvalChunks, c.tasks, c.chunks)
		}
		if p.Label == "" || p.Description == "" {
			t.Errorf("%s: label or description empty", c.name)
		}
	}
}

func presetNames(ps []Preset) []string {
	out := make([]string, len(ps))
	for i, p := range ps {
		out[i] = p.Name
	}
	return out
}

// GetPreset must find capability presets by name (the cell loop and the
// form both resolve through it), and a cell built from a capability
// preset must resolve through it to the right mode.
func TestGetPresetFindsCapabilityPresets(t *testing.T) {
	for _, name := range []string{"perplexity-quick", "kl-divergence-full", "hellaswag-quick", "winogrande-full"} {
		p := GetPreset(name)
		if p.Name != name {
			t.Errorf("GetPreset(%q) = %q, fell back", name, p.Name)
		}
		if p.EffectiveSource() != PresetSourceCapability {
			t.Errorf("GetPreset(%q).EffectiveSource() = %q, want capability", name, p.EffectiveSource())
		}
	}
}

// A job whose only variation is a sweep that never reaches the
// evaluation: expansion must not produce a matrix of identical runs —
// but it is the VALIDATION that refuses such a job, so expansion alone
// still returns one (collapsed) cell per capability preset.
func TestExpansionOfInertSweepProducesOneCellPerCapabilityPreset(t *testing.T) {
	cells := ExpandCellsWithSweeps(
		[]string{"m"}, []string{"b"},
		[]string{"perplexity-quick"},
		[]SweepAxis{{Field: "temperature", Values: []string{"0.1", "0.5", "0.9"}}},
	)
	if len(cells) != 1 {
		t.Fatalf("got %d cells, want 1: %+v", len(cells), cells)
	}
	if !reflect.DeepEqual(cells[0].SweepValues, map[string]string(nil)) {
		t.Errorf("collapsed cell should carry no sweep values, got %+v", cells[0].SweepValues)
	}
}
