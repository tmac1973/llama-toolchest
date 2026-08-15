package models

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// A 2-GPU box where GPU 1 is an iGPU (a desktop APU next to a discrete
// card — the common case this guards).
var igpuAt1 = []bool{false, true}

func optionByValue(opts []GPUOption, value string) *GPUOption {
	for i := range opts {
		if opts[i].Value == value {
			return &opts[i]
		}
	}
	return nil
}

// iGPU guard rails: touching options stay selectable but say so, the
// spanning options are generated from discrete GPUs only, and nothing
// touching an iGPU is ever recommended.
func TestGPUAssignOptionsIGPULabels(t *testing.T) {
	opts := GPUAssignOptions(2, igpuAt1)

	if o := optionByValue(opts, "1"); o == nil || !strings.Contains(o.Label, "(iGPU)") || !o.IncludesIGPU {
		t.Errorf("iGPU single option should be labeled and marked, got %+v", o)
	}
	if o := optionByValue(opts, "0"); o == nil || o.IncludesIGPU || strings.Contains(o.Label, "iGPU") {
		t.Errorf("discrete single option must stay clean, got %+v", o)
	}
	if o := optionByValue(opts, "0-1"); o == nil || !strings.Contains(o.Label, "includes iGPU") {
		t.Errorf("range spanning the iGPU should be labeled, got %+v", o)
	}

	// "All GPUs" becomes "all discrete": value is the explicit dGPU set.
	if o := optionByValue(opts, "all"); o != nil {
		t.Errorf(`"all" must not be offered when an iGPU exists, got %+v`, o)
	}
	span := (*GPUOption)(nil)
	for i := range opts {
		if opts[i].IsSpanAll {
			span = &opts[i]
		}
	}
	if span == nil || span.Value != "0" || !strings.Contains(span.Label, "discrete") {
		t.Errorf("span option should cover only the discrete GPU, got %+v", span)
	}

	// One dGPU → no tensor variants at all.
	for _, o := range opts {
		if o.IsTensor {
			t.Errorf("tensor option offered with a single discrete GPU: %+v", o)
		}
	}
}

// With the iGPU at index 0, the discrete indices aren't a 0..N-1 prefix,
// so tensor and span options must use explicit-set values.
func TestGPUAssignOptionsIGPUFirst(t *testing.T) {
	opts := GPUAssignOptions(3, []bool{true, false, false})

	span := (*GPUOption)(nil)
	for i := range opts {
		if opts[i].IsSpanAll {
			span = &opts[i]
		}
	}
	if span == nil || span.Value != "1,2" {
		t.Fatalf("span option should be the explicit set 1,2, got %+v", span)
	}

	var tensor *GPUOption
	for i := range opts {
		if opts[i].IsTensor {
			tensor = &opts[i]
		}
	}
	if tensor == nil || tensor.Value != "tensor:1,2" {
		t.Fatalf("tensor option should use the explicit set form, got %+v", tensor)
	}

	// And both forms must resolve to splits that skip the iGPU.
	ts, sm, mg := ResolveGPUAssign("1,2", 3)
	if ts != "0,1,1" || sm != "layer" || mg != 1 {
		t.Errorf(`ResolveGPUAssign("1,2") = %q,%q,%d`, ts, sm, mg)
	}
	ts, sm, _ = ResolveGPUAssign("tensor:1,2", 3)
	if ts != "0,1,1" || sm != "tensor" {
		t.Errorf(`ResolveGPUAssign("tensor:1,2") = %q,%q`, ts, sm)
	}
}

// APU-only boxes have no discrete GPUs: the iGPU is all there is, so it
// must not be treated as a second-class device.
func TestGPUAssignOptionsAPUOnly(t *testing.T) {
	opts := GPUAssignOptions(1, []bool{true})
	if o := optionByValue(opts, "all"); o == nil || !o.IsSpanAll {
		t.Errorf(`APU-only box should keep the plain "all" option, got %+v`, opts)
	}
	for _, o := range opts {
		if strings.Contains(o.Label, "iGPU") {
			t.Errorf("APU-only box should not carry iGPU warnings: %+v", o)
		}
	}
}

// No iGPU: output identical to the historic behavior, "all" included.
func TestGPUAssignOptionsNoIGPUUnchanged(t *testing.T) {
	opts := GPUAssignOptions(2, nil)
	if o := optionByValue(opts, "all"); o == nil || !o.IsSpanAll {
		t.Errorf(`"all" option missing on an iGPU-free box`)
	}
	if o := optionByValue(opts, "tensor-2"); o == nil {
		t.Error("legacy tensor-2 value missing on an iGPU-free box")
	}
}

// MarkRecommended must never star an option that touches an iGPU, even
// when it would otherwise win on free VRAM.
func TestMarkRecommendedSkipsIGPU(t *testing.T) {
	opts := GPUAssignOptions(2, igpuAt1)
	MarkRecommended(opts, 8.0, 24.0, nil)
	for _, o := range opts {
		if o.Recommend && o.IncludesIGPU {
			t.Errorf("recommended an iGPU option: %+v", o)
		}
	}
}

// The explicit-set forms must flow through to the device list emission,
// end to end: assignment → padded split → --device.
func TestExplicitSetAssignEmitsDeviceList(t *testing.T) {
	ts, sm, mg := ResolveGPUAssign("tensor:1,2", 3)
	cfg := &ModelConfig{GPUAssign: "tensor:1,2", TensorSplit: ts, SplitMode: sm, MainGPU: mg}
	flags := cfg.EffectiveFlagsFor(false, "rocm")
	if !strings.Contains(flags, "--device ROCm1,ROCm2") {
		t.Errorf("expected a device list skipping the iGPU, got: %s", flags)
	}
	if strings.Contains(flags, "ROCm0") {
		t.Errorf("iGPU device leaked into placement: %s", flags)
	}
}

// AssignedModelGPUs must understand every value form the dropdown can
// produce — it feeds the iGPU build-target warning.
func TestAssignedModelGPUs(t *testing.T) {
	cases := []struct {
		assign string
		want   []int
	}{
		{"1,2", []int{1, 2}},
		{"tensor:1,2", []int{1, 2}},
		{"tensor-2", []int{0, 1}},
		{"0-1", []int{0, 1}},
	}
	for _, c := range cases {
		got := AssignedModelGPUs(&ModelConfig{GPUAssign: c.assign}, 3)
		if len(got) != len(c.want) {
			t.Errorf("%s: got %v, want %v", c.assign, got, c.want)
			continue
		}
		for i := range got {
			if got[i] != c.want[i] {
				t.Errorf("%s: got %v, want %v", c.assign, got, c.want)
				break
			}
		}
	}
}

// AssignGPUsOutOfRange feeds the backup restore engine's topology check.
// The legacy tensor-N form must be checked unclamped — tensorAssignGPUs
// clamps N and would pass "tensor-4" on a 2-GPU box as in range.
func TestAssignGPUsOutOfRange(t *testing.T) {
	cases := []struct {
		assign  string
		numGPUs int
		want    bool
	}{
		{"tensor-4", 2, true},
		{"tensor-2", 2, false},
		{"tensor:1,2", 2, true},
		{"tensor:0,1", 2, false},
		{"2-3", 2, true},
		{"0-1", 4, false},
		{"1,3", 3, true},
		{"all", 1, false},
		{"custom", 1, false},
		{"", 1, false},
		{"garbage", 1, false},
	}
	for _, c := range cases {
		if got := AssignGPUsOutOfRange(c.assign, c.numGPUs); got != c.want {
			t.Errorf("AssignGPUsOutOfRange(%q, %d) = %v, want %v", c.assign, c.numGPUs, got, c.want)
		}
	}
}

// ResolveConfigPaths is shared by restore-apply and pending-claim.
func TestResolveConfigPaths(t *testing.T) {
	dir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dir, "repo"), 0o755); err != nil {
		t.Fatal(err)
	}
	os.WriteFile(filepath.Join(dir, "repo", "mmproj.gguf"), []byte("x"), 0o644)
	abs := filepath.Join(dir, "repo", "mmproj.gguf")

	cfg := &ModelConfig{
		MmprojPath:     filepath.Join("repo", "mmproj.gguf"),  // relative, exists
		MtpPath:        filepath.Join("repo", "missing.gguf"), // relative, missing
		DraftModelPath: abs,                                   // absolute, exists
	}
	warnings := ResolveConfigPaths(cfg, dir)
	if cfg.MmprojPath != abs {
		t.Errorf("relative existing path should resolve: %q", cfg.MmprojPath)
	}
	if cfg.MtpPath != "" {
		t.Errorf("missing path should blank: %q", cfg.MtpPath)
	}
	if cfg.DraftModelPath != abs {
		t.Errorf("absolute existing path should stay: %q", cfg.DraftModelPath)
	}
	if len(warnings) != 1 {
		t.Errorf("want 1 warning, got %v", warnings)
	}
}
