package models

import (
	"strings"
	"testing"
)

// A GPU-subset assignment must reach llama-server as a device list, not
// just a zero-padded tensor-split. A zero split entry keeps weights off
// a GPU but llama.cpp still initializes it, and under split-mode tensor
// the excluded GPUs spin in the collective ops at full utilization while
// the all-reduce falls back to the slow butterfly path (issue observed
// with --tensor-split 1,1,0,0 on a 4-GPU ROCm box).
func TestPresetEmitsDeviceListForTensorSubset(t *testing.T) {
	mods := []*Model{{ID: "a", ModelID: "u/A", Quant: "Q8_0", FilePath: "/m/a.gguf"}}
	cfgs := map[string]*ModelConfig{
		"a": {Enabled: true, ContextSize: 8192, GPULayers: 999, Threads: 8,
			GPUAssign: "tensor-2", TensorSplit: "1,1,0,0", SplitMode: "tensor"},
	}
	ini := GeneratePresetINI("/m", mods, cfgs, "rocm")

	if !strings.Contains(ini, "device = ROCm0,ROCm1") {
		t.Errorf("preset missing device list:\n%s", ini)
	}
	// The visible devices are renumbered 0..N-1, so the split must be
	// trimmed to the active entries.
	if !strings.Contains(ini, "tensor-split = 1,1\n") {
		t.Errorf("preset should carry the trimmed split:\n%s", ini)
	}
	if strings.Contains(ini, "1,1,0,0") {
		t.Errorf("padded split must not survive alongside the device list:\n%s", ini)
	}
	if !strings.Contains(ini, "split-mode = tensor") {
		t.Errorf("split-mode lost:\n%s", ini)
	}
}

// A layer-split range like "2-3" renumbers its first GPU to device index
// 0, so the physical main-gpu index must be dropped with the padding.
func TestPresetDeviceListDropsMainGPU(t *testing.T) {
	mods := []*Model{{ID: "a", ModelID: "u/A", Quant: "Q8_0", FilePath: "/m/a.gguf"}}
	cfgs := map[string]*ModelConfig{
		"a": {Enabled: true, ContextSize: 8192, GPULayers: 999, Threads: 8,
			GPUAssign: "2-3", TensorSplit: "0,0,1,1", SplitMode: "layer", MainGPU: 2},
	}
	ini := GeneratePresetINI("/m", mods, cfgs, "cuda")

	if !strings.Contains(ini, "device = CUDA2,CUDA3") {
		t.Errorf("preset missing device list:\n%s", ini)
	}
	if strings.Contains(ini, "main-gpu") {
		t.Errorf("main-gpu is a physical index; with a device list the first device is already 0:\n%s", ini)
	}
	if !strings.Contains(ini, "tensor-split = 1,1\n") {
		t.Errorf("preset should carry the trimmed split:\n%s", ini)
	}
}

// "custom" means the user typed the split by hand — emit it verbatim
// with no device restriction, whatever it contains.
func TestPresetCustomSplitStaysRaw(t *testing.T) {
	mods := []*Model{{ID: "a", ModelID: "u/A", Quant: "Q8_0", FilePath: "/m/a.gguf"}}
	cfgs := map[string]*ModelConfig{
		"a": {Enabled: true, ContextSize: 8192, GPULayers: 999, Threads: 8,
			GPUAssign: "custom", TensorSplit: "3,1,0,0"},
	}
	ini := GeneratePresetINI("/m", mods, cfgs, "rocm")

	if strings.Contains(ini, "device =") {
		t.Errorf("custom assignment must not emit a device list:\n%s", ini)
	}
	if !strings.Contains(ini, "tensor-split = 3,1,0,0") {
		t.Errorf("custom split must be preserved verbatim:\n%s", ini)
	}
}

// Backends without addressable device names (cpu, metal, unknown) keep
// the legacy padded emission so nothing breaks when the build's backend
// can't be resolved.
func TestPresetFallsBackToPaddedSplitWithoutBackend(t *testing.T) {
	mods := []*Model{{ID: "a", ModelID: "u/A", Quant: "Q8_0", FilePath: "/m/a.gguf"}}
	cfgs := map[string]*ModelConfig{
		"a": {Enabled: true, ContextSize: 8192, GPULayers: 999, Threads: 8,
			GPUAssign: "2-3", TensorSplit: "0,0,1,1", SplitMode: "layer", MainGPU: 2},
	}
	ini := GeneratePresetINI("/m", mods, cfgs, "")

	if strings.Contains(ini, "device =") {
		t.Errorf("no backend, no device list:\n%s", ini)
	}
	if !strings.Contains(ini, "tensor-split = 0,0,1,1") {
		t.Errorf("legacy padded split lost:\n%s", ini)
	}
	if !strings.Contains(ini, "main-gpu = 2") {
		t.Errorf("legacy main-gpu lost:\n%s", ini)
	}
}

// The preset INI and EffectiveFlagsFor are two hand-maintained emitters
// of the same settings; the device list must appear in both or the UI
// preview lies about what the router runs.
func TestFlagPreviewAgreesOnDeviceList(t *testing.T) {
	cfg := &ModelConfig{Enabled: true, ContextSize: 8192, GPULayers: 999, Threads: 8,
		GPUAssign: "tensor-2", TensorSplit: "1,1,0,0", SplitMode: "tensor"}
	flags := cfg.EffectiveFlagsFor(false, "rocm")

	for _, want := range []string{"--device ROCm0,ROCm1", "--tensor-split 1,1", "--split-mode tensor", "--fit off"} {
		if !strings.Contains(flags, want) {
			t.Errorf("flags missing %q, got: %s", want, flags)
		}
	}
	if strings.Contains(flags, "1,1,0,0") {
		t.Errorf("padded split leaked into flags: %s", flags)
	}
}

// A split that doesn't exclude anything ("1,1") needs no device list —
// emitting one would churn every existing all-GPU config.
func TestNoDeviceListWhenSplitCoversAllGPUs(t *testing.T) {
	if got := SplitDeviceIndices("1,1"); got != nil {
		t.Errorf("all-active split should not restrict, got %v", got)
	}
	if got := SplitDeviceIndices(""); got != nil {
		t.Errorf("empty split should not restrict, got %v", got)
	}
	got := SplitDeviceIndices("0,0,1,1")
	if len(got) != 2 || got[0] != 2 || got[1] != 3 {
		t.Errorf("want [2 3], got %v", got)
	}
}

func TestDeviceListBackends(t *testing.T) {
	cases := []struct {
		backend string
		want    string
	}{
		{"rocm", "ROCm0,ROCm2"},
		{"cuda", "CUDA0,CUDA2"},
		{"vulkan", "Vulkan0,Vulkan2"},
		{"metal", ""},
		{"cpu", ""},
		{"", ""},
	}
	for _, c := range cases {
		if got := DeviceList(c.backend, []int{0, 2}); got != c.want {
			t.Errorf("DeviceList(%q) = %q, want %q", c.backend, got, c.want)
		}
	}
}

// GPUPlacementFlags is the exported CLI-formatted view of
// gpuPlacementParams: the same placement the preset INI and
// EffectiveFlagsFor emit, as "--name value" pairs for command lines
// (the capability-evaluation flag path). It must agree with
// EffectiveFlagsFor on what a GPU subset produces.
func TestGPUPlacementFlagsDeviceList(t *testing.T) {
	cfg := &ModelConfig{Enabled: true, ContextSize: 8192, GPULayers: 999, Threads: 8,
		GPUAssign: "tensor-2", TensorSplit: "1,1,0,0", SplitMode: "tensor"}
	got := GPUPlacementFlags(cfg, "rocm")
	want := []string{"--device", "ROCm0,ROCm1", "--tensor-split", "1,1", "--split-mode", "tensor"}
	if len(got) != len(want) {
		t.Fatalf("GPUPlacementFlags = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("GPUPlacementFlags[%d] = %q, want %q (full: %v)", i, got[i], want[i], got)
		}
	}
	// Cross-check against the llama-server arg builder: every pair the
	// wrapper emits must appear in EffectiveFlagsFor verbatim.
	flags := cfg.EffectiveFlagsFor(false, "rocm")
	for i := 0; i+1 < len(got); i += 2 {
		if !strings.Contains(flags, got[i]+" "+got[i+1]) {
			t.Errorf("wrapper emits %q %q but EffectiveFlagsFor lacks it: %s", got[i], got[i+1], flags)
		}
	}
}

// A custom split without a device-name backend falls back to the
// padded split plus split-mode / main-gpu, in the wrapper's CLI form.
func TestGPUPlacementFlagsPaddedSplit(t *testing.T) {
	cfg := &ModelConfig{Enabled: true, ContextSize: 4096, GPULayers: 999, Threads: 8,
		TensorSplit: "0,0,1,1", SplitMode: "layer", MainGPU: 2}
	got := GPUPlacementFlags(cfg, "rocm")
	want := []string{"--tensor-split", "0,0,1,1", "--split-mode", "layer", "--main-gpu", "2"}
	if len(got) != len(want) {
		t.Fatalf("GPUPlacementFlags = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("GPUPlacementFlags[%d] = %q, want %q (full: %v)", i, got[i], want[i], got)
		}
	}
}

// A config with nothing to place emits nothing — the wrapper returns
// nil rather than an empty-but-non-nil slice.
func TestGPUPlacementFlagsEmpty(t *testing.T) {
	if got := GPUPlacementFlags(&ModelConfig{Enabled: true, ContextSize: 4096, GPULayers: 999}, "rocm"); got != nil {
		t.Errorf("unrestricted config should emit no placement flags, got %v", got)
	}
}
