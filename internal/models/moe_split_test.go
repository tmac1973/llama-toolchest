package models

import (
	"strings"
	"testing"
)

// flashNext is Qwen3.8-Flash-Next UD-Q4_K_XL as the registry records it
// on compute2: 48 layers, all with experts, a 26.8 GiB embedding table.
func flashNext() *Model {
	return &Model{
		SizeBytes: 111334654784, NLayers: 48, NEmbd: 2560, NHead: 24, NKVHead: 2,
		ContextLength: 262144, KVFullPerTok: 12288, AttnLayers: 12, IndexerKeyLength: 128,
		PLEBytes: 28800138240, TokenEmbdBytes: 675430400,
		ExpertCount: 512, ExpertUsedCount: 10, ExpertBytes: 77017907200, ExpertLayers: 48,
	}
}

func a4000s(n int) []int {
	out := make([]int, n)
	for i := range out {
		out[i] = 16376
	}
	return out
}

// The case that failed on compute2: experts of 32 layers in system
// memory over three 16 GiB cards. llama.cpp's own split by layer count
// asked one card for 24.9 GiB; a size-balanced split (-ts 36,6,6) loaded
// at 14.5 / 10.7 / 10.5 GiB.
func TestMoESplitBalancesFlashNext(t *testing.T) {
	m := flashNext()
	cfg := &ModelConfig{GPUAssign: "all", SplitMode: "layer", GPULayers: 999, CPUMoE: 32,
		ContextSize: 131072, KVCacheQuant: "q8_0", FlashAttention: true}

	// What llama.cpp does without a split: equal counts.
	b := VRAMBreakdownForConfigOn(m, cfg, 3)
	layers := layerGPUBytes(m, cfg, b)
	even := cardLoads(layers, []int{16, 16, 16}, []float64{0, 0, 0})
	if even[2] < 24 || even[0] > 4 {
		t.Errorf("equal split loads = %.1f, want the last card over 24 GiB of layers", even)
	}

	out := MoESplitConfig(cfg, m, a4000s(3))
	if out == cfg || out.SplitMode != "layer" || len(strings.Split(out.TensorSplit, ",")) != 3 {
		t.Fatalf("split = %+v", out)
	}
	if cfg.TensorSplit != "" {
		t.Error("MoESplitConfig changed its argument; the config form would undo a saved split")
	}
	loads, _, ok := PlanCardLoads(m, cfg, a4000s(3))
	if !ok {
		t.Fatal("PlanCardLoads did not apply")
	}
	// Balanced: no card carries more than a heavy layer or two above another.
	lo, hi := loads[0], loads[0]
	for _, l := range loads {
		lo, hi = min(lo, l), max(hi, l)
	}
	if hi-lo > 3.5 {
		t.Errorf("split %s gives %.1f GiB: unbalanced", out.TensorSplit, loads)
	}
}

// The planner checks each card, so the plan it returns fits every card
// under the split the preset writes.
func TestPlanFitChecksEachCard(t *testing.T) {
	m := flashNext()
	hw := Hardware{LogicalCores: 32, RAMTotalMiB: 64 * 1024}
	for i := range 3 {
		hw.GPUs = append(hw.GPUs, GPUSpec{Index: i, Name: "RTX A4000", VRAMTotalMiB: 16376})
	}
	for _, class := range []ContextClass{ContextMedium, ContextLong} {
		r := PlanFit(m, DefaultConfig(), hw, class)
		if !r.Fits || r.Config.CPUMoE == 0 {
			t.Fatalf("%s: %+v", class, r.Config)
		}
		loads, budgets, ok := PlanCardLoads(m, &r.Config, GPUMiB(hw))
		if !ok {
			t.Fatalf("%s: no per-card plan", class)
		}
		for i := range loads {
			if loads[i] > budgets[i] {
				t.Errorf("%s: card %d holds %.1f of %.1f GiB (cpu_moe %d)", class, i, loads[i], budgets[i], r.Config.CPUMoE)
			}
		}
	}
}

// llama.cpp counts n_layer+1 layers (the output layer last) and puts layer
// il on the first card whose cumulative share exceeds il/(n_layer+1).
func TestLayerDevicesFollowsLlamaCpp(t *testing.T) {
	got := layerDevices([]int{36, 6, 6}, 48)
	count := map[int]int{}
	for _, d := range got {
		count[d]++
	}
	// 36/48 = 0.75: layers 0..36 have il/49 < 0.75, so card 0 takes 37.
	if count[0] != 37 || count[1] != 6 || count[2] != 5 {
		t.Errorf("layers per card = %v", count)
	}
}

func TestMoESplitLeavesOtherConfigsAlone(t *testing.T) {
	m := flashNext()
	base := ModelConfig{GPUAssign: "all", SplitMode: "layer", GPULayers: 999, CPUMoE: 32, ContextSize: 32768}
	for name, mut := range map[string]func(*ModelConfig){
		"no expert offload": func(c *ModelConfig) { c.CPUMoE = 0 },
		"custom split":      func(c *ModelConfig) { c.GPUAssign, c.TensorSplit = "custom", "1,1,1" },
		"tensor split mode": func(c *ModelConfig) { c.SplitMode = "tensor" },
		"one GPU":           func(c *ModelConfig) { c.GPUAssign, c.TensorSplit = "0", "1,0,0" },
	} {
		cfg := base
		mut(&cfg)
		if out := MoESplitConfig(&cfg, m, a4000s(3)); out != &cfg {
			t.Errorf("%s: changed to %q", name, out.TensorSplit)
		}
	}
	if out := MoESplitConfig(&base, m, nil); out != &base {
		t.Error("changed without knowing the GPUs")
	}
}

// An excluded GPU (an integrated one) keeps a zero, so the preset still
// restricts the model to the dedicated cards.
func TestMoESplitKeepsExcludedGPUs(t *testing.T) {
	cfg := &ModelConfig{GPUAssign: "0-2", TensorSplit: "1,1,1,0", SplitMode: "layer", GPULayers: 999, CPUMoE: 32, ContextSize: 32768, KVCacheQuant: "q8_0"}
	out := MoESplitConfig(cfg, flashNext(), append(a4000s(3), 2048))
	parts := strings.Split(out.TensorSplit, ",")
	if len(parts) != 4 || parts[3] != "0" || parts[0] == "1" {
		t.Errorf("TensorSplit = %q", out.TensorSplit)
	}
}

// The preset carries the balanced split; without the GPUs' sizes it
// leaves llama.cpp to split by count, as before.
func TestPresetWritesTheBalancedSplit(t *testing.T) {
	m := flashNext()
	m.ID, m.FilePath = "flash", "/models/flash.gguf"
	cfg := &ModelConfig{Enabled: true, GPUAssign: "all", SplitMode: "layer", GPULayers: 999, CPUMoE: 32, ContextSize: 131072, KVCacheQuant: "q8_0"}
	cfgs := map[string]*ModelConfig{"flash": cfg}

	ini := GeneratePresetINI("/models", []*Model{m}, cfgs, Target{Backend: "cuda", GPUMiB: a4000s(3)})
	want := "tensor-split = " + MoESplitConfig(cfg, m, a4000s(3)).TensorSplit
	if !strings.Contains(ini, want) || !strings.Contains(ini, "split-mode = layer") {
		t.Errorf("preset lacks %q:\n%s", want, ini)
	}
	if ini := GeneratePresetINI("/models", []*Model{m}, cfgs, Target{Backend: "cuda"}); strings.Contains(ini, "tensor-split") {
		t.Errorf("a split was written without the GPU sizes:\n%s", ini)
	}
}
