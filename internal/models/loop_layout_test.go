package models

import (
	"strings"
	"testing"
)

// inklingLoopConfig is Inkling 2 Mini over every A4000 on compute2 with
// its loop core on one card, as measured on 2026-11-23.
func inklingLoopConfig(ctx int) *ModelConfig {
	return &ModelConfig{GPUAssign: "all", SplitMode: "layer", GPULayers: 999, LoopCoreOneGPU: true,
		ContextSize: ctx, UBatchSize: 256, BatchSize: 2048, KVCacheQuant: "q8_0", FlashAttention: true}
}

// The layout measured on compute2: blocks 0-5 on GPU 0, the loop core
// and every cache slot on GPU 1, the output layer on GPU 2, and blocks
// 18-23 moved off GPU 1. 126 graph splits and 29 tokens per second,
// against 592 and 16 with the plain split.
func TestLoopLayoutOnThreeCards(t *testing.T) {
	m := inklingMini()
	out := MoESplitConfig(inklingLoopConfig(65536), m, a4000s(3))
	if out.TensorSplit != "6,42,1" {
		t.Errorf("split %q, want 6,42,1", out.TensorSplit)
	}
	ot := loopOverride(out.LoopBlocks, "cuda")
	for _, il := range []string{"18", "19", "20", "21", "22", "23"} {
		if !strings.Contains(ot, il) {
			t.Errorf("override %q does not move block %s", ot, il)
		}
	}
	if strings.Contains(ot, "CUDA1") || strings.Contains(ot, "(6|") {
		t.Errorf("override %q moves something onto the loop card or out of the loop core", ot)
	}
	flags := out.EffectiveFlagsFor(KindChat, Target{Backend: "cuda"})
	if !strings.Contains(flags, "--override-tensor") || !strings.Contains(flags, "--tensor-split 6,42,1") {
		t.Errorf("flags %q lack the layout", flags)
	}
}

// The loop card holds the most. Measured at 64K and 96K, with about
// 0.3 GiB of CUDA context per card that the estimate leaves to the margin.
func TestLoopLayoutLoads(t *testing.T) {
	m := inklingMini()
	for _, tt := range []struct {
		ctx      int
		measured [3]float64 // GiB, nvidia-smi
	}{
		{65536, [3]float64{6.88, 11.52, 9.21}},
		{98304, [3]float64{8.49, 13.53, 10.82}},
	} {
		loads, budgets, ok := PlanCardLoads(m, inklingLoopConfig(tt.ctx), a4000s(3))
		if !ok {
			t.Fatalf("%d: no per-card loads", tt.ctx)
		}
		t.Logf("ctx %d: loads %.2f / %.2f / %.2f GiB, measured %v", tt.ctx, loads[0], loads[1], loads[2], tt.measured)
		if loads[1] < tt.measured[1]-0.5 || loads[1] > tt.measured[1]+2 {
			t.Errorf("ctx %d: loop card %.2f GiB, measured %.2f", tt.ctx, loads[1], tt.measured[1])
		}
		for i := range loads {
			if loads[i] > budgets[i] {
				t.Errorf("ctx %d: card %d over its budget (%.2f > %.2f)", tt.ctx, i, loads[i], budgets[i])
			}
		}
	}
	// 128K does not fit on the loop card.
	loads, budgets, _ := PlanCardLoads(m, inklingLoopConfig(131072), a4000s(3))
	if loads[1] <= budgets[1] {
		t.Errorf("128K: loop card %.2f GiB within %.2f; on compute2 it did not fit", loads[1], budgets[1])
	}
}

// Two cards: the loop card is the one that leaves the fuller card less
// full, and everything outside the loop core that lands on it moves.
func TestLoopLayoutOnTwoCards(t *testing.T) {
	m := inklingMini()
	cfg := inklingLoopConfig(32768)
	cfg.GPUAssign, cfg.TensorSplit = "1-2", "0,1,1"
	out := MoESplitConfig(cfg, m, a4000s(3))
	parts := strings.Split(out.TensorSplit, ",")
	if len(parts) != 3 || parts[0] != "0" {
		t.Fatalf("split %q, want GPU 0 left out", out.TensorSplit)
	}
	if out.LoopBlocks == "" {
		t.Errorf("nothing moved on two cards: split %q", out.TensorSplit)
	}
	if strings.Contains(loopOverride(out.LoopBlocks, "cuda"), "CUDA0") {
		t.Errorf("a block moved to GPU 0, which the config leaves out")
	}
}

// Without the setting, with experts in system memory, or with a tensor
// override of the user's own, the plain split stays.
func TestLoopLayoutOnlyWhenAsked(t *testing.T) {
	m := inklingMini()
	for name, change := range map[string]func(*ModelConfig){
		"off":          func(c *ModelConfig) { c.LoopCoreOneGPU = false },
		"cpu experts":  func(c *ModelConfig) { c.CPUMoE = 2 },
		"extra -ot":    func(c *ModelConfig) { c.ExtraFlags = `-ot "exps=CPU"` },
		"custom split": func(c *ModelConfig) { c.GPUAssign, c.TensorSplit = "custom", "9,10,29" },
	} {
		cfg := inklingLoopConfig(65536)
		change(cfg)
		if out := MoESplitConfig(cfg, m, a4000s(3)); out.LoopBlocks != "" {
			t.Errorf("%s: layout applied (%q)", name, out.TensorSplit)
		}
	}
	// A model without a loop has no layout.
	cfg := inklingLoopConfig(65536)
	if LoopLayoutApplies(dense20(), cfg, 3) {
		t.Error("layout applies to a model without a loop")
	}
}

func compute2Whole() Hardware {
	hw := Hardware{LogicalCores: 32, RAMTotalMiB: 64 * 1024}
	for i := range 3 {
		hw.GPUs = append(hw.GPUs, GPUSpec{Index: i, Name: "RTX A4000", VRAMTotalMiB: 16376, MarginMiB: WorkingMemoryMiB})
	}
	return hw
}

// Asked for the loop core on one GPU, the plan keeps it there and fits
// the context to the loop card: on compute2 96K ran, and the estimate
// plans a little below that. The plain split fits 128K.
func TestPlanFitWithTheLoopCoreOnOneGPU(t *testing.T) {
	m := inklingMini()
	base := DefaultConfig()
	base.LoopCoreOneGPU = true
	r := PlanFit(m, base, compute2Whole(), ContextMax)
	if !r.Fits || !r.Config.LoopCoreOneGPU {
		t.Fatalf("plan %+v, want the loop core on one GPU", r.Config)
	}
	if c := r.Config.ContextSize; c < 81920 || c > 98304 {
		t.Errorf("context %d, want between 80K and 96K", c)
	}
	if r.Config.CPUMoE != 0 {
		t.Errorf("experts of %d layers in system memory; they end the layout", r.Config.CPUMoE)
	}
	found := false
	for _, n := range r.Notes {
		if n.Field == "loop_core_one_gpu" && strings.Contains(n.Reason, "GPU 1") {
			found = true
		}
	}
	if !found {
		t.Errorf("no note names the loop card: %+v", r.Notes)
	}

	plain := PlanFit(m, DefaultConfig(), compute2Whole(), ContextMax)
	if plain.Config.ContextSize <= r.Config.ContextSize {
		t.Errorf("plain split %d tokens, loop core %d: the plain split should hold more", plain.Config.ContextSize, r.Config.ContextSize)
	}
}

// Where the loop core fits on no card at any context, the plan splits
// the layers the usual way and says why.
func TestPlanFitFallsBackWhenTheLoopCoreDoesNotFit(t *testing.T) {
	m := inklingMini()
	hw := Hardware{LogicalCores: 32, RAMTotalMiB: 64 * 1024}
	for i := range 4 {
		hw.GPUs = append(hw.GPUs, GPUSpec{Index: i, Name: "8 GiB", VRAMTotalMiB: 8192, MarginMiB: WorkingMemoryMiB})
	}
	base := DefaultConfig()
	base.LoopCoreOneGPU = true
	r := PlanFit(m, base, hw, ContextMedium)
	if !r.Fits {
		t.Fatalf("does not fit at all: %+v", r)
	}
	if r.Config.LoopCoreOneGPU {
		t.Errorf("kept the loop core on one 8 GiB card")
	}
	found := false
	for _, n := range r.Notes {
		if n.Field == "loop_core_one_gpu" && strings.Contains(n.Reason, "does not fit") {
			found = true
		}
	}
	if !found {
		t.Errorf("no note says why: %+v", r.Notes)
	}
}
