package models

import "testing"

// A helper model gets fixed settings, with only the context sized to the
// GPU: as much as fits, never below the floor.
func TestHelperConfig(t *testing.T) {
	m := &Model{NLayers: 36, NEmbd: 2560, NHead: 20, NKVHead: 4, ContextLength: 262144, SizeBytes: 5 << 29} // about 2.5 GiB

	cases := []struct {
		name    string
		budget  float64
		wantCtx int
	}{
		{"roomy GPU", 20, HelperContextFull},
		{"middling GPU", 5, HelperContextNormal},
		// Between the estimates at 8K (3.6 GiB) and 16K (4.2 GiB).
		{"small GPU", 3.9, HelperContextSmall},
		{"unknown hardware", 0, HelperContextNormal},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			cfg := HelperConfig(m, HelperGPU{Index: -1, BudgetGiB: c.budget})
			if cfg.ContextSize != c.wantCtx {
				t.Errorf("context = %d, want %d (estimate at that context: %.1f GiB)",
					cfg.ContextSize, c.wantCtx, VRAMEstimateForConfigOn(m, &cfg, 1))
			}
			if !cfg.Enabled || cfg.GPULayers != 999 || !cfg.FlashAttention || !cfg.Jinja {
				t.Errorf("fixed settings not applied: %+v", cfg)
			}
			if cfg.SpecType != "" || cfg.SpecAssist != "" || cfg.CPUMoE != 0 || cfg.Temperature != nil {
				t.Errorf("a helper should carry no tuning or sampling settings: %+v", cfg)
			}
			if cfg.Threads < 1 {
				t.Errorf("threads = %d", cfg.Threads)
			}
		})
	}
}

// A model trained on less context than the helper wants keeps its own
// limit rather than being asked for more than it has.
func TestHelperConfigRespectsTheModelsLimit(t *testing.T) {
	m := &Model{NLayers: 24, NEmbd: 2048, NHead: 16, NKVHead: 4, ContextLength: 4096, SizeBytes: 1 << 30}
	if cfg := HelperConfig(m, HelperGPU{Index: -1, BudgetGiB: 24}); cfg.ContextSize != 4096 {
		t.Errorf("context = %d, want the model's 4096", cfg.ContextSize)
	}
	if cfg := HelperConfig(m, HelperGPU{Index: -1, BudgetGiB: 0.5}); cfg.ContextSize != 4096 {
		t.Errorf("tiny budget: context = %d, want the model's 4096", cfg.ContextSize)
	}
}

// A helper on a machine with several GPUs runs on the one with the most
// room: a card that generation backends have reserved, or that another
// program fills, is passed over, and so is an integrated GPU.
func TestHelperGPUForPicksTheRoomiestCard(t *testing.T) {
	hw := Hardware{GPUs: []GPUSpec{
		{Index: 0, VRAMTotalMiB: 16384, ReservedMiB: 15053},
		{Index: 1, VRAMTotalMiB: 16384},
		{Index: 2, VRAMTotalMiB: 16384, ReservedMiB: 7700},
		{Index: 3, VRAMTotalMiB: 32768, IsIGPU: true},
	}}
	at := HelperGPUFor(hw)
	if at.Index != 1 || at.NumGPUs != 4 {
		t.Fatalf("helper GPU = %+v, want GPU 1 of 4", at)
	}
	if want := 16.0 * 0.92; at.BudgetGiB != want {
		t.Errorf("budget = %.2f GiB, want %.2f", at.BudgetGiB, want)
	}
	m := &Model{NLayers: 36, NEmbd: 2560, NHead: 20, NKVHead: 4, ContextLength: 262144, SizeBytes: 5 << 29}
	cfg := HelperConfig(m, at)
	if cfg.GPUAssign != "1" || cfg.TensorSplit != "0,1,0,0" || cfg.MainGPU != 1 {
		t.Errorf("placement = %q split %q main %d, want GPU 1 alone", cfg.GPUAssign, cfg.TensorSplit, cfg.MainGPU)
	}
}

// With one GPU there is nothing to choose, and the helper keeps the
// default placement.
func TestHelperConfigOneGPUKeepsTheDefaultPlacement(t *testing.T) {
	at := HelperGPUFor(Hardware{GPUs: []GPUSpec{{Index: 0, VRAMTotalMiB: 16384}}})
	cfg := HelperConfig(&Model{ContextLength: 4096}, at)
	if cfg.GPUAssign != "" || cfg.TensorSplit != "" {
		t.Errorf("placement = %q split %q, want the default", cfg.GPUAssign, cfg.TensorSplit)
	}
}
