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
		{"small GPU", 4.2, HelperContextSmall},
		{"unknown hardware", 0, HelperContextNormal},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			cfg := HelperConfig(m, c.budget)
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
	if cfg := HelperConfig(m, 24); cfg.ContextSize != 4096 {
		t.Errorf("context = %d, want the model's 4096", cfg.ContextSize)
	}
	if cfg := HelperConfig(m, 0.5); cfg.ContextSize != 4096 {
		t.Errorf("tiny budget: context = %d, want the model's 4096", cfg.ContextSize)
	}
}
