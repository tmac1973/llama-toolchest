package models

import (
	"strings"
	"testing"
)

func TestEffectiveBatchSizesFallBackToLlamaDefaults(t *testing.T) {
	var c ModelConfig
	if got := c.EffectiveBatchSize(); got != DefaultBatchSize {
		t.Errorf("EffectiveBatchSize = %d, want %d", got, DefaultBatchSize)
	}
	if got := c.EffectiveUBatchSize(); got != DefaultUBatchSize {
		t.Errorf("EffectiveUBatchSize = %d, want %d", got, DefaultUBatchSize)
	}

	c = ModelConfig{BatchSize: 4096, UBatchSize: 1024}
	if got := c.EffectiveBatchSize(); got != 4096 {
		t.Errorf("EffectiveBatchSize = %d, want 4096", got)
	}
	if got := c.EffectiveUBatchSize(); got != 1024 {
		t.Errorf("EffectiveUBatchSize = %d, want 1024", got)
	}
}

func TestValidateBatchSizes(t *testing.T) {
	cases := []struct {
		name    string
		cfg     ModelConfig
		wantErr bool
	}{
		{"both unset", ModelConfig{}, false},
		{"ub below default b", ModelConfig{UBatchSize: 1024}, false},
		{"ub equals b", ModelConfig{BatchSize: 2048, UBatchSize: 2048}, false},
		{"ub below b", ModelConfig{BatchSize: 4096, UBatchSize: 512}, false},
		{"ub above b", ModelConfig{BatchSize: 512, UBatchSize: 1024}, true},
		// The trap this catches: raising -ub past 2048 without raising -b.
		{"ub above default b", ModelConfig{UBatchSize: 4096}, true},
		{"negative", ModelConfig{UBatchSize: -1}, true},
	}
	for _, c := range cases {
		err := c.cfg.ValidateBatchSizes()
		if (err != nil) != c.wantErr {
			t.Errorf("%s: err = %v, wantErr = %v", c.name, err, c.wantErr)
		}
	}
}

// The message must say what to change, since "-ub 4096" looks reasonable
// until you know the batch default is 2048.
func TestValidateBatchSizesExplainsDefaultCase(t *testing.T) {
	c := ModelConfig{UBatchSize: 4096}
	err := c.ValidateBatchSizes()
	if err == nil {
		t.Fatal("expected an error")
	}
	if !strings.Contains(err.Error(), "raise Batch") {
		t.Errorf("error should say how to fix it, got: %v", err)
	}
}

// Zero means "don't emit", so existing models keep llama.cpp's defaults
// and their command lines are unchanged.
func TestBatchFlagsOmittedWhenUnset(t *testing.T) {
	c := ModelConfig{GPULayers: 999, ContextSize: 8192, Threads: 8}
	flags := c.EffectiveFlagsFor(false)
	if strings.Contains(flags, "batch-size") {
		t.Errorf("unset batch sizes should emit no flags, got: %s", flags)
	}
}

func TestBatchFlagsEmittedWhenSet(t *testing.T) {
	c := ModelConfig{GPULayers: 999, ContextSize: 8192, Threads: 8,
		BatchSize: 4096, UBatchSize: 1024}
	flags := c.EffectiveFlagsFor(false)
	if !strings.Contains(flags, "--batch-size 4096") {
		t.Errorf("missing --batch-size, got: %s", flags)
	}
	if !strings.Contains(flags, "--ubatch-size 1024") {
		t.Errorf("missing --ubatch-size, got: %s", flags)
	}
}

// The preset INI and EffectiveFlagsFor are two hand-maintained emitters
// of the same settings; both must carry the new keys or the UI will
// disagree with what the router actually runs.
func TestBatchSizesReachPresetINI(t *testing.T) {
	mods := []*Model{{ID: "a", ModelID: "u/A", Quant: "Q8_0", FilePath: "/m/a.gguf"}}
	cfgs := map[string]*ModelConfig{
		"a": {Enabled: true, ContextSize: 8192, GPULayers: 999, Threads: 8,
			BatchSize: 4096, UBatchSize: 1024},
	}
	ini := GeneratePresetINI("/m", mods, cfgs)

	if !strings.Contains(ini, "batch-size = 4096") {
		t.Errorf("preset missing batch-size:\n%s", ini)
	}
	if !strings.Contains(ini, "ubatch-size = 1024") {
		t.Errorf("preset missing ubatch-size:\n%s", ini)
	}
}

func TestBatchSizesOmittedFromPresetINIWhenUnset(t *testing.T) {
	mods := []*Model{{ID: "a", ModelID: "u/A", Quant: "Q8_0", FilePath: "/m/a.gguf"}}
	cfgs := map[string]*ModelConfig{
		"a": {Enabled: true, ContextSize: 8192, GPULayers: 999, Threads: 8},
	}
	if ini := GeneratePresetINI("/m", mods, cfgs); strings.Contains(ini, "batch-size") {
		t.Errorf("unset batch sizes should not appear in the preset:\n%s", ini)
	}
}

// gpu-layers = 0 means "run on CPU", not "unset". Guarding the emitter on
// > 0 dropped the key entirely, so llama-server applied its own default
// and the model ran fully offloaded while the benchmark recorded 0 — a
// mislabeled measurement, caught only by noticing that a CPU-only cell
// was as fast as a full-GPU one.
func TestZeroGPULayersReachesPreset(t *testing.T) {
	mods := []*Model{{ID: "a", ModelID: "u/A", Quant: "Q8_0", FilePath: "/m/a.gguf"}}
	cfgs := map[string]*ModelConfig{
		"a": {Enabled: true, ContextSize: 4096, GPULayers: 0, Threads: 8},
	}
	ini := GeneratePresetINI("/m", mods, cfgs)

	if !strings.Contains(ini, "gpu-layers = 0") {
		t.Errorf("gpu-layers = 0 missing; llama-server would offload everything:\n%s", ini)
	}
}

// The UI's flag preview and the preset are two hand-maintained emitters
// of the same settings. They disagreed here: the preview showed
// --n-gpu-layers 0 while the preset omitted it.
func TestPresetAndFlagPreviewAgreeOnZero(t *testing.T) {
	cfg := &ModelConfig{Enabled: true, ContextSize: 4096, GPULayers: 0, Threads: 8}
	mods := []*Model{{ID: "a", ModelID: "u/A", Quant: "Q8_0", FilePath: "/m/a.gguf"}}
	ini := GeneratePresetINI("/m", mods, map[string]*ModelConfig{"a": cfg})
	flags := cfg.EffectiveFlagsFor(false)

	iniHasZero := strings.Contains(ini, "gpu-layers = 0")
	flagsHaveZero := strings.Contains(flags, "--n-gpu-layers 0")
	if iniHasZero != flagsHaveZero {
		t.Errorf("emitters disagree: preset=%v preview=%v\nflags: %s", iniHasZero, flagsHaveZero, flags)
	}
}
