package api

import (
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/benchmark"
	"github.com/tmac1973/llama-toolchest/internal/models"
)

func baseConfig() models.ModelConfig {
	return models.ModelConfig{
		Enabled:        true,
		ContextSize:    8192,
		GPULayers:      999,
		Threads:        8,
		FlashAttention: true,
		KVCacheQuant:   "q8_0",
		GPUAssign:      "all",
		Aliases:        []string{"keep-me"},
	}
}

func TestApplySnapshotOverridesSetFields(t *testing.T) {
	snap := benchmark.ConfigSnapshot{
		ContextSize:    65536,
		GPULayers:      50,
		Threads:        16,
		TensorSplit:    "1,1",
		KVCacheQuant:   "q4_0",
		FlashAttention: true,
	}
	got := applySnapshotToConfig(baseConfig(), snap)

	if got.ContextSize != 65536 {
		t.Errorf("ContextSize = %d, want 65536", got.ContextSize)
	}
	if got.GPULayers != 50 {
		t.Errorf("GPULayers = %d, want 50", got.GPULayers)
	}
	if got.Threads != 16 {
		t.Errorf("Threads = %d, want 16", got.Threads)
	}
	if got.TensorSplit != "1,1" {
		t.Errorf("TensorSplit = %q, want 1,1", got.TensorSplit)
	}
	if got.KVCacheQuant != "q4_0" {
		t.Errorf("KVCacheQuant = %q, want q4_0", got.KVCacheQuant)
	}
}

// Fields the snapshot doesn't model must survive untouched — the
// override applies to a copy of the saved config, not a fresh struct.
func TestApplySnapshotPreservesUnmodeledFields(t *testing.T) {
	got := applySnapshotToConfig(baseConfig(), benchmark.ConfigSnapshot{ContextSize: 4096})

	if !got.Enabled {
		t.Error("Enabled was dropped")
	}
	if len(got.Aliases) != 1 || got.Aliases[0] != "keep-me" {
		t.Errorf("Aliases = %v, want [keep-me]", got.Aliases)
	}
	if got.GPUAssign != "all" {
		t.Errorf("GPUAssign = %q, want all", got.GPUAssign)
	}
}

// A zero-valued snapshot field means "not overridden" and must leave the
// saved value alone, otherwise a partial override would blank out config
// the user set.
func TestApplySnapshotZeroValuesDoNotClobber(t *testing.T) {
	got := applySnapshotToConfig(baseConfig(), benchmark.ConfigSnapshot{})

	if got.ContextSize != 8192 {
		t.Errorf("ContextSize = %d, want saved 8192", got.ContextSize)
	}
	if got.GPULayers != 999 {
		t.Errorf("GPULayers = %d, want saved 999", got.GPULayers)
	}
	if got.Threads != 8 {
		t.Errorf("Threads = %d, want saved 8", got.Threads)
	}
	if got.KVCacheQuant != "q8_0" {
		t.Errorf("KVCacheQuant = %q, want saved q8_0", got.KVCacheQuant)
	}
}

// The input must not be mutated — callers pass a value copy of the
// registry's live config and the registry must not observe the change.
func TestApplySnapshotDoesNotMutateInput(t *testing.T) {
	base := baseConfig()
	_ = applySnapshotToConfig(base, benchmark.ConfigSnapshot{ContextSize: 65536, GPULayers: 1})

	if base.ContextSize != 8192 || base.GPULayers != 999 {
		t.Errorf("input mutated: ctx=%d ngl=%d", base.ContextSize, base.GPULayers)
	}
}
