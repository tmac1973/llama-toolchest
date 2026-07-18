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
		GPUAssign:      "all",
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
	got := applySnapshotToConfig(baseConfig(), benchmark.ConfigSnapshot{
		ContextSize: 4096, GPUAssign: "all",
	})

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

// Zero is a value, not "unset". ResolveModel seeds the snapshot from the
// model's saved config and applyOverrides replaces only what the job set,
// so the snapshot is authoritative for every field it models.
//
// This test previously asserted the opposite, which locked in a bug:
// sweeping gpu_layers to 0 (the curated "CPU only" choice) was silently
// discarded while the run still recorded 0 as applied — the same
// mislabeled-result failure this whole mechanism exists to prevent.
func TestApplySnapshotAppliesZeroValues(t *testing.T) {
	got := applySnapshotToConfig(baseConfig(), benchmark.ConfigSnapshot{
		GPULayers:   0,
		ContextSize: 4096,
		Threads:     8,
	})

	if got.GPULayers != 0 {
		t.Errorf("GPULayers = %d, want the overridden 0 (CPU only)", got.GPULayers)
	}
	if got.ContextSize != 4096 {
		t.Errorf("ContextSize = %d, want 4096", got.ContextSize)
	}
}

// A snapshot carrying the saved values reproduces them exactly, which is
// the no-override case.
func TestApplySnapshotRoundTripsSavedValues(t *testing.T) {
	base := baseConfig()
	got := applySnapshotToConfig(base, benchmark.ConfigSnapshot{
		GPULayers:      base.GPULayers,
		ContextSize:    base.ContextSize,
		Threads:        base.Threads,
		KVCacheQuant:   base.KVCacheQuant,
		GPUAssign:      base.GPUAssign,
		FlashAttention: base.FlashAttention,
	})

	if got.ContextSize != 8192 || got.GPULayers != 999 || got.Threads != 8 {
		t.Errorf("round trip changed values: %+v", got)
	}
	if got.KVCacheQuant != "q8_0" {
		t.Errorf("KVCacheQuant = %q, want q8_0", got.KVCacheQuant)
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

// The restore-owed flag is what makes cleanup correct when a restart
// fails after llama-server already came up on the benchmark preset.
// Keying cleanup off "is an override still armed" missed that case: the
// apply error path drops the override, so the deferred cleanup saw
// nothing to do and left the router on benchmark config.
func TestBenchRouterDirtyFlag(t *testing.T) {
	s := &Server{}

	if s.consumeBenchRouterDirty() {
		t.Error("nothing started yet; no restore should be owed")
	}

	s.markBenchRouterDirty()
	if !s.consumeBenchRouterDirty() {
		t.Error("a restore should be owed after the router started for a benchmark")
	}
	if s.consumeBenchRouterDirty() {
		t.Error("consuming twice should not restart twice")
	}
}

// The flag must survive the override being cleared — that combination is
// exactly the failure it exists to cover.
func TestBenchRouterDirtySurvivesOverrideClear(t *testing.T) {
	s := &Server{}
	s.setBenchOverrides(map[string]*models.ModelConfig{"m": {}})
	s.markBenchRouterDirty()

	s.setBenchOverrides(nil)

	if s.benchOverridesSnapshot() != nil {
		t.Fatal("override should be cleared")
	}
	if !s.consumeBenchRouterDirty() {
		t.Error("a restore is still owed once the router has run on a benchmark preset")
	}
}

// A job that switches builds rewrites the user's saved ActiveBuild, so
// the pre-job selection has to be remembered and restored — the same
// leak the ephemeral preset avoids for model config.
func TestCaptureActiveBuildRemembersFirstSelection(t *testing.T) {
	s := &Server{}

	// Captured on the first cell, including when that cell's build is
	// already active and nothing switches.
	s.captureActiveBuild("build-A")
	s.captureActiveBuild("build-B") // later cells must not overwrite it
	s.captureActiveBuild("build-C")

	prev, ok := s.consumeActiveBuild()
	if !ok {
		t.Fatal("expected a build to restore")
	}
	if prev != "build-A" {
		t.Errorf("prev = %q, want the pre-job build-A", prev)
	}
}

func TestConsumeActiveBuildIsOneShot(t *testing.T) {
	s := &Server{}
	if _, ok := s.consumeActiveBuild(); ok {
		t.Error("nothing captured; nothing to restore")
	}

	s.captureActiveBuild("build-A")
	if _, ok := s.consumeActiveBuild(); !ok {
		t.Fatal("expected a restore after capture")
	}
	if _, ok := s.consumeActiveBuild(); ok {
		t.Error("consuming twice would restore again on a later job")
	}
}

// An empty selection is a real state (no build chosen yet) and must be
// distinguishable from "nothing captured".
func TestCaptureActiveBuildHandlesEmptySelection(t *testing.T) {
	s := &Server{}
	s.captureActiveBuild("")

	prev, ok := s.consumeActiveBuild()
	if !ok {
		t.Fatal("an empty pre-job selection is still a selection to restore")
	}
	if prev != "" {
		t.Errorf("prev = %q, want empty", prev)
	}
}
