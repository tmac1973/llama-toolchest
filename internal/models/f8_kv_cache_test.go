package models

import (
	"strings"
	"testing"
)

// f8_e4m3 is one byte per element: no block scale, unlike q8_0.
func TestKVBytesPerElemF8(t *testing.T) {
	for q, want := range map[string]float64{"": 2.0, "f16": 2.0, "q8_0": 1.0625, "q4_0": 0.5625, KVCacheF8: 1.0} {
		if got := kvBytesPerElem(q); got != want {
			t.Errorf("kvBytesPerElem(%q) = %v, want %v", q, got, want)
		}
	}
}

// The preset carries f8_e4m3 for both K and V on a build that knows it,
// and nothing (f16) on an older one, which would refuse to start.
func TestPresetWritesTheF8KVCache(t *testing.T) {
	write := func(cfg *ModelConfig, tgt Target) string {
		var b strings.Builder
		writeConfigParams(&b, cfg, KindChat, tgt)
		return b.String()
	}
	cfg := &ModelConfig{GPULayers: 999, FlashAttention: true, KVCacheQuant: KVCacheF8}
	for _, tgt := range []Target{{}, {Version: versionF8KVCache}} {
		got := write(cfg, tgt)
		if !strings.Contains(got, "cache-type-k = f8_e4m3\n") || !strings.Contains(got, "cache-type-v = f8_e4m3\n") {
			t.Errorf("build %d: f8_e4m3 not written:\n%s", tgt.Version, got)
		}
	}
	if got := write(cfg, Target{Version: versionF8KVCache - 1}); strings.Contains(got, "cache-type") {
		t.Errorf("older build: cache type written, which stops the router at startup:\n%s", got)
	}
	q8 := &ModelConfig{GPULayers: 999, FlashAttention: true, KVCacheQuant: "q8_0"}
	if got := write(q8, Target{Version: versionF8KVCache - 1}); !strings.Contains(got, "cache-type-k = q8_0\n") {
		t.Errorf("older build: q8_0 dropped by the f8_e4m3 gate:\n%s", got)
	}
}

func TestDraftF8KVCacheFlags(t *testing.T) {
	cfg := &ModelConfig{
		GPULayers:         99,
		SpecType:          "draft",
		DraftModelPath:    "/models/qwen-0.5b.gguf",
		DraftKVCacheQuant: KVCacheF8,
	}
	got := cfg.EffectiveFlagsFor(KindChat, Target{Version: versionF8KVCache})
	for _, want := range []string{"--cache-type-k-draft f8_e4m3", "--cache-type-v-draft f8_e4m3"} {
		if !strings.Contains(got, want) {
			t.Errorf("draft flags missing %q in: %s", want, got)
		}
	}
	if got := cfg.EffectiveFlagsFor(KindChat, Target{Version: versionF8KVCache - 1}); strings.Contains(got, "cache-type") {
		t.Errorf("older build: draft cache type written: %s", got)
	}
}

func TestF8KVCacheOption(t *testing.T) {
	for v, want := range map[int]bool{0: true, versionF8KVCache - 1: false, versionF8KVCache: true, versionF8KVCache + 100: true} {
		if got := (Target{Version: v}).F8KVCacheOption(); got != want {
			t.Errorf("Version %d: F8KVCacheOption = %v, want %v", v, got, want)
		}
	}
}
