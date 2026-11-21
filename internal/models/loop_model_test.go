package models

import (
	"math"
	"path/filepath"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/testutil/gguftest"
)

// writeLoopGGUF builds a small looped model in the shape of Inkling 2
// Mini: 6 blocks, blocks 2-3 run 3 times, a sliding window on every block
// but the last of each pair, per-layer head counts and relative-position
// attention.
func writeLoopGGUF(t *testing.T, path string) string {
	t.Helper()
	var b gguftest.Builder
	b.String("general.architecture", "inkling2_loop")
	b.Uint32("inkling2_loop.block_count", 6)
	b.Uint32("inkling2_loop.embedding_length", 256)
	b.Uint32("inkling2_loop.context_length", 65536)
	b.Int32Array("inkling2_loop.attention.head_count", []int32{4, 4, 4, 4, 4, 4})
	b.Int32Array("inkling2_loop.attention.head_count_kv", []int32{2, 2, 2, 2, 2, 2})
	b.Uint32("inkling2_loop.attention.key_length", 64)
	b.Uint32("inkling2_loop.attention.value_length", 64)
	b.Uint32("inkling2_loop.attention.sliding_window", 128)
	b.Int32Array("inkling2_loop.attention.sliding_window_pattern", []int32{1, 0, 1, 0, 1, 0})
	b.Uint32("inkling2_loop.attention.relative_length", 16)
	b.Uint32("inkling2_loop.loop.start", 2)
	b.Uint32("inkling2_loop.loop.end", 4)
	b.Uint32("inkling2_loop.loop.count", 3)
	return b.WriteFile(t, path)
}

// llama.cpp gives a looped model's extra passes cache slots of their own:
// (passes-1) x (end-start) of them, each with the KV cache of the block it
// repeats.
func TestParseLoopedModel(t *testing.T) {
	meta, err := ParseGGUFMeta(writeLoopGGUF(t, filepath.Join(t.TempDir(), "loop.gguf")))
	if err != nil {
		t.Fatal(err)
	}
	if meta.NLayers != 6 || meta.LoopSlots != 4 {
		t.Errorf("layers = %d, loop slots = %d; want 6 and 4", meta.NLayers, meta.LoopSlots)
	}
	// Blocks 0-5 plus blocks 2 and 3 twice more: 10 slots, 5 of them full
	// attention (1, 3, 5, and 3 twice more), 5 with the window.
	if meta.AttnLayers != 10 {
		t.Errorf("attention layers = %d, want 10", meta.AttnLayers)
	}
	perLayer := 2 * (64 + 64)
	if meta.KVFullPerTok != 5*perLayer || meta.KVSWAPerTok != 5*perLayer {
		t.Errorf("KV per token = %d full, %d window; want %d each", meta.KVFullPerTok, meta.KVSWAPerTok, 5*perLayer)
	}
	if meta.RelAttnHeads != 4 {
		t.Errorf("relative-attention heads = %d, want 4 (the head-count array's largest)", meta.RelAttnHeads)
	}
	if meta.NHead != 0 {
		t.Errorf("NHead = %d; an array head count leaves it unset, as before", meta.NHead)
	}

	var m Model
	meta.ApplyTo(&m)
	if m.LoopSlots != 4 || m.RelAttnHeads != 4 {
		t.Errorf("ApplyTo did not copy the loop: %+v", m)
	}
}

// Loop keys that describe no loop add nothing.
func TestLoopSlotsNeedAValidLoop(t *testing.T) {
	for _, tt := range []struct {
		start, end, passes int
	}{{2, 4, 1}, {0, 4, 3}, {4, 4, 3}, {2, 9, 3}} {
		meta := &GGUFMeta{NLayers: 6, LoopStart: tt.start, LoopEnd: tt.end}
		if n := loopSlots(meta, tt.passes); n != 0 || meta.LoopStart != 0 || meta.LoopEnd != 0 {
			t.Errorf("loop %d..%d x%d: %d slots (start %d, end %d), want none", tt.start, tt.end, tt.passes, n, meta.LoopStart, meta.LoopEnd)
		}
	}
}

// Inkling 2 Mini on one A4000 (build b12600): its compute buffer grows
// with context times micro-batch, flash attention or not. The estimate
// stays at or above what llama.cpp reported.
func TestRelativeAttentionScratch(t *testing.T) {
	m := &Model{Arch: "inkling2_loop", NLayers: 24, NEmbd: 2304, NKVHead: 8, RelAttnHeads: 24,
		KVFullPerTok: 16384, KVSWAPerTok: 81920, SlidingWindow: 1024, AttnLayers: 48, LoopSlots: 24,
		SizeBytes: 17627940768, TokenEmbdBytes: 492106752, OutputBytes: 492106752}
	measuredMiB := []struct {
		ub, ctx int
		mib     float64
	}{
		{128, 8192, 246}, {256, 32768, 1682}, {256, 131072, 6434}, {1024, 32768, 6733}, {2048, 524288, 203558},
	}
	for _, p := range measuredMiB {
		cfg := &ModelConfig{ContextSize: p.ctx, UBatchSize: p.ub, BatchSize: 2048, KVCacheQuant: "q8_0", GPULayers: 999}
		b := vramBreakdownOn(&cudaCoefficients, m, cfg, 1, 0)
		got := (b.Compute + b.IndexerScratch) * 1024
		if got < p.mib || got > p.mib*1.15 {
			t.Errorf("ub %d, ctx %d: compute %.0f MiB, measured %.0f", p.ub, p.ctx, got, p.mib)
		}
	}
	// Without the key, nothing is added.
	plain := *m
	plain.RelAttnHeads = 0
	cfg := &ModelConfig{ContextSize: 131072, UBatchSize: 256, KVCacheQuant: "q8_0", GPULayers: 999}
	if b := vramBreakdownOn(&cudaCoefficients, &plain, cfg, 1, 0); b.IndexerScratch != 0 {
		t.Errorf("scratch %.2f GiB on a model without relative attention", b.IndexerScratch)
	}
}

// A looped model's extra cache slots come after its blocks in llama.cpp's
// layer split and hold no weights. With an even split over two cards the
// weights all go to the first; the balanced split gives the first card
// fewer slots.
func TestLoopedModelSplit(t *testing.T) {
	m := &Model{Arch: "inkling2_loop", NLayers: 24, LoopSlots: 24, SizeBytes: 24 << 30}
	cfg := &ModelConfig{ContextSize: 8192, GPULayers: 999, SplitMode: "layer"}
	b := VRAMBreakdown{KVCache: 1.5}
	layers := layerGPUBytes(m, cfg, b)
	if len(layers) != 48 {
		t.Fatalf("%d layers, want 48 (24 blocks and 24 cache slots)", len(layers))
	}
	if layers[30] >= layers[3] || layers[30] <= 0 {
		t.Errorf("a cache slot holds %.0f bytes, a block %.0f: want only its KV cache", layers[30], layers[3])
	}
	even := cardLoads(layers, []int{24, 24}, []float64{0, 0})
	if even[0] < 20 || even[1] > 2 {
		t.Errorf("even split loads = %.1f / %.1f GiB; llama.cpp puts every block on the first card", even[0], even[1])
	}
	if !moeSplitApplies(cfg, m, []int{14000, 14000}) {
		t.Fatal("a looped model's split is not balanced")
	}
	split := balanceSplit(layers, []float64{14, 14}, []float64{0, 0})
	if split[0] >= 24 {
		t.Errorf("balanced split = %v; the first card should get fewer than half the slots", split)
	}
}

// llama.cpp puts the output layer on the card that runs the last layers.
func TestOutputLayerOnTheLastCard(t *testing.T) {
	m := &Model{Arch: "llama", NLayers: 10, SizeBytes: 11 << 30, OutputBytes: 1 << 30, HasClassifierHead: false}
	cfg := &ModelConfig{ContextSize: 4096, GPULayers: 999, SplitMode: "layer", TensorSplit: "1,1", GPUAssign: "all"}
	layers := layerGPUBytes(m, cfg, VRAMBreakdown{})
	var sum float64
	for _, l := range layers {
		sum += l
	}
	if math.Abs(sum/(1<<30)-10) > 0.01 {
		t.Errorf("layers hold %.2f GiB, want 10 (the output layer left out)", sum/(1<<30))
	}
	_, _, fixed := moeSplitInputs(m, cfg, []int{20000, 20000}, []int{0, 1})
	if d := fixed[1] - fixed[0]; math.Abs(d-1) > 0.01 {
		t.Errorf("last card's fixed cost is %.2f GiB above the first's, want the 1 GiB output layer", d)
	}
}

// Experts kept in system memory are sized layer by layer where the file
// was read so: Inkling 2 Mini's first layers carry smaller experts than
// its last.
func TestCPUExpertBytesByLayer(t *testing.T) {
	m := &Model{NLayers: 4, ExpertLayers: 4, ExpertBytes: 1000, ExpertLayerBytes: []int64{100, 100, 300, 500}}
	if got := cpuExpertBytes(m, &ModelConfig{CPUMoE: 2}); got != 200 {
		t.Errorf("experts of the first 2 layers = %d, want 200", got)
	}
	m.ExpertLayerBytes = nil
	if got := cpuExpertBytes(m, &ModelConfig{CPUMoE: 2}); got != 500 {
		t.Errorf("without sizes by layer = %d, want an even share (500)", got)
	}
}

// The tensor scan records expert sizes by layer and the output layer's
// size.
func TestParseExpertBytesByLayer(t *testing.T) {
	path := writeMoEGGUF(t, filepath.Join(t.TempDir(), "moe.gguf"), map[string]uint32{
		"moearch.block_count": 3, "moearch.expert_count": 8,
	}, []string{
		"token_embd.weight",
		"blk.0.ffn_up.weight",
		"blk.1.ffn_up_exps.weight", "blk.1.ffn_down_exps.weight",
		"blk.2.ffn_up_exps.weight",
		"output.weight",
	}, 64)
	meta, err := ParseGGUFMeta(path)
	if err != nil {
		t.Fatal(err)
	}
	want := []int64{0, 128, 64}
	if len(meta.ExpertLayerBytes) != 3 || meta.ExpertLayerBytes[1] != want[1] || meta.ExpertLayerBytes[2] != want[2] {
		t.Errorf("expert bytes by layer = %v, want %v", meta.ExpertLayerBytes, want)
	}
	if meta.OutputBytes != 64 {
		t.Errorf("output bytes = %d, want 64", meta.OutputBytes)
	}
}
