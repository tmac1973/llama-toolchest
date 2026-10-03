package models

import (
	"bytes"
	"path/filepath"
	"testing"
)

// gpt-oss leaves its sliding-window layout out of the file; llama.cpp
// sets every other layer to the window in code, and so does the parser.
func TestBuiltinSlidingWindowLayout(t *testing.T) {
	var b ggufBuilder
	b.s("general.architecture", "gpt-oss")
	b.u32("gpt-oss.block_count", 24)
	b.u32("gpt-oss.embedding_length", 2880)
	b.u32("gpt-oss.attention.head_count", 64)
	b.u32("gpt-oss.attention.head_count_kv", 8)
	b.u32("gpt-oss.attention.key_length", 64)
	b.u32("gpt-oss.attention.value_length", 64)
	b.u32("gpt-oss.attention.sliding_window", 128)
	b.u32("gpt-oss.context_length", 131072)

	meta, err := ParseGGUFMetaOnly(bytes.NewReader(b.bytes()))
	if err != nil {
		t.Fatal(err)
	}
	// 12 window layers and 12 full ones, 8 KV heads of 64+64 each.
	if meta.KVFullPerTok != 12*8*128 || meta.KVSWAPerTok != 12*8*128 || meta.SlidingWindow != 128 {
		t.Errorf("full %d, window %d (size %d)", meta.KVFullPerTok, meta.KVSWAPerTok, meta.SlidingWindow)
	}
}

// A model without output.weight ties its output to the input embedding,
// and that table then counts as GPU memory.
func TestTiedOutputEmbedding(t *testing.T) {
	dir := t.TempDir()
	tied := writeMoEGGUF(t, filepath.Join(dir, "tied.gguf"), map[string]uint32{"moearch.block_count": 1},
		[]string{"token_embd.weight", "blk.0.ffn_up.weight"}, 64)
	untied := writeMoEGGUF(t, filepath.Join(dir, "untied.gguf"), map[string]uint32{"moearch.block_count": 1},
		[]string{"token_embd.weight", "blk.0.ffn_up.weight", "output.weight"}, 64)

	for path, want := range map[string]bool{tied: true, untied: false} {
		meta, err := ParseGGUFMeta(path)
		if err != nil {
			t.Fatal(err)
		}
		var m Model
		meta.ApplyTo(&m)
		if m.OutputTied != want || m.EmbeddingsTied() != want {
			t.Errorf("%s: OutputTied = %v, want %v", filepath.Base(path), m.OutputTied, want)
		}
	}
	// Without a tensor table, the gemma family is taken as tied.
	if !(&Model{Arch: "gemma4"}).EmbeddingsTied() || (&Model{Arch: "llama"}).EmbeddingsTied() {
		t.Error("architecture fallback wrong")
	}
}

// The coefficients follow the backend the build runs on; anything without
// a set of its own gets ROCm's, the more cautious.
func TestVRAMBackendSelectsCoefficients(t *testing.T) {
	defer SetVRAMBackend("")
	m := &Model{SizeBytes: 8 << 30, NLayers: 36, AttnLayers: 36, NEmbd: 4096, NHead: 32, NKVHead: 8, KVFullPerTok: 36 * 8 * 256}
	cfg := &ModelConfig{ContextSize: 32768, GPULayers: 999, SplitMode: "layer"}

	SetVRAMBackend("cuda")
	cuda := VRAMEstimateForConfigOn(m, cfg, 3)
	SetVRAMBackend("vulkan")
	other := VRAMEstimateForConfigOn(m, cfg, 3)
	SetVRAMBackend("")
	rocm := VRAMEstimateForConfigOn(m, cfg, 3)

	if other != rocm {
		t.Errorf("an unfitted backend got %.2f, ROCm %.2f", other, rocm)
	}
	if cuda >= rocm-2 {
		t.Errorf("CUDA %.2f GiB is not well below ROCm's %.2f on a layer split", cuda, rocm)
	}
}
