package models

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

func approx(t *testing.T, label string, got, want, tol float64) {
	t.Helper()
	if math.Abs(got-want) > tol {
		t.Errorf("%s: got %.4f, want %.4f (±%.4f)", label, got, want, tol)
	}
}

// computeKVScaling should reduce uniform full-attention params to the same KV
// scaling the legacy formula assumes: nLayers · nKVHead · (k_dim+v_dim) per tok.
func TestComputeKVScalingUniform(t *testing.T) {
	meta := &GGUFMeta{NLayers: 4, NEmbd: 512, NHead: 8} // headDim = 64
	computeKVScaling(meta, 2 /*kv*/, nil, 0, 0, 0, 0, 0, nil, 0, nil)

	if want := 4 * 2 * (64 + 64); meta.KVFullPerTok != want {
		t.Errorf("KVFullPerTok = %d, want %d", meta.KVFullPerTok, want)
	}
	if meta.KVSWAPerTok != 0 {
		t.Errorf("KVSWAPerTok = %d, want 0", meta.KVSWAPerTok)
	}
	if meta.SlidingWindow != 0 {
		t.Errorf("SlidingWindow = %d, want 0", meta.SlidingWindow)
	}
}

// Explicit key_length/value_length should override n_embd/n_head, which matters
// when the latter isn't a whole number (e.g. Qwen3.6: 5120/24 = 213.3).
func TestComputeKVScalingExplicitHeadDim(t *testing.T) {
	meta := &GGUFMeta{NLayers: 64, NEmbd: 5120, NHead: 24}
	computeKVScaling(meta, 4 /*kv*/, nil, 256 /*keyLen*/, 256 /*valLen*/, 0, 0, 0, nil, 0, nil)

	if want := 64 * 4 * (256 + 256); meta.KVFullPerTok != want {
		t.Errorf("KVFullPerTok = %d, want %d", meta.KVFullPerTok, want)
	}
}

// gemma-style: per-layer GQA (head_count_kv array) plus sliding-window layers
// with their own smaller head dims.
func TestComputeKVScalingGemma(t *testing.T) {
	meta := &GGUFMeta{NLayers: 6, NEmbd: 3840, NHead: 16}
	kvHeads := []int{8, 8, 8, 8, 8, 1}
	swa := []bool{true, true, true, true, true, false}
	computeKVScaling(meta, 0, kvHeads, 512, 512, 256, 256, 1024, swa, 0, nil)

	// 5 sliding-window layers: 8 kv heads × (256+256)
	if want := 5 * 8 * (256 + 256); meta.KVSWAPerTok != want {
		t.Errorf("KVSWAPerTok = %d, want %d", meta.KVSWAPerTok, want)
	}
	// 1 full-attention layer: 1 kv head × (512+512)
	if want := 1 * 1 * (512 + 512); meta.KVFullPerTok != want {
		t.Errorf("KVFullPerTok = %d, want %d", meta.KVFullPerTok, want)
	}
	if meta.SlidingWindow != 1024 {
		t.Errorf("SlidingWindow = %d, want 1024", meta.SlidingWindow)
	}
	// Representative scalar for display = max per-layer kv head count.
	if meta.NKVHead != 8 {
		t.Errorf("NKVHead = %d, want 8", meta.NKVHead)
	}
}

// When a model carries no per-token factors (old record), KVCacheGB must match
// the legacy uniform estimate exactly.
func TestKVCacheGBLegacyFallback(t *testing.T) {
	m := &Model{NLayers: 48, NKVHead: 8, NHead: 40, NEmbd: 5120}
	for _, ctx := range []int{8192, 32768, 131072} {
		for _, q := range []string{"", "q8_0", "q4_0"} {
			got := m.KVCacheGB(ctx, q)
			want := EstimateKVCacheGB(m.NLayers, m.NKVHead, m.NHead, m.NEmbd, ctx, q)
			approx(t, "KVCacheGB fallback", got, want, 1e-9)
		}
	}
}

// Real gemma-4-12b factors (parsed from the GGUF): 8 global layers @ 1 kv head,
// 40 sliding-window layers @ 8 kv heads, window 1024. At 128K/q8_0 this is ~1.2
// GB — versus the ~48 GB the old uniform formula produced.
func TestKVCacheGBGemma4(t *testing.T) {
	m := &Model{
		NLayers:       48,
		NEmbd:         3840,
		NHead:         16,
		NKVHead:       8,
		ContextLength: 262144,
		KVFullPerTok:  8192,
		KVSWAPerTok:   163840,
		SlidingWindow: 1024,
	}
	approx(t, "gemma4 KV @128K q8_0", m.KVCacheGB(131072, "q8_0"), 1.23, 0.05)

	// Below the sliding window, local layers stop growing; KV is tiny.
	small := m.KVCacheGB(512, "q8_0")
	if small > 0.1 {
		t.Errorf("KV @512 tokens = %.4f GB, expected < 0.1", small)
	}

	// Past the window, only the 8 global layers keep scaling with context, so
	// doubling context from 128K to 256K should less-than-double total KV.
	if kv256 := m.KVCacheGB(262144, "q8_0"); kv256 >= 2*m.KVCacheGB(131072, "q8_0") {
		t.Errorf("sliding window not capping local layers: kv256=%.3f", kv256)
	}
}

// Auxiliary files (mmproj / MTP head / draft) count toward VRAM only when both
// downloaded (present on disk) and activated (toggle on / draft mode selected).
func TestAuxFilesVRAMGB(t *testing.T) {
	dir := t.TempDir()
	const oneGiB = 1024 * 1024 * 1024
	write := func(name string, gb int) string {
		p := filepath.Join(dir, name)
		if err := os.Truncate(p, 0); err != nil {
			f, _ := os.Create(p)
			f.Close()
		}
		if err := os.Truncate(p, int64(gb)*oneGiB); err != nil {
			t.Fatal(err)
		}
		return p
	}
	mmproj := write("mmproj.gguf", 1)
	mtp := write("mtp.gguf", 2)
	draft := write("draft.gguf", 3)

	cases := []struct {
		name string
		cfg  ModelConfig
		want float64
	}{
		{"all activated", ModelConfig{MmprojPath: mmproj, MtpPath: mtp, SpecType: "draft", DraftModelPath: draft}, 6},
		{"mmproj disabled", ModelConfig{MmprojPath: mmproj, MmprojDisabled: true}, 0},
		{"mtp disabled", ModelConfig{MtpPath: mtp, MtpDisabled: true}, 0},
		{"draft path but not draft mode", ModelConfig{SpecType: "draft-mtp", DraftModelPath: draft}, 0},
		{"mtp head under draft-mtp", ModelConfig{SpecType: "draft-mtp", MtpPath: mtp}, 2},
		{"missing file counts zero", ModelConfig{MmprojPath: filepath.Join(dir, "nope.gguf")}, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			approx(t, "AuxFilesVRAMGB", AuxFilesVRAMGB(&tc.cfg), tc.want, 1e-9)
		})
	}
}

// The sliding window caps the local-layer token count at min(ctx, window).
func TestKVCacheGBSlidingWindowCap(t *testing.T) {
	m := &Model{NLayers: 1, KVSWAPerTok: 1000, SlidingWindow: 1024}
	// ctx well past the window — local contribution should use 1024, not ctx.
	got := m.KVCacheGB(100000, "")
	want := 1024.0 * 1000.0 * 2.0 / (1024 * 1024 * 1024)
	approx(t, "SWA cap", got, want, 1e-9)
}
