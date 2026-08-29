package models

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// writeHybridGGUF builds a minimal GGUF carrying the hparams the KV
// scaling needs plus full_attention_interval, so the backfill has a real
// file to re-parse.
func writeHybridGGUF(t *testing.T, dir, name string, layers, interval, kvHeads, headDim int) string {
	t.Helper()
	var b bytes.Buffer
	wstr := func(s string) {
		binary.Write(&b, binary.LittleEndian, uint64(len(s)))
		b.WriteString(s)
	}
	kv := func(k string, v uint32) {
		wstr(k)
		binary.Write(&b, binary.LittleEndian, ggufTypeUint32)
		binary.Write(&b, binary.LittleEndian, v)
	}
	b.WriteString("GGUF")
	binary.Write(&b, binary.LittleEndian, uint32(3))
	binary.Write(&b, binary.LittleEndian, uint64(0)) // no tensors
	binary.Write(&b, binary.LittleEndian, uint64(8))
	wstr("general.architecture")
	binary.Write(&b, binary.LittleEndian, ggufTypeString)
	wstr("hyb")
	kv("hyb.block_count", uint32(layers))
	kv("hyb.embedding_length", uint32(kvHeads*headDim))
	kv("hyb.attention.head_count", uint32(kvHeads))
	kv("hyb.attention.head_count_kv", uint32(kvHeads))
	kv("hyb.attention.key_length", uint32(headDim))
	kv("hyb.attention.value_length", uint32(headDim))
	kv("hyb.full_attention_interval", uint32(interval))
	p := filepath.Join(dir, name)
	if err := os.WriteFile(p, b.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

// A hybrid model interleaves recurrent (linear-attention) layers, which
// hold no KV cache. Counting them made the estimate too large by the
// full-attention interval — four times over on both Qwen hybrids measured.
//
// Both cases below are pinned to hardware. The predicted cache is compared
// against what llama.cpp actually allocated, read from its own buffer
// report at verbosity 4 (see plan/ple-vram-findings.md).
func TestHybridKVExcludesRecurrentLayers(t *testing.T) {
	tests := []struct {
		name        string
		layers      int
		kvHeads     int
		headDim     int
		interval    int
		ctx         int
		quant       string
		measuredGiB float64 // summed across the cards llama.cpp reported
	}{
		// Qwen3.8-27B: 65 layers, interval 4, key/value_length 256, f16 cache.
		// llama.cpp allocated 4.00 GiB per card across 4 cards.
		{"qwen35 27B", 65, 4, 256, 4, 262144, "", 16.00},
		// Qwen3.8-Flash-Next: 48 layers, interval 4, q8_0 cache.
		// 816 MiB per card across 4 cards = 3.19 GiB.
		{"qwen4exp Flash-Next", 48, 2, 256, 4, 262144, "q8_0", 3.19},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			meta := &GGUFMeta{NLayers: tt.layers, NEmbd: tt.kvHeads * tt.headDim, NHead: tt.kvHeads}
			computeKVScaling(meta, tt.kvHeads, nil, tt.headDim, tt.headDim, 0, 0, 0, nil, tt.interval, nil)

			m := &Model{NLayers: tt.layers, NEmbd: meta.NEmbd, NKVHead: meta.NKVHead,
				KVFullPerTok: meta.KVFullPerTok, KVSWAPerTok: meta.KVSWAPerTok}
			got := m.KVCacheGB(tt.ctx, tt.quant)
			if math.Abs(got-tt.measuredGiB) > 0.35 {
				t.Errorf("KV cache = %.2f GiB, llama.cpp allocated %.2f", got, tt.measuredGiB)
			}

			// And confirm the old behaviour would have been badly wrong,
			// so this test fails if the exclusion is ever dropped.
			all := &GGUFMeta{NLayers: tt.layers, NEmbd: meta.NEmbd, NHead: tt.kvHeads}
			computeKVScaling(all, tt.kvHeads, nil, tt.headDim, tt.headDim, 0, 0, 0, nil, 0, nil)
			if all.KVFullPerTok <= meta.KVFullPerTok {
				t.Fatal("counting every layer did not produce a larger cache; the fixture is wrong")
			}
			if ratio := float64(all.KVFullPerTok) / float64(meta.KVFullPerTok); math.Abs(ratio-float64(tt.interval)) > 0.3 {
				t.Errorf("over-count ratio %.2f, want about the interval %d", ratio, tt.interval)
			}
		})
	}
}

// An explicit array wins over the interval, matching llama.cpp's own
// precedence.
func TestRecurrentLayerArrayTakesPrecedence(t *testing.T) {
	// Interval 4 would make layers 0,1,2 recurrent; the array says otherwise.
	arr := []bool{false, false, false, true}
	for il, wantRecr := range arr {
		if got := isRecurrentLayer(il, 4, arr); got != wantRecr {
			t.Errorf("layer %d: recurrent = %v, want %v", il, got, wantRecr)
		}
	}
}

// A model with neither key is not hybrid: every layer attends, and the
// estimate must be exactly what it was before.
func TestNonHybridUnaffected(t *testing.T) {
	meta := &GGUFMeta{NLayers: 32, NEmbd: 4096, NHead: 32}
	computeKVScaling(meta, 8, nil, 128, 128, 0, 0, 0, nil, 0, nil)
	want := 32 * 8 * (128 + 128)
	if meta.KVFullPerTok != want {
		t.Errorf("KVFullPerTok = %d, want %d — a non-hybrid model must be untouched", meta.KVFullPerTok, want)
	}
	for il := 0; il < 32; il++ {
		if isRecurrentLayer(il, 0, nil) {
			t.Fatalf("layer %d treated as recurrent without either key", il)
		}
	}
}

// The correction is worthless if it cannot reach a model already in the
// registry, and those records cannot be spotted by their values: the old
// factors are non-zero and plausible, just too large. Keyed on a flag
// instead — the same mistake as the per-layer embedding backfill, which
// went unnoticed because the value it wrote was also merely absent.
func TestBackfillCorrectsStaleHybridKV(t *testing.T) {
	dir := t.TempDir()
	cfgDir := filepath.Join(dir, "config")
	modelsDir := filepath.Join(dir, "models")
	for _, d := range []string{cfgDir, modelsDir} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	path := writeHybridGGUF(t, modelsDir, "hybrid.gguf", 8, 4, 2, 256)

	// A record as an older build wrote it: every layer counted.
	const stale = 8 * 2 * (256 + 256)
	raw := fmt.Sprintf(`{"models":{"h":{"id":"h","filename":"hybrid.gguf","file_path":%q,`+
		`"n_layers":8,"n_embd":512,"n_head":2,"kv_full_per_tok":%d}}}`, path, stale)
	if err := os.WriteFile(filepath.Join(cfgDir, "models.json"), []byte(raw), 0o644); err != nil {
		t.Fatal(err)
	}

	r := NewRegistry(dir, modelsDir)
	r.BackfillGGUFMeta()

	m, err := r.Get("h")
	if err != nil {
		t.Fatal(err)
	}
	if !m.KVRecurrentChecked {
		t.Error("KVRecurrentChecked not set — the model would be re-parsed at every startup")
	}
	// Interval 4 over 8 layers leaves 2 attending.
	if want := 2 * 2 * (256 + 256); m.KVFullPerTok != want {
		t.Errorf("KVFullPerTok = %d, want %d (stale value was %d)", m.KVFullPerTok, want, stale)
	}
}
