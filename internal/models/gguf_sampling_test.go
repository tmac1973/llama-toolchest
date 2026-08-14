package models

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

// writeTestGGUF builds a minimal GGUF v3 file with the given metadata KVs.
type testKV struct {
	key       string
	valueType uint32
	value     any // string, float32, float64, uint32, int32
}

func writeTestGGUF(t *testing.T, kvs []testKV) string {
	t.Helper()
	var buf bytes.Buffer
	buf.WriteString("GGUF")
	binary.Write(&buf, binary.LittleEndian, uint32(3))        // version
	binary.Write(&buf, binary.LittleEndian, uint64(0))        // tensor count
	binary.Write(&buf, binary.LittleEndian, uint64(len(kvs))) // kv count
	writeStr := func(s string) {
		binary.Write(&buf, binary.LittleEndian, uint64(len(s)))
		buf.WriteString(s)
	}
	for _, kv := range kvs {
		writeStr(kv.key)
		binary.Write(&buf, binary.LittleEndian, kv.valueType)
		switch v := kv.value.(type) {
		case string:
			writeStr(v)
		default:
			binary.Write(&buf, binary.LittleEndian, v)
		}
	}
	path := filepath.Join(t.TempDir(), "test.gguf")
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestParseGGUFSamplingKeys(t *testing.T) {
	path := writeTestGGUF(t, []testKV{
		{"general.architecture", ggufTypeString, "qwen3"},
		{"general.base_model.0.repo_url", ggufTypeString, "https://huggingface.co/Qwen/Qwen3.8-27B"},
		{"general.sampling.temp", ggufTypeFloat32, float32(1.0)},
		{"general.sampling.top_p", ggufTypeFloat32, float32(0.95)},
		{"general.sampling.top_k", ggufTypeUint32, uint32(20)},
		{"general.sampling.min_p", ggufTypeFloat64, float64(0.0)},
		{"general.sampling.penalty_repeat", ggufTypeFloat32, float32(1.05)},
		{"general.sampling.mirostat", ggufTypeUint32, uint32(0)}, // uncaptured — must be skipped cleanly
		{"qwen3.block_count", ggufTypeUint32, uint32(48)},
	})

	meta, err := ParseGGUFMeta(path)
	if err != nil {
		t.Fatal(err)
	}
	if !meta.SamplingChecked {
		t.Error("SamplingChecked should be true after a successful parse")
	}
	if meta.BaseModelRepo != "Qwen/Qwen3.8-27B" {
		t.Errorf("BaseModelRepo = %q, want Qwen/Qwen3.8-27B", meta.BaseModelRepo)
	}
	if meta.NLayers != 48 {
		t.Errorf("NLayers = %d, want 48 (parser misaligned after sampling keys?)", meta.NLayers)
	}
	if meta.SamplingTemp == nil || *meta.SamplingTemp != 1.0 {
		t.Errorf("SamplingTemp = %v, want 1.0", meta.SamplingTemp)
	}
	if meta.SamplingTopP == nil || *meta.SamplingTopP < 0.949 || *meta.SamplingTopP > 0.951 {
		t.Errorf("SamplingTopP = %v, want ~0.95", meta.SamplingTopP)
	}
	if meta.SamplingTopK == nil || *meta.SamplingTopK != 20 {
		t.Errorf("SamplingTopK = %v, want 20", meta.SamplingTopK)
	}
	if meta.SamplingMinP == nil || *meta.SamplingMinP != 0.0 {
		t.Errorf("SamplingMinP = %v, want 0.0 (zero is meaningful, not absent)", meta.SamplingMinP)
	}
	if meta.SamplingRepeatPenalty == nil || *meta.SamplingRepeatPenalty < 1.049 || *meta.SamplingRepeatPenalty > 1.051 {
		t.Errorf("SamplingRepeatPenalty = %v, want ~1.05", meta.SamplingRepeatPenalty)
	}

	p := meta.EmbeddedSamplingPreset()
	if p == nil {
		t.Fatal("EmbeddedSamplingPreset returned nil")
	}
	if p.Name != "default" || p.Source != "gguf" {
		t.Errorf("preset name/source = %q/%q, want default/gguf", p.Name, p.Source)
	}

	var m Model
	meta.ApplyTo(&m)
	if len(m.SamplingPresets) != 1 || m.SamplingPresets[0].Source != "gguf" {
		t.Errorf("ApplyTo presets = %+v, want one gguf-source preset", m.SamplingPresets)
	}
	if !m.SamplingChecked || m.BaseModelRepo != "Qwen/Qwen3.8-27B" {
		t.Errorf("ApplyTo checked/base = %v/%q", m.SamplingChecked, m.BaseModelRepo)
	}
}

func TestParseGGUFNoSamplingKeys(t *testing.T) {
	path := writeTestGGUF(t, []testKV{
		{"general.architecture", ggufTypeString, "llama"},
		{"llama.block_count", ggufTypeUint32, uint32(32)},
	})
	meta, err := ParseGGUFMeta(path)
	if err != nil {
		t.Fatal(err)
	}
	if !meta.SamplingChecked {
		t.Error("SamplingChecked should be true even when no keys are present")
	}
	if p := meta.EmbeddedSamplingPreset(); p != nil {
		t.Errorf("EmbeddedSamplingPreset = %+v, want nil", p)
	}
}

func TestParseGGUFSamplingOutOfRange(t *testing.T) {
	path := writeTestGGUF(t, []testKV{
		{"general.architecture", ggufTypeString, "llama"},
		{"general.sampling.temp", ggufTypeFloat32, float32(9.5)},  // > 4: dropped
		{"general.sampling.top_k", ggufTypeUint32, uint32(0)},     // 0: dropped
		{"general.sampling.top_p", ggufTypeFloat32, float32(0.9)}, // valid — proves alignment survived
		{"llama.block_count", ggufTypeUint32, uint32(32)},
	})
	meta, err := ParseGGUFMeta(path)
	if err != nil {
		t.Fatal(err)
	}
	if meta.SamplingTemp != nil || meta.SamplingTopK != nil {
		t.Errorf("out-of-range values kept: temp=%v top_k=%v", meta.SamplingTemp, meta.SamplingTopK)
	}
	if meta.SamplingTopP == nil || meta.NLayers != 32 {
		t.Errorf("parser misaligned after dropped values: top_p=%v layers=%d", meta.SamplingTopP, meta.NLayers)
	}
}

func TestRepoFromURL(t *testing.T) {
	cases := map[string]string{
		"https://huggingface.co/Qwen/Qwen3.8-27B":  "Qwen/Qwen3.8-27B",
		"https://huggingface.co/Qwen/Qwen3.8-27B/": "Qwen/Qwen3.8-27B",
		"http://huggingface.co/org/repo":           "org/repo",
		"https://huggingface.co/onlyorg":           "",
		"https://example.com/org/repo":             "",
		"":                                         "",
	}
	for in, want := range cases {
		if got := repoFromURL(in); got != want {
			t.Errorf("repoFromURL(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestUpsertSamplingPreset(t *testing.T) {
	presets := []SamplingPreset{{Name: "thinking", Source: "unsloth-docs"}}
	presets = UpsertSamplingPreset(presets, SamplingPreset{Name: "default", Source: "gguf"})
	if len(presets) != 2 || presets[0].Name != "default" {
		t.Fatalf("default not inserted first: %+v", presets)
	}
	// Replacing keeps one entry per name.
	presets = UpsertSamplingPreset(presets, SamplingPreset{Name: "default", Source: "gguf", Temperature: fptr(0.7)})
	if len(presets) != 2 || presets[0].Temperature == nil {
		t.Fatalf("replace failed: %+v", presets)
	}
}

func TestEffectiveSamplingPresetsFallback(t *testing.T) {
	// Attached presets win.
	m := &Model{ModelID: "unsloth/Qwen3-32B-GGUF",
		SamplingPresets: []SamplingPreset{{Name: "default", Source: "gguf"}}}
	if got := m.EffectiveSamplingPresets(); len(got) != 1 || got[0].Source != "gguf" {
		t.Errorf("attached presets should win: %+v", got)
	}
	// Empty field falls back to the legacy compiled-in snapshot.
	legacy := &Model{ModelID: "unsloth/Qwen3-32B-GGUF"}
	if got := legacy.EffectiveSamplingPresets(); len(got) == 0 {
		t.Error("expected fallback to legacy scrape data for unsloth/Qwen3-32B-GGUF")
	}
	// Unknown repo, no attached presets: nil.
	unknown := &Model{ModelID: "does/not-exist"}
	if got := unknown.EffectiveSamplingPresets(); len(got) != 0 {
		t.Errorf("expected no presets, got %+v", got)
	}
}
