package models

import (
	"os"
	"path/filepath"
	"testing"
)

func TestParseQuant(t *testing.T) {
	cases := []struct {
		filename string
		want     string
	}{
		// UD ultra-dynamic quants — prefix must be preserved consistently.
		{"UD-IQ1_M/MiniMax-M2.5-UD-IQ1_M-00001-of-00003.gguf", "UD_IQ1_M"},
		{"UD-IQ1_S/MiniMax-M2.5-UD-IQ1_S-00001-of-00003.gguf", "UD_IQ1_S"},
		{"UD-IQ2_M/MiniMax-M2.5-UD-IQ2_M-00001-of-00003.gguf", "UD_IQ2_M"},
		{"UD-IQ2_XXS/MiniMax-M2.5-UD-IQ2_XXS-00001-of-00003.gguf", "UD_IQ2_XXS"},
		{"UD-IQ3_XXS/MiniMax-M2.5-UD-IQ3_XXS-00001-of-00003.gguf", "UD_IQ3_XXS"},
		// Previously dropped the XL suffix and reported a bare Q2_K.
		{"UD-Q2_K_XL/MiniMax-M2.5-UD-Q2_K_XL-00001-of-00003.gguf", "UD_Q2_K_XL"},
		{"UD-Q3_K_XL/MiniMax-M2.5-UD-Q3_K_XL-00001-of-00004.gguf", "UD_Q3_K_XL"},
		{"UD-Q4_K_XL/MiniMax-M2.5-UD-Q4_K_XL-00001-of-00004.gguf", "UD_Q4_K_XL"},
		{"UD-Q5_K_XL/MiniMax-M2.5-UD-Q5_K_XL-00001-of-00005.gguf", "UD_Q5_K_XL"},
		{"UD-Q6_K_XL/MiniMax-M2.5-UD-Q6_K_XL-00001-of-00005.gguf", "UD_Q6_K_XL"},

		// MXFP4 — previously reported "unknown". The bare-FP4 alternative
		// must not steal these: MXFP4 starts earlier, so leftmost-first wins.
		{"gpt-oss-20b-MXFP4.gguf", "MXFP4"},
		{"openai-gpt-oss-120b-UD-MXFP4.gguf", "UD_MXFP4"},

		// FP4 family — vendor-prefixed and bare, previously "unknown".
		{"Qwen3.5-122B-A10B-Heretic-ROCmFP4-iMatrix.gguf", "FP4"},
		{"Qwen3.5-122B-A10B-Heretic-ROCmFP4-MTP.gguf", "FP4"},
		{"model-NVFP4.gguf", "FP4"},
		{"model-FP4.gguf", "FP4"},

		// Plain (non-UD) quants.
		{"model-Q4_K_M.gguf", "Q4_K_M"},
		{"model-Q2_K.gguf", "Q2_K"},
		{"model-Q2_K_S.gguf", "Q2_K_S"},
		{"model-Q8_0.gguf", "Q8_0"},
		{"model-Q4_0.gguf", "Q4_0"},
		{"model-IQ4_NL.gguf", "IQ4_NL"},
		{"model-IQ2_XS.gguf", "IQ2_XS"},
		{"model-TQ1_0.gguf", "TQ1_0"},
		{"model-F16.gguf", "F16"},
		{"model-BF16.gguf", "BF16"},
		{"model-F32.gguf", "F32"},

		// No recognizable quant.
		{"some-random-model.gguf", "unknown"},
	}

	for _, c := range cases {
		if got := ParseQuant(c.filename); got != c.want {
			t.Errorf("ParseQuant(%q) = %q, want %q", c.filename, got, c.want)
		}
	}
}

// TestLoadBackfillsStaleQuant verifies that load() re-derives the persisted
// Quant field from the filename. Models registered before a ParseQuant
// improvement keep their stale value (ScanModels skips known paths), so load
// must backfill it for the Models tab to match the live-parsed Search HF tab.
func TestLoadBackfillsStaleQuant(t *testing.T) {
	dir := t.TempDir()
	cfgDir := filepath.Join(dir, "config")
	if err := os.MkdirAll(cfgDir, 0o755); err != nil {
		t.Fatal(err)
	}
	// Persist a registry whose MXFP4 model was frozen as "unknown" by the old
	// hand-maintained list, and a UD model truncated to its base quant.
	const stale = `{
  "models": {
    "a": {"id": "a", "filename": "gpt-oss-20b-MXFP4.gguf", "quant": "unknown"},
    "b": {"id": "b", "filename": "model-UD-Q2_K_XL.gguf", "quant": "Q2_K"},
    "c": {"id": "c", "filename": "model-Q8_0.gguf", "quant": "Q8_0"}
  }
}`
	if err := os.WriteFile(filepath.Join(cfgDir, "models.json"), []byte(stale), 0o644); err != nil {
		t.Fatal(err)
	}

	r := NewRegistry(dir, filepath.Join(dir, "models"))

	want := map[string]string{"a": "MXFP4", "b": "UD_Q2_K_XL", "c": "Q8_0"}
	for id, q := range want {
		m, err := r.Get(id)
		if err != nil {
			t.Fatalf("Get(%q): %v", id, err)
		}
		if m.Quant != q {
			t.Errorf("model %q Quant = %q, want %q", id, m.Quant, q)
		}
	}
}
