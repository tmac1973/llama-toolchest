package models

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The generation backends keep their model files in gen-models under the
// models directory. Their GGUFs and partial downloads must not show up as
// language models or as incomplete LLM downloads.
func TestScanSkipsGenModelsDir(t *testing.T) {
	r, _, modelsDir := newTestRegistry(t)
	stub, err := os.ReadFile(writeTestGGUF(t, nil))
	if err != nil {
		t.Fatal(err)
	}
	write := func(path string, data []byte) {
		t.Helper()
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, data, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write(filepath.Join(modelsDir, "org--llm-GGUF", "llm-Q4_K_M.gguf"), stub)
	gen := filepath.Join(modelsDir, GenModelsDirName)
	write(filepath.Join(gen, "leejet--Z-Image-Turbo-GGUF", "z_image_turbo-Q8_0.gguf"), stub)
	write(filepath.Join(gen, "silveroxides--Chroma-GGUF", "Chroma1-HD", "Chroma1-HD-Q8_0.gguf.part"), []byte("x"))
	// The forks' model sets, likewise.
	fork := filepath.Join(modelsDir, ForkModelsDirName)
	write(filepath.Join(fork, "freriadoc--Qwen3.8-Flash-Next-Avalon-GGUF", "Qwen3.8-Flash-Next-Avalon-IQ3KS-00001-of-00033.gguf"), stub)
	write(filepath.Join(fork, "freriadoc--Qwen3.8-Flash-Next-Avalon-GGUF", "MTP", "mtp.gguf.part"), []byte("x"))

	if n := r.ScanModels(); n != 1 {
		t.Errorf("ScanModels found %d models (%v), want only the LLM", n, modelIDs(r.List()))
	}
	for _, m := range r.List() {
		for _, dir := range []string{gen, fork} {
			if rel, _ := filepath.Rel(dir, m.FilePath); !strings.HasPrefix(rel, "..") {
				t.Errorf("registered %s from %s: %s", m.ID, filepath.Base(dir), m.FilePath)
			}
		}
	}
	if got := r.OrphanParts(); len(got) != 0 {
		t.Errorf("OrphanParts = %+v, want none from gen-models or fork-models", got)
	}
}
