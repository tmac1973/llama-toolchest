package models

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

// A model registered before the split-shard scan existed has architecture
// metadata but no PLE table recorded. BackfillGGUFMeta must re-parse it and
// store both the table size and the "we looked" flag, or the Per-Layer
// Embeddings selector never appears for any already-downloaded model.
func TestBackfillPopulatesPLE(t *testing.T) {
	dir := t.TempDir()
	cfgDir := filepath.Join(dir, "config")
	modelsDir := filepath.Join(dir, "models")
	if err := os.MkdirAll(cfgDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(modelsDir, 0o755); err != nil {
		t.Fatal(err)
	}

	const pleSize = int64(6) << 30
	first := writeSplitGGUF(t, modelsDir, "Qwen3.8-Flash-Next-UD-Q4_K_XL", 4, 2, pleSize)

	// The registry as it looks on a box that downloaded this model before
	// the scan shipped: layers known, PLE fields absent.
	stale := fmt.Sprintf(`{"models":{"q":{"id":"q","filename":%q,"file_path":%q,"n_layers":48}}}`,
		filepath.Base(first), first)
	if err := os.WriteFile(filepath.Join(cfgDir, "models.json"), []byte(stale), 0o644); err != nil {
		t.Fatal(err)
	}

	r := NewRegistry(dir, modelsDir)
	r.BackfillGGUFMeta()

	m, err := r.Get("q")
	if err != nil {
		t.Fatal(err)
	}
	if m.PLEBytes != pleSize {
		t.Errorf("PLEBytes = %d, want %d — the selector stays hidden without it", m.PLEBytes, pleSize)
	}
	if !m.PLEChecked {
		t.Error("PLEChecked = false — every startup will re-scan this model's shards forever")
	}

	// And it must actually persist, not just live in memory.
	raw, err := os.ReadFile(filepath.Join(cfgDir, "models.json"))
	if err != nil {
		t.Fatal(err)
	}
	var saved struct {
		Models map[string]struct {
			PLEBytes   int64 `json:"ple_bytes"`
			PLEChecked bool  `json:"ple_checked"`
		} `json:"models"`
	}
	if err := json.Unmarshal(raw, &saved); err != nil {
		t.Fatal(err)
	}
	if saved.Models["q"].PLEBytes != pleSize || !saved.Models["q"].PLEChecked {
		t.Errorf("persisted ple_bytes=%d ple_checked=%v, want %d/true",
			saved.Models["q"].PLEBytes, saved.Models["q"].PLEChecked, pleSize)
	}
}
