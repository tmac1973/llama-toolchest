package models

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

// The failure this design exists to prevent: a record holding a value that
// is present, plausible and wrong. Every per-field trigger it replaced
// asked "does this look unset?", which cannot see such a record — and
// twice did not, shipping a fix that changed nothing on disk.
func TestVersionReReadsPlausibleButWrongValues(t *testing.T) {
	dir, modelsDir := t.TempDir(), ""
	cfgDir := filepath.Join(dir, "config")
	modelsDir = filepath.Join(dir, "models")
	for _, d := range []string{cfgDir, modelsDir} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	path := writeHybridGGUF(t, modelsDir, "hybrid.gguf", 8, 4, 2, 256)

	// Nothing here is zero or empty. Under the old triggers every one of
	// these fields looked already-populated, so none of them was re-read.
	const wrongKV = 8 * 2 * (256 + 256) // every layer counted
	raw := fmt.Sprintf(`{"models":{"h":{"id":"h","filename":"hybrid.gguf","file_path":%q,`+
		`"n_layers":8,"n_embd":512,"n_head":2,"n_kv_head":2,"kv_full_per_tok":%d,`+
		`"vocab_checked":true,"ple_checked":true,"reasoning_checked":true,"sampling_checked":true}}}`,
		path, wrongKV)
	if err := os.WriteFile(filepath.Join(cfgDir, "models.json"), []byte(raw), 0o644); err != nil {
		t.Fatal(err)
	}

	r := NewRegistry(dir, modelsDir)
	r.BackfillGGUFMeta()

	m, err := r.Get("h")
	if err != nil {
		t.Fatal(err)
	}
	if want := 2 * 2 * (256 + 256); m.KVFullPerTok != want {
		t.Errorf("KVFullPerTok = %d, want %d — the stale value survived", m.KVFullPerTok, want)
	}
	if m.GGUFMetaVersion != GGUFMetaVersion {
		t.Errorf("version = %d, want %d", m.GGUFMetaVersion, GGUFMetaVersion)
	}
	if m.AttnLayers != 2 {
		t.Errorf("AttnLayers = %d, want 2 — a field added with this version was not filled", m.AttnLayers)
	}
}

// Re-reading must happen once, not at every startup. The old vision
// trigger asked whether a model lacked vision, so every model without it
// was re-parsed forever — and for a split model that reopens every shard.
func TestVersionStopsReReadingOnceCurrent(t *testing.T) {
	dir := t.TempDir()
	cfgDir, modelsDir := filepath.Join(dir, "config"), filepath.Join(dir, "models")
	for _, d := range []string{cfgDir, modelsDir} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	path := writeHybridGGUF(t, modelsDir, "m.gguf", 8, 4, 2, 256)
	raw := fmt.Sprintf(`{"models":{"m":{"id":"m","filename":"m.gguf","file_path":%q,"n_layers":8}}}`, path)
	if err := os.WriteFile(filepath.Join(cfgDir, "models.json"), []byte(raw), 0o644); err != nil {
		t.Fatal(err)
	}

	r := NewRegistry(dir, modelsDir)
	r.BackfillGGUFMeta()

	// Remove the file. A second pass that still wanted to re-read would
	// fail to parse and log; one that is satisfied never opens it.
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	r.BackfillGGUFMeta()

	m, _ := r.Get("m")
	if m.NLayers == 0 {
		t.Error("the second pass re-read and clobbered the record")
	}
	if m.GGUFMetaVersion != GGUFMetaVersion {
		t.Errorf("version = %d, want %d", m.GGUFMetaVersion, GGUFMetaVersion)
	}
}

// An unreadable file must not be versioned: an unmounted volume or a
// download still moving into place should be re-read next start, not
// frozen at whatever an older parser wrote.
func TestUnreadableFileIsNotVersioned(t *testing.T) {
	dir := t.TempDir()
	cfgDir := filepath.Join(dir, "config")
	if err := os.MkdirAll(cfgDir, 0o755); err != nil {
		t.Fatal(err)
	}
	raw := `{"models":{"gone":{"id":"gone","filename":"gone.gguf","file_path":"/nonexistent/gone.gguf","n_layers":8}}}`
	if err := os.WriteFile(filepath.Join(cfgDir, "models.json"), []byte(raw), 0o644); err != nil {
		t.Fatal(err)
	}
	r := NewRegistry(dir, filepath.Join(dir, "models"))
	r.BackfillGGUFMeta()
	m, _ := r.Get("gone")
	if m.GGUFMetaVersion != 0 {
		t.Errorf("version = %d, want 0 so the record is retried when the file returns", m.GGUFMetaVersion)
	}
}

// The version must persist, or every startup re-reads every model.
func TestVersionPersists(t *testing.T) {
	dir := t.TempDir()
	cfgDir, modelsDir := filepath.Join(dir, "config"), filepath.Join(dir, "models")
	for _, d := range []string{cfgDir, modelsDir} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	path := writeHybridGGUF(t, modelsDir, "m.gguf", 8, 4, 2, 256)
	raw := fmt.Sprintf(`{"models":{"m":{"id":"m","filename":"m.gguf","file_path":%q,"n_layers":8}}}`, path)
	if err := os.WriteFile(filepath.Join(cfgDir, "models.json"), []byte(raw), 0o644); err != nil {
		t.Fatal(err)
	}
	NewRegistry(dir, modelsDir).BackfillGGUFMeta()

	b, err := os.ReadFile(filepath.Join(cfgDir, "models.json"))
	if err != nil {
		t.Fatal(err)
	}
	var saved struct {
		Models map[string]struct {
			V int `json:"gguf_meta_version"`
		} `json:"models"`
	}
	if err := json.Unmarshal(b, &saved); err != nil {
		t.Fatal(err)
	}
	if saved.Models["m"].V != GGUFMetaVersion {
		t.Errorf("persisted version %d, want %d", saved.Models["m"].V, GGUFMetaVersion)
	}
}
