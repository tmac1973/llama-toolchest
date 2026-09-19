package models

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// writeRegistryFile puts raw bytes where NewRegistry(dataDir, …) reads them.
func writeRegistryFile(t *testing.T, dataDir, contents string) string {
	t.Helper()
	path := filepath.Join(dataDir, "config", "models.json")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(contents), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

// assertRefusesEveryWrite checks that each mutator refuses with
// ErrRegistryReadOnly and that the file on disk is byte-for-byte unchanged.
func assertRefusesEveryWrite(t *testing.T, reg *Registry, path, before string) {
	t.Helper()
	if reg.ReadOnlyReason() == "" {
		t.Fatal("ReadOnlyReason is empty; want the registry read-only")
	}
	m := &Model{ID: "new--m.gguf", ModelID: "org/new-GGUF", FilePath: "/nowhere/m.gguf"}
	checks := map[string]error{
		"Add":                reg.Add(m),
		"SetConfig":          reg.SetConfig("kept--m.gguf", &ModelConfig{ContextSize: 1}),
		"SetSamplingPresets": reg.SetSamplingPresets("kept--m.gguf", nil, time.Now()),
		"SetPendingConfig":   reg.SetPendingConfig(pendingEntry(4096)),
		"Remove":             reg.Remove("kept--m.gguf"),
		"Delete":             reg.Delete("kept--m.gguf"),
	}
	_, discardErr := reg.DiscardPendingConfig("org/a-GGUF", "Q4_K_M")
	checks["DiscardPendingConfig"] = discardErr
	for name, err := range checks {
		if !errors.Is(err, ErrRegistryReadOnly) {
			t.Errorf("%s: err = %v, want ErrRegistryReadOnly", name, err)
		}
	}
	if n := reg.DeduplicateModels() + reg.AutoDetectMMProj() + reg.AutoDetectMTP() +
		reg.BackfillSpecAssist() + reg.ScanModels(); n != 0 {
		t.Errorf("startup backfills changed %d entries on a read-only registry", n)
	}
	reg.BackfillGGUFMeta()

	after, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(after) != before {
		t.Errorf("models.json changed on disk:\n%s", after)
	}
}

const keptModelJSON = `"kept--m.gguf": {"id": "kept--m.gguf", "model_id": "org/kept-GGUF", "file_path": "/nowhere/kept.gguf"}`

// A file that does not parse used to be replaced by an empty registry on
// the next save. Now nothing is written and the reason is reported.
func TestRegistryCorruptFileIsReadOnly(t *testing.T) {
	dataDir := t.TempDir()
	corrupt := `{"models": {` + keptModelJSON + `,` // truncated mid-write
	path := writeRegistryFile(t, dataDir, corrupt)

	reg := NewRegistry(dataDir, filepath.Join(dataDir, "models"))
	if !strings.Contains(reg.ReadOnlyReason(), "could not be parsed") {
		t.Errorf("ReadOnlyReason = %q, want it to say the file could not be parsed", reg.ReadOnlyReason())
	}
	assertRefusesEveryWrite(t, reg, path, corrupt)
}

// A file from a newer build may carry fields this build would drop, so it
// is served but never rewritten.
func TestRegistryNewerSchemaIsReadOnly(t *testing.T) {
	dataDir := t.TempDir()
	newer := `{"schema_version": 99, "models": {` + keptModelJSON + `}, "configs": {}, "future_field": [1, 2]}`
	path := writeRegistryFile(t, dataDir, newer)

	reg := NewRegistry(dataDir, filepath.Join(dataDir, "models"))
	if !strings.Contains(reg.ReadOnlyReason(), "newer version") {
		t.Errorf("ReadOnlyReason = %q, want it to name a newer version", reg.ReadOnlyReason())
	}
	if _, err := reg.Get("kept--m.gguf"); err != nil {
		t.Errorf("model from the newer file is not listed: %v", err)
	}
	assertRefusesEveryWrite(t, reg, path, newer)
}

// Every file written before the version existed has none; it loads as
// current and is saved with the version from then on.
func TestRegistryUnversionedFileLoadsAndGainsVersion(t *testing.T) {
	dataDir := t.TempDir()
	path := writeRegistryFile(t, dataDir, `{"models": {`+keptModelJSON+`}, "configs": {"kept--m.gguf": {"context_size": 4096}}}`)

	reg := NewRegistry(dataDir, filepath.Join(dataDir, "models"))
	if reason := reg.ReadOnlyReason(); reason != "" {
		t.Fatalf("unversioned file made the registry read-only: %s", reason)
	}
	if err := reg.SetConfig("kept--m.gguf", &ModelConfig{ContextSize: 8192}); err != nil {
		t.Fatal(err)
	}

	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var onDisk struct {
		SchemaVersion int                     `json:"schema_version"`
		Configs       map[string]*ModelConfig `json:"configs"`
	}
	if err := json.Unmarshal(raw, &onDisk); err != nil {
		t.Fatal(err)
	}
	if onDisk.SchemaVersion != RegistrySchemaVersion {
		t.Errorf("schema_version = %d, want %d", onDisk.SchemaVersion, RegistrySchemaVersion)
	}
	if got := onDisk.Configs["kept--m.gguf"].ContextSize; got != 8192 {
		t.Errorf("saved context_size = %d, want 8192", got)
	}

	entries, err := os.ReadDir(filepath.Dir(path))
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range entries {
		if strings.HasSuffix(e.Name(), ".tmp") {
			t.Errorf("temporary file left behind: %s", e.Name())
		}
	}
}

// A fresh install has no file at all, which is not an error.
func TestRegistryMissingFileIsWritable(t *testing.T) {
	reg, _ := pendingRegistry(t)
	if reason := reg.ReadOnlyReason(); reason != "" {
		t.Fatalf("missing models.json made the registry read-only: %s", reason)
	}
	if err := reg.SetPendingConfig(pendingEntry(4096)); err != nil {
		t.Fatal(err)
	}
}

// Delete must refuse before it removes the model's files: a delete that
// removed the GGUF and then failed to record it would leave an entry for a
// file that no longer exists.
func TestRegistryReadOnlyDeleteKeepsFiles(t *testing.T) {
	dataDir := t.TempDir()
	gguf := filepath.Join(dataDir, "models", "kept.gguf")
	if err := os.MkdirAll(filepath.Dir(gguf), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(gguf, []byte("GGUF"), 0o644); err != nil {
		t.Fatal(err)
	}
	writeRegistryFile(t, dataDir, `{"schema_version": 99, "models": {"kept--m.gguf": {"id": "kept--m.gguf", "file_path": "`+gguf+`"}}}`)

	reg := NewRegistry(dataDir, filepath.Join(dataDir, "models"))
	if err := reg.Delete("kept--m.gguf"); !errors.Is(err, ErrRegistryReadOnly) {
		t.Fatalf("Delete err = %v, want ErrRegistryReadOnly", err)
	}
	if _, err := os.Stat(gguf); err != nil {
		t.Errorf("model file was removed by a refused delete: %v", err)
	}
}
