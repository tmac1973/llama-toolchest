package models

import (
	"path/filepath"
	"testing"
	"time"
)

func seedRegistry(t *testing.T) (*Registry, string) {
	t.Helper()
	dir := t.TempDir()
	r := NewRegistry(dir, filepath.Join(dir, "models"))
	m := &Model{ID: "m", ModelID: "org/M-GGUF", Filename: "m.gguf"}
	if err := r.Add(m); err != nil {
		t.Fatal(err)
	}
	return r, m.ID
}

func TestSeedConfigOnlyReplacesTheDefault(t *testing.T) {
	r, id := seedRegistry(t)
	cfg := DefaultConfig()
	cfg.ContextSize, cfg.KVCacheQuant = 131072, "q8_0"
	note := SeedNote{Requested: 131072, Context: 131072, At: time.Now()}

	ok, err := r.SeedConfig(id, cfg, note)
	if err != nil || !ok {
		t.Fatalf("SeedConfig = %v, %v", ok, err)
	}
	got, _ := r.GetConfig(id)
	if got.ContextSize != 131072 || got.KVCacheQuant != "q8_0" {
		t.Errorf("config = %+v", got)
	}
	if m, _ := r.Get(id); m.Seeded == nil {
		t.Error("not marked")
	}

	// A second seed finds a config that is no longer the default.
	cfg.ContextSize = 8192
	if ok, _ := r.SeedConfig(id, cfg, note); ok {
		t.Error("a seeded config was seeded again")
	}
}

// A config someone else put there first (a backup restore, an edit) is
// never replaced.
func TestSeedConfigLeavesAChangedConfigAlone(t *testing.T) {
	r, id := seedRegistry(t)
	cur, _ := r.GetConfig(id)
	changed := *cur
	changed.ContextSize = 65536
	if err := r.SetConfig(id, &changed); err != nil {
		t.Fatal(err)
	}
	if ok, _ := r.SeedConfig(id, DefaultConfig(), SeedNote{}); ok {
		t.Error("a changed config was replaced")
	}
	if got, _ := r.GetConfig(id); got.ContextSize != 65536 {
		t.Errorf("context = %d", got.ContextSize)
	}
}

func TestApplyingAProfileClearsTheSeededMark(t *testing.T) {
	r, id := seedRegistry(t)
	cfg := DefaultConfig()
	cfg.ContextSize = 32768
	if ok, err := r.SeedConfig(id, cfg, SeedNote{Requested: 32768}); err != nil || !ok {
		t.Fatalf("SeedConfig = %v, %v", ok, err)
	}
	if _, err := r.SaveProfile(id, "Mine", ProfileSourceUser, ""); err != nil {
		t.Fatal(err)
	}
	if err := r.ApplyProfile(id, "Mine"); err != nil {
		t.Fatal(err)
	}
	if m, _ := r.Get(id); m.Seeded != nil {
		t.Error("applying a profile kept the mark")
	}
}

func TestClearSeeded(t *testing.T) {
	r, id := seedRegistry(t)
	if _, err := r.SeedConfig(id, DefaultConfig(), SeedNote{}); err != nil {
		t.Fatal(err)
	}
	if err := r.ClearSeeded(id); err != nil {
		t.Fatal(err)
	}
	if m, _ := r.Get(id); m.Seeded != nil {
		t.Error("not cleared")
	}
	if err := r.ClearSeeded("nobody"); err != nil {
		t.Errorf("clearing an unknown model: %v", err)
	}
}
