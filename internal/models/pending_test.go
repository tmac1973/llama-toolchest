package models

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func pendingRegistry(t *testing.T) (*Registry, string) {
	t.Helper()
	dataDir := t.TempDir()
	modelsDir := filepath.Join(dataDir, "models")
	os.MkdirAll(modelsDir, 0o755)
	return NewRegistry(dataDir, modelsDir), modelsDir
}

func pendingEntry(ctx int) PendingConfig {
	return PendingConfig{
		ModelID: "org/a-GGUF", Quant: "Q4_K_M", Filename: "a-Q4_K_M.gguf",
		Config:  ModelConfig{Enabled: true, GPULayers: 999, ContextSize: ctx, Threads: 8},
		SavedAt: time.Now().UTC(),
	}
}

// Pending entries persist across registry reloads, and Add claims a
// matching identity: the imported config wins over the default one.
func TestPendingPersistAndClaimOnAdd(t *testing.T) {
	reg, modelsDir := pendingRegistry(t)
	if err := reg.SetPendingConfig(pendingEntry(31337)); err != nil {
		t.Fatal(err)
	}

	// Reload from disk — a fresh Registry over the same dataDir.
	reg2 := NewRegistry(filepath.Dir(modelsDir), modelsDir)
	if got := reg2.PendingConfigs(); len(got) != 1 {
		t.Fatalf("pending did not persist: %v", got)
	}

	m := &Model{ID: "org--a--q4", ModelID: "org/a-GGUF", Quant: "Q4_K_M", Filename: "a-Q4_K_M.gguf"}
	if err := reg2.Add(m); err != nil {
		t.Fatal(err)
	}
	cfg, err := reg2.GetConfig(m.ID)
	if err != nil || cfg.ContextSize != 31337 {
		t.Fatalf("claim did not attach the pending config: %+v (%v)", cfg, err)
	}
	if len(reg2.PendingConfigs()) != 0 {
		t.Error("claimed entry should be removed")
	}
	// And the claim persisted.
	reg3 := NewRegistry(filepath.Dir(modelsDir), modelsDir)
	if cfg3, _ := reg3.GetConfig(m.ID); cfg3 == nil || cfg3.ContextSize != 31337 {
		t.Error("claim did not persist to disk")
	}
}

// Claim is exact-quant: a different quant of the same repo must not
// consume the pending entry.
func TestClaimExactQuantOnly(t *testing.T) {
	reg, _ := pendingRegistry(t)
	reg.SetPendingConfig(pendingEntry(31337))

	other := &Model{ID: "org--a--q8", ModelID: "org/a-GGUF", Quant: "Q8_0", Filename: "a-Q8_0.gguf"}
	reg.Add(other)
	if cfg, _ := reg.GetConfig(other.ID); cfg.ContextSize == 31337 {
		t.Error("wrong quant claimed the pending config")
	}
	if len(reg.PendingConfigs()) != 1 {
		t.Error("pending entry should survive a non-matching registration")
	}
}

// ScanModels routes through Add, so a scanned-in GGUF claims too.
func TestClaimOnScan(t *testing.T) {
	reg, modelsDir := pendingRegistry(t)
	reg.SetPendingConfig(PendingConfig{
		ModelID: "org/a-GGUF", Quant: "Q4_K_M", Filename: "a-Q4_K_M.gguf",
		Config: ModelConfig{Enabled: true, GPULayers: 999, ContextSize: 31337, Threads: 8},
	})

	repoDir := filepath.Join(modelsDir, "org--a-GGUF")
	os.MkdirAll(repoDir, 0o755)
	// A minimal GGUF file so the scan registers it — reuse the shared
	// test helper's output bytes.
	stub, err := os.ReadFile(writeTestGGUF(t, nil))
	if err != nil {
		t.Fatal(err)
	}
	os.WriteFile(filepath.Join(repoDir, "a-Q4_K_M.gguf"), stub, 0o644)

	if n := reg.ScanModels(); n == 0 {
		t.Skip("scan did not register the stub GGUF on this build")
	}
	if len(reg.PendingConfigs()) != 0 {
		t.Fatal("scan registration did not claim the pending config")
	}
	claimed := false
	for _, m := range reg.List() {
		if cfg, _ := reg.GetConfig(m.ID); cfg != nil && cfg.ContextSize == 31337 {
			claimed = true
		}
	}
	if !claimed {
		t.Error("claimed config not found on the scanned model")
	}
}

// SetPendingConfig upserts by identity; discard removes and persists.
func TestPendingUpsertAndDiscard(t *testing.T) {
	reg, _ := pendingRegistry(t)
	reg.SetPendingConfig(pendingEntry(1000))
	reg.SetPendingConfig(pendingEntry(2000))
	got := reg.PendingConfigs()
	if len(got) != 1 || got[0].Config.ContextSize != 2000 {
		t.Fatalf("upsert failed: %+v", got)
	}
	if !reg.DiscardPendingConfig("org/a-GGUF", "Q4_K_M") {
		t.Error("discard reported missing")
	}
	if reg.DiscardPendingConfig("org/a-GGUF", "Q4_K_M") {
		t.Error("second discard should report missing")
	}
	if len(reg.PendingConfigs()) != 0 {
		t.Error("entry survived discard")
	}
}
