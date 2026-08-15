package models

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// benchRegistry builds a two-model registry backed by a temp data dir.
func benchRegistry(t *testing.T) *Registry {
	t.Helper()
	r := NewRegistry(t.TempDir(), "/data/models")
	for _, m := range []*Model{
		{ID: "a", ModelID: "u/A-GGUF", Quant: "Q8_0", FilePath: "/data/models/u--A-GGUF/A-Q8_0.gguf"},
		{ID: "b", ModelID: "u/B-GGUF", Quant: "Q8_0", FilePath: "/data/models/u--B-GGUF/B-Q8_0.gguf"},
	} {
		if err := r.Add(m); err != nil {
			t.Fatalf("add %s: %v", m.ID, err)
		}
	}
	if err := r.SetConfig("a", &ModelConfig{Enabled: true, ContextSize: 8192, GPULayers: 999, Threads: 8}); err != nil {
		t.Fatalf("set config a: %v", err)
	}
	if err := r.SetConfig("b", &ModelConfig{Enabled: true, ContextSize: 4096, GPULayers: 999, Threads: 8}); err != nil {
		t.Fatalf("set config b: %v", err)
	}
	return r
}

func sectionOf(t *testing.T, ini, id string) string {
	t.Helper()
	start := strings.Index(ini, "["+id+"]")
	if start < 0 {
		t.Fatalf("section [%s] missing from:\n%s", id, ini)
	}
	rest := ini[start:]
	if end := strings.Index(rest[1:], "\n["); end >= 0 {
		rest = rest[:end+1]
	}
	return rest
}

// The override must reach the generated INI for the targeted model and
// leave every other model on its saved config.
func TestWriteBenchPresetINIAppliesOverrideToOneModel(t *testing.T) {
	r := benchRegistry(t)

	path, err := r.WriteBenchPresetINI(map[string]*ModelConfig{
		"a": {Enabled: true, ContextSize: 65536, GPULayers: 999, Threads: 8},
	}, "")
	if err != nil {
		t.Fatalf("write: %v", err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	ini := string(raw)

	if got := sectionOf(t, ini, "a"); !strings.Contains(got, "ctx-size = 65536") {
		t.Errorf("override not applied to model a:\n%s", got)
	}
	if got := sectionOf(t, ini, "b"); !strings.Contains(got, "ctx-size = 4096") {
		t.Errorf("model b should keep its saved config:\n%s", got)
	}
}

// The whole point of the ephemeral layer: the user's saved config and
// the real preset must be untouched by a benchmark override.
func TestWriteBenchPresetINIDoesNotMutateSavedConfig(t *testing.T) {
	r := benchRegistry(t)

	realPath, err := r.WritePresetINI("")
	if err != nil {
		t.Fatalf("write real preset: %v", err)
	}
	realBefore, err := os.ReadFile(realPath)
	if err != nil {
		t.Fatalf("read real preset: %v", err)
	}

	benchPath, err := r.WriteBenchPresetINI(map[string]*ModelConfig{
		"a": {Enabled: true, ContextSize: 65536, GPULayers: 1, Threads: 1},
	}, "")
	if err != nil {
		t.Fatalf("write bench preset: %v", err)
	}

	if benchPath == realPath {
		t.Fatalf("bench preset must not share a path with the real preset (%s)", realPath)
	}

	// Registry config unchanged.
	cfg, err := r.GetConfig("a")
	if err != nil {
		t.Fatalf("get config: %v", err)
	}
	if cfg.ContextSize != 8192 || cfg.GPULayers != 999 || cfg.Threads != 8 {
		t.Errorf("saved config mutated: ctx=%d ngl=%d threads=%d",
			cfg.ContextSize, cfg.GPULayers, cfg.Threads)
	}

	// Real preset file unchanged on disk.
	realAfter, err := os.ReadFile(realPath)
	if err != nil {
		t.Fatalf("re-read real preset: %v", err)
	}
	if string(realBefore) != string(realAfter) {
		t.Error("real preset.ini was rewritten by a bench override")
	}

	// And regenerating the real preset still reflects saved values.
	if got := sectionOf(t, string(realAfter), "a"); !strings.Contains(got, "ctx-size = 8192") {
		t.Errorf("real preset lost the saved ctx-size:\n%s", got)
	}
}

// A nil map is the "no overrides" case and must produce content
// equivalent to the real preset, so clearing overrides is a plain
// regeneration rather than a special path.
func TestWriteBenchPresetININilOverridesMatchesReal(t *testing.T) {
	r := benchRegistry(t)

	realPath, err := r.WritePresetINI("")
	if err != nil {
		t.Fatalf("write real: %v", err)
	}
	benchPath, err := r.WriteBenchPresetINI(nil, "")
	if err != nil {
		t.Fatalf("write bench: %v", err)
	}
	realRaw, _ := os.ReadFile(realPath)
	benchRaw, _ := os.ReadFile(benchPath)
	if string(realRaw) != string(benchRaw) {
		t.Errorf("nil overrides should match the real preset\nreal:\n%s\nbench:\n%s", realRaw, benchRaw)
	}
}

// An override for a model that isn't registered must not invent a
// section — a stale model ID in a job should degrade to "no override",
// not corrupt the preset.
func TestWriteBenchPresetINIIgnoresUnknownModel(t *testing.T) {
	r := benchRegistry(t)

	path, err := r.WriteBenchPresetINI(map[string]*ModelConfig{
		"ghost": {Enabled: true, ContextSize: 65536},
		"a":     nil,
	}, "")
	if err != nil {
		t.Fatalf("write: %v", err)
	}
	raw, _ := os.ReadFile(path)
	ini := string(raw)

	if strings.Contains(ini, "[ghost]") {
		t.Errorf("unknown model produced a section:\n%s", ini)
	}
	// A nil override for a known model falls through to its saved config.
	if got := sectionOf(t, ini, "a"); !strings.Contains(got, "ctx-size = 8192") {
		t.Errorf("nil override should leave saved config in place:\n%s", got)
	}
}

func TestBenchPresetFileNameIsDistinct(t *testing.T) {
	if BenchPresetFileName == PresetFileName {
		t.Fatal("bench preset must be a separate file from the real preset")
	}
	if filepath.Ext(BenchPresetFileName) != ".ini" {
		t.Errorf("unexpected extension on %q", BenchPresetFileName)
	}
}
