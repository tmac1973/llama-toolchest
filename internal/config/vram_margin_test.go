package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

func TestVRAMMarginFor(t *testing.T) {
	c := &Config{VRAMMarginMiB: 512, VRAMMarginByGPU: map[int]int{1: 2048, 2: 0}}
	for gpu, want := range map[int]int{0: 512, 1: 2048, 2: 0, 3: 512} {
		if got := c.VRAMMarginFor(gpu); got != want {
			t.Errorf("GPU %d: %d MiB, want %d", gpu, got, want)
		}
	}
	if got := (&Config{}).VRAMMarginFor(0); got != 0 {
		t.Errorf("default margin = %d MiB, want 0", got)
	}
}

// A copy's per-GPU margins are its own.
func TestCloneCopiesVRAMMargins(t *testing.T) {
	c := &Config{VRAMMarginByGPU: map[int]int{0: 1024}}
	out := c.Clone()
	out.VRAMMarginByGPU[0] = 4096
	if c.VRAMMarginByGPU[0] != 1024 {
		t.Error("changing the copy changed the original")
	}
}

// The margins survive a save and a load; the default writes nothing.
func TestVRAMMarginRoundTrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "llama-toolchest.yaml")
	save := func(c *Config) string {
		t.Helper()
		b, err := yaml.Marshal(c.FileValues())
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, b, 0o600); err != nil {
			t.Fatal(err)
		}
		return string(b)
	}
	save(&Config{DataDir: dir, VRAMMarginMiB: 768, VRAMMarginByGPU: map[int]int{1: 2048}})
	got, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if got.VRAMMarginMiB != 768 || got.VRAMMarginByGPU[1] != 2048 {
		t.Errorf("loaded %d and %v", got.VRAMMarginMiB, got.VRAMMarginByGPU)
	}
	if out := save(&Config{DataDir: dir}); strings.Contains(out, "vram_margin") {
		t.Errorf("a default config writes the margin:\n%s", out)
	}
}
