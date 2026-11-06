package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

func TestVRAMMarginFor(t *testing.T) {
	margin := 512
	c := &Config{VRAMMarginMiB: &margin, VRAMMarginByGPU: map[int]int{1: 2048, 2: 0}}
	for gpu, want := range map[int]int{0: 512, 1: 2048, 2: 0, 3: 512} {
		if got := c.VRAMMarginFor(gpu); got != want {
			t.Errorf("GPU %d: %d MiB, want %d", gpu, got, want)
		}
	}
	if got := (&Config{}).VRAMMarginFor(0); got != DefaultVRAMMarginMiB {
		t.Errorf("default margin = %d MiB, want %d", got, DefaultVRAMMarginMiB)
	}
	zero := 0
	if got := (&Config{VRAMMarginMiB: &zero}).VRAMMarginFor(0); got != 0 {
		t.Errorf("margin set to 0 = %d MiB, want 0", got)
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

// The margins survive a save and a load, 0 too; the default writes
// nothing.
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
	for _, mib := range []int{768, 0} {
		save(&Config{DataDir: dir, VRAMMarginMiB: &mib, VRAMMarginByGPU: map[int]int{1: 2048, 2: 0}})
		got, err := Load(path)
		if err != nil {
			t.Fatal(err)
		}
		if got.VRAMMarginAll() != mib || got.VRAMMarginFor(1) != 2048 || got.VRAMMarginFor(2) != 0 {
			t.Errorf("saved %d: loaded %d and %v", mib, got.VRAMMarginAll(), got.VRAMMarginByGPU)
		}
	}
	if out := save(&Config{DataDir: dir}); strings.Contains(out, "vram_margin") {
		t.Errorf("a default config writes the margin:\n%s", out)
	}
}
