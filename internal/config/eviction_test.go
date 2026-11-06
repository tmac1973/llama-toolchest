package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"go.yaml.in/yaml/v3"
)

// A config from before the eviction setting gets the mode that keeps its
// behaviour, and router_memory is not written back.
func TestLoadMigratesEviction(t *testing.T) {
	cases := []struct {
		yaml string
		want string
	}{
		{"", EvictionOff},
		{"router_memory: false\n", EvictionOff},
		{"router_memory: true\n", EvictionLlamaCpp},
		{"router_memory: true\ngeneration_enabled: true\ngen_evict: true\n", EvictionLlamaCpp},
		{"generation_enabled: true\ngen_evict: true\n", EvictionToolchest},
		{"gen_evict: true\n", EvictionOff},
		{"eviction: toolchest\nrouter_memory: false\n", EvictionToolchest},
		{"eviction: sideways\n", EvictionOff},
	}
	for _, c := range cases {
		path := filepath.Join(t.TempDir(), "cfg.yaml")
		if err := os.WriteFile(path, []byte(c.yaml), 0o644); err != nil {
			t.Fatal(err)
		}
		cfg, err := Load(path)
		if err != nil {
			t.Fatal(err)
		}
		if cfg.Eviction != c.want {
			t.Errorf("%q: Eviction = %q, want %q", c.yaml, cfg.Eviction, c.want)
		}
		data, err := yaml.Marshal(cfg.FileValues())
		if err != nil {
			t.Fatal(err)
		}
		if strings.Contains(string(data), "router_memory") {
			t.Errorf("%q: saved config still has router_memory:\n%s", c.yaml, data)
		}
	}
}

func TestEffectiveEviction(t *testing.T) {
	cases := []struct {
		eviction        string
		generation, fks bool
		want            string
	}{
		{EvictionOff, true, false, EvictionOff},
		{EvictionLlamaCpp, false, false, EvictionLlamaCpp},
		{EvictionLlamaCpp, true, false, EvictionToolchest},
		{EvictionLlamaCpp, false, true, EvictionToolchest},
		{EvictionToolchest, false, false, EvictionToolchest},
		{"", false, false, EvictionOff},
	}
	for _, c := range cases {
		cfg := Config{Eviction: c.eviction, GenerationEnabled: c.generation, ForksEnabled: c.fks}
		if got := cfg.EffectiveEviction(); got != c.want {
			t.Errorf("%+v: EffectiveEviction = %q, want %q", c, got, c.want)
		}
	}
}

// Taking turns needs toolchest to decide: with eviction off the switch
// does nothing.
func TestGenEvictOnNeedsToolchest(t *testing.T) {
	cfg := Config{GenerationEnabled: true, GenEvict: true, Eviction: EvictionOff}
	if cfg.GenEvictOn() {
		t.Error("eviction off: GenEvictOn = true")
	}
	for _, e := range []string{EvictionLlamaCpp, EvictionToolchest} {
		cfg.Eviction = e
		if !cfg.GenEvictOn() {
			t.Errorf("eviction %s with generation on: GenEvictOn = false", e)
		}
	}
	cfg.GenEvict = false
	if cfg.GenEvictOn() {
		t.Error("switch off: GenEvictOn = true")
	}
}
