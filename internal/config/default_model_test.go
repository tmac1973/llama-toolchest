package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"go.yaml.in/yaml/v3"
)

func TestDefaultModelNameAndOn(t *testing.T) {
	c := &Config{}
	if !c.DefaultModelOn() || c.DefaultModelName() != DefaultModelNameDefault {
		t.Errorf("empty config: on %v, name %q", c.DefaultModelOn(), c.DefaultModelName())
	}
	c.DefaultModel = DefaultModelConfig{Off: true, Name: "  "}
	if c.DefaultModelOn() || c.DefaultModelName() != DefaultModelNameDefault {
		t.Errorf("off, blank name: on %v, name %q", c.DefaultModelOn(), c.DefaultModelName())
	}
	c.DefaultModel.Name = " chat "
	if got := c.DefaultModelName(); got != "chat" {
		t.Errorf("name = %q, want chat", got)
	}
}

func TestValidDefaultModelName(t *testing.T) {
	for _, name := range []string{"default", "my-model:latest", "team/chat", "a.b_c", strings.Repeat("x", 64)} {
		if err := ValidDefaultModelName(name); err != nil {
			t.Errorf("%q refused: %v", name, err)
		}
	}
	for _, name := range []string{"", "  ", "two words", strings.Repeat("x", 65), "ä"} {
		if ValidDefaultModelName(name) == nil {
			t.Errorf("%q accepted", name)
		}
	}
}

// The settings survive a save and a load; a config that sets none
// writes nothing and loads with the name on.
func TestDefaultModelRoundTrip(t *testing.T) {
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
	want := DefaultModelConfig{Off: true, Name: "chat", Pin: DefaultModelForkPrefix + "ik-1"}
	save(&Config{DataDir: dir, DefaultModel: want})
	got, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if got.DefaultModel != want {
		t.Errorf("loaded %+v, want %+v", got.DefaultModel, want)
	}
	if out := save(&Config{DataDir: dir}); strings.Contains(out, "default_model:") {
		t.Errorf("a default config writes default_model:\n%s", out)
	}
	got, err = Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if !got.DefaultModelOn() {
		t.Error("a config without the setting loads with the name off")
	}
}

func TestCloneCopiesDefaultModel(t *testing.T) {
	c := &Config{DefaultModel: DefaultModelConfig{Pin: "a"}}
	out := c.Clone()
	out.DefaultModel.Pin = "b"
	if c.DefaultModel.Pin != "a" {
		t.Error("changing the copy changed the original")
	}
}
