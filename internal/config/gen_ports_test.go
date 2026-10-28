package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

func TestGenPortsDefaultsAndContainer(t *testing.T) {
	c := &Config{}
	if start, count := c.GenPorts(false); start != 8081 || count != 8 {
		t.Errorf("unset = %d,%d, want 8081,8", start, count)
	}
	c.GenPortStart, c.GenPortCount = 9100, 4
	if start, count := c.GenPorts(false); start != 9100 || count != 4 {
		t.Errorf("host = %d,%d, want 9100,4", start, count)
	}
	// Inside a container the range starts at 8081 and holds the ports
	// setup.sh chose, so a saved value cannot move or resize it.
	if start, count := c.GenPorts(true); start != 8081 || count != 8 {
		t.Errorf("container = %d,%d, want 8081,8", start, count)
	}
	c.GenContainerPortCount = 40
	if start, count := c.GenPorts(true); start != 8081 || count != 40 {
		t.Errorf("container with a count = %d,%d, want 8081,40", start, count)
	}
}

func TestGenPortCountFromEnv(t *testing.T) {
	for _, tc := range []struct {
		value string
		want  int
	}{{"", 0}, {"40", 40}, {"64", 64}, {"65", 0}, {"0", 0}, {"many", 0}} {
		t.Setenv("LLAMA_TOOLCHEST_GEN_PORT_COUNT", tc.value)
		c := &Config{}
		applyEnvOverrides(c)
		if c.GenContainerPortCount != tc.want {
			t.Errorf("LLAMA_TOOLCHEST_GEN_PORT_COUNT=%q gives %d, want %d", tc.value, c.GenContainerPortCount, tc.want)
		}
	}
}

func TestGenPublicStart(t *testing.T) {
	c := &Config{GenPortStart: 9100}
	if got := c.GenPublicStart(false); got != 9100 {
		t.Errorf("unset public start = %d, want the listening start 9100", got)
	}
	c.GenPublicPortStart = 18081
	if got := c.GenPublicStart(true); got != 18081 {
		t.Errorf("public start = %d, want 18081", got)
	}
}

func TestValidateGenPorts(t *testing.T) {
	for _, tc := range []struct {
		start, count int
		want         string // "" means valid
	}{
		{8081, 8, ""},
		{80, 8, "between 1024 and 65535"},
		{8081, 0, "between 1 and 64"},
		{8081, 65, "between 1 and 64"},
		{65530, 8, "past 65535"},
		{8075, 8, "llama-server port 8080"},
		{2995, 8, "web UI port 3000"},
	} {
		err := ValidateGenPorts(tc.start, tc.count, 8080, ":3000")
		switch {
		case tc.want == "" && err != nil:
			t.Errorf("%d+%d: %v, want valid", tc.start, tc.count, err)
		case tc.want != "" && (err == nil || !strings.Contains(err.Error(), tc.want)):
			t.Errorf("%d+%d: %v, want an error containing %q", tc.start, tc.count, err, tc.want)
		}
	}
}

// An install that never touches generation settings must not gain new
// keys in its config file when it is saved.
func TestGenFieldsOmittedFromYAMLByDefault(t *testing.T) {
	out, err := yaml.Marshal(&Config{LlamaPort: 8080})
	if err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"generation_enabled", "gen_port_start", "gen_port_count", "gen_public_port_start"} {
		if strings.Contains(string(out), key) {
			t.Errorf("default config YAML contains %q:\n%s", key, out)
		}
	}
}

func TestGenPublicPortEnvOverride(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cfg.yaml")
	if err := os.WriteFile(path, []byte("llama_port: 8080\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("LLAMA_TOOLCHEST_GEN_PUBLIC_PORT_START", "9081")
	cfg, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.GenPublicPortStart != 9081 {
		t.Errorf("GenPublicPortStart = %d, want 9081 from the environment", cfg.GenPublicPortStart)
	}

	t.Setenv("LLAMA_TOOLCHEST_GEN_PUBLIC_PORT_START", "not-a-port")
	if cfg, _ = Load(path); cfg.GenPublicPortStart != 0 {
		t.Errorf("invalid value was used: %d", cfg.GenPublicPortStart)
	}
}

// The llama port override gets the same 1-65535 check as the other ports:
// 0 or 70000 is ignored and the YAML value kept.
func TestLlamaPortEnvOverrideChecksRange(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cfg.yaml")
	if err := os.WriteFile(path, []byte("llama_port: 8080\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	for _, v := range []string{"0", "70000", "x"} {
		t.Setenv("LLAMA_TOOLCHEST_LLAMA_PORT", v)
		if cfg, err := Load(path); err != nil || cfg.LlamaPort != 8080 {
			t.Errorf("LLAMA_TOOLCHEST_LLAMA_PORT=%s: LlamaPort = %d (err %v), want 8080", v, cfg.LlamaPort, err)
		}
	}
	t.Setenv("LLAMA_TOOLCHEST_LLAMA_PORT", "8090")
	if cfg, _ := Load(path); cfg.LlamaPort != 8090 {
		t.Errorf("LlamaPort = %d, want 8090 from the environment", cfg.LlamaPort)
	}
}

// A Clone keeps the saved file values a later Set changes, so a reader
// holding a copy never sees a save in progress.
func TestCloneCopiesTheSavedFileValues(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cfg.yaml")
	if err := os.WriteFile(path, []byte("external_url: http://saved:3000\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("LLAMA_TOOLCHEST_EXTERNAL_URL", "http://env:3000")
	cfg, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	c := cfg.Clone()
	cfg.SetExternalURL("http://changed:3000")
	if got := c.SavedExternalURL(); got != "http://saved:3000" {
		t.Errorf("the copy's saved URL = %q, want it unchanged by the later Set", got)
	}
	if got := cfg.SavedExternalURL(); got != "http://changed:3000" {
		t.Errorf("the config's saved URL = %q, want the new value", got)
	}
}
