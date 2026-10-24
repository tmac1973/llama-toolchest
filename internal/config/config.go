package config

import (
	"errors"
	"fmt"
	"log/slog"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"gopkg.in/yaml.v3"
)

type Config struct {
	ListenAddr  string `yaml:"listen_addr"`  // default ":3000"
	DataDir     string `yaml:"data_dir"`     // default "/data"
	ModelsDir   string `yaml:"models_dir"`   // optional override for the models dir; empty = <DataDir>/models
	LlamaPort   int    `yaml:"llama_port"`   // default 8080
	ExternalURL string `yaml:"external_url"` // e.g. "http://myserver:3000" for links
	HFToken     string `yaml:"hf_token"`     // optional HuggingFace token
	// DefaultModelSource preselects one of the model sources on the
	// browse page. It only sets the radio's initial position; the choice
	// is still per-search. Empty means HuggingFace.
	DefaultModelSource string `yaml:"default_model_source,omitempty"`

	// MSToken is the ModelScope access token. Only needed for gated or
	// private repositories; public ones, which is everything the browse
	// UI surfaces, download anonymously.
	MSToken     string `yaml:"ms_token,omitempty"`
	APIKey      string `yaml:"api_key"`      // optional API key for /v1/* proxy
	LogLevel    string `yaml:"log_level"`    // default "info"
	ActiveBuild string `yaml:"active_build"` // which llama.cpp build to use
	ModelsMax   int    `yaml:"models_max"`   // max loaded models, 0 = unlimited
	AutoStart   bool   `yaml:"auto_start"`   // start the llama-server on container startup

	// RuntimeEnv holds curated environment variables applied to the
	// llama-server router process. Router-wide by nature: every spawned
	// model instance inherits this environment identically, so per-model
	// env is not expressible. See RuntimeEnvOptions for the allowed set.
	RuntimeEnv map[string]string `yaml:"runtime_env,omitempty"`

	// RuntimeEnvExtra holds free-form KEY=VALUE lines applied alongside
	// RuntimeEnv (and overriding it when the same name is set in both).
	// Any variable is accepted; known-risky ones warn on save.
	RuntimeEnvExtra string `yaml:"runtime_env_extra,omitempty"`

	// PendingHelper is the "<repo>|<filename>" of a helper model being
	// downloaded, so the model is marked as the app's own when it
	// finishes even if the server restarted meanwhile. The helper model
	// is not a user choice; see api/helper_model.go.
	PendingHelper string `yaml:"pending_helper,omitempty"`

	// GenerationEnabled turns on the image and music generation backends
	// (stable-diffusion.cpp, acestep.cpp). Off by default; while off,
	// nothing about them is shown or run.
	GenerationEnabled bool `yaml:"generation_enabled,omitempty"`

	// GenPortStart and GenPortCount set the port range the generation
	// backends listen on, one port per backend instance. Zero means the
	// default (8081, 8 ports); use GenPorts for the effective values.
	GenPortStart int `yaml:"gen_port_start,omitempty"`
	GenPortCount int `yaml:"gen_port_count,omitempty"`

	// GenPublicPortStart is the first port of that range as seen from
	// outside, for the backend web UI links. Zero means the same as the
	// listening range; it differs when a container publishes the range on
	// other host ports.
	GenPublicPortStart int `yaml:"gen_public_port_start,omitempty"`
}

// Default generation backend port range. Container images always publish
// this range, so inside a container it cannot be changed.
const (
	DefaultGenPortStart = 8081
	DefaultGenPortCount = 8
)

// GenPorts returns the effective generation backend port range. In a
// container the range is fixed at the defaults, because that is the range
// the image publishes.
func (c *Config) GenPorts(inContainer bool) (start, count int) {
	start, count = c.GenPortStart, c.GenPortCount
	if inContainer || start == 0 {
		start = DefaultGenPortStart
	}
	if inContainer || count == 0 {
		count = DefaultGenPortCount
	}
	return start, count
}

// GenPublicStart returns the first port of the generation range as seen
// from outside: GenPublicPortStart when set, else the listening start.
func (c *Config) GenPublicStart(inContainer bool) int {
	if c.GenPublicPortStart != 0 {
		return c.GenPublicPortStart
	}
	start, _ := c.GenPorts(inContainer)
	return start
}

// ValidateGenPorts checks a generation port range against the ports
// toolchest already uses: the start must be 1024–65535, the count 1–32,
// the whole range must fit below 65536, and it must not include the
// llama-server port or the web UI's listen port.
func ValidateGenPorts(start, count, llamaPort int, listenAddr string) error {
	if start < 1024 || start > 65535 {
		return fmt.Errorf("port range start %d must be between 1024 and 65535", start)
	}
	if count < 1 || count > 32 {
		return fmt.Errorf("port count %d must be between 1 and 32", count)
	}
	end := start + count - 1
	if end > 65535 {
		return fmt.Errorf("port range %d–%d goes past 65535", start, end)
	}
	if llamaPort >= start && llamaPort <= end {
		return fmt.Errorf("port range %d–%d includes the llama-server port %d", start, end, llamaPort)
	}
	if p := listenPort(listenAddr); p >= start && p <= end {
		return fmt.Errorf("port range %d–%d includes the web UI port %d", start, end, p)
	}
	return nil
}

// listenPort returns the port of a listen address like ":3000" or
// "0.0.0.0:3000", or 0 when there is none.
func listenPort(addr string) int {
	i := strings.LastIndexByte(addr, ':')
	if i < 0 {
		return 0
	}
	n, err := strconv.Atoi(addr[i+1:])
	if err != nil {
		return 0
	}
	return n
}

// ModelsPath returns the directory where GGUF models live. ModelsDir wins
// when set; otherwise it's <DataDir>/models. Pulling it into a single
// helper means downstream code doesn't have to repeat the fallback logic.
func (c *Config) ModelsPath() string {
	if c.ModelsDir != "" {
		return c.ModelsDir
	}
	return filepath.Join(c.DataDir, "models")
}

func Load(path string) (*Config, error) {
	cfg := &Config{
		ListenAddr:  ":3000",
		DataDir:     DefaultDataDir(),
		LlamaPort:   8080,
		ExternalURL: "http://localhost:3000",
		LogLevel:    "info",
		ModelsMax:   1,
	}

	data, err := os.ReadFile(path)
	if err != nil {
		if !errors.Is(err, os.ErrNotExist) {
			return nil, err
		}
		// no file → use defaults, fall through to validation
	} else if err := yaml.Unmarshal(data, cfg); err != nil {
		return nil, err
	}

	applyEnvOverrides(cfg)
	salvageModelsDir(cfg)
	return cfg, nil
}

// applyEnvOverrides lets a few deployment-critical fields be set from the
// environment, overriding the YAML. This exists so a containerized deploy can
// inject values through compose `environment:` without editing the config file
// baked into the image's /data volume — notably ExternalURL, which a reverse
// proxy (see docs/secure.md) must set so the UI's chat/API links render with
// the externally reachable scheme and host. Empty/unset env vars are ignored,
// so they never clobber a YAML value.
func applyEnvOverrides(cfg *Config) {
	if v := os.Getenv("LLAMA_TOOLCHEST_EXTERNAL_URL"); v != "" {
		// The chat and API links are built from it; a value that is not
		// an http(s) URL with a host would render broken links, so it is
		// ignored with a warning, like a non-numeric port below.
		if u, err := url.Parse(v); err == nil && (u.Scheme == "http" || u.Scheme == "https") && u.Host != "" {
			cfg.ExternalURL = v
		} else {
			slog.Warn("ignoring LLAMA_TOOLCHEST_EXTERNAL_URL: not an http(s) URL with a host", "value", v)
		}
	}
	if v := os.Getenv("LLAMA_TOOLCHEST_LLAMA_PORT"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			cfg.LlamaPort = n
		} else {
			slog.Warn("ignoring non-numeric LLAMA_TOOLCHEST_LLAMA_PORT", "value", v)
		}
	}
	// Set by the container's compose file when the generation web UI
	// ports are published on different host ports, so the dashboard links
	// point at the host side.
	if v := os.Getenv("LLAMA_TOOLCHEST_GEN_PUBLIC_PORT_START"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 1 && n <= 65535 {
			cfg.GenPublicPortStart = n
		} else {
			slog.Warn("ignoring LLAMA_TOOLCHEST_GEN_PUBLIC_PORT_START: not a port number", "value", v)
		}
	}
}

// salvageModelsDir blanks ModelsDir if it points to a path that doesn't exist
// or isn't a directory. This catches a previously-shipped bug where the host
// bind-mount source LLAMA_TOOLCHEST_MODELS_DIR leaked from .env into the
// container's environment and was misinterpreted as the in-container path,
// causing downloads to land in the container's writable layer (lost on
// rebuild). With the env-override path removed, runtime models_dir comes
// only from YAML — but registries written under the broken behavior may
// still hold a host-shaped models_dir value, so we drop it here at load
// time and fall back to <DataDir>/models, which is the bind-mount target.
func salvageModelsDir(cfg *Config) {
	if cfg.ModelsDir == "" {
		return
	}
	info, err := os.Stat(cfg.ModelsDir)
	if err == nil && info.IsDir() {
		return
	}
	reason := "does not exist"
	if err == nil {
		reason = "is not a directory"
	} else if !errors.Is(err, os.ErrNotExist) {
		reason = fmt.Sprintf("stat failed: %v", err)
	}
	fallback := filepath.Join(cfg.DataDir, "models")
	slog.Warn("configured models_dir is unusable; falling back to default",
		"models_dir", cfg.ModelsDir, "reason", reason, "fallback", fallback,
		"hint", "save the Settings page to clear the bad value from the config file")
	cfg.ModelsDir = ""
}
