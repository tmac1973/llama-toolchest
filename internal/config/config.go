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

	// GenEvict lets LLMs and generation backends take a GPU from each
	// other when both do not fit: a chat request for an LLM that is not
	// loaded stops idle backends in its way, and a job for a stopped
	// backend unloads idle LLMs and starts it. Reservations then count
	// only while a backend runs. Off by default: fixed reservations, and
	// a load that does not fit is refused.
	GenEvict bool `yaml:"gen_evict,omitempty"`

	// GenKeepPromptCache saves an LLM's prompt cache to disk before an
	// eviction unloads it, and restores it once the model is loaded
	// again, so the next turn does not process the whole conversation
	// again. Only with GenEvict; the router needs a restart to pick it up.
	GenKeepPromptCache bool `yaml:"gen_keep_prompt_cache,omitempty"`

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

	// ChatPublicPort is the llama-server chat UI port as seen from
	// outside, for the "Open Chat UI" link. Zero means LlamaPort. A
	// container sets it from the environment, because the port it
	// publishes (or Caddy's chat port in secure mode) can differ from the
	// port llama-server listens on inside. Never written to the YAML: it
	// describes the deploy, not a setting.
	ChatPublicPort int `yaml:"-"`

	// GenPortsPublished is set in a container whose generation backend
	// ports are published (LLAMA_TOOLCHEST_GEN_PORTS=1, from
	// docker-compose.gen-ports.yml or the Quadlet unit). Never written to
	// the YAML.
	GenPortsPublished bool `yaml:"-"`

	// GenContainerPortCount is the number of generation backend ports in a
	// container, from LLAMA_TOOLCHEST_GEN_PORT_COUNT (written by setup.sh,
	// which publishes that many ports). Zero means the default. Never
	// written to the YAML: it describes the deploy, not a setting.
	GenContainerPortCount int `yaml:"-"`

	// ROCmBaseImage is the ROCm image a rocm-next container was built
	// from (LLAMA_TOOLCHEST_ROCM_BASE_IMAGE, set by Dockerfile.rocm-next).
	// Never written to the YAML.
	ROCmBaseImage string `yaml:"-"`

	// envVars maps the YAML name of each field applyEnvOverrides set to
	// the environment variable that set it, and file holds the values
	// those fields had before the override. Saving writes file's values
	// for those fields (see FileValues), so a value that only came from
	// the environment never ends up in the YAML and outlives it.
	envVars map[string]string
	file    *Config
}

// Clone returns a copy of c that a later Set call on c does not change:
// the saved file values are copied too. Maps and slices are shared, since
// their writers replace them rather than editing them in place.
func (c *Config) Clone() Config {
	out := *c
	if c.file != nil {
		f := *c.file
		out.file = &f
	}
	return out
}

// YAML names of the fields an environment variable can override.
const (
	FieldExternalURL        = "external_url"
	FieldLlamaPort          = "llama_port"
	FieldGenPublicPortStart = "gen_public_port_start"
)

// envFieldCopy copies one overridable field from src to dst, keyed by
// the field's YAML name.
var envFieldCopy = map[string]func(dst, src *Config){
	FieldExternalURL:        func(dst, src *Config) { dst.ExternalURL = src.ExternalURL },
	FieldLlamaPort:          func(dst, src *Config) { dst.LlamaPort = src.LlamaPort },
	FieldGenPublicPortStart: func(dst, src *Config) { dst.GenPublicPortStart = src.GenPublicPortStart },
}

// EnvVar returns the environment variable that overrides the field with
// this YAML name, or "" when the field's value came from the file.
func (c *Config) EnvVar(field string) string {
	return c.envVars[field]
}

// FileValues returns a copy of the config to write to the YAML file:
// every field an environment variable overrides has the file's own value
// (or the value saved through SetSaved since), not the overridden one.
func (c *Config) FileValues() *Config {
	out := *c
	for field := range c.envVars {
		envFieldCopy[field](&out, c.file)
	}
	out.envVars, out.file = nil, nil
	return &out
}

// SavedExternalURL returns the external URL the config file holds, which
// differs from ExternalURL while LLAMA_TOOLCHEST_EXTERNAL_URL overrides it.
func (c *Config) SavedExternalURL() string {
	if c.envVars[FieldExternalURL] != "" {
		return c.file.ExternalURL
	}
	return c.ExternalURL
}

// SetExternalURL changes the external URL. While an environment variable
// overrides it, the variable keeps winning: only the value saved to the
// file changes, and it takes effect once the variable is removed.
func (c *Config) SetExternalURL(v string) {
	c.setSaved(FieldExternalURL, func(cfg *Config) { cfg.ExternalURL = v })
}

// SetLlamaPort changes the llama-server port, with the same rule as
// SetExternalURL while LLAMA_TOOLCHEST_LLAMA_PORT overrides it.
func (c *Config) SetLlamaPort(n int) {
	c.setSaved(FieldLlamaPort, func(cfg *Config) { cfg.LlamaPort = n })
}

// setSaved applies set to the file's values when an environment variable
// overrides field, and to the config in use otherwise.
func (c *Config) setSaved(field string, set func(*Config)) {
	if c.envVars[field] != "" {
		set(c.file)
		return
	}
	set(c)
}

// ChatPort returns the chat UI port as seen from outside: ChatPublicPort
// when set, else LlamaPort.
func (c *Config) ChatPort() int {
	if c.ChatPublicPort != 0 {
		return c.ChatPublicPort
	}
	return c.LlamaPort
}

// Default generation backend port range, and the most ports it can hold.
// Inside a container the range always starts at DefaultGenPortStart.
const (
	DefaultGenPortStart = 8081
	DefaultGenPortCount = 8
	MaxGenPortCount     = 64
)

// GenPorts returns the effective generation backend port range. In a
// container the range starts at the default, and its size is the one
// setup.sh chose (GenContainerPortCount), because that is what the
// container publishes; Settings cannot change either.
func (c *Config) GenPorts(inContainer bool) (start, count int) {
	start, count = c.GenPortStart, c.GenPortCount
	if inContainer {
		start, count = DefaultGenPortStart, c.GenContainerPortCount
	}
	if start == 0 {
		start = DefaultGenPortStart
	}
	if count == 0 {
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
// toolchest already uses: the start must be 1024–65535, the count 1–64,
// the whole range must fit below 65536, and it must not include the
// llama-server port or the web UI's listen port.
func ValidateGenPorts(start, count, llamaPort int, listenAddr string) error {
	if start < 1024 || start > 65535 {
		return fmt.Errorf("port range start %d must be between 1024 and 65535", start)
	}
	if count < 1 || count > MaxGenPortCount {
		return fmt.Errorf("port count %d must be between 1 and %d", count, MaxGenPortCount)
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
//
// Each field set here is recorded with the value the file had, so saving
// the config writes that value back instead of the overridden one.
func applyEnvOverrides(cfg *Config) {
	file := *cfg
	set := func(field, envVar string) {
		if cfg.envVars == nil {
			cfg.envVars = map[string]string{}
			cfg.file = &file
		}
		cfg.envVars[field] = envVar
	}
	if v := os.Getenv("LLAMA_TOOLCHEST_EXTERNAL_URL"); v != "" {
		// The chat and API links are built from it; a value that is not
		// an http(s) URL with a host would render broken links, so it is
		// ignored with a warning, like a non-numeric port below.
		if u, err := url.Parse(v); err == nil && (u.Scheme == "http" || u.Scheme == "https") && u.Host != "" {
			cfg.ExternalURL = v
			set(FieldExternalURL, "LLAMA_TOOLCHEST_EXTERNAL_URL")
		} else {
			slog.Warn("ignoring LLAMA_TOOLCHEST_EXTERNAL_URL: not an http(s) URL with a host", "value", v)
		}
	}
	if n, ok := envPort(envLlamaPort); ok {
		cfg.LlamaPort = n
		set(FieldLlamaPort, envLlamaPort)
	}
	// Set by the container's compose file when the generation web UI
	// ports are published on different host ports, so the dashboard links
	// point at the host side.
	if n, ok := envPort(envGenPublicPortStart); ok {
		cfg.GenPublicPortStart = n
		set(FieldGenPublicPortStart, envGenPublicPortStart)
	}
	// Set by setup.sh for a container: the host port llama-server's 8080
	// is published on, or Caddy's chat port in secure mode.
	if n, ok := envPort("LLAMA_TOOLCHEST_CHAT_PUBLIC_PORT"); ok {
		cfg.ChatPublicPort = n
	}
	cfg.GenPortsPublished = os.Getenv("LLAMA_TOOLCHEST_GEN_PORTS") == "1"
	// Set by setup.sh: how many generation backend ports the container has
	// (and publishes, when they are published).
	if v := os.Getenv(envGenPortCount); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 1 && n <= MaxGenPortCount {
			cfg.GenContainerPortCount = n
		} else {
			slog.Warn("ignoring environment variable: not a port count from 1 to 64", "name", envGenPortCount, "value", v)
		}
	}
	cfg.ROCmBaseImage = os.Getenv("LLAMA_TOOLCHEST_ROCM_BASE_IMAGE")
}

const (
	envLlamaPort          = "LLAMA_TOOLCHEST_LLAMA_PORT"
	envGenPublicPortStart = "LLAMA_TOOLCHEST_GEN_PUBLIC_PORT_START"
	envGenPortCount       = "LLAMA_TOOLCHEST_GEN_PORT_COUNT"
)

// envPort reads a port number from the environment variable name. An
// unset variable is not ok; a value that is not a port from 1 to 65535 is
// not ok either, and is logged as ignored.
func envPort(name string) (int, bool) {
	v := os.Getenv(name)
	if v == "" {
		return 0, false
	}
	n, err := strconv.Atoi(v)
	if err != nil || n < 1 || n > 65535 {
		slog.Warn("ignoring environment variable: not a port number", "name", name, "value", v)
		return 0, false
	}
	return n, true
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
