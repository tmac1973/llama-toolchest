// Package backup defines the versioned configuration backup file format
// and the engines that produce (Assemble) and consume (Parse/Apply) it.
//
// The backup carries intent, not artifacts: server preference settings,
// the runtime environment, saved build flag sets, and per-model launch
// configs keyed by stable HF identity (ModelID, Quant). It deliberately
// excludes anything machine-specific or rebuildable — build binaries,
// GGUF files, registry metadata, deployment-identity settings (exported
// for reference only, never applied).
package backup

import (
	"encoding/json"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/builder"
	"github.com/tmac1973/llama-toolchest/internal/config"
	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/monitor"
)

// Version is the current backup schema version. Parse rejects files
// carrying any other value.
const Version = 1

// File is the top-level backup document. Every field carries an explicit
// snake_case JSON tag: this is a versioned wire format, and the Settings
// page's client-side preview reads these exact keys.
type File struct {
	Version      int                  `json:"version"`
	ExportedAt   time.Time            `json:"exported_at"`
	Source       SourceInfo           `json:"source"` // reference only, never applied
	Settings     *Settings            `json:"settings,omitempty"`
	RuntimeEnv   *RuntimeEnv          `json:"runtime_env,omitempty"`
	FlagPresets  []builder.FlagPreset `json:"flag_presets,omitempty"`
	ModelConfigs []ModelConfigExport  `json:"model_configs,omitempty"`
}

// SourceInfo documents the origin server. Restore ignores it entirely —
// deployment-identity fields are exported so the file describes its
// source, never so they can be applied to a target.
type SourceInfo struct {
	ListenAddr  string   `json:"listen_addr,omitempty"`
	LlamaPort   int      `json:"llama_port,omitempty"`
	ExternalURL string   `json:"external_url,omitempty"`
	DataDir     string   `json:"data_dir,omitempty"`
	ModelsDir   string   `json:"models_dir,omitempty"`
	ActiveBuild string   `json:"active_build,omitempty"`
	NumGPUs     int      `json:"num_gpus"`
	GPUs        []string `json:"gpus,omitempty"` // marketing names, for topology warnings
}

// Settings holds the preference fields restore may apply. All pointers:
// an absent field means "leave the target untouched", a present zero
// (auto_start: false) is a real value — the merge-never-deletes
// guarantee depends on the distinction.
type Settings struct {
	ModelsMax *int    `json:"models_max,omitempty"`
	AutoStart *bool   `json:"auto_start,omitempty"`
	LogLevel  *string `json:"log_level,omitempty"`
	HFToken   *string `json:"hf_token,omitempty"` // only with includeSecrets, and only when non-empty
	MSToken   *string `json:"ms_token,omitempty"` // ModelScope; same rules as HFToken
	APIKey    *string `json:"api_key,omitempty"`  // only with includeSecrets, and only when non-empty
}

// RuntimeEnv mirrors the global EnvSet: curated variable values plus the
// free-form extra block.
type RuntimeEnv struct {
	Curated map[string]string `json:"curated,omitempty"`
	Extra   string            `json:"extra,omitempty"`
}

// ModelConfigExport is one model's launch config keyed by stable HF
// identity — never a registry ID or an absolute path. Filename is
// required: it makes the missing-model download offer precise (the
// registry derives Quant from the filename on arrival, so downloading
// exactly this file is what lets a pending config claim).
type ModelConfigExport struct {
	ModelID  string             `json:"model_id"` // HF org/repo
	Quant    string             `json:"quant"`
	Filename string             `json:"filename"`
	Config   models.ModelConfig `json:"config"` // path fields relativized to the models dir
}

// Assemble builds a backup of the current configuration state. The
// output is deterministic: identical state produces byte-identical JSON
// except exported_at.
func Assemble(cfg *config.Config, b *builder.Builder, reg *models.Registry, gpus []monitor.GPUInfo, includeSecrets bool) File {
	f := File{
		Version:    Version,
		ExportedAt: time.Now().UTC(),
		Source: SourceInfo{
			ListenAddr:  cfg.ListenAddr,
			LlamaPort:   cfg.LlamaPort,
			ExternalURL: cfg.ExternalURL,
			DataDir:     cfg.DataDir,
			ModelsDir:   cfg.ModelsDir,
			ActiveBuild: cfg.ActiveBuild,
			NumGPUs:     len(gpus),
		},
	}
	for _, g := range gpus {
		f.Source.GPUs = append(f.Source.GPUs, g.Name)
	}

	s := &Settings{
		ModelsMax: ptr(cfg.ModelsMax),
		AutoStart: ptr(cfg.AutoStart),
		LogLevel:  ptr(cfg.LogLevel),
	}
	// Secrets: only with the explicit flag, and never as empty strings —
	// an empty emitted secret could blank a target's credential on
	// restore, so absence of the key is the only representation of
	// "no secret".
	if includeSecrets {
		if cfg.HFToken != "" {
			s.HFToken = ptr(cfg.HFToken)
		}
		if cfg.MSToken != "" {
			s.MSToken = ptr(cfg.MSToken)
		}
		if cfg.APIKey != "" {
			s.APIKey = ptr(cfg.APIKey)
		}
	}
	f.Settings = s

	f.RuntimeEnv = &RuntimeEnv{Curated: cfg.RuntimeEnv, Extra: cfg.RuntimeEnvExtra}

	f.FlagPresets = b.FlagPresets("") // already name-sorted

	f.ModelConfigs = assembleModelConfigs(reg, cfg.ModelsPath())

	return f
}

// assembleModelConfigs emits one entry per (ModelID, Quant) identity.
// Duplicate registrations of the same identity exist in practice
// (DeduplicateModels exists because of them) and Registry.List order is
// not stable among them, so the duplicate with the lexicographically
// smallest registry ID wins deterministically.
func assembleModelConfigs(reg *models.Registry, modelsDir string) []ModelConfigExport {
	type candidate struct {
		regID string
		entry ModelConfigExport
	}
	byIdentity := map[string]candidate{}
	for _, m := range reg.List() {
		cfg, err := reg.GetConfig(m.ID)
		if err != nil {
			continue
		}
		key := m.ModelID + "\x00" + m.Quant
		if prev, ok := byIdentity[key]; ok && prev.regID <= m.ID {
			continue
		}
		exported := *cfg
		relativizePaths(&exported, modelsDir)
		byIdentity[key] = candidate{
			regID: m.ID,
			entry: ModelConfigExport{
				ModelID:  m.ModelID,
				Quant:    m.Quant,
				Filename: m.Filename,
				Config:   exported,
			},
		}
	}
	out := make([]ModelConfigExport, 0, len(byIdentity))
	for _, c := range byIdentity {
		out = append(out, c.entry)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].ModelID != out[j].ModelID {
			return out[i].ModelID < out[j].ModelID
		}
		return out[i].Quant < out[j].Quant
	})
	return out
}

// relativizePaths rewrites the machine-local GGUF path fields to be
// relative to the models dir. A path counts as inside only when Rel
// succeeds and doesn't escape upward; anything else exports the original
// absolute path verbatim (import keeps it only if it exists on the
// target). The export therefore emits exactly two shapes — clean
// relatives and absolutes — which is the dispatch ResolveConfigPaths
// switches on at import/claim time.
func relativizePaths(cfg *models.ModelConfig, modelsDir string) {
	rel := func(p string) string {
		if p == "" || modelsDir == "" {
			return p
		}
		r, err := filepath.Rel(modelsDir, p)
		if err != nil || r == ".." || strings.HasPrefix(r, ".."+string(filepath.Separator)) {
			return p
		}
		return r
	}
	cfg.MmprojPath = rel(cfg.MmprojPath)
	cfg.MtpPath = rel(cfg.MtpPath)
	cfg.DraftModelPath = rel(cfg.DraftModelPath)
}

// Marshal renders the file as indented JSON.
func (f File) Marshal() ([]byte, error) {
	return json.MarshalIndent(f, "", "  ")
}

func ptr[T any](v T) *T { return &v }
