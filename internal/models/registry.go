package models

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

// OrgAndBase returns the HuggingFace organization and base model name
// (the repo name with any "-GGUF" suffix stripped).
func (m *Model) OrgAndBase() (org, base string) {
	base = m.ModelID
	if i := strings.Index(m.ModelID, "/"); i >= 0 {
		org = m.ModelID[:i]
		base = m.ModelID[i+1:]
	}
	base = strings.TrimSuffix(base, "-GGUF")
	base = strings.TrimSuffix(base, "-gguf")
	return org, base
}

// PublicName returns a short, human-friendly model identifier for the
// /v1/models API and preset aliases. It strips the redundant "-GGUF" suffix,
// collapses any dash-segmented prefix shared between the HuggingFace org
// and repo name (e.g. "nomic-ai" + "nomic-embed-text-v1.5" → "nomic-ai-embed-text-v1.5"),
// and appends the quant. Multi-file shard suffixes are dropped because the
// name is derived from ModelID, not the on-disk filename.
func (m *Model) PublicName() string {
	org, base := m.OrgAndBase()

	var name string
	if org == "" {
		name = base
	} else {
		orgParts := strings.Split(org, "-")
		baseParts := strings.Split(base, "-")
		k := 0
		for k < len(orgParts) && k < len(baseParts) && strings.EqualFold(orgParts[k], baseParts[k]) {
			k++
		}
		combined := append([]string{}, orgParts...)
		combined = append(combined, baseParts[k:]...)
		name = strings.Join(combined, "-")
	}

	if m.Quant != "" {
		name += "." + m.Quant
	}
	return name
}

// Model represents a locally downloaded GGUF model.
type Model struct {
	ID           string    `json:"id"`
	ModelID      string    `json:"model_id"`
	Filename     string    `json:"filename"`
	Quant        string    `json:"quant"`
	SizeBytes    int64     `json:"size_bytes"`
	FilePath     string    `json:"file_path"`
	VRAMEstGB    float64   `json:"vram_est_gb"`
	DownloadedAt time.Time `json:"downloaded_at"`

	// Architecture parameters parsed from GGUF header.
	Arch             string `json:"arch,omitempty"`
	NLayers          int    `json:"n_layers,omitempty"`
	NEmbd            int    `json:"n_embd,omitempty"`
	NHead            int    `json:"n_head,omitempty"`
	NKVHead          int    `json:"n_kv_head,omitempty"`
	ContextLength    int    `json:"context_length,omitempty"`     // max trained context
	SupportsTools    bool   `json:"supports_tools,omitempty"`     // chat template handles tools
	HasBuiltinVision bool   `json:"has_builtin_vision,omitempty"` // vision encoder baked into model

	// Reasoning / thinking mode detected from the chat template at parse time.
	// ReasoningChecked distinguishes a genuine "no reasoning" from a record
	// parsed before detection existed (BackfillGGUFMeta re-parses the latter).
	Reasoning        ReasoningCapability `json:"reasoning,omitempty"`
	ReasoningChecked bool                `json:"reasoning_checked,omitempty"`

	// KV-cache scaling factors (see GGUFMeta). Persisted so VRAM estimates
	// don't re-parse the GGUF on every render. Zero on records parsed before
	// these existed — BackfillGGUFMeta repopulates them, and KVCacheGB falls
	// back to the uniform estimate until then.
	KVFullPerTok  int `json:"kv_full_per_tok,omitempty"`
	KVSWAPerTok   int `json:"kv_swa_per_tok,omitempty"`
	SlidingWindow int `json:"sliding_window,omitempty"`

	// Sampling presets discovered at download/scan time: GGUF-embedded
	// defaults now, network sources (publisher docs, generation_config.json)
	// as they run. SamplingChecked records that the GGUF header was inspected
	// for general.sampling.* keys (distinguishes "file has none" from "parsed
	// before these existed" during backfill). PresetsCheckedAt records the
	// last network fetch attempt; zero = never attempted.
	SamplingPresets  []SamplingPreset `json:"sampling_presets,omitempty"`
	SamplingChecked  bool             `json:"sampling_checked,omitempty"`
	BaseModelRepo    string           `json:"base_model_repo,omitempty"`
	PresetsCheckedAt time.Time        `json:"presets_checked_at,omitzero"`
}

// ModelConfig holds per-model launch configuration for llama-server.
type ModelConfig struct {
	Enabled     bool   `json:"enabled"`
	GPULayers   int    `json:"gpu_layers"`
	TensorSplit string `json:"tensor_split"`
	SplitMode   string `json:"split_mode,omitempty"` // "layer", "tensor", or ""
	MainGPU     int    `json:"main_gpu,omitempty"`
	GPUAssign   string `json:"gpu_assign,omitempty"` // "all", "0", "0-1", "custom", etc.
	ContextSize int    `json:"context_size"`
	Parallel    int    `json:"parallel,omitempty"` // n parallel sequence slots; 0/1 = no extra slots, >1 divides ctx_size across slots
	// BatchSize/UBatchSize map to --batch-size / --ubatch-size. Zero means
	// "don't emit", leaving llama.cpp on its own defaults (2048 / 512), so
	// existing models are unaffected. UBatchSize is the physical compute
	// batch and is the main prompt-processing tuning knob; BatchSize is
	// the logical batch and must be >= UBatchSize.
	BatchSize      int    `json:"batch_size,omitempty"`
	UBatchSize     int    `json:"ubatch_size,omitempty"`
	Threads        int    `json:"threads"`
	FlashAttention bool   `json:"flash_attention"`
	Jinja          bool   `json:"jinja"`
	KVCacheQuant   string `json:"kv_cache_quant"`            // "", "q8_0", "q4_0"
	DirectIO       bool   `json:"direct_io"`                 // bypass page cache, load straight to VRAM
	MmprojPath     string `json:"mmproj_path,omitempty"`     // path to mmproj GGUF for vision models
	MmprojDisabled bool   `json:"mmproj_disabled,omitempty"` // skip --mmproj at launch even when MmprojPath is set; preserves the path so it can be re-enabled without retyping
	MtpPath        string `json:"mtp_path,omitempty"`        // path to a separate MTP drafter-head GGUF (gemma-4 style); loaded via --model-draft under spec_type=draft-mtp. Empty for self-speculation MTP (Qwen3.6/DeepSeek-V3) where the head is baked into the main GGUF.
	MtpDisabled    bool   `json:"mtp_disabled,omitempty"`    // skip the separate --model-draft MTP head at launch even when MtpPath is set; preserves the path so it can be re-enabled

	// Speculative decoding
	SpecType       string `json:"spec_type,omitempty"`        // "", "draft", "draft-mtp", "ngram-simple", "ngram-cache", etc.
	DraftModelPath string `json:"draft_model_path,omitempty"` // path to draft model (when spec_type="draft")
	DraftMax       int    `json:"draft_max,omitempty"`        // max draft tokens per step
	DraftMin       int    `json:"draft_min,omitempty"`        // min draft tokens per step
	DraftPMin      string `json:"draft_p_min,omitempty"`      // min probability threshold (string to allow empty=default)
	NgramSizeN     int    `json:"ngram_size_n,omitempty"`     // n-gram lookup length
	NgramSizeM     int    `json:"ngram_size_m,omitempty"`     // n-gram draft length

	// Draft model resource overrides. Apply to spec_type="draft" and to
	// gemma-4-style draft-mtp (separate head loaded via --model-draft). Not
	// used by self-speculation MTP, where the draft *is* the main model.
	DraftCtxSize      int    `json:"draft_ctx_size,omitempty"`       // --ctx-size-draft
	DraftGPULayers    int    `json:"draft_gpu_layers,omitempty"`     // --gpu-layers-draft
	DraftDevice       string `json:"draft_device,omitempty"`         // --device-draft
	DraftCPUMoE       int    `json:"draft_cpu_moe,omitempty"`        // --n-cpu-moe-draft
	DraftKVCacheQuant string `json:"draft_kv_cache_quant,omitempty"` // --cache-type-{k,v}-draft

	Aliases    []string `json:"aliases,omitempty"` // user-defined friendly names
	ExtraFlags string   `json:"extra_flags"`

	// ReasoningOverride lets a user correct or supply reasoning capability when
	// chat-template auto-detection is wrong or absent. nil = use detection.
	ReasoningOverride *ReasoningCapability `json:"reasoning,omitempty"`

	// Sampling parameters — nil means use llama.cpp server default.
	Temperature     *float64 `json:"temperature,omitempty"`
	TopP            *float64 `json:"top_p,omitempty"`
	TopK            *int     `json:"top_k,omitempty"`
	MinP            *float64 `json:"min_p,omitempty"`
	PresencePenalty *float64 `json:"presence_penalty,omitempty"`
	RepeatPenalty   *float64 `json:"repeat_penalty,omitempty"`
}

// DefaultBatchSize and DefaultUBatchSize mirror llama.cpp's own defaults,
// used when the field is left blank so validation can reason about the
// value that will actually be in effect.
const (
	DefaultBatchSize  = 2048
	DefaultUBatchSize = 512
)

// EffectiveBatchSize returns the batch size that will be in effect.
func (c *ModelConfig) EffectiveBatchSize() int {
	if c.BatchSize > 0 {
		return c.BatchSize
	}
	return DefaultBatchSize
}

// EffectiveUBatchSize returns the micro-batch size that will be in effect.
func (c *ModelConfig) EffectiveUBatchSize() int {
	if c.UBatchSize > 0 {
		return c.UBatchSize
	}
	return DefaultUBatchSize
}

// ValidateBatchSizes rejects a micro-batch larger than the batch it has
// to fit inside. Checked against the effective values, so raising -ub
// past 2048 without also raising -b is caught rather than silently
// clamped by llama.cpp at load time.
func (c *ModelConfig) ValidateBatchSizes() error {
	if c.BatchSize < 0 || c.UBatchSize < 0 {
		return fmt.Errorf("batch sizes cannot be negative")
	}
	b, ub := c.EffectiveBatchSize(), c.EffectiveUBatchSize()
	if ub > b {
		if c.BatchSize == 0 {
			return fmt.Errorf("micro-batch %d exceeds the default batch size of %d — raise Batch (-b) to at least %d", ub, DefaultBatchSize, ub)
		}
		return fmt.Errorf("micro-batch %d exceeds batch size %d — micro-batch must be less than or equal to batch", ub, b)
	}
	return nil
}

// SamplingOverrides returns a map of non-nil sampling parameters suitable
// for merging into an OpenAI-compatible request body.
func (c *ModelConfig) SamplingOverrides() map[string]any {
	m := make(map[string]any)
	if c.Temperature != nil {
		m["temperature"] = *c.Temperature
	}
	if c.TopP != nil {
		m["top_p"] = *c.TopP
	}
	if c.TopK != nil {
		m["top_k"] = *c.TopK
	}
	if c.MinP != nil {
		m["min_p"] = *c.MinP
	}
	if c.PresencePenalty != nil {
		m["presence_penalty"] = *c.PresencePenalty
	}
	if c.RepeatPenalty != nil {
		m["repeat_penalty"] = *c.RepeatPenalty
	}
	return m
}

// EffectiveFlags returns the full set of llama-server flags (excluding
// binary, model path, host, and port) that will be used at launch.
// EffectiveFlagsFor returns the flags that will be used at launch, filtering
// out chat-specific flags for embedding models.
func (c *ModelConfig) EffectiveFlagsFor(isEmbedding bool) string {
	var parts []string
	parts = append(parts, "--n-gpu-layers", strconv.Itoa(c.GPULayers))
	if isEmbedding {
		parts = append(parts, "--embeddings")
	}
	if !isEmbedding && c.ContextSize > 0 {
		parts = append(parts, "--ctx-size", strconv.Itoa(c.ContextSize))
	}
	parts = append(parts, "--threads", strconv.Itoa(c.Threads))
	if c.BatchSize > 0 {
		parts = append(parts, "--batch-size", strconv.Itoa(c.BatchSize))
	}
	if c.UBatchSize > 0 {
		parts = append(parts, "--ubatch-size", strconv.Itoa(c.UBatchSize))
	}
	if c.Parallel > 1 {
		parts = append(parts, "--parallel", strconv.Itoa(c.Parallel))
	}
	if c.TensorSplit != "" {
		parts = append(parts, "--tensor-split", c.TensorSplit)
	}
	if c.SplitMode != "" {
		parts = append(parts, "--split-mode", c.SplitMode)
		// Upstream auto-fit (common_fit_params) is not implemented for
		// SPLIT_MODE_TENSOR and aborts model load with "llama_params_fit
		// is not implemented for SPLIT_MODE_TENSOR". Disable it so the
		// user's explicit --n-gpu-layers / --tensor-split values are
		// honored verbatim. Drop this when llama.cpp adds the fitter for
		// tensor mode.
		if c.SplitMode == "tensor" {
			parts = append(parts, "--fit", "off")
		}
	}
	if c.MainGPU > 0 {
		parts = append(parts, "--main-gpu", strconv.Itoa(c.MainGPU))
	}
	if !isEmbedding {
		if c.FlashAttention {
			parts = append(parts, "--flash-attn", "on")
		}
		if c.Jinja {
			parts = append(parts, "--jinja")
		}
		if c.KVCacheQuant != "" {
			parts = append(parts, "--cache-type-k", c.KVCacheQuant, "--cache-type-v", c.KVCacheQuant)
		}
		if c.DirectIO {
			parts = append(parts, "--direct-io")
		}
		if c.MmprojPath != "" && !c.MmprojDisabled {
			parts = append(parts, "--mmproj", c.MmprojPath)
		}
		// Speculative decoding (see specDecodingParams for the SpecType rules).
		for _, p := range specDecodingParams(c) {
			parts = append(parts, "--"+p.Name, p.Value)
		}
	}
	if c.ExtraFlags != "" {
		parts = append(parts, strings.Fields(c.ExtraFlags)...)
	}
	return strings.Join(parts, " ")
}

// EffectiveFlags returns the flags for a chat model (backward compat).
func (c *ModelConfig) EffectiveFlags() string {
	return c.EffectiveFlagsFor(false)
}

type registryData struct {
	Models  map[string]*Model       `json:"models"`
	Configs map[string]*ModelConfig `json:"configs"`
}

// Registry manages local model storage and metadata.
type Registry struct {
	mu        sync.RWMutex
	dataDir   string
	modelsDir string
	data      registryData
}

// NewRegistry creates a registry and loads persisted state. modelsDir is
// where GGUF files live; this can be inside dataDir (default) or somewhere
// else entirely (config.ModelsDir override).
func NewRegistry(dataDir, modelsDir string) *Registry {
	r := &Registry{
		dataDir:   dataDir,
		modelsDir: modelsDir,
		data: registryData{
			Models:  make(map[string]*Model),
			Configs: make(map[string]*ModelConfig),
		},
	}
	r.load()
	return r
}

// Add registers a new model.
func (r *Registry) Add(m *Model) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.data.Models[m.ID] = m
	// Set default config
	if _, exists := r.data.Configs[m.ID]; !exists {
		r.data.Configs[m.ID] = &ModelConfig{
			Enabled:        true,
			GPULayers:      999,
			TensorSplit:    "",
			SplitMode:      "",
			ContextSize:    8192,
			Threads:        8,
			FlashAttention: true,
			Jinja:          true,
		}
	}
	r.save()
	return nil
}

// List returns all models, sorted alphabetically by ModelID.
func (r *Registry) List() []*Model {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]*Model, 0, len(r.data.Models))
	for _, m := range r.data.Models {
		out = append(out, m)
	}
	sort.Slice(out, func(i, j int) bool {
		return out[i].ModelID < out[j].ModelID
	})
	return out
}

// HasFile reports whether the registry already contains a model matching the
// given HuggingFace repo + filename. Used to mark "already downloaded" in the
// HF browse UI so we don't show a Download button for files we already have.
func (r *Registry) HasFile(modelID, filename string) (*Model, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	for _, m := range r.data.Models {
		if m.ModelID == modelID && m.Filename == filename {
			return m, true
		}
	}
	return nil, false
}

// Get returns a model by ID.
func (r *Registry) Get(id string) (*Model, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	m, ok := r.data.Models[id]
	if !ok {
		return nil, fmt.Errorf("model not found: %s", id)
	}
	return m, nil
}

// FindByAny resolves any name a client might use — registry ID, OpenAI
// public name, or user alias — to the underlying model and its config.
// Returns nil, nil when nothing matches. The OpenAI v1 endpoints, the
// chat-completion auto-loader, and the /api/models/{id}/* handlers all
// route through here so a single canonical lookup applies everywhere.
func (r *Registry) FindByAny(name string) (*Model, *ModelConfig) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	if m, ok := r.data.Models[name]; ok {
		return m, r.data.Configs[m.ID]
	}
	for _, m := range r.data.Models {
		if m.PublicName() == name {
			return m, r.data.Configs[m.ID]
		}
		if cfg := r.data.Configs[m.ID]; cfg != nil {
			for _, alias := range cfg.Aliases {
				if alias == name {
					return m, cfg
				}
			}
		}
	}
	return nil, nil
}

// ResolveID returns the canonical registry ID for a name supplied as a
// registry ID, public name, or alias. Returns the input unchanged when no
// model matches, so callers can still pass it on to error paths.
func (r *Registry) ResolveID(name string) string {
	if m, _ := r.FindByAny(name); m != nil {
		return m.ID
	}
	return name
}

// Remove removes a model entry from the registry without deleting files.
func (r *Registry) Remove(id string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if _, ok := r.data.Models[id]; !ok {
		return fmt.Errorf("model not found: %s", id)
	}

	delete(r.data.Models, id)
	delete(r.data.Configs, id)
	r.save()
	return nil
}

// Delete removes a model entry and deletes its files from disk.
func (r *Registry) Delete(id string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	m, ok := r.data.Models[id]
	if !ok {
		return fmt.Errorf("model not found: %s", id)
	}

	// Delete the GGUF file(s) — for sharded models, delete all parts
	shards := findShards(filepath.Dir(m.FilePath), filepath.Base(m.FilePath))
	for _, shard := range shards {
		os.Remove(shard)
		os.Remove(shard + ".part") // clean up any partial downloads too
	}

	// Remove empty directories left behind
	dir := filepath.Dir(m.FilePath)
	removeEmptyDirs(dir)

	delete(r.data.Models, id)
	delete(r.data.Configs, id)
	r.save()
	return nil
}

// removeEmptyDirs removes dir and its parent if they're empty, stopping at the models dir.
func removeEmptyDirs(dir string) {
	for {
		entries, err := os.ReadDir(dir)
		if err != nil || len(entries) > 0 {
			break
		}
		parent := filepath.Dir(dir)
		os.Remove(dir) // only succeeds if empty
		if parent == dir {
			break
		}
		dir = parent
	}
}

// GetConfig returns the launch config for a model.
func (r *Registry) GetConfig(id string) (*ModelConfig, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	cfg, ok := r.data.Configs[id]
	if !ok {
		return nil, fmt.Errorf("config not found: %s", id)
	}
	return cfg, nil
}

// SetConfig updates the launch config for a model.
func (r *Registry) SetConfig(id string, cfg *ModelConfig) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.data.Models[id]; !ok {
		return fmt.Errorf("model not found: %s", id)
	}
	r.data.Configs[id] = cfg
	r.save()
	return nil
}

// SetSamplingPresets replaces the sampling presets on a model record and
// stamps the network-fetch attempt time. Called from the async download-time
// enrichment goroutine, so all mutation happens under the registry lock.
func (r *Registry) SetSamplingPresets(id string, presets []SamplingPreset, checkedAt time.Time) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	m, ok := r.data.Models[id]
	if !ok {
		return fmt.Errorf("model not found: %s", id)
	}
	m.SamplingPresets = presets
	m.PresetsCheckedAt = checkedAt
	r.save()
	return nil
}

// ListNeedingPresetFetch returns IDs of models that have never had a network
// preset-fetch attempt (zero PresetsCheckedAt). Embedding models are skipped
// by the enrichment itself, not here, so the marker still gets stamped.
func (r *Registry) ListNeedingPresetFetch() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	var ids []string
	for id, m := range r.data.Models {
		if m.PresetsCheckedAt.IsZero() {
			ids = append(ids, id)
		}
	}
	sort.Strings(ids)
	return ids
}

// BackfillGGUFMeta parses GGUF metadata for any models missing architecture
// info. Called at startup to handle models downloaded before GGUF parsing existed.
func (r *Registry) BackfillGGUFMeta() {
	r.mu.Lock()
	defer r.mu.Unlock()

	changed := false
	for _, m := range r.data.Models {
		needsFull := m.NLayers == 0
		needsVision := m.NLayers > 0 && !m.HasBuiltinVision // re-check for vision field
		// Records parsed before KV scaling factors existed have layers but no
		// per-token factors — re-parse once to repopulate them.
		needsKV := m.NLayers > 0 && m.KVFullPerTok == 0 && m.KVSWAPerTok == 0
		// Records parsed before reasoning detection existed have no reasoning
		// verdict — re-parse once to inspect the chat template.
		needsReasoning := m.NLayers > 0 && !m.ReasoningChecked
		// Records parsed before general.sampling.* keys were read — re-parse
		// once to pick up embedded sampling defaults and the base-model repo.
		needsSampling := m.NLayers > 0 && !m.SamplingChecked

		if !needsFull && !needsVision && !needsKV && !needsReasoning && !needsSampling {
			continue
		}
		meta, err := ParseGGUFMeta(m.FilePath)
		if err != nil {
			if needsFull {
				slog.Warn("failed to parse GGUF metadata", "model", m.ID, "error", err)
			}
			continue
		}
		if needsFull {
			meta.ApplyTo(m)
			changed = true
			slog.Info("backfilled GGUF metadata", "model", m.ID, "arch", meta.Architecture,
				"layers", meta.NLayers, "kv_heads", meta.NKVHead, "ctx", meta.ContextLength,
				"vision", meta.HasVision)
		} else {
			if meta.HasVision && !m.HasBuiltinVision {
				m.HasBuiltinVision = true
				changed = true
				slog.Info("detected built-in vision", "model", m.ID)
			}
			if needsKV && (meta.KVFullPerTok > 0 || meta.KVSWAPerTok > 0) {
				m.KVFullPerTok = meta.KVFullPerTok
				m.KVSWAPerTok = meta.KVSWAPerTok
				m.SlidingWindow = meta.SlidingWindow
				if m.NKVHead == 0 {
					m.NKVHead = meta.NKVHead
				}
				changed = true
				slog.Info("backfilled KV scaling", "model", m.ID,
					"kv_full_per_tok", meta.KVFullPerTok, "kv_swa_per_tok", meta.KVSWAPerTok,
					"sliding_window", meta.SlidingWindow)
			}
			if needsReasoning && meta.ReasoningChecked {
				m.Reasoning = meta.Reasoning
				m.ReasoningChecked = true
				changed = true
				slog.Info("backfilled reasoning capability", "model", m.ID,
					"supported", meta.Reasoning.Supported, "toggle", meta.Reasoning.Toggle)
			}
			if needsSampling && meta.SamplingChecked {
				if m.BaseModelRepo == "" {
					m.BaseModelRepo = meta.BaseModelRepo
				}
				if p := meta.EmbeddedSamplingPreset(); p != nil {
					m.SamplingPresets = UpsertSamplingPreset(m.SamplingPresets, *p)
					slog.Info("backfilled embedded sampling defaults", "model", m.ID)
				}
				m.SamplingChecked = true
				changed = true
			}
		}
	}
	if changed {
		r.save()
	}
}

// DeduplicateModels removes duplicate registry entries that point to the same file.
// Keeps the first entry found (by ID sort order) and removes the rest.
func (r *Registry) DeduplicateModels() int {
	r.mu.Lock()
	defer r.mu.Unlock()

	seen := make(map[string]string) // file path → first model ID
	var dupes []string

	// Sort IDs for deterministic behavior
	ids := make([]string, 0, len(r.data.Models))
	for id := range r.data.Models {
		ids = append(ids, id)
	}
	sort.Strings(ids)

	for _, id := range ids {
		m := r.data.Models[id]
		if existing, ok := seen[m.FilePath]; ok {
			slog.Info("removing duplicate model entry", "id", id, "kept", existing, "path", m.FilePath)
			dupes = append(dupes, id)
		} else {
			seen[m.FilePath] = id
		}
	}

	for _, id := range dupes {
		delete(r.data.Models, id)
		delete(r.data.Configs, id)
	}

	if len(dupes) > 0 {
		r.save()
	}
	return len(dupes)
}

// FindOrphans returns registry entries whose model files no longer exist on disk.
func (r *Registry) FindOrphans() []*Model {
	r.mu.RLock()
	defer r.mu.RUnlock()

	var orphans []*Model
	for _, m := range r.data.Models {
		if _, err := os.Stat(m.FilePath); os.IsNotExist(err) {
			orphans = append(orphans, m)
		}
	}
	return orphans
}

// IncompleteRegistered returns registered multi-shard models whose primary file
// exists but whose shard set is incomplete (one or more shard files missing).
// These are the silently-broken downloads: ScanModels registers a model off its
// first shard, and FindOrphans only stats that first shard, so a model missing
// later shards looks healthy until it fails to load. Single-file models, and
// models whose primary file is entirely missing (those are orphans — see
// FindOrphans), are not reported here.
func (r *Registry) IncompleteRegistered() []*Model {
	r.mu.RLock()
	defer r.mu.RUnlock()

	var incomplete []*Model
	for _, m := range r.data.Models {
		// Primary file missing → orphan, not incomplete.
		if _, err := os.Stat(m.FilePath); err != nil {
			continue
		}
		shards := findShards(filepath.Dir(m.FilePath), filepath.Base(m.FilePath))
		if len(shards) < 2 {
			continue // single-file model with its file present → complete
		}
		for _, shard := range shards {
			if _, err := os.Stat(shard); os.IsNotExist(err) {
				incomplete = append(incomplete, m)
				break
			}
		}
	}
	return incomplete
}

// OrphanPart describes a partially-downloaded model that has on-disk `.part`
// files but no registry entry — a download that failed before completing and so
// never appears in the model list. Filename is a shard member (the first shard
// for multi-part sets, or the bare filename) suitable for passing back to the
// downloader, which resumes from the existing `.part` data.
type OrphanPart struct {
	ModelID     string `json:"model_id"`
	Filename    string `json:"filename"`
	BytesOnDisk int64  `json:"bytes_on_disk"`
	PartCount   int    `json:"part_count"`
}

// OrphanParts scans the models directory for `.part` files that don't belong to
// any registered model, grouping multi-shard partials into a single entry. The
// returned entries can be resumed via the normal download path. Partials that
// belong to a registered-but-incomplete model are excluded — those surface as a
// per-card indicator via IncompleteRegistered instead.
func (r *Registry) OrphanParts() []OrphanPart {
	modelsDir := r.modelsDir
	if _, err := os.Stat(modelsDir); err != nil {
		return nil
	}

	// filepath.Walk won't descend a symlinked dir; resolve then remap returned
	// paths back under modelsDir (same approach as ScanModels).
	walkRoot := modelsDir
	if resolved, err := filepath.EvalSymlinks(modelsDir); err == nil && resolved != modelsDir {
		walkRoot = resolved
	}

	// All shard paths of every registered model, so a `.part` that's really a
	// missing shard of a registered model isn't double-reported here.
	r.mu.RLock()
	registered := make(map[string]bool)
	for _, m := range r.data.Models {
		for _, sh := range findShards(filepath.Dir(m.FilePath), filepath.Base(m.FilePath)) {
			registered[sh] = true
		}
	}
	r.mu.RUnlock()

	type group struct {
		modelID  string
		filename string   // first shard / bare filename to resume with
		set      []string // full shard paths under modelsDir
		parts    int
	}
	groups := make(map[string]*group)

	filepath.Walk(walkRoot, func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() || !strings.HasSuffix(path, ".part") {
			return nil
		}
		// Remap back under modelsDir so derived IDs match registry conventions.
		if walkRoot != modelsDir {
			if rel, relErr := filepath.Rel(walkRoot, path); relErr == nil {
				path = filepath.Join(modelsDir, rel)
			}
		}
		finalPath := strings.TrimSuffix(path, ".part")
		if registered[finalPath] {
			return nil // belongs to a registered (incomplete) model
		}
		rel, relErr := filepath.Rel(modelsDir, finalPath)
		if relErr != nil {
			return nil
		}
		parts := strings.SplitN(rel, string(filepath.Separator), 2)
		if len(parts) < 2 {
			return nil // .part directly under modelsDir — no owning model dir
		}
		modelID := strings.ReplaceAll(parts[0], "--", "/")
		dir := filepath.Dir(finalPath)
		shards := findShards(dir, filepath.Base(finalPath))
		firstName := filepath.Base(shards[0])
		key := dir + "::" + firstName
		g := groups[key]
		if g == nil {
			g = &group{modelID: modelID, filename: firstName, set: shards}
			groups[key] = g
		}
		g.parts++
		return nil
	})

	var out []OrphanPart
	for _, g := range groups {
		var bytes int64
		for _, sh := range g.set {
			if info, err := os.Stat(sh); err == nil {
				bytes += info.Size()
			} else if pinfo, perr := os.Stat(sh + ".part"); perr == nil {
				bytes += pinfo.Size()
			}
		}
		out = append(out, OrphanPart{
			ModelID:     g.modelID,
			Filename:    g.filename,
			BytesOnDisk: bytes,
			PartCount:   g.parts,
		})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ModelID < out[j].ModelID })
	return out
}

// ScanModels walks the models directory for GGUF files not already in the
// registry and adds them. Returns the number of new models found.
func (r *Registry) ScanModels() int {
	modelsDir := r.modelsDir
	if _, err := os.Stat(modelsDir); err != nil {
		return 0
	}

	// filepath.Walk treats a symlink as a single non-directory entry and
	// does not descend into it, so a symlinked models dir would scan as
	// empty. Resolve the symlink before walking; remap returned paths back
	// under modelsDir so registry entries are stable across symlink-target
	// changes.
	walkRoot := modelsDir
	if resolved, err := filepath.EvalSymlinks(modelsDir); err == nil && resolved != modelsDir {
		walkRoot = resolved
		slog.Debug("scanning via resolved symlink", "models_dir", modelsDir, "resolved", resolved)
	}

	// Build set of known file paths for fast lookup
	r.mu.RLock()
	knownPaths := make(map[string]bool, len(r.data.Models))
	for _, m := range r.data.Models {
		knownPaths[m.FilePath] = true
	}
	r.mu.RUnlock()

	// Walk looking for .gguf files
	var found []*Model
	filepath.Walk(walkRoot, func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() {
			return nil
		}
		// Remap path back under modelsDir so registry entries reference
		// the user's data dir, not the resolved-symlink target.
		if walkRoot != modelsDir {
			if rel, relErr := filepath.Rel(walkRoot, path); relErr == nil {
				path = filepath.Join(modelsDir, rel)
			}
		}
		if !strings.HasSuffix(strings.ToLower(info.Name()), ".gguf") {
			return nil
		}
		// Skip .part files (incomplete downloads)
		if strings.HasSuffix(path, ".part") {
			return nil
		}
		// Skip if already registered
		if knownPaths[path] {
			return nil
		}
		// Skip shard parts beyond the first (we'll register the first shard as the model)
		if isNonFirstShard(info.Name()) {
			return nil
		}
		// Skip mmproj files — they're vision projectors, not models
		if IsMMProjFile(info.Name()) {
			return nil
		}

		// Derive model info from directory structure and filename
		// Expected: /data/models/{org--repo}/{filename}.gguf
		// or:       /data/models/{org--repo}/{subdir}/{filename}.gguf
		rel, _ := filepath.Rel(modelsDir, path)
		parts := strings.SplitN(rel, string(filepath.Separator), 2)

		dirName := parts[0]                               // e.g., "unsloth--Qwen3.5-27B-GGUF"
		filename := info.Name()                           // e.g., "Qwen3.5-27B-Q4_K_M.gguf"
		modelID := strings.ReplaceAll(dirName, "--", "/") // e.g., "unsloth/Qwen3.5-27B-GGUF"

		safeName := dirName
		safeFilename := strings.ReplaceAll(strings.TrimSuffix(rel, ".gguf"), string(filepath.Separator), "--")
		id := fmt.Sprintf("%s--%s", safeName, strings.TrimSuffix(filename, ".gguf"))
		if len(parts) > 1 {
			// Has subdirectory — use the full relative path for the ID
			id = safeFilename
			// Prefix with org--repo if not already
			if !strings.HasPrefix(id, safeName) {
				id = safeName + "--" + id
			}
		}

		// Calculate total size (sum shards if multi-part)
		totalSize := info.Size()
		shardFiles := findShards(filepath.Dir(path), filename)
		if len(shardFiles) > 1 {
			totalSize = 0
			for _, sf := range shardFiles {
				if si, err := os.Stat(sf); err == nil {
					totalSize += si.Size()
				}
			}
		}

		m := &Model{
			ID:           id,
			ModelID:      modelID,
			Filename:     filename,
			Quant:        ParseQuant(filename),
			SizeBytes:    totalSize,
			FilePath:     path,
			VRAMEstGB:    EstimateVRAM(totalSize),
			DownloadedAt: info.ModTime(),
		}

		// Parse GGUF metadata
		if meta, err := ParseGGUFMeta(path); err == nil {
			meta.ApplyTo(m)
		}

		// Skip standalone MTP / drafter "assistant" heads (e.g. gemma-4's
		// gemma4-assistant). They're loaded via --model-draft alongside a main
		// model, not served on their own — auto-associated by AutoDetectMTP below.
		if IsMTPHeadArch(m.Arch) {
			return nil
		}

		found = append(found, m)
		return nil
	})

	for _, m := range found {
		r.Add(m)
		slog.Info("scanned model", "id", m.ID, "file", m.FilePath,
			"size_gb", fmt.Sprintf("%.1f", float64(m.SizeBytes)/(1024*1024*1024)),
			"arch", m.Arch)
	}

	// Auto-associate mmproj files and separate MTP drafter heads with models
	if len(found) > 0 {
		r.AutoDetectMMProj()
		r.AutoDetectMTP()
	}

	return len(found)
}

// IsMMProjFile returns true if the filename looks like a multimodal projector.
func IsMMProjFile(filename string) bool {
	return strings.Contains(strings.ToLower(filename), "mmproj")
}

// embeddingPattern matches common embedding model name patterns.
var embeddingPattern = regexp.MustCompile(`(?i)([-/]embed[-/]|[-/]embed$|nomic-embed|^bge-|[-/]bge[-/]|[-/]e5[-/]|[-/]gte[-/]|snowflake-arctic-embed|mxbai-embed|jina-embed)`)

// IsEmbeddingModel returns true if the model name/ID suggests it's an embedding model.
func IsEmbeddingModel(name string) bool {
	return embeddingPattern.MatchString(name)
}

// IsEmbedding reports whether the model looks like an embedding model.
// It checks both the HF repo ID and the registry ID because legacy records
// may have only one populated.
func (m *Model) IsEmbedding() bool {
	return IsEmbeddingModel(m.ModelID) || IsEmbeddingModel(m.ID)
}

// HasVision reports whether the model can accept images — either built-in
// or via a configured mmproj projector.
func (m *Model) HasVision(cfg *ModelConfig) bool {
	return m.HasBuiltinVision || (cfg != nil && cfg.MmprojPath != "")
}

// Capabilities returns the capability list for a model: "chat" or
// "embedding", plus "tools" and "vision" when supported.
func (m *Model) Capabilities(cfg *ModelConfig) []string {
	var caps []string
	if m.IsEmbedding() {
		caps = append(caps, "embedding")
	} else {
		caps = append(caps, "chat")
	}
	if m.SupportsTools {
		caps = append(caps, "tools")
	}
	if m.HasVision(cfg) {
		caps = append(caps, "vision")
	}
	return caps
}

// FindMMProj looks for mmproj GGUF files in the same directory as the model,
// then checks the parent directory (for repos where mmproj is at the root
// and model GGUFs are in subdirectories, e.g. Mistral-Small-4-119B).
// Returns the path to the first one found, or empty string.
func FindMMProj(modelFilePath string) string {
	dir := filepath.Dir(modelFilePath)
	if found := findMMProjInDir(dir); found != "" {
		return found
	}
	// Check parent directory — handles repos where mmproj lives at the
	// repo root while model files are in quant subdirectories.
	parent := filepath.Dir(dir)
	if parent != dir {
		return findMMProjInDir(parent)
	}
	return ""
}

// findMMProjInDir scans a single directory for mmproj GGUF files.
func findMMProjInDir(dir string) string {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return ""
	}
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if strings.HasSuffix(strings.ToLower(name), ".gguf") && IsMMProjFile(name) {
			return filepath.Join(dir, name)
		}
	}
	return ""
}

// AutoDetectMMProj scans all registered models and sets MmprojPath on
// configs where an mmproj file exists in the model directory but isn't
// configured yet.
func (r *Registry) AutoDetectMMProj() int {
	r.mu.Lock()
	defer r.mu.Unlock()

	found := 0
	for id, m := range r.data.Models {
		cfg := r.data.Configs[id]
		if cfg == nil || cfg.MmprojPath != "" {
			continue
		}
		if mmproj := FindMMProj(m.FilePath); mmproj != "" {
			cfg.MmprojPath = mmproj
			found++
		}
	}
	if found > 0 {
		r.save()
	}
	return found
}

// IsMTPHeadArch reports whether a GGUF architecture identifies a standalone
// MTP / speculative "assistant" drafter head (e.g. gemma-4's "gemma4-assistant")
// rather than a runnable model. Such files load via --model-draft under
// spec_type=draft-mtp, not served on their own.
//
// Detection is by architecture, not filename: Qwen's self-speculation MTP
// models also carry "MTP" in their name but ARE runnable (the head is baked
// into a normal qwen3 arch), so they must keep registering as ordinary models.
func IsMTPHeadArch(arch string) bool {
	return strings.Contains(strings.ToLower(arch), "assistant")
}

// FindMTP looks for a separate MTP drafter-head GGUF associated with the given
// model: the same directory, an "MTP/" subdirectory (unsloth's gemma-4 layout),
// or the parent directory and its "MTP/" subdir (for quant-in-subdir repos).
// Returns the path to the first one found, or empty string.
func FindMTP(modelFilePath string) string {
	dir := filepath.Dir(modelFilePath)
	for _, d := range []string{dir, filepath.Join(dir, "MTP")} {
		if p := findMTPInDir(d); p != "" {
			return p
		}
	}
	parent := filepath.Dir(dir)
	if parent != dir {
		for _, d := range []string{parent, filepath.Join(parent, "MTP")} {
			if p := findMTPInDir(d); p != "" {
				return p
			}
		}
	}
	return ""
}

// findMTPInDir scans a single directory for a GGUF whose architecture marks it
// as an MTP drafter head.
func findMTPInDir(dir string) string {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return ""
	}
	// unsloth ships heads in an "MTP/" subdirectory; arch is authoritative but
	// reading every GGUF header is the expensive part, so use location/name as
	// a cheap pre-filter.
	inMTPDir := strings.EqualFold(filepath.Base(dir), "MTP")
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		lname := strings.ToLower(name)
		if !strings.HasSuffix(lname, ".gguf") {
			continue
		}
		// Only crack open headers for files that plausibly *are* a drafter head.
		// Skips parsing every multi-GB main quant on each config open / scan.
		// The IsMTPHeadArch check below stays authoritative — e.g. a Qwen
		// self-speculation model named "...-MTP-..." passes this pre-filter but
		// is correctly rejected by its runnable (non-assistant) architecture.
		if !inMTPDir && !strings.Contains(lname, "mtp") && !strings.Contains(lname, "assistant") {
			continue
		}
		path := filepath.Join(dir, name)
		if meta, err := ParseGGUFMeta(path); err == nil && IsMTPHeadArch(meta.Architecture) {
			return path
		}
	}
	return ""
}

// AutoDetectMTP scans all registered models and sets MtpPath on configs where a
// separate MTP drafter head exists in or near the model directory but isn't
// configured yet.
func (r *Registry) AutoDetectMTP() int {
	r.mu.Lock()
	defer r.mu.Unlock()

	found := 0
	for id, m := range r.data.Models {
		cfg := r.data.Configs[id]
		if cfg == nil || cfg.MtpPath != "" {
			continue
		}
		if mtp := FindMTP(m.FilePath); mtp != "" {
			cfg.MtpPath = mtp
			found++
		}
	}
	if found > 0 {
		r.save()
	}
	return found
}

// DraftCandidate represents a model that could serve as a speculative draft.
type DraftCandidate struct {
	ID       string
	Filename string
	FilePath string
	SizeGB   float64
	Arch     string
}

// FindDraftCandidates returns models that could serve as draft models for
// the given model: same architecture family, significantly smaller.
func (r *Registry) FindDraftCandidates(id string) []DraftCandidate {
	r.mu.RLock()
	defer r.mu.RUnlock()

	target, ok := r.data.Models[id]
	if !ok || target.Arch == "" {
		return nil
	}

	var candidates []DraftCandidate
	for _, m := range r.data.Models {
		if m.ID == id {
			continue
		}
		// Same architecture family
		if m.Arch != target.Arch {
			continue
		}
		// Must be significantly smaller (< 40% of target size)
		if m.SizeBytes >= target.SizeBytes*4/10 {
			continue
		}
		// Skip embedding models
		if m.IsEmbedding() {
			continue
		}
		candidates = append(candidates, DraftCandidate{
			ID:       m.ID,
			Filename: m.Filename,
			FilePath: m.FilePath,
			SizeGB:   BytesToGB(m.SizeBytes),
			Arch:     m.Arch,
		})
	}
	return candidates
}

// shardRe matches shard filenames like "model-00002-of-00005.gguf"
var shardRe = regexp.MustCompile(`-(\d{5})-of-(\d{5})\.gguf$`)

// isNonFirstShard returns true if filename is a shard part other than 00001.
func isNonFirstShard(filename string) bool {
	m := shardRe.FindStringSubmatch(filename)
	if m == nil {
		return false
	}
	return m[1] != "00001"
}

// findShards returns all shard file paths if filename is part of a multi-part set.
// Returns a single-element slice for non-sharded files.
func findShards(dir, filename string) []string {
	m := shardRe.FindStringSubmatch(filename)
	if m == nil {
		return []string{filepath.Join(dir, filename)}
	}

	total, err := strconv.Atoi(m[2])
	if err != nil || total < 2 {
		return []string{filepath.Join(dir, filename)}
	}

	// Extract the base name (everything before -NNNNN-of-NNNNN.gguf)
	loc := shardRe.FindStringIndex(filename)
	base := filename[:loc[0]]

	var shards []string
	for i := 1; i <= total; i++ {
		shard := filepath.Join(dir, fmt.Sprintf("%s-%05d-of-%05d.gguf", base, i, total))
		shards = append(shards, shard)
	}
	return shards
}

// quantRe matches a quantization token in a normalized (uppercased,
// dashes->underscores) GGUF filename, with an optional "UD_" ultra-dynamic
// prefix captured separately. The body alternatives are written so the greedy
// optional suffix groups absorb variants generically (e.g. Q2_K_XL, Q3_K_L),
// which the previous hand-maintained list dropped. Ordering keeps longer
// tokens ahead of their prefixes where alternation could otherwise short-cut
// (e.g. BF16 before F16, MXFP4 before the bare FP4 that also matches vendor
// names like ROCmFP4 / NVFP4).
var quantRe = regexp.MustCompile(`(UD_)?(MXFP4|FP4|BF16|F16|F32|TQ[1-4]_[01]|IQ[1-4]_(?:XXS|XS|NL|S|M|L)|Q[2-8]_K(?:_(?:XL|XS|S|M|L))?|Q[2-8]_[01])`)

// ParseQuant extracts the quantization type from a GGUF filename. It preserves
// the "UD_" (Unsloth ultra-dynamic) prefix when present so UD quants are
// labelled consistently. Exported so it can be shared across packages.
func ParseQuant(filename string) string {
	name := strings.TrimSuffix(filepath.Base(filename), ".gguf")
	name = strings.TrimSuffix(name, ".GGUF")

	// Remove shard suffix if present
	if idx := strings.LastIndex(name, "-00001-of-"); idx > 0 {
		name = name[:idx]
	}

	// Normalize dashes to underscores so "UD-Q8_K_XL" matches "UD_Q8_K_XL"
	upper := strings.ToUpper(strings.ReplaceAll(name, "-", "_"))

	m := quantRe.FindStringSubmatch(upper)
	if m == nil {
		return "unknown"
	}
	return m[1] + m[2] // m[1] is "UD_" or "", m[2] is the quant body
}

func (r *Registry) registryPath() string {
	return filepath.Join(r.dataDir, "config", "models.json")
}

func (r *Registry) load() {
	data, err := os.ReadFile(r.registryPath())
	if err != nil {
		return
	}
	if err := json.Unmarshal(data, &r.data); err != nil {
		slog.Error("failed to load model registry", "error", err)
	}
	if r.data.Models == nil {
		r.data.Models = make(map[string]*Model)
	}
	if r.data.Configs == nil {
		r.data.Configs = make(map[string]*ModelConfig)
	}

	// Re-derive Quant from the filename for every registered model. The field
	// is persisted, but ScanModels skips already-known paths, so entries added
	// before a ParseQuant improvement keep their stale value (e.g. MXFP4 frozen
	// as "unknown", or UD-Q2_K_XL truncated to Q2_K). The Search HF tab parses
	// live and stays correct; the Models tab reads this field, so backfill it
	// on load to keep both tabs consistent and self-healing.
	for _, m := range r.data.Models {
		if m.Filename == "" {
			continue
		}
		if q := ParseQuant(m.Filename); q != m.Quant {
			m.Quant = q
		}
	}
}

func (r *Registry) save() {
	os.MkdirAll(filepath.Dir(r.registryPath()), 0o755)
	data, err := json.MarshalIndent(r.data, "", "  ")
	if err != nil {
		slog.Error("failed to marshal model registry", "error", err)
		return
	}
	os.WriteFile(r.registryPath(), data, 0o644)
}
