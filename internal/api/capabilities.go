package api

import "github.com/tmac1973/llama-toolchest/internal/models"

// CapabilitiesSchemaVersion is the version of the capabilities object shape.
// Clients branch on it; bump only on breaking changes to the schema.
const CapabilitiesSchemaVersion = 1

// buildCapabilities assembles the versioned capability object documented in the
// README. It is the single source of truth shared by /api/models/{id}/info and
// /api/service/loaded-models, so a client sees identical data from either and
// can configure itself in one round-trip.
//
// Per the contract, keys are emitted with explicit null/false rather than
// omitted, so a client can tell "known to be absent" from "server too old to
// report" (the opposite of Go's omitempty default).
func (s *Server) buildCapabilities(m *models.Model, cfg *models.ModelConfig) map[string]any {
	// Served context: prefer the live snapshot (what the router is actually
	// running) over the configured value, and resolve 0 → the model's trained
	// max — matching how /info reports context_size. This is the number a
	// client must compact against, never the trained max (context_length).
	servedCtx := m.ContextLength
	parallel := 1
	if cfg != nil {
		live := cfg
		if snap, ok := s.runningConfigs[m.ID]; ok && s.process.IsRunning() {
			live = snap
		}
		if live.ContextSize > 0 {
			servedCtx = live.ContextSize
		}
		if live.Parallel > 1 {
			parallel = live.Parallel
		}
	}

	return map[string]any{
		"schema_version": CapabilitiesSchemaVersion,

		// context (see README §"served context vs. trained max")
		"context_size":        servedCtx,       // SERVED n_ctx — compact on this
		"context_length":      m.ContextLength, // trained max, informational
		"parallel":            parallel,        // slot count; 1 = no extra slots
		"context_per_request": servedCtx / parallel,

		// modalities / tools
		"vision":    m.HasVision(cfg),
		"tools":     m.SupportsTools,
		"embedding": m.IsEmbedding(),

		// reasoning / thinking
		"reasoning": m.EffectiveReasoning(cfg),

		// recommended sampling
		"sampling": buildSampling(m, cfg),

		// generation limits — no server-imposed cap today
		"max_output_tokens": nil,
	}
}

// buildSampling resolves the recommended sampling block. `default` is the
// effective per-parameter value: a per-model override (cfg) wins, else the
// model card recommendation, else null. `presets` is the published preset list
// verbatim, and source/source_url carry provenance from the card.
func buildSampling(m *models.Model, cfg *models.ModelConfig) map[string]any {
	presets := models.LookupSamplingPresets(m.ModelID)
	if presets == nil {
		presets = []models.SamplingPreset{}
	}

	// Card recommendation: prefer a preset named "default", else the first.
	var card *models.SamplingPreset
	for i := range presets {
		if presets[i].Name == "default" {
			card = &presets[i]
			break
		}
	}
	if card == nil && len(presets) > 0 {
		card = &presets[0]
	}

	// Per-field resolver: override → card → null.
	resolveF := func(override func(*models.ModelConfig) *float64, cardVal func(*models.SamplingPreset) *float64) any {
		if cfg != nil {
			if v := override(cfg); v != nil {
				return *v
			}
		}
		if card != nil {
			if v := cardVal(card); v != nil {
				return *v
			}
		}
		return nil
	}
	resolveI := func(override func(*models.ModelConfig) *int, cardVal func(*models.SamplingPreset) *int) any {
		if cfg != nil {
			if v := override(cfg); v != nil {
				return *v
			}
		}
		if card != nil {
			if v := cardVal(card); v != nil {
				return *v
			}
		}
		return nil
	}

	def := map[string]any{
		"temperature":      resolveF(func(c *models.ModelConfig) *float64 { return c.Temperature }, func(p *models.SamplingPreset) *float64 { return p.Temperature }),
		"top_p":            resolveF(func(c *models.ModelConfig) *float64 { return c.TopP }, func(p *models.SamplingPreset) *float64 { return p.TopP }),
		"top_k":            resolveI(func(c *models.ModelConfig) *int { return c.TopK }, func(p *models.SamplingPreset) *int { return p.TopK }),
		"min_p":            resolveF(func(c *models.ModelConfig) *float64 { return c.MinP }, func(p *models.SamplingPreset) *float64 { return p.MinP }),
		"presence_penalty": resolveF(func(c *models.ModelConfig) *float64 { return c.PresencePenalty }, func(p *models.SamplingPreset) *float64 { return p.PresencePenalty }),
		"repeat_penalty":   resolveF(func(c *models.ModelConfig) *float64 { return c.RepeatPenalty }, func(p *models.SamplingPreset) *float64 { return p.RepeatPenalty }),
	}

	out := map[string]any{
		"source":     nil,
		"source_url": nil,
		"default":    def,
		"presets":    presets,
	}
	if card != nil {
		out["source"] = card.Source
		if card.SourceURL != "" {
			out["source_url"] = card.SourceURL
		}
	}
	return out
}
