package api

import (
	"net/http"

	"github.com/go-chi/chi/v5"
	"github.com/tmac1973/llama-toolchest/internal/models"
)

// openAIModel builds an OpenAI-compatible Model object with a meta extension.
func (s *Server) openAIModel(m *models.Model, cfg *models.ModelConfig) map[string]any {
	meta := map[string]any{
		"arch":         m.Arch,
		"quant":        m.Quant,
		"n_layers":     m.NLayers,
		"n_embd":       m.NEmbd,
		"n_ctx_train":  m.ContextLength,
		"size":         m.SizeBytes,
		"capabilities": m.Capabilities(cfg),
	}
	if cfg != nil {
		// 0 means "use model default" (n_ctx_train)
		if cfg.ContextSize > 0 {
			meta["context_size"] = cfg.ContextSize
		} else {
			meta["context_size"] = m.ContextLength
		}
	}

	obj := map[string]any{
		"id":       m.PublicName(),
		"object":   "model",
		"created":  m.DownloadedAt.Unix(),
		"owned_by": "llama-toolchest",
		"meta":     meta,
	}
	if cfg != nil && len(cfg.Aliases) > 0 {
		obj["aliases"] = cfg.Aliases
	}
	return obj
}

// handleV1Models returns an OpenAI-compatible model list with meta extensions.
func (s *Server) handleV1Models(w http.ResponseWriter, r *http.Request) {
	var data []map[string]any
	for _, m := range s.registry.List() {
		cfg, _ := s.registry.GetConfig(m.ID)
		if advertisedToClients(m, cfg) {
			data = append(data, s.openAIModel(m, cfg))
		}
	}

	respondJSON(w, map[string]any{
		"object": "list",
		"data":   data,
	})
}

// handleV1Model returns a single OpenAI-compatible model object with meta extensions.
func (s *Server) handleV1Model(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "model")

	m, cfg := s.findModelByAny(id)
	if m == nil || !advertisedToClients(m, cfg) {
		respondJSONStatus(w, http.StatusNotFound, map[string]any{
			"error": map[string]any{
				"message": "model not found: " + id,
				"type":    "invalid_request_error",
				"code":    "model_not_found",
			},
		})
		return
	}

	respondJSON(w, s.openAIModel(m, cfg))
}

// advertisedToClients reports whether the OpenAI-compatible API offers a
// model: enabled, and not one of the app's own helper models. The list and
// the single-model lookup both use it, so a client sees the same set of
// models either way; a hidden model answers 404 like an unknown name.
func advertisedToClients(m *models.Model, cfg *models.ModelConfig) bool {
	return !m.HelperRole && (cfg == nil || cfg.Enabled)
}

// findModelByAny looks up a model by registry ID, public name, router name,
// or user-defined alias. Thin wrapper around registry.FindByAny so existing
// API-package call sites stay readable.
func (s *Server) findModelByAny(name string) (*models.Model, *models.ModelConfig) {
	return s.registry.FindByAny(name)
}
