package api

import (
	"encoding/json"
	"fmt"
	"net/http"
	"slices"

	"github.com/go-chi/chi/v5"
	"github.com/tmac1973/llama-toolchest/internal/generation"
	"github.com/tmac1973/llama-toolchest/internal/models"
)

// modelArchitecture is what a model reads and makes, in the shape of
// llama-server's own /v1/models architecture object.
func modelArchitecture(m *models.Model, cfg *models.ModelConfig) map[string]any {
	in, out := m.Modalities(cfg)
	return map[string]any{"input_modalities": in, "output_modalities": out}
}

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
		"architecture": modelArchitecture(m, cfg),
		// The reasoning_effort values the template accepts, lowest first;
		// null until the model has been loaded on a llama.cpp build that
		// reports them.
		"reasoning_effort_levels": m.EffortLevels(),
	}
	if cfg != nil {
		// The same context fields as the capabilities object, so a client
		// that reads only /v1/models compacts against what one
		// conversation can use, not the whole pool.
		layout := s.servedContextLayout(m, cfg)
		meta["context_size"] = layout.Pool
		meta["parallel"] = layout.Slots
		meta["context_shared"] = layout.Shared
		meta["context_per_request"] = layout.PerConversation
	}

	// Only while generation backends are on, so the list is unchanged for
	// everyone else.
	if s.generationOn() {
		meta["kind"] = "text"
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

// v1ModelKinds are the meta.kind values /v1/models?kind= accepts.
var v1ModelKinds = []string{"text", "image", "video", "music", "upscale"}

// handleV1Models returns an OpenAI-compatible model list with meta extensions.
// ?kind= keeps only the models of one meta.kind.
func (s *Server) handleV1Models(w http.ResponseWriter, r *http.Request) {
	kind := r.URL.Query().Get("kind")
	if kind != "" && !slices.Contains(v1ModelKinds, kind) {
		writeOpenAIError(w, http.StatusBadRequest, "invalid_kind",
			fmt.Sprintf("unknown kind %q: use text, image, video, music or upscale", kind))
		return
	}

	// Not nil, so an empty list is [] rather than null, which some
	// OpenAI clients cannot read.
	data := []map[string]any{}
	// LLMs carry meta.kind only while generation is on, so they are
	// filtered here rather than by their meta.
	if kind == "" || kind == "text" {
		for _, m := range s.registry.List() {
			cfg, _ := s.registry.GetConfig(m.ID)
			if advertisedToClients(m, cfg) {
				data = append(data, s.openAIModel(m, cfg))
			}
		}
	}

	for _, obj := range s.genV1Models() {
		if kind == "" || obj["meta"].(map[string]any)["kind"] == kind {
			data = append(data, obj)
		}
	}

	respondJSON(w, map[string]any{
		"object": "list",
		"data":   data,
	})
}

// genV1Models lists the generation instances that can take requests as
// models, with meta.kind "image", "video", "music" or "upscale": enabled,
// with a VRAM reservation (pickInstance refuses any other). Clients that
// only chat should skip kinds other than "text". Empty while generation
// backends are off.
func (s *Server) genV1Models() []map[string]any {
	if !s.generationOn() || s.gen == nil {
		return nil
	}
	var out []map[string]any
	for _, inst := range s.gen.Instances() {
		if !inst.Enabled || inst.ReservedMiB == 0 {
			continue
		}
		if obj := s.genV1Model(inst); obj != nil {
			out = append(out, obj)
		}
	}
	return out
}

func (s *Server) genV1Model(inst generation.Instance) map[string]any {
	set, err := s.gen.ModelSet(inst.ModelSetID)
	if err != nil {
		return nil
	}
	caps := set.CurrentCapabilities()
	if caps == nil {
		caps = []string{}
	}
	// api lists the route groups that serve this model, so a client can
	// tell what this build supports without trying each route.
	kind, api := genKindLabel(set.Kind), []string{"images.sync", "images.jobs"}
	if slices.Contains(caps, generation.CapEdit) {
		// images.jobs.edits: /v1/images/jobs takes edits as multipart
		// forms. Builds that refused them do not list it.
		api = append(api, "images.edits", "images.jobs.edits")
	}
	switch set.Kind {
	case generation.KindAce:
		api = []string{"music.jobs", "music.ace_step"}
	case generation.KindVideo:
		api = []string{"video.jobs", "video.sync"}
	case generation.KindUpscale:
		api = []string{"images.upscale"}
	}
	meta := map[string]any{
		"kind":         kind,
		"model_set":    set.ID,
		"recipe":       set.Recipe,
		"capabilities": caps,
		"gpu":          inst.GPU,
		"reserved_mib": inst.ReservedMiB,
		"state":        s.genMgr.Status(inst.ID).State,
		// measuring is true while a measured test run has the instance;
		// requests are refused with backend_unavailable until it ends.
		"measuring": s.genMgr.IsMeasuring(inst.ID),
		"api":       api,
	}
	// max_edit_images is how many image[] parts one edit takes; left out
	// when no limit is known.
	if n := generation.EditImageLimit(inst, set); n > 0 {
		meta["max_edit_images"] = n
	}
	// scale is how many times larger an upscaler makes an image; left out
	// for a custom upscaler set.
	if n := set.UpscaleScale(); n > 0 {
		meta["scale"] = n
	}
	if set.Kind == generation.KindAce {
		// tasks lists the music task_type values this model runs, and
		// tracks the stems lego, extract and complete take.
		tasks := aceTasks(set)
		meta["tasks"] = tasks
		if slices.Contains(tasks, taskLego) {
			meta["tracks"] = musicTracks
		}
	}
	if set.Kind == generation.KindVideo {
		// video gives the defaults a request gets and the largest video it
		// may ask for (left out when the set has no ceiling).
		video := map[string]any{
			"default_size":   fmt.Sprintf("%dx%d", set.DefaultWidth, set.DefaultHeight),
			"default_frames": set.DefaultFrames,
			"default_fps":    set.DefaultFPS,
		}
		if set.MaxFrames > 0 {
			video["max_frames"] = set.MaxFrames
		}
		if set.MaxWidth > 0 && set.MaxHeight > 0 {
			video["max_size"] = fmt.Sprintf("%dx%d", set.MaxWidth, set.MaxHeight)
		}
		// size_multiple is what sd-server rounds width and height up to.
		if e, ok := generation.LookupCatalog(set.CatalogID); ok && set.CatalogID != "" && e.SizeMultiple > 0 {
			video["size_multiple"] = e.SizeMultiple
		}
		// measured_* is what this instance will take: the size it was
		// measured for; larger requests are refused.
		limit, _ := generation.VideoLimit(inst, set)
		video["measured_frames"] = limit.Frames
		video["measured_size"] = fmt.Sprintf("%dx%d", limit.Width, limit.Height)
		video["measured_pixel_frames"] = generation.PixelFrames(limit.Frames, limit.Width, limit.Height)
		meta["video"] = video
	}
	return map[string]any{
		"id":       inst.ID,
		"object":   "model",
		"created":  set.Created.Unix(),
		"owned_by": "llama-toolchest",
		"meta":     meta,
	}
}

// handleV1Model returns a single OpenAI-compatible model object with meta extensions.
func (s *Server) handleV1Model(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "model")

	m, cfg := s.findModelByAny(id)
	if m == nil && s.generationOn() && s.gen != nil {
		if inst, err := s.gen.Instance(id); err == nil && inst.Enabled && inst.ReservedMiB > 0 {
			if obj := s.genV1Model(inst); obj != nil {
				respondJSON(w, obj)
				return
			}
		}
	}
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
// model: enabled, and neither one of the app's own helper models nor a
// draft head, which only another model can load. The list and
// the single-model lookup both use it, so a client sees the same set of
// models either way; a hidden model answers 404 like an unknown name.
func advertisedToClients(m *models.Model, cfg *models.ModelConfig) bool {
	return !m.HelperRole && !m.IsDraftHead() && (cfg == nil || cfg.Enabled)
}

// findModelByAny looks up a model by registry ID, public name, router name,
// or user-defined alias. Thin wrapper around registry.FindByAny so existing
// API-package call sites stay readable.
func (s *Server) findModelByAny(name string) (*models.Model, *models.ModelConfig) {
	return s.registry.FindByAny(name)
}

// modelFromBody finds the model a JSON request body's "model" field names,
// as findModelByAny does. Both are nil when the body is not JSON, names no
// model or names one we do not know.
func (s *Server) modelFromBody(body []byte) (*models.Model, *models.ModelConfig) {
	var req struct {
		Model string `json:"model"`
	}
	if json.Unmarshal(body, &req) != nil || req.Model == "" {
		return nil, nil
	}
	return s.findModelByAny(req.Model)
}

// genKindLabel names a set's kind as /v1/models and the UI show it.
func genKindLabel(k generation.BackendKind) string {
	switch k {
	case generation.KindAce:
		return "music"
	case generation.KindVideo:
		return "video"
	case generation.KindUpscale:
		return "upscale"
	}
	return "image"
}
