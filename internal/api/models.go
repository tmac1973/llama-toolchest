package api

import (
	"fmt"
	"html"
	"log/slog"
	"net/http"
	"sort"
	"strings"

	"github.com/go-chi/chi/v5"
	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/modelsource"
)

func (s *Server) handleListEmbeddingModels(w http.ResponseWriter, r *http.Request) {
	embeddingModels := filterModels(s.registry.List(), true)
	pending := filterPending(s.registry.PendingConfigs(), true)

	if isHTMX(r) {
		respondHTML(w)
		if len(embeddingModels) == 0 && len(pending) == 0 {
			return
		}
		s.renderModelList(w, r, embeddingModels, pending, false)
		return
	}

	respondJSON(w, withPublicNames(embeddingModels))
}

func (s *Server) handleListModels(w http.ResponseWriter, r *http.Request) {
	// Filter out embedding models — they have their own section
	modelList := filterModels(s.registry.List(), false)
	pending := filterPending(s.registry.PendingConfigs(), false)

	if isHTMX(r) {
		respondHTML(w)
		// Pending entries must render even with an empty registry — a
		// fresh target server right after a backup restore is exactly
		// the case ghost cards exist for.
		if len(modelList) == 0 && len(pending) == 0 {
			w.Write([]byte(`<p>No models downloaded yet. <a href="/models/browse">Browse HuggingFace</a> to download models.</p>`))
			return
		}

		s.renderModelList(w, r, modelList, pending, true)
		return
	}

	respondJSON(w, withPublicNames(modelList))
}

// filterPending splits pending backup configs by model kind using the
// same name predicate the real lists use, so a pending embedding model
// ghosts next to its installed siblings.
func filterPending(all []models.PendingConfig, embedding bool) []models.PendingConfig {
	var out []models.PendingConfig
	for _, p := range all {
		if models.IsEmbeddingModel(p.ModelID) == embedding {
			out = append(out, p)
		}
	}
	return out
}

// filterModels splits the registry list by model kind: embedding=true keeps
// only embedding models, embedding=false keeps only chat models.
func filterModels(all []*models.Model, embedding bool) []*models.Model {
	var out []*models.Model
	for _, m := range all {
		if m.IsEmbedding() == embedding {
			out = append(out, m)
		}
	}
	return out
}

// withPublicNames wraps each model with its OpenAI-style public name so
// the JSON listing surfaces the same ID that /v1/models advertises. Without
// this, a client that discovers models via /api/models/ has no way to map
// them to /v1/chat/completions request bodies.
func withPublicNames(list []*models.Model) []map[string]any {
	out := make([]map[string]any, 0, len(list))
	for _, m := range list {
		entry := map[string]any{
			"id":                 m.ID,
			"public_name":        m.PublicName(),
			"model_id":           m.ModelID,
			"filename":           m.Filename,
			"quant":              m.Quant,
			"size_bytes":         m.SizeBytes,
			"file_path":          m.FilePath,
			"vram_est_gb":        m.VRAMEstGB,
			"downloaded_at":      m.DownloadedAt,
			"arch":               m.Arch,
			"n_layers":           m.NLayers,
			"n_embd":             m.NEmbd,
			"n_head":             m.NHead,
			"n_kv_head":          m.NKVHead,
			"context_length":     m.ContextLength,
			"supports_tools":     m.SupportsTools,
			"has_builtin_vision": m.HasBuiltinVision,
			// The per-layer embedding table drives both the config
			// selector's visibility and the VRAM estimate, so a client
			// reading this list cannot explain either number without it.
			"ple_bytes":   m.PLEBytes,
			"ple_checked": m.PLEChecked,
		}
		out = append(out, entry)
	}
	return out
}

func (s *Server) handleEmbeddingPresets(w http.ResponseWriter, r *http.Request) {
	presets := models.CuratedEmbeddingModels()

	// Mark which ones are already downloaded
	allModels := s.registry.List()
	downloaded := make(map[string]bool)
	for _, m := range allModels {
		downloaded[m.ModelID] = true
	}

	if isHTMX(r) {
		respondHTML(w)
		s.renderPartial(w, "embedding_presets", struct {
			Presets    []models.EmbeddingModelPreset
			Downloaded map[string]bool
		}{Presets: presets, Downloaded: downloaded})
		return
	}

	respondJSON(w, presets)
}

func (s *Server) handleDownloadEmbeddingPreset(w http.ResponseWriter, r *http.Request) {
	r.ParseForm()
	repo := r.FormValue("repo")
	filename := r.FormValue("filename")

	if repo == "" || filename == "" {
		http.Error(w, "missing repo or filename", http.StatusBadRequest)
		return
	}

	// Curated embedding presets are HuggingFace repos by construction.
	downloadID, err := s.downloader.Start(r.Context(), modelsource.SourceHuggingFace, repo, filename, 0)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	if isHTMX(r) {
		respondHTML(w)
		s.renderPartial(w, "download_progress", struct {
			DownloadID string
			Filename   string
		}{DownloadID: downloadID, Filename: filename})
		return
	}

	respondJSON(w, map[string]string{"download_id": downloadID})
}

func (s *Server) handleScanModels(w http.ResponseWriter, r *http.Request) {
	found := s.registry.ScanModels()
	if found > 0 {
		go s.backfillPresets()
	}

	if isHTMX(r) {
		// Re-render the model list with any newly discovered models
		s.handleListModels(w, r)
		return
	}

	respondJSON(w, map[string]int{"new_models": found})
}

func (s *Server) handleGetModel(w http.ResponseWriter, r *http.Request) {
	id := s.registry.ResolveID(chi.URLParam(r, "id"))
	m, err := s.registry.Get(id)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}

	respondJSON(w, m)
}

// handleModelInfo returns enriched model metadata with capabilities and config.
func (s *Server) handleModelInfo(w http.ResponseWriter, r *http.Request) {
	id := s.registry.ResolveID(chi.URLParam(r, "id"))
	m, err := s.registry.Get(id)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}

	cfg, _ := s.registry.GetConfig(id)

	info := map[string]any{
		"id":             m.ID,
		"public_name":    m.PublicName(),
		"model_id":       m.ModelID,
		"filename":       m.Filename,
		"arch":           m.Arch,
		"quant":          m.Quant,
		"context_length": m.ContextLength,
		"size_bytes":     m.SizeBytes,
		"vram_est_gb":    m.VRAMEstGB,
		"capabilities":   m.Capabilities(cfg),
		"downloaded_at":  m.DownloadedAt,
	}

	if cfg != nil {
		// The router reads preset.ini only at startup, so any field that's
		// baked into the launch command line is "what will be live after
		// the next restart" — not necessarily what's live now. When we have
		// a snapshot from the last router start, prefer it as the primary
		// value and add a `<field>_pending` key whenever the configured
		// value differs. Sampling params and aliases are not snapshotted
		// because the proxy applies them per-request.
		live := cfg
		hasSnapshot := false
		if snap, ok := s.runningConfigFor(id); ok && s.process.IsRunning() {
			live = snap
			hasSnapshot = true
		}

		// context_size resolves 0 → model's trained context length in every
		// other surface (see f6be200). Do the same on both sides so we don't
		// flag a spurious "pending" diff between e.g. configured=0 and
		// live=131072 when they mean the same thing.
		resolveCtx := func(v int) int {
			if v == 0 {
				return m.ContextLength
			}
			return v
		}
		liveCtx := resolveCtx(live.ContextSize)
		configuredCtx := resolveCtx(cfg.ContextSize)

		// parallel >1 divides ctx_size across slots, so each request gets only
		// liveCtx/parallel tokens of KV. 0 and 1 both mean "one slot"; normalize
		// so context_per_request never divides by zero and clients compact on
		// the right number. See the §"per-request context" note in the plan.
		resolvePar := func(v int) int {
			if v < 1 {
				return 1
			}
			return v
		}
		liveParallel := resolvePar(live.Parallel)
		configuredParallel := resolvePar(cfg.Parallel)

		configMap := map[string]any{
			"enabled":             live.Enabled,
			"gpu_layers":          live.GPULayers,
			"context_size":        liveCtx,
			"parallel":            liveParallel,
			"context_per_request": liveCtx / liveParallel,
			"threads":             live.Threads,
			"flash_attention":     live.FlashAttention,
		}
		// Optional string fields: include the key when either side has a
		// value, so an edit that *clears* the field still shows up as a
		// pending change rather than silently disappearing.
		if live.TensorSplit != "" || cfg.TensorSplit != "" {
			configMap["tensor_split"] = live.TensorSplit
		}
		if live.KVCacheQuant != "" || cfg.KVCacheQuant != "" {
			configMap["kv_cache_quant"] = live.KVCacheQuant
		}
		if live.MmprojPath != "" || cfg.MmprojPath != "" {
			configMap["mmproj_path"] = live.MmprojPath
		}

		if hasSnapshot {
			if cfg.Enabled != live.Enabled {
				configMap["enabled_pending"] = cfg.Enabled
			}
			if cfg.GPULayers != live.GPULayers {
				configMap["gpu_layers_pending"] = cfg.GPULayers
			}
			if liveCtx != configuredCtx {
				configMap["context_size_pending"] = configuredCtx
			}
			if liveParallel != configuredParallel {
				configMap["parallel_pending"] = configuredParallel
				configMap["context_per_request_pending"] = configuredCtx / configuredParallel
			} else if liveCtx != configuredCtx {
				configMap["context_per_request_pending"] = configuredCtx / liveParallel
			}
			if cfg.Threads != live.Threads {
				configMap["threads_pending"] = cfg.Threads
			}
			if cfg.FlashAttention != live.FlashAttention {
				configMap["flash_attention_pending"] = cfg.FlashAttention
			}
			if cfg.TensorSplit != live.TensorSplit {
				configMap["tensor_split_pending"] = cfg.TensorSplit
			}
			if cfg.KVCacheQuant != live.KVCacheQuant {
				configMap["kv_cache_quant_pending"] = cfg.KVCacheQuant
			}
			if cfg.MmprojPath != live.MmprojPath {
				configMap["mmproj_path_pending"] = cfg.MmprojPath
			}
		}
		info["config"] = configMap
	}

	respondJSON(w, info)
}

func (s *Server) handleDeleteModel(w http.ResponseWriter, r *http.Request) {
	id := s.registry.ResolveID(chi.URLParam(r, "id"))

	var err error
	if r.URL.Query().Get("keep_files") == "true" {
		err = s.registry.Remove(id)
	} else {
		err = s.registry.Delete(id)
	}
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}

	// Regenerate preset INI so the router doesn't reference a deleted model
	if _, err := s.registry.WritePresetINI(s.activeBackend()); err != nil {
		slog.Warn("failed to regenerate preset INI after delete", "error", err)
	}

	if isHTMX(r) {
		w.Header().Set("HX-Trigger", `{"gpuMapChanged":true,"modelsChanged":true}`)
		w.WriteHeader(http.StatusOK)
		return
	}

	w.WriteHeader(http.StatusNoContent)
}

// routerKnownStates queries the router for all known models and returns a map
// from {model ID, name, alias} → status value. Empty map if the router is down.
func (s *Server) routerKnownStates() map[string]string {
	routerKnown := make(map[string]string)
	routerModels, err := s.process.ListModels()
	if err != nil {
		return routerKnown
	}
	for _, rm := range routerModels {
		routerKnown[rm.ID] = rm.Status.Value
		if rm.Model != "" {
			routerKnown[rm.Model] = rm.Status.Value
		}
		for _, alias := range rm.Aliases {
			routerKnown[alias] = rm.Status.Value
		}
	}
	return routerKnown
}

// routerStateFor returns the status for the first of names present in a
// routerKnownStates() map, and whether any of the names was found.
func routerStateFor(states map[string]string, names ...string) (string, bool) {
	for _, n := range names {
		if st, ok := states[n]; ok {
			return st, true
		}
	}
	return "", false
}

// renderModelCard writes one model_card partial.
func (s *Server) renderModelCard(w http.ResponseWriter, m *models.Model, routerKnown map[string]string, isOrphan, isIncomplete bool) {
	// Look up router state under any of the names the router might know this
	// model by. The router's primary ID is the auto-discovery section name
	// (RouterName), but it may also surface m.ID or PublicName via aliases.
	// Falling back through all three avoids a false-positive "restart needed"
	// indicator when the router does know about the model under a different key.
	routerName := s.registry.RouterName(m.ID)
	state := routerKnown[routerName]
	if state == "" {
		state = routerKnown[m.ID]
	}
	if state == "" {
		state = routerKnown[m.PublicName()]
	}

	vramGB := models.BytesToGiB(m.SizeBytes) + 0.2
	enabled := true
	hasVision := m.HasBuiltinVision
	gpuLabel := ""
	var aliases []string
	if cfg, err := s.registry.GetConfig(m.ID); err == nil {
		vramGB = models.VRAMEstimateForConfigOn(m, cfg, models.DeviceCountForConfig(cfg, len(s.monitor.Current().GPU)))
		enabled = cfg.Enabled
		if cfg.MmprojPath != "" {
			hasVision = true
		}
		if cfg.GPUAssign != "" && cfg.GPUAssign != "all" {
			gpuLabel = models.GPUAssignLabel(cfg.GPUAssign)
		}
		aliases = cfg.Aliases
	}

	pendingEnable := enabled && state == "" && s.process.IsRunning()
	pendingDisable := !enabled && state != "" && s.process.IsRunning()
	configChanged := s.isDirty(m.ID) && state != "" && s.process.IsRunning()

	org, base := m.OrgAndBase()
	searchText := strings.ToLower(strings.Join([]string{
		org, base, m.Quant, m.PublicName(), m.ModelID, m.Arch,
		strings.Join(aliases, " "),
	}, " "))

	data := struct {
		models.Model
		IsActive       bool
		IsEnabled      bool
		PendingEnable  bool
		PendingDisable bool
		NeedsReload    bool
		HasVision      bool
		GPULabel       string
		ServiceState   string
		VRAMGB         float64
		IsOrphan       bool
		IsIncomplete   bool
		ResumeFilename string
		SearchText     string
	}{
		Model:          *m,
		IsActive:       state == "loaded" || state == "loading",
		IsEnabled:      enabled,
		PendingEnable:  pendingEnable,
		PendingDisable: pendingDisable,
		NeedsReload:    configChanged,
		HasVision:      hasVision,
		GPULabel:       gpuLabel,
		ServiceState:   state,
		VRAMGB:         vramGB,
		IsOrphan:       isOrphan,
		IsIncomplete:   isIncomplete,
		ResumeFilename: m.Filename,
		SearchText:     searchText,
	}
	s.renderPartial(w, "model_card", data)
}

// renderModelList renders the shared model list used by both chat and embedding
// sections as a flat list of cards, sorted by base name then quant. When
// withFilter is true, a filter input is emitted above the list.
func (s *Server) renderModelList(w http.ResponseWriter, r *http.Request, modelList []*models.Model, pending []models.PendingConfig, withFilter bool) {
	routerKnown := s.routerKnownStates()

	orphanSet := make(map[string]bool)
	for _, m := range s.registry.FindOrphans() {
		orphanSet[m.ID] = true
	}

	incompleteSet := make(map[string]bool)
	for _, m := range s.registry.IncompleteRegistered() {
		incompleteSet[m.ID] = true
	}

	// Merge installed models and pending ghost entries into one sequence
	// sorted by the same (base, quant) key, so a ghost sits next to
	// installed quants of its base model or stands alone for an absent
	// one. OrgAndBase is pure ModelID text, so a stub Model computes the
	// key for pending entries.
	type listEntry struct {
		base, quant string
		m           *models.Model
		p           *models.PendingConfig
	}
	entries := make([]listEntry, 0, len(modelList)+len(pending))
	for _, m := range modelList {
		_, base := m.OrgAndBase()
		entries = append(entries, listEntry{base: strings.ToLower(base), quant: strings.ToLower(m.Quant), m: m})
	}
	for i := range pending {
		p := &pending[i]
		_, base := (&models.Model{ModelID: p.ModelID}).OrgAndBase()
		entries = append(entries, listEntry{base: strings.ToLower(base), quant: strings.ToLower(p.Quant), p: p})
	}
	sort.SliceStable(entries, func(i, j int) bool {
		if entries[i].base != entries[j].base {
			return entries[i].base < entries[j].base
		}
		return entries[i].quant < entries[j].quant
	})

	if withFilter {
		w.Write([]byte(`<div class="model-list-controls"><input type="search" class="model-filter" placeholder="Filter by name, quant, architecture…" oninput="filterModels(this.value)" autocomplete="off"></div>`))
	}

	w.Write([]byte(`<div class="model-card-list">`))
	w.Write([]byte(`<div class="model-card-header"><span></span><span>Model</span><span>Quant</span><span title="Estimated VRAM at the configured context size">VRAM Est.</span><span>Size</span><span></span></div>`))
	for _, e := range entries {
		if e.m != nil {
			s.renderModelCard(w, e.m, routerKnown, orphanSet[e.m.ID], incompleteSet[e.m.ID])
		} else {
			s.renderPendingCard(w, e.p)
		}
	}
	w.Write([]byte(`</div>`))
}

// renderPendingCard renders a ghost card for a backup-imported config
// whose model isn't installed yet: greyed, identity + imported date, a
// Download button (inline response mode — the table-shaped progress
// partial can't live in a card) and a Discard button whose 204 +
// HX-Trigger response refreshes the whole listing.
func (s *Server) renderPendingCard(w http.ResponseWriter, p *models.PendingConfig) {
	ident := domID(p.ModelID + "-" + p.Quant)
	fmt.Fprintf(w, `<article class="model-card" style="opacity:0.6;" id="pending-card-%s" data-search="%s">
	<div class="model-card-row">
		<div class="model-card-toggle" title="Not installed — this is a config imported from a backup, waiting for its model."><span>&#x23F3;</span></div>
		<div class="model-card-name">%s
			<small style="display:block;color:var(--pico-muted-color);">not installed &mdash; config waiting (imported %s)</small>
			<span id="pending-dl-%s"></span>
		</div>
		<div>%s</div>
		<div>&mdash;</div>
		<div>&mdash;</div>
		<div style="display:flex;gap:0.4rem;justify-content:flex-end;">
			<button type="button" class="outline" style="padding:0.1rem 0.6rem;font-size:0.8em;margin:0;"
			        title="Download %s from HuggingFace; the waiting config attaches automatically when it arrives."
			        hx-post="/api/hf/download"
			        hx-vals='{"model_id": %q, "filename": %q, "inline": "1"}'
			        hx-target="#pending-dl-%s" hx-swap="innerHTML" hx-disabled-elt="this">Download</button>
			<button type="button" class="outline secondary" style="padding:0.1rem 0.6rem;font-size:0.8em;margin:0;"
			        title="Discard this waiting config."
			        hx-post="/api/backup/pending/discard"
			        hx-vals='{"model_id": %q, "quant": %q}'
			        hx-swap="none">Discard</button>
		</div>
	</div>
</article>`,
		ident, html.EscapeString(strings.ToLower(p.ModelID+" "+p.Quant+" pending")),
		html.EscapeString(p.ModelID), p.SavedAt.Format("2006-01-02"), ident,
		html.EscapeString(p.Quant),
		html.EscapeString(p.Filename),
		p.ModelID, p.Filename, ident,
		p.ModelID, p.Quant)
}
