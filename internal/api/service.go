package api

import (
	"encoding/json"
	"fmt"
	"html"
	"log/slog"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/process"
)

// applySpecDefaults resets speculative decoding parameters to recommended
// values for the selected mode. Call this only on a mode *change* — calling
// it on every save would clobber user-tuned values within an existing mode
// (the form parser already loaded them from the request into cfg).
func applySpecDefaults(cfg *models.ModelConfig) {
	// Zero everything, then apply the mode's recommended defaults from
	// the shared table — the same one the benchmark job form renders, so
	// the two surfaces cannot disagree.
	cfg.DraftMax = 0
	cfg.DraftMin = 0
	cfg.DraftPMin = ""
	cfg.NgramSizeN = 0
	cfg.NgramSizeM = 0
	for _, p := range models.SpecModeParams(cfg.SpecType) {
		switch p.Key {
		case "draft_max":
			cfg.DraftMax, _ = strconv.Atoi(p.Default)
		case "draft_min":
			cfg.DraftMin, _ = strconv.Atoi(p.Default)
		case "draft_p_min":
			cfg.DraftPMin = p.Default
		case "ngram_size_n":
			cfg.NgramSizeN, _ = strconv.Atoi(p.Default)
		case "ngram_size_m":
			cfg.NgramSizeM, _ = strconv.Atoi(p.Default)
		}
	}
}

// parseOptionalFloat returns a *float64 if s is non-empty and valid, else nil.
func parseOptionalFloat(s string) *float64 {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil
	}
	v, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return nil
	}
	return &v
}

// parseOptionalInt returns a *int if s is non-empty and valid, else nil.
func parseOptionalInt(s string) *int {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil
	}
	v, err := strconv.Atoi(s)
	if err != nil {
		return nil
	}
	return &v
}

// countNonZeroSplit counts the non-zero comma-separated entries in a
// tensor-split string. Used to derive GPU count from legacy configs.
func countNonZeroSplit(s string) int {
	if s == "" {
		return 0
	}
	n := 0
	for _, p := range strings.Split(s, ",") {
		p = strings.TrimSpace(p)
		if p != "" && p != "0" && p != "0.0" {
			n++
		}
	}
	return n
}

// resolveRouterModel finds the registry model (and config) behind a router
// entry, matching the router-section ID first, then each alias. The router's
// primary ID may not equal the registry ID, so both are tried. Shared by
// handlePS and renderLoadedModelsJSON.
func (s *Server) resolveRouterModel(rm process.ModelStatus) (*models.Model, *models.ModelConfig) {
	reg, cfg := s.findModelByAny(rm.ID)
	if reg == nil {
		for _, a := range rm.Aliases {
			if reg, cfg = s.findModelByAny(a); reg != nil {
				break
			}
		}
	}
	return reg, cfg
}

// handlePS returns currently loaded models with resource info, similar to Ollama's /api/ps.
func (s *Server) handlePS(w http.ResponseWriter, r *http.Request) {
	type psModel struct {
		Name        string  `json:"name"`
		Status      string  `json:"status"`
		VRAMEstGB   float64 `json:"vram_est_gb,omitempty"`
		ContextSize int     `json:"context_size,omitempty"`
		Arch        string  `json:"arch,omitempty"`
	}

	var result []psModel

	if s.process.IsRunning() {
		routerModels, err := s.process.ListModels()
		if err == nil {
			for _, rm := range routerModels {
				pm := psModel{
					Name:   rm.ID,
					Status: rm.Status.Value,
				}
				// Enrich with registry metadata if available
				if regModel, cfg := s.resolveRouterModel(rm); regModel != nil {
					pm.Arch = regModel.Arch
					pm.VRAMEstGB = regModel.VRAMEstGB
					if cfg != nil {
						pm.ContextSize = cfg.ContextSize
						pm.VRAMEstGB = models.VRAMEstimateForConfig(regModel, cfg)
					}
				}
				result = append(result, pm)
			}
		}
	}

	respondJSON(w, map[string]any{"models": result})
}

func (s *Server) handleServiceStatus(w http.ResponseWriter, r *http.Request) {
	status := s.process.GetStatus()

	if isHTMX(r) {
		respondHTML(w)
		var badge string
		switch status.State {
		case process.StateRunning:
			badge = `<ins>Running</ins>`
		case process.StateStarting:
			badge = `<mark>Starting...</mark>`
		case process.StateFailed:
			badge = fmt.Sprintf(`<del>Failed</del> <small style="color:var(--pico-del-color)">%s</small>`, status.Error)
		default:
			badge = `Stopped`
		}
		if status.Uptime != "" {
			badge += fmt.Sprintf(` <small>(%s)</small>`, status.Uptime)
		}
		fmt.Fprint(w, badge)
		return
	}

	respondJSON(w, status)
}

// routerBusyWithJob reports whether a benchmark job is currently running
// and therefore driving the router. Interactive start/stop/restart
// refuses rather than fighting it: restarting mid-cell would leave that
// cell measuring something other than what it reports.
//
// Deliberately sourced from the queue rather than from jobEnv's
// ownership flag. That flag is sticky by design — a failed restore
// leaves it set so cleanup knows work is still owed — and using it here
// meant one failed restore locked the user out of starting the router
// at all, with no job to cancel.
func (s *Server) routerBusyWithJob() bool {
	// Both signals, because each is blind alone. The queue's view goes
	// false if the running job's record is deleted mid-run, which would
	// disarm every guard while the job kept driving the router. jobEnv's
	// ownership covers that window — and it can no longer strand the
	// guards, because cleanup releases it unconditionally, including when
	// the restore restart fails.
	if s.jobs != nil {
		if _, running := s.jobs.Status(); running {
			return true
		}
	}
	if env, ok := s.jobEnv(); ok {
		return env.routerOwnedByJob()
	}
	return false
}

func (s *Server) handleServiceStart(w http.ResponseWriter, r *http.Request) {
	if s.routerBusyWithJob() {
		slog.Info("refused router action: a benchmark job is using the router", "path", r.URL.Path)
		http.Error(w, "a benchmark job is currently running the router — cancel it first", http.StatusConflict)
		return
	}
	if !s.process.IsRunning() {
		if err := s.startRouter(); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
	}
	s.handleServiceStatus(w, r)
}

// Stop is deliberately NOT guarded against a running job. It is the
// user's only way out of a hung cell holding all the VRAM, and cleanup
// already handles finding the router stopped. Guards belong on the
// actions that *start* the router, because those are the ones that can
// put a cell on the wrong config while it reports another.
func (s *Server) handleServiceStop(w http.ResponseWriter, r *http.Request) {
	if err := s.process.Stop(); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	// Nothing is live anymore — drop the snapshot so /info falls back to
	// reporting just the configured value.
	s.setRunningConfigs(make(map[string]*models.ModelConfig))
	s.handleServiceStatus(w, r)
}

func (s *Server) handleServiceRestart(w http.ResponseWriter, r *http.Request) {
	if s.routerBusyWithJob() {
		slog.Info("refused router action: a benchmark job is using the router", "path", r.URL.Path)
		http.Error(w, "a benchmark job is currently running the router — cancel it first", http.StatusConflict)
		return
	}
	// Stop + startRouter rather than process.Restart(): the latter relaunches
	// with the RouterConfig captured at the original Start, so a build
	// switched in the UI between Start and Restart would be ignored. Going
	// through startRouter re-resolves cfg.ActiveBuild and rewrites the
	// preset INI from the current registry state.
	if s.process.IsRunning() {
		if err := s.process.Stop(); err != nil {
			slog.Debug("stop during restart", "error", err)
		}
	}
	if err := s.startRouter(); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	s.handleServiceStatus(w, r)
}

func (s *Server) handleServiceLogs(w http.ResponseWriter, r *http.Request) {
	ch := s.process.Subscribe()
	defer s.process.Unsubscribe(ch)
	StreamLines(w, r.Context(), ch, "Router exited")
}

func (s *Server) handleServiceLogsClear(w http.ResponseWriter, r *http.Request) {
	s.process.ClearLogs()
	w.WriteHeader(http.StatusNoContent)
}

// handleLoadedModels returns a list of models known to the router. HTMX
// requests get an HTML fragment for the server page; everything else gets
// a JSON payload suitable for programmatic clients.
func (s *Server) handleLoadedModels(w http.ResponseWriter, r *http.Request) {
	if isHTMX(r) {
		s.renderLoadedModelsHTML(w)
		return
	}
	s.renderLoadedModelsJSON(w)
}

// renderLoadedModelsJSON returns the loaded-models list as JSON.
// Each entry surfaces every name a client might use to address the model
// (router section ID, OpenAI public name, registry ID, aliases) so callers
// can correlate with /v1/models without parsing HTML glyphs.
func (s *Server) renderLoadedModelsJSON(w http.ResponseWriter) {
	type loadedModel struct {
		ID           string         `json:"id"`                    // router section ID (the name /models/load accepts)
		Status       string         `json:"status"`                // "loaded", "loading", "unloaded"
		Aliases      []string       `json:"aliases,omitempty"`     // every other name the router will accept
		PublicName   string         `json:"public_name,omitempty"` // canonical OpenAI-style ID (matches /v1/models)
		RegistryID   string         `json:"registry_id,omitempty"` // long internal ID (matches /api/models/)
		Capabilities map[string]any `json:"capabilities,omitempty"`
	}

	out := struct {
		SchemaVersion int           `json:"schema_version"`
		Running       bool          `json:"running"`
		Models        []loadedModel `json:"models"`
	}{SchemaVersion: CapabilitiesSchemaVersion, Models: []loadedModel{}}

	if !s.process.IsRunning() {
		respondJSON(w, out)
		return
	}
	out.Running = true

	routerModels, err := s.process.ListModels()
	if err != nil {
		slog.Debug("failed to list router models", "error", err)
		respondJSON(w, out)
		return
	}

	for _, rm := range routerModels {
		entry := loadedModel{
			ID:      rm.ID,
			Status:  rm.Status.Value,
			Aliases: rm.Aliases,
		}
		// Find the registry model behind this router entry so we can surface
		// its canonical IDs and capability block. Folding capabilities in
		// here lets a client auto-configure from this single request instead
		// of fanning out to /info per model.
		reg, cfg := s.resolveRouterModel(rm)
		if reg != nil {
			entry.RegistryID = reg.ID
			entry.PublicName = reg.PublicName()
			entry.Capabilities = s.buildCapabilities(reg, cfg)
		}
		out.Models = append(out.Models, entry)
	}

	respondJSON(w, out)
}

// renderLoadedModelsHTML returns the htmx fragment for the server page.
func (s *Server) renderLoadedModelsHTML(w http.ResponseWriter) {
	respondHTML(w)
	if !s.process.IsRunning() {
		return
	}

	routerModels, err := s.process.ListModels()
	if err != nil {
		slog.Debug("failed to list router models", "error", err)
		return
	}
	if len(routerModels) == 0 {
		return
	}

	fmt.Fprint(w, `<div style="margin-top: 0.5rem;"><small><strong>Models:</strong></small>`)
	for _, m := range routerModels {
		name := html.EscapeString(m.ID)
		switch m.Status.Value {
		case "loaded":
			fmt.Fprintf(w, `<br><small>&nbsp;&nbsp;● %s</small>`, name)
		case "loading":
			fmt.Fprintf(w, `<br><small>&nbsp;&nbsp;● %s <mark style="padding:0 0.2rem;">loading</mark></small>`, name)
		default: // "unloaded" or empty
			fmt.Fprintf(w, `<br><small>&nbsp;&nbsp;○ %s</small>`, name)
		}
	}
	fmt.Fprint(w, `</div>`)
}

func (s *Server) handleServiceLogTabs(w http.ResponseWriter, r *http.Request) {
	// With the native router, logs are combined — no tabs needed.
	// Return empty to hide the tab bar.
	respondHTML(w)
}

func (s *Server) handleServiceHealth(w http.ResponseWriter, r *http.Request) {
	healthy := s.process.CheckHealth()
	respondJSON(w, map[string]bool{"healthy": healthy})
}

// handleActivateModel loads a model via the router.
func (s *Server) handleActivateModel(w http.ResponseWriter, r *http.Request) {
	id := s.registry.ResolveID(chi.URLParam(r, "id"))

	// Third startRouter caller, and the easiest to miss: it fires only
	// when the router happens to be down, which includes the window
	// between Stop and Start inside a job's own restart. Starting there
	// would launch the user's preset under a job, so the cell would
	// measure saved config while reporting the override.
	if s.routerBusyWithJob() {
		slog.Info("refused router action: a benchmark job is using the router", "path", r.URL.Path)
		http.Error(w, "a benchmark job is currently running the router — cancel it first", http.StatusConflict)
		return
	}

	// Ensure router is running
	if !s.process.IsRunning() {
		if err := s.startRouter(); err != nil {
			http.Error(w, "Failed to start router: "+err.Error(), http.StatusInternalServerError)
			return
		}
		// Wait for router to be ready (up to 10 seconds)
		for i := 0; i < 20; i++ {
			time.Sleep(500 * time.Millisecond)
			if s.process.CheckHealth() {
				break
			}
		}
		if !s.process.IsRunning() {
			http.Error(w, "Router failed to start", http.StatusInternalServerError)
			return
		}
	}

	// Try router name, fall back to file path
	routerName := s.registry.RouterName(id)
	if err := s.process.LoadModel(routerName); err != nil {
		filePath := s.registry.ModelFilePath(id)
		if filePath == "" || s.process.LoadModel(filePath) != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
	}

	if isHTMX(r) {
		s.handleListModels(w, r)
		return
	}

	respondJSON(w, map[string]string{"status": "loading", "model": id})
}

// handleDeactivateModel unloads a model via the router.
func (s *Server) handleDeactivateModel(w http.ResponseWriter, r *http.Request) {
	id := s.registry.ResolveID(chi.URLParam(r, "id"))

	// Unloading the model a cell is measuring corrupts that measurement,
	// which makes this more disruptive than Activate — it was left
	// unguarded while Activate was not.
	if s.routerBusyWithJob() {
		slog.Info("refused model action: a benchmark job is using the router", "path", r.URL.Path)
		http.Error(w, "a benchmark job is currently using the router — cancel it first", http.StatusConflict)
		return
	}

	if err := s.process.UnloadModel(s.registry.RouterName(id)); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	if isHTMX(r) {
		s.handleListModels(w, r)
		return
	}

	w.WriteHeader(http.StatusNoContent)
}

// routerOptions describes what one particular router start should run.
//
// The zero value means "the user's saved build and config", which is
// what every interactive path wants. A benchmark job passes its own
// build and config explicitly.
//
// This used to be ambient server state that startRouter consulted, and
// that design was the root of a family of bugs: a user hitting Restart
// during a job got the benchmark's config, the job's build selection was
// persisted to disk, and the launch snapshot had to be suppressed by a
// flag. Making it a parameter means an interactive start structurally
// cannot pick up a job's settings.
type routerOptions struct {
	// buildID overrides which build to launch. Empty uses the user's
	// saved selection. Deliberately not persisted: a job's build choice
	// is transient, so a crash mid-job can't rewrite the user's config.
	buildID string
	// overrides substitutes model configs for this start only. Nil uses
	// the user's saved config, written to the normal preset.
	overrides map[string]*models.ModelConfig
}

// startRouter starts the llama-server on the user's saved build and
// config. Interactive callers use this.
func (s *Server) startRouter() error {
	return s.startRouterWith(routerOptions{})
}

// startRouterWith starts the llama-server under the given options.
func (s *Server) startRouterWith(opt routerOptions) error {
	// Read config under the lock: the benchmark queue and the settings
	// handler both write these fields from other goroutines.
	s.cfgMu.Lock()
	buildID := opt.buildID
	if buildID == "" {
		buildID = s.cfg.ActiveBuild
	}
	modelsMax := s.cfg.ModelsMax
	port := s.cfg.LlamaPort
	extraEnv := s.cfg.RuntimeEnvPairs()
	s.cfgMu.Unlock()

	// Find the build binary. Explicit selection wins; otherwise fall back
	// to the successful build with the newest GitRef.
	build := s.resolveBuild(buildID)
	if build == nil || build.BinaryPath == "" {
		return fmt.Errorf("no compiled build available — build llama.cpp first")
	}

	// A benchmark start writes its substitute config to a separate preset
	// file; everything else regenerates the user's.
	var presetPath string
	var err error
	if opt.overrides != nil {
		presetPath, err = s.registry.WriteBenchPresetINI(opt.overrides, buildBackend(build))
		if err != nil {
			// Falling back to the saved preset would silently benchmark
			// the wrong config — the exact failure this mechanism exists
			// to prevent. Refuse to start instead.
			return fmt.Errorf("write benchmark preset: %w", err)
		}
	} else {
		presetPath, err = s.registry.WritePresetINI(buildBackend(build))
		if err != nil {
			slog.Warn("failed to write preset INI", "error", err)
		}
	}

	overridden := make([]string, 0, len(opt.overrides))
	for id := range opt.overrides {
		overridden = append(overridden, id)
	}
	sort.Strings(overridden)

	slog.Info("starting router",
		"build", build.ID,
		"preset", presetPath,
		"for_benchmark", opt.overrides != nil,
		"substitute_configs", overridden,
		"env", extraEnv,
	)

	if err := s.process.Start(process.RouterConfig{
		BinaryPath: build.BinaryPath,
		PresetPath: presetPath,
		ModelsMax:  modelsMax,
		Port:       port,
		ExtraEnv:   extraEnv,
	}); err != nil {
		return err
	}

	// Record which build is actually serving, so callers can tell the
	// user's saved selection from what a job temporarily launched.
	s.setRunningBuild(build.ID)

	// Snapshot each model's launch config so /api/models/{id}/info can
	// report live values for restart-requiring fields vs. a subsequent
	// edit. Value-copy (not a shared pointer) because
	// handleUpdateModelConfig mutates the registry's config struct in
	// place.
	//
	// Built from the configs this start actually used, substitutes
	// included. Returning early for override starts left the previous
	// snapshot in place, so /info reported the user's saved values as
	// live while llama-server ran the job's — the same "recorded config
	// is a lie" failure this mechanism exists to prevent.
	snapshot := make(map[string]*models.ModelConfig)
	for _, m := range s.registry.List() {
		if sub, ok := opt.overrides[m.ID]; ok && sub != nil {
			cp := *sub
			snapshot[m.ID] = &cp
			continue
		}
		cfg, err := s.registry.GetConfig(m.ID)
		if err != nil {
			continue
		}
		cp := *cfg
		snapshot[m.ID] = &cp
	}
	s.setRunningConfigs(snapshot)

	// Pending-reload badges only make sense against the user's own
	// config. A job's start doesn't apply the user's edits, so it must
	// not clear them.
	if opt.overrides == nil {
		s.clearDirty()
	}
	return nil
}

// handleModelEnable toggles a model's enabled state and updates the preset.
func (s *Server) handleModelEnable(w http.ResponseWriter, r *http.Request) {
	id := s.registry.ResolveID(chi.URLParam(r, "id"))

	r.ParseForm()
	enabled := r.FormValue("enabled") == "true"

	cfg, err := s.registry.GetConfig(id)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}

	cfg.Enabled = enabled
	if err := s.registry.SetConfig(id, cfg); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Regenerate preset so the router picks up the change
	if _, err := s.registry.WritePresetINI(s.activeBackend()); err != nil {
		slog.Warn("failed to regenerate preset INI", "error", err)
	}

	// The router reads preset.ini at startup only. Changing which models
	// are available requires a restart. If the router is running, mark
	// that a restart is needed so the UI can show an indicator.
	// Note: /models/load and /models/unload control VRAM, not the
	// available list, so we don't call them here.

	if isHTMX(r) {
		// Re-render only the toggled row so the rest of the list keeps its
		// expand/collapse state. Initial=false omits display:none so the new
		// row stays visible (the user's click proves the row was visible).
		m, err := s.registry.Get(id)
		if err != nil {
			http.Error(w, err.Error(), http.StatusNotFound)
			return
		}
		isOrphan := false
		for _, om := range s.registry.FindOrphans() {
			if om.ID == id {
				isOrphan = true
				break
			}
		}
		isIncomplete := false
		for _, im := range s.registry.IncompleteRegistered() {
			if im.ID == id {
				isIncomplete = true
				break
			}
		}
		w.Header().Set("HX-Trigger-After-Swap", `{"gpuMapChanged":true}`)
		respondHTML(w)
		s.renderModelCard(w, m, s.routerKnownStates(), isOrphan, isIncomplete)
		return
	}

	respondJSON(w, map[string]bool{"enabled": enabled})
}

// handleModelVRAMEstimate returns a VRAM estimate for a model with given config params.
func (s *Server) handleModelVRAMEstimate(w http.ResponseWriter, r *http.Request) {
	id := s.registry.ResolveID(chi.URLParam(r, "id"))

	model, err := s.registry.Get(id)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}

	r.ParseForm()
	contextSize, _ := strconv.Atoi(r.FormValue("context_size"))
	kvCacheQuant := r.FormValue("kv_cache_quant")

	// Start from the saved config so the estimate includes the model's
	// activated auxiliary files (mmproj / MTP head / draft), then apply the
	// context and cache-type the user is currently adjusting. Copy so we don't
	// mutate the registry's live config.
	cfg := &models.ModelConfig{}
	if saved, err := s.registry.GetConfig(id); err == nil && saved != nil {
		*cfg = *saved
	}
	cfg.ContextSize = contextSize
	cfg.KVCacheQuant = kvCacheQuant

	total := models.VRAMEstimateForConfig(model, cfg)
	kvGB := model.KVCacheGB(contextSize, kvCacheQuant)
	weightsGB := models.BytesToGiB(model.SizeBytes)
	extraGB := models.AuxFilesVRAMGB(cfg)

	if isHTMX(r) {
		respondHTML(w)
		extra := ""
		if extraGB > 0.05 {
			extra = fmt.Sprintf(" + aux files: %.1f GiB", extraGB)
		}
		fmt.Fprintf(w, `<strong>%.1f GiB</strong> <small>(weights: %.1f GiB + KV cache: %.1f GiB%s + overhead)</small>`,
			total, weightsGB, kvGB, extra)
		return
	}

	respondJSON(w, map[string]any{
		"total_gb":     total,
		"weights_gb":   weightsGB,
		"kv_cache_gb":  kvGB,
		"aux_files_gb": extraGB,
	})
}

// handleGetModelConfig returns the launch config for a model.
func (s *Server) handleGetModelConfig(w http.ResponseWriter, r *http.Request) {
	id := s.registry.ResolveID(chi.URLParam(r, "id"))

	cfg, err := s.registry.GetConfig(id)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}

	model, _ := s.registry.Get(id)

	if isHTMX(r) {
		respondHTML(w)

		maxContext := 0
		detectedMMProj := ""
		detectedMTP := ""
		isEmbedding := false
		var draftCandidates []models.DraftCandidate
		if model != nil {
			maxContext = model.ContextLength
			detectedMMProj = models.FindMMProj(model.FilePath)
			detectedMTP = models.FindMTP(model.FilePath)
			isEmbedding = model.IsEmbedding()
			if !isEmbedding {
				draftCandidates = s.registry.FindDraftCandidates(id)
			}
		}

		hasBuiltinVision := model != nil && model.HasBuiltinVision

		// GPU assignment options
		metrics := s.monitor.Current()
		numGPUs := len(metrics.GPU)
		gpuOptions := models.GPUAssignOptions(numGPUs, igpuFlags(metrics.GPU))

		// Migration: map legacy and pre-iGPU-audit configs onto the
		// current dropdown values.
		migrateGPUAssign(cfg, gpuOptions, numGPUs)

		// Mark disabled/recommended options
		if numGPUs > 0 && model != nil {
			perGPUGB := float64(metrics.GPU[0].VRAMTotalMB) / 1024.0
			modelVRAM := models.VRAMEstimateForConfig(model, cfg)
			allModels := s.registry.List()
			allConfigs := make(map[string]*models.ModelConfig)
			for _, m := range allModels {
				if c, err := s.registry.GetConfig(m.ID); err == nil {
					allConfigs[m.ID] = c
				}
			}
			existing := models.ComputeAllocations(allModels, allConfigs, numGPUs)
			// Exclude the current model from existing allocations
			var filtered []models.GPUAllocation
			for _, a := range existing {
				if a.ModelID != id {
					filtered = append(filtered, a)
				}
			}
			models.MarkRecommended(gpuOptions, modelVRAM, perGPUGB, filtered)
		}

		var samplingPresets []models.SamplingPreset
		var samplingPresetsJSON string
		var hasEmbeddedDefault bool
		if model != nil && !isEmbedding {
			samplingPresets = model.EffectiveSamplingPresets()
			if len(samplingPresets) > 0 {
				if b, err := json.Marshal(samplingPresets); err == nil {
					samplingPresetsJSON = string(b)
				}
			}
			for _, p := range samplingPresets {
				if p.Source == "gguf" {
					hasEmbeddedDefault = true
					break
				}
			}
		}

		data := struct {
			ModelID             string
			Config              *models.ModelConfig
			EffectiveFlags      string
			MaxContext          int
			HasMMProj           bool
			HasMTP              bool
			HasBuiltinVision    bool
			IsEmbedding         bool
			DraftCandidates     []models.DraftCandidate
			GPUOptions          []models.GPUOption
			GPUAssignWarning    string
			NumGPUs             int
			SamplingPresets     []models.SamplingPreset
			SamplingPresetsJSON string
			HasEmbeddedDefault  bool
			HasPLE              bool
			PLESizeLabel        string
		}{
			ModelID:             id,
			Config:              cfg,
			EffectiveFlags:      cfg.EffectiveFlagsFor(isEmbedding, s.activeBackend()),
			MaxContext:          maxContext,
			HasMMProj:           cfg.MmprojPath != "" || detectedMMProj != "",
			HasMTP:              cfg.MtpPath != "" || detectedMTP != "",
			HasBuiltinVision:    hasBuiltinVision,
			IsEmbedding:         isEmbedding,
			DraftCandidates:     draftCandidates,
			GPUOptions:          gpuOptions,
			GPUAssignWarning:    s.gpuAssignWarning(cfg, metrics.GPU),
			NumGPUs:             numGPUs,
			SamplingPresets:     samplingPresets,
			SamplingPresetsJSON: samplingPresetsJSON,
			HasEmbeddedDefault:  hasEmbeddedDefault,
			// The per-layer embedding control is only meaningful for the
			// handful of architectures that carry such a table, so it is
			// rendered only when this model actually has one.
			HasPLE:       model != nil && model.PLEBytes > 0,
			PLESizeLabel: pleSizeLabel(model),
		}
		s.renderPartial(w, "model_config", data)
		return
	}

	respondJSON(w, cfg)
}

// handleUpdateModelConfig updates the launch config for a model.
func (s *Server) handleUpdateModelConfig(w http.ResponseWriter, r *http.Request) {
	id := s.registry.ResolveID(chi.URLParam(r, "id"))

	// Fetch existing config to preserve fields not in the form (e.g. Enabled).
	cfg, err := s.registry.GetConfig(id)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}

	if r.Header.Get("Content-Type") == "application/json" {
		if err := json.NewDecoder(r.Body).Decode(cfg); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
	} else {
		r.ParseForm()
		cfg.GPULayers, _ = strconv.Atoi(r.FormValue("gpu_layers"))

		// GPU assignment — single dropdown drives tensor-split, split-mode,
		// and main-gpu. "custom" preserves the raw tensor_split.
		gpuAssign := r.FormValue("gpu_assign")
		cfg.GPUAssign = gpuAssign
		numGPUs := len(s.monitor.Current().GPU)
		if gpuAssign == "custom" {
			cfg.TensorSplit = r.FormValue("tensor_split")
			cfg.SplitMode = ""
			cfg.MainGPU = 0
		} else {
			ts, sm, mg := models.ResolveGPUAssign(gpuAssign, numGPUs)
			cfg.TensorSplit = ts
			cfg.SplitMode = sm
			cfg.MainGPU = mg
		}
		cfg.ContextSize, _ = strconv.Atoi(r.FormValue("context_size"))
		cfg.Parallel, _ = strconv.Atoi(r.FormValue("parallel"))
		cfg.BatchSize, _ = strconv.Atoi(r.FormValue("batch_size"))
		cfg.UBatchSize, _ = strconv.Atoi(r.FormValue("ubatch_size"))
		cfg.Threads, _ = strconv.Atoi(r.FormValue("threads"))
		cfg.FlashAttention = r.FormValue("flash_attention") == "on"
		cfg.Jinja = r.FormValue("jinja") == "on"
		cfg.KVCacheQuant = r.FormValue("kv_cache_quant")
		cfg.DirectIO = r.FormValue("direct_io") == "on"
		// Anything other than the two explicit modes means auto, which is
		// stored as empty so the preset omits the flag entirely.
		switch v := r.FormValue("ple_mode"); v {
		case "on", "off":
			cfg.PLEMode = v
		default:
			cfg.PLEMode = ""
		}
		cfg.ExtraFlags = r.FormValue("extra_flags")

		// "__clear__" is the picker's action entry, not a preset name; it
		// never reaches here in normal flow (the JS resets the picker before
		// the save fires) but must not be stored if it does.
		if preset := r.FormValue("sampling_preset"); preset != "__clear__" {
			cfg.SamplingPreset = preset
		} else {
			cfg.SamplingPreset = ""
		}
		cfg.Temperature = parseOptionalFloat(r.FormValue("temperature"))
		cfg.TopP = parseOptionalFloat(r.FormValue("top_p"))
		cfg.TopK = parseOptionalInt(r.FormValue("top_k"))
		cfg.MinP = parseOptionalFloat(r.FormValue("min_p"))
		cfg.PresencePenalty = parseOptionalFloat(r.FormValue("presence_penalty"))
		cfg.RepeatPenalty = parseOptionalFloat(r.FormValue("repeat_penalty"))

		if r.Form.Has("mmproj_path") {
			cfg.MmprojPath = r.FormValue("mmproj_path")
			// Inverted storage: form sends mmproj_enabled=on when checked.
			// Default-false on a missing checkbox means "disabled", which is
			// exactly the unchecked state.
			cfg.MmprojDisabled = r.FormValue("mmproj_enabled") != "on"
		}
		if r.Form.Has("mtp_path") {
			cfg.MtpPath = r.FormValue("mtp_path")
			// Inverted storage, same as mmproj: form sends mtp_enabled=on when
			// checked; a missing checkbox means disabled.
			cfg.MtpDisabled = r.FormValue("mtp_enabled") != "on"
		}
		// Parse aliases (comma-separated, trimmed)
		if aliasStr := strings.TrimSpace(r.FormValue("aliases")); aliasStr != "" {
			var aliases []string
			for _, a := range strings.Split(aliasStr, ",") {
				a = strings.TrimSpace(a)
				if a != "" {
					aliases = append(aliases, a)
				}
			}
			cfg.Aliases = aliases
		} else {
			cfg.Aliases = nil
		}

		// Speculative decoding. Capture the previous SpecType so we can tell
		// whether the user just switched modes vs. is saving an existing one
		// — applySpecDefaults wipes user-tuned values, so we only want to
		// run it on a mode change.
		prevSpecType := cfg.SpecType
		cfg.SpecType = r.FormValue("spec_type")
		if r.Form.Has("draft_model_path") {
			cfg.DraftModelPath = r.FormValue("draft_model_path")
		}
		if v, err := strconv.Atoi(r.FormValue("draft_max")); err == nil && v > 0 {
			cfg.DraftMax = v
		} else {
			cfg.DraftMax = 0
		}
		if v, err := strconv.Atoi(r.FormValue("draft_min")); err == nil && v > 0 {
			cfg.DraftMin = v
		} else {
			cfg.DraftMin = 0
		}
		cfg.DraftPMin = r.FormValue("draft_p_min")
		if v, err := strconv.Atoi(r.FormValue("ngram_size_n")); err == nil && v > 0 {
			cfg.NgramSizeN = v
		} else {
			cfg.NgramSizeN = 0
		}
		if v, err := strconv.Atoi(r.FormValue("ngram_size_m")); err == nil && v > 0 {
			cfg.NgramSizeM = v
		} else {
			cfg.NgramSizeM = 0
		}

		// Draft model resource overrides (spec_type=draft only).
		if v, err := strconv.Atoi(r.FormValue("draft_ctx_size")); err == nil && v > 0 {
			cfg.DraftCtxSize = v
		} else {
			cfg.DraftCtxSize = 0
		}
		if v, err := strconv.Atoi(r.FormValue("draft_gpu_layers")); err == nil && v > 0 {
			cfg.DraftGPULayers = v
		} else {
			cfg.DraftGPULayers = 0
		}
		cfg.DraftDevice = strings.TrimSpace(r.FormValue("draft_device"))
		if v, err := strconv.Atoi(r.FormValue("draft_cpu_moe")); err == nil && v > 0 {
			cfg.DraftCPUMoE = v
		} else {
			cfg.DraftCPUMoE = 0
		}
		cfg.DraftKVCacheQuant = r.FormValue("draft_kv_cache_quant")

		// Populate recommended defaults only when the user actually switched
		// modes — preserves any custom values they tuned within an existing
		// mode (e.g. lowering ngram-mod's draft_min from 48 to 12).
		if cfg.SpecType != prevSpecType {
			applySpecDefaults(cfg)
		}
	}

	// Reject an unusable batch pair here rather than letting llama-server
	// clamp or fail at model load, where the cause is far less obvious.
	if err := cfg.ValidateBatchSizes(); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	if err := s.registry.SetConfig(id, cfg); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Regenerate preset INI so the router picks up changes on next load/reload
	if _, err := s.registry.WritePresetINI(s.activeBackend()); err != nil {
		slog.Warn("failed to regenerate preset INI", "error", err)
	}

	// Mark model as needing reload (config changed but model not reloaded yet).
	// Sampling params are injected at the proxy layer and don't need a reload.
	if cfg.Enabled && s.process.IsRunning() {
		s.markDirty(id)
	}

	// Update VRAM estimate in model list
	if isHTMX(r) {
		if model, err := s.registry.Get(id); err == nil {
			vramGB := models.VRAMEstimateForConfig(model, cfg)
			w.Header().Set("HX-Trigger", fmt.Sprintf(
				`{"vramUpdated":{"id":%q,"vram":"%.1f GiB"},"gpuMapChanged":true}`,
				id, vramGB))
		}
	}

	s.handleGetModelConfig(w, r)
}

// pleSizeLabel renders the per-layer embedding table's size for the model
// config form, e.g. "28.8 GB". Empty when the model has no such table.
func pleSizeLabel(m *models.Model) string {
	if m == nil || m.PLEBytes <= 0 {
		return ""
	}
	return models.FormatVRAM(models.BytesToGiB(m.PLEBytes))
}
