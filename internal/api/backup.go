package api

import (
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/backup"
	"github.com/tmac1973/llama-toolchest/internal/models"
)

// handleBackupExport serves the configuration backup as a JSON download.
// Same-origin UI route like the rest of /api (apiKeyAuth guards /v1
// only) — a secrets-bearing export has the same exposure as the existing
// settings API, which the help page states; secrets are included only on
// the explicit ?secrets=1 opt-in.
func (s *Server) handleBackupExport(w http.ResponseWriter, r *http.Request) {
	includeSecrets := r.URL.Query().Get("secrets") == "1"

	s.cfgMu.Lock()
	f := backup.Assemble(s.cfg, s.builder, s.registry, s.monitor.Current().GPU, includeSecrets)
	s.cfgMu.Unlock()

	data, err := f.Marshal()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Content-Disposition",
		fmt.Sprintf("attachment; filename=llama-toolchest-backup-%s.json", time.Now().Format("2006-01-02")))
	w.Write(data)
}

// restoreFileLimit bounds the uploaded backup; real backups are
// kilobytes, so 10 MB is generous.
const restoreFileLimit = 10 << 20

// handleRestore applies an uploaded backup file with merge semantics and
// an itemized report.
//
// Response contract — failures must be visible, and htmx doesn't swap
// non-2xx responses: htmx callers always get 200 with the report
// partial, refusals rendered as a report carrying only Error. Non-htmx
// clients get real status codes (400 structural/empty-selection, 409
// job-busy) with a JSON body.
func (s *Server) handleRestore(w http.ResponseWriter, r *http.Request) {
	fail := func(status int, msg string) {
		if isHTMX(r) {
			respondHTML(w)
			s.renderPartial(w, "restore_report", backup.Report{Error: msg})
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		fmt.Fprintf(w, `{"error": %q}`, msg)
	}

	// A restore mid-benchmark would change configs a running cell
	// reports as fixed — same guard as router start/restart.
	if s.routerBusyWithJob() {
		fail(http.StatusConflict, "a benchmark job is currently running — cancel it before restoring")
		return
	}

	if err := r.ParseMultipartForm(restoreFileLimit); err != nil {
		fail(http.StatusBadRequest, "invalid upload: "+err.Error())
		return
	}
	sel := backup.Selections{
		Settings:     r.FormValue("sec_settings") == "on",
		RuntimeEnv:   r.FormValue("sec_env") == "on",
		FlagPresets:  r.FormValue("sec_flags") == "on",
		ModelConfigs: r.FormValue("sec_models") == "on",
	}
	if sel.None() {
		fail(http.StatusBadRequest, "select at least one section to restore")
		return
	}

	file, _, err := r.FormFile("file")
	if err != nil {
		fail(http.StatusBadRequest, "no backup file in upload")
		return
	}
	defer file.Close()
	data, err := io.ReadAll(io.LimitReader(file, restoreFileLimit))
	if err != nil {
		fail(http.StatusBadRequest, "reading upload: "+err.Error())
		return
	}

	f, err := backup.Parse(data)
	if err != nil {
		fail(http.StatusBadRequest, err.Error())
		return
	}

	report := backup.Apply(f, sel, s.restoreDeps())

	if report.AppliedModelConfigs > 0 {
		if _, err := s.registry.WritePresetINI(s.activeBackend()); err != nil {
			report.Warnings = append(report.Warnings, "failed to regenerate preset: "+err.Error())
		}
	}

	if isHTMX(r) {
		respondHTML(w)
		s.renderPartial(w, "restore_report", report)
		return
	}
	respondJSON(w, report)
}

// restoreDeps wires the engine's collaborators to the live server. Every
// mutation goes through an existing single-source setter.
func (s *Server) restoreDeps() backup.Deps {
	return backup.Deps{
		ApplySettings: func(in backup.Settings) ([]string, error) {
			s.cfgMu.Lock()
			defer s.cfgMu.Unlock()
			var changed []string
			if in.ModelsMax != nil && s.cfg.ModelsMax != *in.ModelsMax {
				s.cfg.ModelsMax = *in.ModelsMax
				changed = append(changed, "models_max")
			}
			if in.AutoStart != nil && s.cfg.AutoStart != *in.AutoStart {
				s.cfg.AutoStart = *in.AutoStart
				changed = append(changed, "auto_start")
			}
			if in.LogLevel != nil && *in.LogLevel != "" && s.cfg.LogLevel != *in.LogLevel {
				s.cfg.LogLevel = *in.LogLevel
				changed = append(changed, "log_level")
			}
			if in.HFToken != nil && s.cfg.HFToken != *in.HFToken {
				s.cfg.HFToken = *in.HFToken
				changed = append(changed, "hf_token")
			}
			if in.APIKey != nil && s.cfg.APIKey != *in.APIKey {
				s.cfg.APIKey = *in.APIKey
				changed = append(changed, "api_key")
			}
			if len(changed) > 0 {
				s.saveConfigLocked()
			}
			return changed, nil
		},
		CurrentEnv: func() backup.RuntimeEnv {
			s.cfgMu.Lock()
			defer s.cfgMu.Unlock()
			cur := make(map[string]string, len(s.cfg.RuntimeEnv))
			for k, v := range s.cfg.RuntimeEnv {
				cur[k] = v
			}
			return backup.RuntimeEnv{Curated: cur, Extra: s.cfg.RuntimeEnvExtra}
		},
		ApplyEnv: func(merged backup.RuntimeEnv) error {
			s.cfgMu.Lock()
			defer s.cfgMu.Unlock()
			s.cfg.RuntimeEnv = merged.Curated
			s.cfg.RuntimeEnvExtra = merged.Extra
			s.saveConfigLocked()
			return nil
		},
		SaveFlagPreset: s.builder.SaveFlagPreset,
		InstalledModels: func(modelID, quant string) []string {
			var ids []string
			for _, m := range s.registry.List() {
				if m.ModelID == modelID && m.Quant == quant {
					ids = append(ids, m.ID)
				}
			}
			return ids
		},
		ApplyModelConfig: func(id string, cfg models.ModelConfig) error {
			if err := s.registry.SetConfig(id, &cfg); err != nil {
				return err
			}
			s.markDirty(id)
			return nil
		},
		SavePending: func(m backup.MissingModel) error {
			return s.registry.SetPendingConfig(models.PendingConfig{
				ModelID:  m.ModelID,
				Quant:    m.Quant,
				Filename: m.Filename,
				Config:   m.Config,
				SavedAt:  time.Now().UTC(),
			})
		},
		NumGPUs:   len(s.monitor.Current().GPU),
		ModelsDir: s.cfg.ModelsPath(),
	}
}

// handleDiscardPending removes a pending config. Responds 204 with an
// HX-Trigger so the models page's listing container (#model-list,
// hx-trigger="modelsChanged from:body") re-fetches itself — the ghost
// card disappears through existing machinery.
func (s *Server) handleDiscardPending(w http.ResponseWriter, r *http.Request) {
	r.ParseForm()
	modelID, quant := r.FormValue("model_id"), r.FormValue("quant")
	if !s.registry.DiscardPendingConfig(modelID, quant) {
		http.Error(w, fmt.Sprintf("no pending config for %s %s", modelID, quant), http.StatusNotFound)
		return
	}
	w.Header().Set("HX-Trigger", "modelsChanged")
	w.WriteHeader(http.StatusNoContent)
}
