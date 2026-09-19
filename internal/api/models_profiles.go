package api

import (
	"errors"
	"fmt"
	"net/http"

	"github.com/go-chi/chi/v5"

	"github.com/tmac1973/llama-toolchest/internal/models"
)

// profileOption is one entry in the saved-profile picker.
type profileOption struct {
	Name     string
	Label    string // "Name — saved 2026-09-19, build b1234 (autoconfig)"
	Selected bool
}

// panelBanner is the result line a profile action shows above the bar.
// Every action re-renders the whole panel with one of these, rather than
// swapping an error over the form.
type panelBanner struct {
	Kind string // "ok" | "warning" | "error"
	Text string
}

// profileBarData lists the model's profiles for the picker, with the
// active one selected, and reports whether the live config was edited
// since that profile.
func (s *Server) profileBarData(id string) ([]profileOption, string, bool) {
	active, edited := s.registry.ActiveProfileState(id)
	var opts []profileOption
	for _, p := range s.registry.Profiles(id) {
		label := p.Name + " — saved " + p.SavedAt.Local().Format("2006-01-02")
		if p.BuildID != "" {
			label += ", build " + p.BuildID
		}
		if p.Source != "" && p.Source != models.ProfileSourceUser {
			label += " (" + p.Source + ")"
		}
		opts = append(opts, profileOption{Name: p.Name, Label: label, Selected: p.Name == active})
	}
	return opts, active, edited
}

// renderConfigPanel re-renders the whole config panel (profile bar and
// form) with a banner.
func (s *Server) renderConfigPanel(w http.ResponseWriter, id string, banner panelBanner) {
	data, err := s.configPanelData(id)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	data.Banner = &banner
	respondHTML(w)
	s.renderPartial(w, "model_config", data)
}

// handleSaveProfile saves the live config as a profile (form: profile_name).
func (s *Server) handleSaveProfile(w http.ResponseWriter, r *http.Request) {
	id := s.registry.ResolveID(chi.URLParam(r, "id"))
	r.ParseForm()
	name := models.NormalizeProfileName(r.FormValue("profile_name"))

	replaced, err := s.registry.SaveProfile(id, name, models.ProfileSourceUser, s.activeBuild())
	switch {
	case errors.Is(err, models.ErrProfileName):
		s.renderConfigPanel(w, id, panelBanner{"error", "Type a name for the profile first."})
	case err != nil:
		s.renderConfigPanel(w, id, panelBanner{"error", "Profile not saved: " + err.Error()})
	case replaced:
		s.renderConfigPanel(w, id, panelBanner{"ok", fmt.Sprintf("Replaced profile %q with the current settings.", name)})
	default:
		s.renderConfigPanel(w, id, panelBanner{"ok", fmt.Sprintf("Saved the current settings as profile %q.", name)})
	}
}

// handleApplyProfile restores a profile over the live config (form:
// profile). It runs the checks a config save runs, plus that every file
// the profile names still exists, and refuses on failure: a restore has no
// undo.
func (s *Server) handleApplyProfile(w http.ResponseWriter, r *http.Request) {
	id := s.registry.ResolveID(chi.URLParam(r, "id"))
	r.ParseForm()
	name := r.FormValue("profile")

	p, err := s.registry.GetProfile(id, name)
	if err != nil {
		s.renderConfigPanel(w, id, panelBanner{"error", "Choose a saved profile to restore."})
		return
	}
	if err := models.ValidateProfileConfig(p.Config); err != nil {
		s.renderConfigPanel(w, id, panelBanner{"error", fmt.Sprintf("Profile %q was not restored: %s", p.Name, err)})
		return
	}
	if err := s.registry.ApplyProfile(id, p.Name); err != nil {
		s.renderConfigPanel(w, id, panelBanner{"error", fmt.Sprintf("Profile %q was not restored: %s", p.Name, err)})
		return
	}
	if cfg, err := s.registry.GetConfig(id); err == nil {
		s.afterConfigChange(w, r, id, cfg)
	}

	msg := fmt.Sprintf("Restored profile %q. The new settings take effect the next time this model loads.", p.Name)
	if active := s.activeBuild(); p.BuildID != "" && active != "" && p.BuildID != active {
		s.renderConfigPanel(w, id, panelBanner{"warning", msg + fmt.Sprintf(
			" It was saved with build %s and the active build is %s; speculative decoding methods and extra flags may behave differently.",
			p.BuildID, active)})
		return
	}
	s.renderConfigPanel(w, id, panelBanner{"ok", msg})
}

// handleDeleteProfile deletes a profile (form: profile). The live config
// is not changed. POST rather than DELETE: htmx sends included fields in
// the body, where ParseForm does not look for a DELETE.
func (s *Server) handleDeleteProfile(w http.ResponseWriter, r *http.Request) {
	id := s.registry.ResolveID(chi.URLParam(r, "id"))
	r.ParseForm()
	name := r.FormValue("profile")

	if err := s.registry.DeleteProfile(id, name); err != nil {
		if errors.Is(err, models.ErrProfileNotFound) {
			s.renderConfigPanel(w, id, panelBanner{"error", "Choose a saved profile to delete."})
			return
		}
		s.renderConfigPanel(w, id, panelBanner{"error", "Profile not deleted: " + err.Error()})
		return
	}
	s.renderConfigPanel(w, id, panelBanner{"ok", fmt.Sprintf("Deleted profile %q. The settings below were not changed.", models.NormalizeProfileName(name))})
}
