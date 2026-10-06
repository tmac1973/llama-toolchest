package api

import (
	"fmt"
	"net/http"
	"strings"

	"github.com/tmac1973/llama-toolchest/internal/builder"
)

// flagPresetRowData feeds the flag_preset_row partial.
type flagPresetRowData struct {
	Presets  []builder.FlagPreset
	Selected string // name of the preset to show as chosen, if any
	Msg      string // status line under the controls, if any
}

// renderFlagPresetRow writes the saved-flag-set controls: a dropdown of
// the profile's presets, Save and Delete buttons, and an optional status
// message. The dropdown applies a preset by re-fetching #build-options
// with a preset= param (which also fills the Build Tag via an OOB swap);
// Save posts the whole build form so it captures the live toggle states.
func (s *Server) renderFlagPresetRow(w http.ResponseWriter, profile, selected, msg string) {
	respondHTML(w)
	s.renderPartial(w, "flag_preset_row", flagPresetRowData{
		Presets:  s.builder.FlagPresets(profile),
		Selected: selected,
		Msg:      msg,
	})
}

// handleFlagPresetRow renders the controls for the selected profile.
func (s *Server) handleFlagPresetRow(w http.ResponseWriter, r *http.Request) {
	s.renderFlagPresetRow(w, r.URL.Query().Get("profile"), "", "")
}

// handleSaveFlagPreset saves the posted build form as a flag preset
// named by the Build Tag. Upserts: an existing name is updated.
func (s *Server) handleSaveFlagPreset(w http.ResponseWriter, r *http.Request) {
	r.ParseForm()
	profile := r.FormValue("profile")
	name := strings.TrimSpace(r.FormValue("tag"))
	if name == "" {
		s.renderFlagPresetRow(w, profile, "", "Enter a Build Tag first — it names the saved set.")
		return
	}
	// Checkbox semantics: a present opt_* field means checked; absent
	// means unchecked. Record every option explicitly so applying the
	// preset later doesn't fall back to defaults for the off ones.
	options := map[string]bool{}
	for _, opt := range builder.ProfileOptions(profile) {
		options[opt.Flag] = r.Form.Get("opt_"+opt.Flag) == "on"
	}
	p := builder.FlagPreset{
		Name:       name,
		Profile:    profile,
		Options:    options,
		ExtraCMake: strings.TrimSpace(r.FormValue("extra_cmake")),
		Version:    builder.FlagPresetVersion,
	}
	if err := s.builder.SaveFlagPreset(p); err != nil {
		s.renderFlagPresetRow(w, profile, "", err.Error())
		return
	}
	s.renderFlagPresetRow(w, profile, name, fmt.Sprintf("Saved %q.", name))
}

// handleDeleteFlagPreset removes the selected preset.
func (s *Server) handleDeleteFlagPreset(w http.ResponseWriter, r *http.Request) {
	r.ParseForm()
	profile := r.FormValue("profile")
	name := r.FormValue("preset")
	if name == "" {
		s.renderFlagPresetRow(w, profile, "", "Select a saved set to delete.")
		return
	}
	if s.builder.DeleteFlagPreset(name) {
		s.renderFlagPresetRow(w, profile, "", fmt.Sprintf("Deleted %q.", name))
		return
	}
	s.renderFlagPresetRow(w, profile, "", fmt.Sprintf("No saved set named %q.", name))
}
