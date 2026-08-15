package api

import (
	"fmt"
	"html"
	"net/http"
	"strings"

	"github.com/tmac1973/llama-toolchest/internal/builder"
)

// renderFlagPresetRow writes the saved-flag-set controls: a dropdown of
// the profile's presets, Save and Delete buttons, and an optional status
// message. The dropdown applies a preset by re-fetching #build-options
// with a preset= param (which also fills the Build Tag via an OOB swap);
// Save posts the whole build form so it captures the live toggle states.
func (s *Server) renderFlagPresetRow(w http.ResponseWriter, profile, selected, msg string) {
	respondHTML(w)
	fmt.Fprintf(w, `<div class="grid" style="align-items:end;">
		<label style="margin-bottom:0;" title="Apply a saved flag set: fills the toggles, Extra CMake Flags, and Build Tag. Saved sets belong to the selected profile and carry no git ref, so they replay against any ref.">
			Saved Flag Sets
			<select id="flag-preset-select" name="preset" style="margin-bottom:0;"
			        hx-get="/api/builds/options"
			        hx-target="#build-options"
			        hx-swap="outerHTML"
			        hx-include="#build-profile"
			        hx-trigger="change[target.value!='']">
				<option value="">&mdash; apply a saved set &mdash;</option>`)
	for _, p := range s.builder.FlagPresets(profile) {
		sel := ""
		if p.Name == selected {
			sel = "selected"
		}
		fmt.Fprintf(w, `<option value="%s" %s>%s</option>`, html.EscapeString(p.Name), sel, html.EscapeString(p.Name))
	}
	fmt.Fprintf(w, `</select>
		</label>
		<div style="display:flex;gap:0.5rem;align-items:end;">
			<button type="button" class="outline" style="margin-bottom:0;"
			        title="Save the current toggles and Extra CMake Flags under the Build Tag name (tag required)."
			        hx-post="/api/builds/flag-presets"
			        hx-include="closest form"
			        hx-target="#flag-presets" hx-swap="innerHTML">Save Flags</button>
			<button type="button" class="outline secondary" style="margin-bottom:0;"
			        title="Delete the selected saved set."
			        hx-post="/api/builds/flag-presets/delete"
			        hx-include="#flag-preset-select, #build-profile"
			        hx-target="#flag-presets" hx-swap="innerHTML">Delete</button>
		</div>
	</div>`)
	if msg != "" {
		fmt.Fprintf(w, `<small style="color:var(--pico-muted-color);">%s</small>`, html.EscapeString(msg))
	}
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
