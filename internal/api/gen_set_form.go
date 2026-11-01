package api

import (
	"errors"
	"fmt"
	"html"
	"net/http"
	"os"
	"slices"
	"strings"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/generation"
)

// maxSuggestedFiles caps the file list the custom set form suggests.
const maxSuggestedFiles = 5000

// genSetSlotField is one file field of the custom set form.
type genSetSlotField struct {
	Slot, Label string
	Required    bool
	// OneOf names the group the slot belongs to, e.g. "vae, tae", when at
	// least one of the group must be given.
	OneOf string
	Value string
	// Hint is the field's placeholder.
	Hint string
	// List is the datalist that suggests files for the field.
	List string
}

// genSetForm is what the gen_set_form and gen_set_form_fields partials
// render: the custom set form, with the values typed so far.
type genSetForm struct {
	Name    string
	Kind    generation.BackendKind
	Recipe  string
	Recipes []string
	Slots   []genSetSlotField
	// LaunchArgs is the set's extra backend flags, as typed.
	LaunchArgs                     string
	Width, Height, Frames, FPS     string
	MaxFrames, MaxWidth, MaxHeight string
	Scale                          string
	StemTasks                      bool
	ModelsDir                      string
}

// genSetFormFromRequest reads the custom set form's values from a request,
// so the form can be drawn again for another backend or recipe with what
// the user typed kept. A missing or unknown kind is an image set; a recipe
// that does not fit the kind becomes the kind's first.
func (s *Server) genSetFormFromRequest(r *http.Request) genSetForm {
	f := genSetForm{
		Name:       r.FormValue("name"),
		Kind:       generation.BackendKind(r.FormValue("kind")),
		Recipe:     r.FormValue("recipe"),
		LaunchArgs: r.FormValue("launch_args"),
		Width:      r.FormValue("width"), Height: r.FormValue("height"),
		Frames: r.FormValue("frames"), FPS: r.FormValue("fps"),
		MaxFrames: r.FormValue("max_frames"), MaxWidth: r.FormValue("max_width"), MaxHeight: r.FormValue("max_height"),
		Scale:     r.FormValue("scale"),
		StemTasks: r.FormValue("stem_tasks") == "1",
		ModelsDir: s.cfg.ModelsPath(),
	}
	switch f.Kind {
	case generation.KindSD:
		f.Recipes = generation.Recipes()
	case generation.KindVideo:
		f.Recipes = generation.VideoRecipeIDs()
	case generation.KindAce, generation.KindUpscale:
	default:
		f.Kind, f.Recipes = generation.KindSD, generation.Recipes()
	}
	if len(f.Recipes) == 0 {
		f.Recipe = ""
	} else if !slices.Contains(f.Recipes, f.Recipe) {
		f.Recipe = f.Recipes[0]
	}

	slots := generation.KindSlots(f.Kind, f.Recipe)
	for _, slot := range slots.Slots() {
		field := genSetSlotField{Slot: slot, Label: generation.SlotLabel(f.Recipe, slot),
			Required: slices.Contains(slots.Required, slot), Value: strings.TrimSpace(r.FormValue("slot_" + slot)),
			Hint: "/path/to/file", List: "gen-model-file-list"}
		if slices.Contains(slots.OneOf, slot) {
			field.OneOf = strings.Join(slots.OneOf, ", ")
		}
		if slot == "tokenizer" {
			field.Hint, field.List = "/path/to/tokenizer.json", "gen-tokenizer-file-list"
		}
		f.Slots = append(f.Slots, field)
	}
	return f
}

// handleGenSetForm serves the custom set form (the dialog body).
func (s *Server) handleGenSetForm(w http.ResponseWriter, r *http.Request) {
	if s.genStore(w) == nil {
		return
	}
	respondHTML(w)
	s.renderPartial(w, "gen_set_form", s.genSetFormFromRequest(r))
}

// handleGenSetFormFields serves the part of the custom set form that
// changes with the backend and recipe: the selects, the file fields and
// the kind's own fields.
func (s *Server) handleGenSetFormFields(w http.ResponseWriter, r *http.Request) {
	if s.genStore(w) == nil {
		return
	}
	respondHTML(w)
	s.renderPartial(w, "gen_set_form_fields", s.genSetFormFromRequest(r))
}

// handleGenModelFiles lists the files the custom set form suggests: the
// model files and tokenizer files under the models folder. htmx gets the
// two datalists; other clients get JSON.
func (s *Server) handleGenModelFiles(w http.ResponseWriter, r *http.Request) {
	if s.genStore(w) == nil {
		return
	}
	models, tokenizers, truncated := generation.ListModelFiles(s.cfg.ModelsPath(), maxSuggestedFiles)
	if !isHTMX(r) {
		respondJSON(w, map[string]any{"models": models, "tokenizers": tokenizers, "truncated": truncated})
		return
	}
	respondHTML(w)
	var b strings.Builder
	for _, list := range []struct {
		id    string
		paths []string
	}{{"gen-model-file-list", models}, {"gen-tokenizer-file-list", tokenizers}} {
		fmt.Fprintf(&b, `<datalist id="%s">`, list.id)
		for _, p := range list.paths {
			fmt.Fprintf(&b, `<option value="%s">`, html.EscapeString(p))
		}
		b.WriteString("</datalist>")
	}
	w.Write([]byte(b.String()))
}

// setFromForm builds a custom set from the custom set form, with every
// check an add makes: the slots fit the recipe, the files exist and are
// model files, the LoRAs share one folder, the flags are allowed, and the
// video ceiling and upscale factor make sense. The files come from the
// slot_<slot> fields, or, for scripts written before them, from the files
// field's "slot = /path" lines. On an error it also returns the HTTP
// status to send.
func (s *Server) setFromForm(r *http.Request) (generation.ModelSet, int, error) {
	bad := func(err error) (generation.ModelSet, int, error) {
		return generation.ModelSet{}, http.StatusBadRequest, err
	}
	name := strings.TrimSpace(r.FormValue("name"))
	kind := generation.BackendKind(r.FormValue("kind"))
	recipe := r.FormValue("recipe")
	if kind == generation.KindAce || kind == generation.KindUpscale {
		recipe = ""
	}
	set := generation.ModelSet{
		ID: generation.Slug(name), Name: name, Kind: kind, Recipe: recipe, Created: time.Now(), Complete: true,
	}
	if set.ID == "" {
		return bad(errors.New("the set needs a name"))
	}
	switch kind {
	case generation.KindSD:
		set.DefaultWidth, set.DefaultHeight = formInt(r, "width", 1024), formInt(r, "height", 1024)
	case generation.KindVideo:
		// The ceiling is optional: without it, Max fields stay zero and
		// only the measured size holds requests back.
		set.DefaultWidth, set.DefaultHeight = formInt(r, "width", 832), formInt(r, "height", 480)
		set.DefaultFrames, set.DefaultFPS = formInt(r, "frames", 33), formInt(r, "fps", 24)
		set.MaxFrames = formInt(r, "max_frames", 0)
		set.MaxWidth, set.MaxHeight = formInt(r, "max_width", 0), formInt(r, "max_height", 0)
		if err := checkVideoCeiling(set); err != nil {
			return bad(err)
		}
	case generation.KindUpscale:
		set.Scale = formInt(r, "scale", 4)
		if set.Scale > 8 {
			return bad(errors.New("the upscale factor must be between 1 and 8"))
		}
	}

	comps, err := formComponents(r)
	if err != nil {
		return bad(err)
	}
	set.Components = comps
	if err := generation.ValidateComponents(kind, recipe, set.Components); err != nil {
		return bad(err)
	}
	if err := generation.CheckFiles(set.Components, s.cfg.ModelsPath(), s.inContainer()); err != nil {
		return bad(err)
	}
	for _, c := range set.Components {
		if err := generation.CheckFileFormat(c.Slot, c.Path); err != nil {
			return bad(err)
		}
	}
	if _, err := generation.LoraDir(set); err != nil {
		return bad(err)
	}
	args, err := generation.ValidateExtraArgs(r.FormValue("launch_args"))
	if err != nil {
		return bad(err)
	}
	if err := generation.CheckLoraArgs(set, args); err != nil {
		return bad(err)
	}
	set.LaunchArgs = args
	if kind == generation.KindSD || kind == generation.KindVideo {
		set.Capabilities = generation.RecipeCapabilities(recipe, set.Components)
	}
	if kind == generation.KindAce {
		stems := r.FormValue("stem_tasks") == "1"
		set.StemTasks = &stems
	}
	return set, 0, nil
}

// formComponents reads a set's files from the form: one slot_<slot> field
// per file, or the older files field with one "slot = /path" line per
// file. Not both.
func formComponents(r *http.Request) ([]generation.Component, error) {
	var comps []generation.Component
	add := func(slot, path string) {
		c := generation.Component{Slot: slot, Path: path}
		if info, err := os.Stat(c.Path); err == nil {
			c.Bytes = info.Size()
		}
		comps = append(comps, c)
	}
	for key, values := range r.Form {
		slot, ok := strings.CutPrefix(key, "slot_")
		if !ok || len(values) == 0 || strings.TrimSpace(values[0]) == "" {
			continue
		}
		add(slot, strings.TrimSpace(values[0]))
	}
	files := strings.TrimSpace(r.FormValue("files"))
	if files != "" && len(comps) > 0 {
		return nil, errors.New("send the files as slot_<name> fields or as the files list, not both")
	}
	for _, line := range strings.Split(files, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		slot, path, ok := strings.Cut(line, "=")
		if !ok {
			return nil, fmt.Errorf("%q: write each file as slot = /path/to/file", line)
		}
		add(strings.TrimSpace(slot), strings.TrimSpace(path))
	}
	slices.SortFunc(comps, func(a, b generation.Component) int { return strings.Compare(a.Slot, b.Slot) })
	return comps, nil
}
