package api

import (
	"encoding/json"
	"errors"
	"fmt"
	"html"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/generation"
	"github.com/tmac1973/llama-toolchest/internal/huggingface"
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
	// Tab is the tab the form's dialog is on: "models", or "catalog"
	// (Download Models), where a save answers with a note, not the set
	// list.
	Tab string
	// CopiedFrom names the set or catalog entry the form was filled from;
	// empty for a new set.
	CopiedFrom string
	// VideoSizes (JSON) and License are copied from the source and sent
	// back unchanged in hidden fields.
	VideoSizes, License string
}

// genSetFormFromRequest reads the custom set form's values from a request,
// so the form can be drawn again for another backend or recipe with what
// the user typed kept. A missing or unknown kind is an image set; a recipe
// that does not fit the kind becomes the kind's first.
func (s *Server) genSetFormFromRequest(r *http.Request) genSetForm {
	return s.genSetFormFrom(r.FormValue)
}

// genSetFormFrom is genSetFormFromRequest for values from anywhere: get
// returns a form field's value.
func (s *Server) genSetFormFrom(get func(string) string) genSetForm {
	f := genSetForm{
		Name:       get("name"),
		Kind:       generation.BackendKind(get("kind")),
		Recipe:     get("recipe"),
		LaunchArgs: get("launch_args"),
		Width:      get("width"), Height: get("height"),
		Frames: get("frames"), FPS: get("fps"),
		MaxFrames: get("max_frames"), MaxWidth: get("max_width"), MaxHeight: get("max_height"),
		Scale:      get("scale"),
		StemTasks:  get("stem_tasks") == "1",
		ModelsDir:  s.cfg.ModelsPath(),
		Tab:        get("tab"),
		VideoSizes: get("video_sizes"),
		License:    get("license"),
	}
	if f.Tab != "catalog" {
		f.Tab = "models"
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
			Required: slices.Contains(slots.Required, slot), Value: strings.TrimSpace(get("slot_" + slot)),
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

// handleGenSetForm serves the custom set form (the dialog body). With
// copy=<set id> or copy_catalog=<entry id> it is filled in from that set
// or catalog entry.
func (s *Server) handleGenSetForm(w http.ResponseWriter, r *http.Request) {
	store := s.genStore(w)
	if store == nil {
		return
	}
	f := s.genSetFormFromRequest(r)
	if r.FormValue("copy") != "" || r.FormValue("copy_catalog") != "" {
		var code int
		var err error
		if f, code, err = s.genSetCopyForm(r, store); err != nil {
			http.Error(w, err.Error(), code)
			return
		}
	}
	respondHTML(w)
	s.renderPartial(w, "gen_set_form", f)
}

// genSetCopyForm fills the custom set form from the set (copy) or catalog
// entry (copy_catalog) the request names. A catalog set, and a catalog
// entry, give the entry's current flags, sizes, limits and licence, as a
// catalog set runs with those. Each file that is on this machine fills its
// slot; one that is not leaves the slot empty, with the catalog file's
// name as the hint.
func (s *Server) genSetCopyForm(r *http.Request, store *generation.Store) (genSetForm, int, error) {
	copyID, catalogID := r.FormValue("copy"), r.FormValue("copy_catalog")
	if copyID != "" && catalogID != "" {
		return genSetForm{}, http.StatusBadRequest, errors.New("copy a set or a catalog entry, not both")
	}
	var set generation.ModelSet
	entry, fromCatalog := generation.CatalogEntry{}, false
	if copyID != "" {
		var err error
		if set, err = store.ModelSet(copyID); err != nil {
			code := http.StatusInternalServerError
			if errors.Is(err, generation.ErrNotFound) {
				code = http.StatusNotFound
			}
			return genSetForm{}, code, err
		}
		if set.CatalogID != "" {
			entry, fromCatalog = generation.LookupCatalog(set.CatalogID)
		}
	} else {
		var ok bool
		if entry, ok = generation.LookupCatalog(catalogID); !ok {
			return genSetForm{}, http.StatusNotFound, errors.New("unknown catalog set")
		}
		fromCatalog = true
		set = generation.ModelSet{Name: entry.Name, Kind: entry.Kind, Recipe: entry.Recipe, Components: entry.Components}
	}

	// The source's values: the catalog entry's for a catalog set or entry,
	// otherwise the custom set's own.
	src := set
	src.LaunchArgs = set.RunArgs()
	src.Scale, src.VideoSizes = set.UpscaleScale(), set.VideoSizes
	stems := set.Kind == generation.KindAce && set.HasStemTasks()
	if fromCatalog {
		src.LaunchArgs = entry.LaunchArgs
		src.DefaultWidth, src.DefaultHeight, src.DefaultFrames, src.DefaultFPS = entry.DefaultWidth, entry.DefaultHeight, entry.DefaultFrames, entry.DefaultFPS
		src.MaxFrames, src.MaxWidth, src.MaxHeight = entry.MaxFrames, entry.MaxWidth, entry.MaxHeight
		src.Scale, src.VideoSizes, src.License = entry.Scale, entry.VideoSizes, entry.License
		stems = entry.StemTasks
	}

	vals := url.Values{
		"name":        {s.freeSetName(store, set.Name+" custom")},
		"kind":        {string(set.Kind)},
		"recipe":      {set.Recipe},
		"launch_args": {generation.JoinArgs(src.LaunchArgs)},
		"tab":         {r.FormValue("tab")},
		"license":     {src.License},
	}
	for k, n := range map[string]int{"width": src.DefaultWidth, "height": src.DefaultHeight, "frames": src.DefaultFrames, "fps": src.DefaultFPS,
		"max_frames": src.MaxFrames, "max_width": src.MaxWidth, "max_height": src.MaxHeight, "scale": src.Scale} {
		if n > 0 {
			vals.Set(k, strconv.Itoa(n))
		}
	}
	if stems {
		vals.Set("stem_tasks", "1")
	}
	if set.Kind == generation.KindVideo && len(src.VideoSizes) > 0 {
		b, err := json.Marshal(src.VideoSizes)
		if err != nil {
			return genSetForm{}, http.StatusInternalServerError, err
		}
		vals.Set("video_sizes", string(b))
	}
	hints := map[string]string{}
	for _, c := range set.Components {
		path := c.Path
		if path == "" && c.Repo != "" {
			if p := huggingface.FilePath(generation.ModelsDir(s.cfg), c.Repo, c.File); fileExists(p) {
				path = p
			} else {
				hints[c.Slot] = filepath.Base(c.File)
			}
		}
		if path != "" {
			vals.Set("slot_"+c.Slot, path)
		}
	}

	f := s.genSetFormFrom(vals.Get)
	f.CopiedFrom = set.Name
	for i, field := range f.Slots {
		if h := hints[field.Slot]; h != "" {
			f.Slots[i].Hint = h
		}
	}
	return f, 0, nil
}

// freeSetName is base, or base with " 2", " 3" and so on, whose ID no set
// has yet.
func (s *Server) freeSetName(store *generation.Store, base string) string {
	name := base
	for n := 2; ; n++ {
		if _, err := store.ModelSet(generation.Slug(name)); errors.Is(err, generation.ErrNotFound) {
			return name
		}
		name = fmt.Sprintf("%s %d", base, n)
	}
}

func fileExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && info.Mode().IsRegular()
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
		sizes, err := formVideoSizes(r, set)
		if err != nil {
			return bad(err)
		}
		set.VideoSizes = sizes
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
	set.License = strings.TrimSpace(r.FormValue("license"))
	return set, 0, nil
}

// formVideoSizes reads the Measure sizes a copied video set keeps (the
// video_sizes field, JSON). The first is the test size, so it takes the
// form's test size, which the user may have changed; none may be larger
// than the set's ceiling. No field gives none: Measure then offers only
// the test size.
func formVideoSizes(r *http.Request, set generation.ModelSet) ([]generation.VideoSize, error) {
	raw := strings.TrimSpace(r.FormValue("video_sizes"))
	if raw == "" {
		return nil, nil
	}
	var sizes []generation.VideoSize
	if err := json.Unmarshal([]byte(raw), &sizes); err != nil {
		return nil, fmt.Errorf("video_sizes: %w", err)
	}
	if len(sizes) == 0 {
		return nil, nil
	}
	sizes[0].Frames, sizes[0].Width, sizes[0].Height = set.DefaultFrames, set.DefaultWidth, set.DefaultHeight
	for _, v := range sizes {
		check := set
		check.DefaultFrames, check.DefaultWidth, check.DefaultHeight = v.Frames, v.Width, v.Height
		if v.Frames <= 0 || v.Width <= 0 || v.Height <= 0 {
			return nil, fmt.Errorf("video size %q has no frames or size", v.Label)
		}
		if err := checkVideoCeiling(check); err != nil {
			return nil, fmt.Errorf("video size %s: %w", v.Label, err)
		}
	}
	return sizes, nil
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
