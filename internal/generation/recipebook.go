package generation

import (
	"embed"
	"encoding/json"
	"fmt"
	"io/fs"
	"log/slog"
	"os"
	"path"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
)

// Recipe is one sd-server recipe: upstream's recipe file
// (stable-diffusion.cpp examples/server/recipes/<id>.json) with
// toolchest's overlay applied (recipe_overlay.go).
type Recipe struct {
	ID          string
	Name        string
	Description string
	// Kind is KindVideo for a recipe with video_constraints, else KindSD.
	Kind BackendKind
	// Slots are the files the recipe takes, in the file's order. An
	// optional slot that sd-server has no flag for is left out; a required
	// one makes the recipe unavailable.
	Slots RecipeSlotSet
	// UpstreamLabels are the slot labels the recipe file gives.
	UpstreamLabels map[string]string
	Features       []string
	// SizeMultiple is what sd-server rounds a size up to (size_multiple).
	SizeMultiple int
	// Unavailable is empty when toolchest can run the recipe, else the
	// reason it cannot.
	Unavailable string

	// rawDefaults and rawContext keep the file's defaults and context for
	// the custom set form.
	rawDefaults map[string]any
	rawContext  map[string]any
}

// slotLabel names a slot of this recipe for people: toolchest's own
// label, else upstream's, else the shared one, else the slot ID.
func (r *Recipe) slotLabel(slot string) string {
	if l := recipeLabels[r.ID][slot]; l != "" {
		return l
	}
	if l := r.UpstreamLabels[slot]; l != "" {
		return l
	}
	if l := SlotLabels[slot]; l != "" {
		return l
	}
	return slot
}

// HasFeature reports whether the recipe file lists feature.
func (r *Recipe) HasFeature(feature string) bool {
	return slices.Contains(r.Features, feature)
}

// RecipeBook is a set of recipes read from one folder of recipe files:
// a build's, or the snapshot built into llama-toolchest.
type RecipeBook struct {
	// Source says where the recipes come from, e.g. "snapshot
	// master-1047-984eb06".
	Source  string
	recipes map[string]*Recipe
}

// Get returns a recipe by ID.
func (b *RecipeBook) Get(id string) (*Recipe, bool) {
	r, ok := b.recipes[id]
	return r, ok
}

// Len is the number of recipes in the book, available or not.
func (b *RecipeBook) Len() int { return len(b.recipes) }

// IDs returns the available recipes of a kind: the catalog's recipes
// first, in catalog order, then the rest A–Z.
func (b *RecipeBook) IDs(kind BackendKind) []string {
	var ids []string
	for _, e := range Catalog() {
		if e.Kind != kind || e.Recipe == "" || slices.Contains(ids, e.Recipe) {
			continue
		}
		if r, ok := b.recipes[e.Recipe]; ok && r.Kind == kind && r.Unavailable == "" {
			ids = append(ids, e.Recipe)
		}
	}
	var rest []string
	for id, r := range b.recipes {
		if r.Kind == kind && r.Unavailable == "" && !slices.Contains(ids, id) {
			rest = append(rest, id)
		}
	}
	slices.Sort(rest)
	return append(ids, rest...)
}

// Unavailable returns the recipes of a kind that toolchest cannot run,
// A–Z.
func (b *RecipeBook) Unavailable(kind BackendKind) []*Recipe {
	var out []*Recipe
	for _, r := range b.recipes {
		if r.Kind == kind && r.Unavailable != "" {
			out = append(out, r)
		}
	}
	slices.SortFunc(out, func(a, b *Recipe) int { return strings.Compare(a.ID, b.ID) })
	return out
}

// recipeFile is the format of upstream's recipe files (schema_version 1).
// Keys toolchest does not use are ignored.
type recipeFile struct {
	SchemaVersion int    `json:"schema_version"`
	ID            string `json:"id"`
	Name          string `json:"name"`
	Description   string `json:"description"`
	Slots         []struct {
		ID       string `json:"id"`
		Required bool   `json:"required"`
		Label    string `json:"label"`
	} `json:"slots"`
	RequiredAny      [][]string      `json:"required_any"`
	Defaults         map[string]any  `json:"defaults"`
	Context          map[string]any  `json:"context"`
	Features         []string        `json:"features"`
	SizeMultiple     int             `json:"size_multiple"`
	VideoConstraints json.RawMessage `json:"video_constraints"`
}

// ParseRecipes reads every *.json recipe file in the root of fsys and
// applies the overlay. A file that does not parse, has another schema
// version, or whose id is not its file name is skipped with a warning.
// It fails only when no recipe could be read.
func ParseRecipes(fsys fs.FS) (*RecipeBook, error) {
	names, err := fs.Glob(fsys, "*.json")
	if err != nil {
		return nil, err
	}
	book := &RecipeBook{recipes: map[string]*Recipe{}}
	for _, name := range names {
		r, err := parseRecipeFile(fsys, name)
		if err != nil {
			slog.Warn("skipping sd-server recipe file", "file", name, "error", err)
			continue
		}
		book.recipes[r.ID] = r
	}
	if len(book.recipes) == 0 {
		return nil, fmt.Errorf("no recipe files could be read (%d files)", len(names))
	}
	return book, nil
}

func parseRecipeFile(fsys fs.FS, name string) (*Recipe, error) {
	data, err := fs.ReadFile(fsys, name)
	if err != nil {
		return nil, err
	}
	var f recipeFile
	if err := json.Unmarshal(data, &f); err != nil {
		return nil, err
	}
	if f.SchemaVersion != 1 {
		return nil, fmt.Errorf("schema version %d, not 1", f.SchemaVersion)
	}
	if want := strings.TrimSuffix(path.Base(name), ".json"); f.ID != want {
		return nil, fmt.Errorf("id %q is not the file name %q", f.ID, want)
	}
	r := &Recipe{ID: f.ID, Name: f.Name, Description: f.Description, Kind: KindSD,
		Features: f.Features, SizeMultiple: f.SizeMultiple, UpstreamLabels: map[string]string{},
		rawDefaults: f.Defaults, rawContext: f.Context}
	if len(f.VideoConstraints) > 0 && string(f.VideoConstraints) != "null" {
		r.Kind = KindVideo
	}
	var missing []string // required slots sd-server has no flag for
	for _, s := range f.Slots {
		if s.Label != "" {
			r.UpstreamLabels[s.ID] = s.Label
		}
		if sdSlotFlags[s.ID] == "" {
			if s.Required {
				missing = append(missing, s.ID)
			}
			continue
		}
		if s.Required {
			r.Slots.Required = append(r.Slots.Required, s.ID)
		} else {
			r.Slots.Optional = append(r.Slots.Optional, s.ID)
		}
	}
	for _, group := range f.RequiredAny {
		// A group is kept with the slots toolchest can pass; one with none
		// left would make the recipe unfillable.
		var g []string
		for _, s := range group {
			if slices.Contains(r.Slots.Slots(), s) {
				g = append(g, s)
			}
		}
		if len(g) == 0 {
			missing = append(missing, group...)
			continue
		}
		r.Slots.OneOf = append(r.Slots.OneOf, g)
	}
	r.Unavailable = unavailableReason(r, missing)
	applyOverlay(r)
	return r, nil
}

// unavailableReason says why toolchest cannot run a recipe, or "" when it
// can. missing are its required slots without an sd-server flag.
func unavailableReason(r *Recipe, missing []string) string {
	if reason := recipeLeftOut[r.ID]; reason != "" {
		return reason
	}
	for _, f := range r.Features {
		if what := recipeNeedsInput[f]; what != "" {
			return "needs " + what + " input, which llama-toolchest does not send"
		}
	}
	if len(missing) > 0 {
		return "needs the " + r.slotLabel(missing[0]) + " file, which llama-toolchest cannot pass to sd-server"
	}
	keys := make([]string, 0, len(r.rawContext))
	for k := range r.rawContext {
		keys = append(keys, k)
	}
	slices.Sort(keys)
	for _, k := range keys {
		if k != "vae_format" && k != "model_args" {
			return "uses a recipe setting (" + k + ") that llama-toolchest cannot pass to sd-server"
		}
	}
	return ""
}

// LoadRecipes reads the recipe files in a folder.
func LoadRecipes(dir string) (*RecipeBook, error) {
	book, err := ParseRecipes(os.DirFS(dir))
	if err != nil {
		return nil, fmt.Errorf("%s: %w", dir, err)
	}
	book.Source = dir
	return book, nil
}

// The recipe snapshot built into llama-toolchest: upstream's recipe files
// at the commit SOURCE names. make update-sd-recipes refreshes it.
//
//go:embed sdrecipes/*.json sdrecipes/SOURCE
var snapshotFS embed.FS

var snapshotOnce = sync.OnceValue(func() *RecipeBook {
	sub, err := fs.Sub(snapshotFS, "sdrecipes")
	if err != nil {
		panic(err)
	}
	book, err := ParseRecipes(sub)
	if err != nil {
		panic("built-in recipe snapshot: " + err.Error())
	}
	book.Source = "snapshot"
	if src, err := fs.ReadFile(sub, "SOURCE"); err == nil {
		if f := strings.Fields(string(src)); len(f) >= 2 {
			book.Source = "snapshot " + f[1]
		}
	}
	return book
})

// SnapshotRecipes returns the recipe snapshot built into llama-toolchest.
func SnapshotRecipes() *RecipeBook { return snapshotOnce() }

var activeRecipes atomic.Pointer[RecipeBook]

// ActiveRecipes returns the recipe book the custom set form, validation
// and Details use: the snapshot until SetActiveRecipes swaps it.
func ActiveRecipes() *RecipeBook {
	if b := activeRecipes.Load(); b != nil {
		return b
	}
	return SnapshotRecipes()
}

// SetActiveRecipes swaps the active recipe book.
func SetActiveRecipes(b *RecipeBook) { activeRecipes.Store(b) }
