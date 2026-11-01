package generation

import (
	"os"
	"regexp"
	"slices"
	"strings"

	"github.com/tmac1973/llama-toolchest/internal/huggingface"
)

// SlotDetail is one file of a set, for its Details.
type SlotDetail struct {
	Slot     string `json:"slot"`
	Label    string `json:"label"`
	Required bool   `json:"required"`
	// Path is the file on this machine, or where a catalog file will be
	// saved when it is not downloaded yet.
	Path string `json:"path"`
	// Repo and File are the Hugging Face source; empty for a custom set.
	Repo    string `json:"repo,omitempty"`
	File    string `json:"file,omitempty"`
	Bytes   int64  `json:"bytes,omitempty"`
	Present bool   `json:"present"`
}

// FlagDetail is one backend flag of a set, with why it is there.
type FlagDetail struct {
	Flag   string `json:"flag"`
	Value  string `json:"value,omitempty"`
	Reason string `json:"reason,omitempty"`
}

// SetDetails says how a set runs: its files, its flags with reasons, and
// the command an instance of it starts. Values that differ per instance
// are markers in the command, such as "<port>".
type SetDetails struct {
	ID        string      `json:"id"`
	Name      string      `json:"name"`
	Kind      BackendKind `json:"kind"`
	Recipe    string      `json:"recipe,omitempty"`
	CatalogID string      `json:"catalog_id,omitempty"`
	Custom    bool        `json:"custom"`
	// Program is "sd-server" or "ace-server".
	Program string       `json:"program"`
	Slots   []SlotDetail `json:"slots"`
	// Flags are the set's own flags: RunArgs for a set, the entry's
	// LaunchArgs for a catalog entry.
	Flags []FlagDetail `json:"flags"`
	// FixedFlags are the flags toolchest adds for each instance.
	FixedFlags []FlagDetail `json:"fixed_flags"`
	// Command is the whole command line, shell-quoted, with markers.
	Command string `json:"command"`
	Notes   string `json:"notes,omitempty"`
	License string `json:"license,omitempty"`
}

// The markers the command shows for values each instance sets itself.
const (
	markerHost      = "<listen address>"
	markerPort      = "<port>"
	markerVRAM      = "<vram limit>"
	markerInstance  = "<instance folder>"
	upscalersFolder = markerInstance + "/upscalers"
	aceModelsFolder = markerInstance + "/models"
)

// DetailsForSet describes a set on this machine. modelsDir is where a
// catalog set's missing files will be saved (ModelsDir).
func DetailsForSet(set ModelSet, modelsDir string) (SetDetails, error) {
	entry, _ := set.catalogEntry()
	d := SetDetails{ID: set.ID, Name: set.Name, Kind: set.Kind, Recipe: set.Recipe, CatalogID: set.CatalogID,
		Custom: set.CatalogID == "", Notes: entry.Notes, License: set.License}
	if entry.License != "" {
		d.License = entry.License
	}
	set.Components = slices.Clone(set.Components) // the caller's set keeps its paths
	for i, c := range set.Components {
		if c.Path == "" && c.Repo != "" {
			set.Components[i].Path = huggingface.FilePath(modelsDir, c.Repo, c.File)
		}
	}
	return d, d.fill(set, entry)
}

// DetailsForCatalog describes a catalog entry, downloaded or not. Its files
// are where the downloader saves them under modelsDir (ModelsDir).
func DetailsForCatalog(e CatalogEntry, modelsDir string) (SetDetails, error) {
	set := ModelSet{ID: e.ID, Name: e.Name, Kind: e.Kind, CatalogID: e.ID, Recipe: e.Recipe, LaunchArgs: e.LaunchArgs}
	for _, c := range e.Components {
		c.Path = huggingface.FilePath(modelsDir, c.Repo, c.File)
		set.Components = append(set.Components, c)
	}
	d := SetDetails{ID: e.ID, Name: e.Name, Kind: e.Kind, Recipe: e.Recipe, CatalogID: e.ID, Notes: e.Notes, License: e.License}
	return d, d.fill(set, e)
}

// fill adds the slots, flags and command. set has a path for every file.
func (d *SetDetails) fill(set ModelSet, entry CatalogEntry) error {
	// Empty lists, not nil: the JSON gives [] for a set with no flags of
	// its own (ACE-Step), not null.
	d.Slots, d.Flags, d.FixedFlags = []SlotDetail{}, []FlagDetail{}, []FlagDetail{}
	d.Program = "sd-server"
	if set.Kind == KindAce {
		d.Program = "ace-server"
	}

	slots := KindSlots(set.Kind, set.Recipe)
	order := slots.Slots()
	comps := slices.Clone(set.Components)
	slices.SortStableFunc(comps, func(a, b Component) int {
		return slotIndex(order, a.Slot) - slotIndex(order, b.Slot)
	})
	for _, c := range comps {
		label := SlotLabel(set.Recipe, c.Slot)
		if slices.Contains(slots.OneOf, c.Slot) {
			label += " (one of: " + strings.Join(slots.OneOf, ", ") + ")"
		}
		_, err := os.Stat(c.Path)
		d.Slots = append(d.Slots, SlotDetail{Slot: c.Slot, Label: label, Required: slices.Contains(slots.Required, c.Slot),
			Path: c.Path, Repo: c.Repo, File: c.File, Bytes: c.Bytes, Present: err == nil})
	}

	for _, f := range SplitFlags(set.RunArgs()) {
		d.Flags = append(d.Flags, FlagDetail{Flag: f[0], Value: f[1], Reason: FlagReason(f[0], f[1], entry)})
	}

	p := argPaths{Host: markerHost, Port: markerPort, UpscalerDir: upscalersFolder, AceModelsDir: aceModelsFolder}
	if set.Kind == KindSD || set.Kind == KindVideo {
		p.MaxVRAM = markerVRAM
		var err error
		if p.LoraDir, err = LoraDir(set); err != nil {
			return err
		}
	}
	args, fixed, err := buildArgs(set, nil, p)
	if err != nil {
		return err
	}
	for _, f := range fixed {
		d.FixedFlags = append(d.FixedFlags, FlagDetail{Flag: f[0], Value: f[1], Reason: FlagReasons[f[0]]})
	}
	words := []string{d.Program}
	for _, a := range args {
		words = append(words, shellWord(a))
	}
	d.Command = strings.Join(words, " ")
	return nil
}

// KindSlots returns the slots a set of a kind and recipe takes: the
// recipe's for image and video sets, the fixed ones for music and
// upscaler sets.
func KindSlots(kind BackendKind, recipe string) RecipeSlotSet {
	switch kind {
	case KindVideo:
		return VideoRecipes[recipe].Slots
	case KindAce:
		return RecipeSlotSet{Required: AceSlots}
	case KindUpscale:
		return RecipeSlotSet{Required: []string{UpscalerSlot}}
	}
	return RecipeSlots[recipe]
}

// slotIndex is a slot's place in the recipe's order; unknown slots last.
func slotIndex(order []string, slot string) int {
	if i := slices.Index(order, slot); i >= 0 {
		return i
	}
	return len(order)
}

// plainWord is a word a shell reads as it is.
var plainWord = regexp.MustCompile(`^[A-Za-z0-9_@%+=:,./-]+$`)

// shellWord quotes a word for a POSIX shell, except the "<...>" markers,
// which are not meant to be run as they are.
func shellWord(w string) string {
	if plainWord.MatchString(w) || strings.HasPrefix(w, "<") {
		return w
	}
	return "'" + strings.ReplaceAll(w, "'", `'\''`) + "'"
}
