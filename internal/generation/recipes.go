package generation

import (
	"errors"
	"fmt"
	"os"
	"regexp"
	"slices"
	"strings"
)

// RecipeSlotSet describes which component slots an sd-server recipe takes.
type RecipeSlotSet struct {
	Required []string
	Optional []string
	// OneOf lists slots of which at least one must be filled.
	OneOf []string
}

// Slots returns every slot the recipe accepts, required first.
func (r RecipeSlotSet) Slots() []string {
	return append(slices.Clone(r.Required), r.Optional...)
}

// RecipeSlots holds the slots of the sd-server recipes a set may use,
// copied from stable-diffusion.cpp examples/server/recipes/*.json at
// master-1000-95ff762 (2026-10-24).
var RecipeSlots = map[string]RecipeSlotSet{
	"ming-image":          {Required: []string{"diffusion_model", "llm", "tokenizer"}, Optional: []string{"vae", "tae"}, OneOf: []string{"vae", "tae"}},
	"qwen-image-21-turbo": {Required: []string{"diffusion_model", "llm"}, Optional: []string{"llm_vision", "vae", "tae"}, OneOf: []string{"vae", "tae"}},
	"qwen-image-21":       {Required: []string{"diffusion_model", "llm"}, Optional: []string{"llm_vision", "vae", "tae"}, OneOf: []string{"vae", "tae"}},
	"flux2-klein-4b":      {Required: []string{"diffusion_model", "llm"}, Optional: []string{"vae", "tae"}, OneOf: []string{"vae", "tae"}},
	"z-image":             {Required: []string{"diffusion_model", "llm"}, Optional: []string{"vae", "tae"}, OneOf: []string{"vae", "tae"}},
	"chroma":              {Required: []string{"diffusion_model", "t5xxl"}, Optional: []string{"vae", "tae"}, OneOf: []string{"vae", "tae"}},
	"flux1-dev":           {Required: []string{"diffusion_model", "clip_l", "t5xxl"}, Optional: []string{"vae", "tae"}, OneOf: []string{"vae", "tae"}},
	"flux1-schnell":       {Required: []string{"diffusion_model", "clip_l", "t5xxl"}, Optional: []string{"vae", "tae"}, OneOf: []string{"vae", "tae"}},
	"sdxl-checkpoint":     {Required: []string{"model"}, Optional: []string{"vae", "tae"}},
}

// Recipes returns the recipe IDs a custom set may use: the catalog's
// recipes first, in catalog order, then the rest alphabetically.
func Recipes() []string {
	var ids []string
	for _, e := range Catalog() {
		if e.Recipe != "" && !slices.Contains(ids, e.Recipe) {
			ids = append(ids, e.Recipe)
		}
	}
	var rest []string
	for id := range RecipeSlots {
		if !slices.Contains(ids, id) {
			rest = append(rest, id)
		}
	}
	slices.Sort(rest)
	return append(ids, rest...)
}

// AceSlots are the four files every ACE-Step set needs.
var AceSlots = []string{"lm", "dit", "embedding", "vae"}

// RecipeCapabilities returns the capabilities a custom set gets from its
// recipe and files: alpha output for Ming-Image, and editing for the
// recipes whose sd.cpp docs describe it. Qwen-Image edits need the vision
// weights, so those recipes edit only with an llm_vision file.
func RecipeCapabilities(recipe string, comps []Component) []string {
	hasSlot := func(slot string) bool {
		return slices.ContainsFunc(comps, func(c Component) bool { return c.Slot == slot })
	}
	switch recipe {
	case "ming-image":
		return []string{"alpha"}
	case "flux2-klein-4b":
		return []string{"edit"}
	case "qwen-image-21", "qwen-image-21-turbo":
		if hasSlot("llm_vision") {
			return []string{"edit"}
		}
	}
	return nil
}

// ValidateComponents checks that a set's components fill its slots: every
// required slot (and one of each OneOf group) is present, no slot is
// unknown or repeated. It does not look at the files.
func ValidateComponents(kind BackendKind, recipe string, comps []Component) error {
	var slots RecipeSlotSet
	switch kind {
	case KindSD:
		var ok bool
		if slots, ok = RecipeSlots[recipe]; !ok {
			return fmt.Errorf("unknown recipe %q", recipe)
		}
	case KindAce:
		slots = RecipeSlotSet{Required: AceSlots}
	default:
		return fmt.Errorf("unknown backend kind %q", kind)
	}
	have := map[string]bool{}
	for _, c := range comps {
		if !slices.Contains(slots.Slots(), c.Slot) {
			return fmt.Errorf("slot %q is not used by %s", c.Slot, describeSet(kind, recipe))
		}
		if have[c.Slot] {
			return fmt.Errorf("slot %q is given twice", c.Slot)
		}
		have[c.Slot] = true
	}
	for _, s := range slots.Required {
		if !have[s] {
			return fmt.Errorf("%s needs a %q file", describeSet(kind, recipe), s)
		}
	}
	if len(slots.OneOf) > 0 && !slices.ContainsFunc(slots.OneOf, func(s string) bool { return have[s] }) {
		return fmt.Errorf("%s needs one of: %s", describeSet(kind, recipe), strings.Join(slots.OneOf, ", "))
	}
	return nil
}

func describeSet(kind BackendKind, recipe string) string {
	if kind == KindAce {
		return "an ACE-Step set"
	}
	return "recipe " + recipe
}

// CheckFiles checks that every component path is an absolute path to an
// existing regular file.
func CheckFiles(comps []Component) error {
	for _, c := range comps {
		if !strings.HasPrefix(c.Path, "/") {
			return fmt.Errorf("%s: %q is not an absolute path", c.Slot, c.Path)
		}
		info, err := os.Stat(c.Path)
		if err != nil {
			return fmt.Errorf("%s: %w", c.Slot, err)
		}
		if !info.Mode().IsRegular() {
			return fmt.Errorf("%s: %s is not a regular file", c.Slot, c.Path)
		}
	}
	return nil
}

// deniedArgs are flags toolchest sets itself; letting a set or instance
// set them would break routing, the port range or the VRAM reservation.
var deniedArgs = []string{"--listen-ip", "--listen-port", "--host", "--port", "--max-vram", "--models"}

// ValidateExtraArgs splits extra backend flags the way a shell would
// (spaces separate words; single or double quotes group them) and refuses
// the flags toolchest sets itself.
func ValidateExtraArgs(s string) ([]string, error) {
	args, err := splitWords(s)
	if err != nil {
		return nil, err
	}
	for _, a := range args {
		name, _, _ := strings.Cut(a, "=")
		if slices.Contains(deniedArgs, name) {
			return nil, fmt.Errorf("%s is set by llama-toolchest and cannot be given here", name)
		}
	}
	return args, nil
}

func splitWords(s string) ([]string, error) {
	var (
		words []string
		cur   strings.Builder
		quote rune
		inArg bool
	)
	for _, r := range s {
		switch {
		case quote != 0:
			if r == quote {
				quote = 0
			} else {
				cur.WriteRune(r)
			}
		case r == '"' || r == '\'':
			quote, inArg = r, true
		case r == ' ' || r == '\t' || r == '\n':
			if inArg {
				words = append(words, cur.String())
				cur.Reset()
				inArg = false
			}
		default:
			cur.WriteRune(r)
			inArg = true
		}
	}
	if quote != 0 {
		return nil, errors.New("unclosed quote in the extra arguments")
	}
	if inArg {
		words = append(words, cur.String())
	}
	return words, nil
}

var slugRE = regexp.MustCompile(`[^a-z0-9]+`)

// Slug turns a name into an ID: lower case, with runs of anything other
// than letters and digits replaced by one hyphen.
func Slug(name string) string {
	return strings.Trim(slugRE.ReplaceAllString(strings.ToLower(name), "-"), "-")
}
