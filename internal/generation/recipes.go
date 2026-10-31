package generation

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
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
	"ming-image":           {Required: []string{"diffusion_model", "llm", "tokenizer"}, Optional: []string{"vae", "tae"}, OneOf: []string{"vae", "tae"}},
	"qwen-image-21-turbo":  {Required: []string{"diffusion_model", "llm"}, Optional: []string{"llm_vision", "vae", "tae"}, OneOf: []string{"vae", "tae"}},
	"qwen-image-21":        {Required: []string{"diffusion_model", "llm"}, Optional: []string{"llm_vision", "vae", "tae"}, OneOf: []string{"vae", "tae"}},
	"flux2-klein-4b":       {Required: []string{"diffusion_model", "llm"}, Optional: []string{"vae", "tae"}, OneOf: []string{"vae", "tae"}},
	"z-image-turbo":        {Required: []string{"diffusion_model", "llm"}, Optional: []string{"vae", "tae"}, OneOf: []string{"vae", "tae"}},
	"flux3-klein":          {Required: []string{"diffusion_model", "llm"}, Optional: []string{"vae", "tae"}, OneOf: []string{"vae", "tae"}},
	"qwen-image-edit-2511": {Required: []string{"diffusion_model", "llm"}, Optional: []string{"llm_vision", "vae", "tae"}, OneOf: []string{"vae", "tae"}},
	"z-image":              {Required: []string{"diffusion_model", "llm"}, Optional: []string{"vae", "tae"}, OneOf: []string{"vae", "tae"}},
	"chroma":               {Required: []string{"diffusion_model", "t5xxl"}, Optional: []string{"vae", "tae"}, OneOf: []string{"vae", "tae"}},
	"flux1-dev":            {Required: []string{"diffusion_model", "clip_l", "t5xxl"}, Optional: []string{"vae", "tae"}, OneOf: []string{"vae", "tae"}},
	"flux1-schnell":        {Required: []string{"diffusion_model", "clip_l", "t5xxl"}, Optional: []string{"vae", "tae"}, OneOf: []string{"vae", "tae"}},
	"sdxl-checkpoint":      {Required: []string{"model"}, Optional: []string{"vae", "tae"}},
}

// Video capabilities: what a video set can start from. A set without
// CapText needs a start image.
const (
	CapText       = "text"
	CapStartImage = "start_image"
	CapEndImage   = "end_image"
)

// VideoRecipe is a video recipe's slots, and what its model can start
// from.
type VideoRecipe struct {
	Slots RecipeSlotSet
	// Caps comes from the recipe's features (init_image gives
	// CapStartImage, end_image CapEndImage), plus CapText when the model
	// can work from a prompt alone.
	Caps []string
}

// VideoRecipes holds the video recipes a set may use, copied from
// stable-diffusion.cpp examples/server/recipes/*.json at
// master-1000-95ff762 (2026-10-24). wan21-vace and wan22-s2v are left
// out: they need control videos or audio, which toolchest does not send.
var VideoRecipes = map[string]VideoRecipe{
	"wan22-ti2v-5b":        {wanSlots(), []string{CapText, CapStartImage}},
	"wan22-i2v":            {wanMoESlots(), []string{CapStartImage, CapEndImage}},
	"wan22-t2v":            {wanMoESlots(), []string{CapText}},
	"wan21-t2v":            {wanSlots(), []string{CapText}},
	"wan21-i2v":            {RecipeSlotSet{Required: []string{"diffusion_model", "t5xxl"}, Optional: []string{"vae", "tae", "clip_vision"}, OneOf: []string{"vae", "tae"}}, []string{CapStartImage}},
	"wan21-flf2v":          {RecipeSlotSet{Required: []string{"diffusion_model", "t5xxl", "clip_vision"}, Optional: []string{"vae", "tae"}, OneOf: []string{"vae", "tae"}}, []string{CapStartImage, CapEndImage}},
	"ltx23-distilled":      {ltxSlots(), []string{CapText, CapStartImage, CapEndImage}},
	"ltx23-dev":            {ltxSlots(), []string{CapText, CapStartImage, CapEndImage}},
	"ltx25-dev":            {RecipeSlotSet{Required: []string{"diffusion_model", "llm"}, Optional: []string{"vae", "tae", "audio_vae"}, OneOf: []string{"vae", "tae"}}, []string{CapText, CapStartImage, CapEndImage}},
	"hunyuan-video-15-t2v": {hunyuanSlots(), []string{CapText}},
	"hunyuan-video-15-i2v": {hunyuanSlots(), []string{CapStartImage}},
	"lingbot-video-dense":  {llmVideoSlots(), []string{CapText, CapStartImage}},
	"lingbot-video-moe":    {llmVideoSlots(), []string{CapText, CapStartImage}},
}

func wanSlots() RecipeSlotSet {
	return RecipeSlotSet{Required: []string{"diffusion_model", "t5xxl"}, Optional: []string{"vae", "tae"}, OneOf: []string{"vae", "tae"}}
}

// wanMoESlots are the Wan2.2 A14B pairs: a low-noise and a high-noise
// expert, each with an optional LoRA (such as the Lightning 4-step ones).
func wanMoESlots() RecipeSlotSet {
	return RecipeSlotSet{Required: []string{"diffusion_model", "high_noise_diffusion_model", "t5xxl"},
		Optional: []string{"vae", "tae", "lora", "high_noise_lora"}, OneOf: []string{"vae", "tae"}}
}

// LoraSlots are the LoRA slots, and whether each one is for the
// high-noise expert. sd-server has no flag per LoRA file: it is given the
// folder the files are in (--lora-model-dir), and each request names the
// LoRAs to apply.
var LoraSlots = map[string]bool{"lora": false, "high_noise_lora": true}

// Loras returns the set's LoRA components.
func (m ModelSet) Loras() []Component {
	var out []Component
	for _, c := range m.Components {
		if _, ok := LoraSlots[c.Slot]; ok {
			out = append(out, c)
		}
	}
	return out
}

// LoraDir returns the folder all of the set's LoRA files are in, or "" for
// a set without LoRAs. sd-server only uses LoRA files inside its
// --lora-model-dir and skips links that point outside it, so the files
// must share one folder.
func LoraDir(set ModelSet) (string, error) {
	dir := ""
	for _, c := range set.Loras() {
		d := filepath.Dir(c.Path)
		if dir != "" && d != dir {
			return "", fmt.Errorf("the LoRA files of a set must be in one folder: %s and %s", dir, d)
		}
		dir = d
	}
	return dir, nil
}

// CheckLoraArgs refuses --lora-model-dir in a set's or instance's flags
// when the set has LoRAs of its own: toolchest sets that flag for them.
func CheckLoraArgs(set ModelSet, args []string) error {
	if len(set.Loras()) == 0 {
		return nil
	}
	for _, a := range args {
		if name, _, _ := strings.Cut(a, "="); name == "--lora-model-dir" {
			return fmt.Errorf("--lora-model-dir is set by llama-toolchest for %s, which has LoRA files of its own", set.Name)
		}
	}
	return nil
}

func ltxSlots() RecipeSlotSet {
	return RecipeSlotSet{Required: []string{"diffusion_model", "llm", "embeddings_connectors"}, Optional: []string{"vae", "tae", "audio_vae"}, OneOf: []string{"vae", "tae"}}
}

func hunyuanSlots() RecipeSlotSet {
	return RecipeSlotSet{Required: []string{"diffusion_model", "llm", "t5xxl"}, Optional: []string{"vae", "tae"}, OneOf: []string{"vae", "tae"}}
}

func llmVideoSlots() RecipeSlotSet {
	return RecipeSlotSet{Required: []string{"diffusion_model", "llm"}, Optional: []string{"vae", "tae"}, OneOf: []string{"vae", "tae"}}
}

// Recipes returns the image recipe IDs a custom set may use: the
// catalog's recipes first, in catalog order, then the rest alphabetically.
func Recipes() []string {
	return recipeIDs(KindSD, func(id string) bool { _, ok := RecipeSlots[id]; return ok }, keys(RecipeSlots))
}

// VideoRecipeIDs is Recipes for video sets.
func VideoRecipeIDs() []string {
	return recipeIDs(KindVideo, func(id string) bool { _, ok := VideoRecipes[id]; return ok }, keys(VideoRecipes))
}

func recipeIDs(kind BackendKind, known func(string) bool, all []string) []string {
	var ids []string
	for _, e := range Catalog() {
		if e.Kind == kind && e.Recipe != "" && known(e.Recipe) && !slices.Contains(ids, e.Recipe) {
			ids = append(ids, e.Recipe)
		}
	}
	var rest []string
	for _, id := range all {
		if !slices.Contains(ids, id) {
			rest = append(rest, id)
		}
	}
	slices.Sort(rest)
	return append(ids, rest...)
}

func keys[V any](m map[string]V) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}

// AceSlots are the four files every ACE-Step set needs.
var AceSlots = []string{"lm", "dit", "embedding", "vae"}

// UpscalerSlot is the one slot of an upscaler set: an ESRGAN model file
// (.pth or .safetensors).
const UpscalerSlot = "upscaler"

// RecipeCapabilities returns the capabilities a custom set gets from its
// recipe and files: alpha output for Ming-Image, and editing for the
// recipes whose sd.cpp docs describe it. Qwen-Image edits need the vision
// weights, so those recipes edit only with an llm_vision file. A video
// recipe gives what its model can start from (VideoRecipe.Caps).
func RecipeCapabilities(recipe string, comps []Component) []string {
	if r, ok := VideoRecipes[recipe]; ok {
		return slices.Clone(r.Caps)
	}
	hasSlot := func(slot string) bool {
		return slices.ContainsFunc(comps, func(c Component) bool { return c.Slot == slot })
	}
	switch recipe {
	case "ming-image":
		return []string{"alpha"}
	case "flux2-klein-4b", "flux3-klein":
		return []string{"edit"}
	case "qwen-image-21", "qwen-image-21-turbo", "qwen-image-edit-2511":
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
			if _, video := VideoRecipes[recipe]; video {
				return fmt.Errorf("recipe %s is a video recipe; make the set a video set", recipe)
			}
			return fmt.Errorf("unknown recipe %q", recipe)
		}
	case KindVideo:
		r, ok := VideoRecipes[recipe]
		if !ok {
			if _, image := RecipeSlots[recipe]; image {
				return fmt.Errorf("recipe %s is an image recipe; make the set an image set", recipe)
			}
			return fmt.Errorf("unknown video recipe %q", recipe)
		}
		slots = r.Slots
	case KindAce:
		slots = RecipeSlotSet{Required: AceSlots}
	case KindUpscale:
		slots = RecipeSlotSet{Required: []string{UpscalerSlot}}
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
	switch kind {
	case KindAce:
		return "an ACE-Step set"
	case KindUpscale:
		return "an upscaler set"
	case KindVideo:
		return "video recipe " + recipe
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
