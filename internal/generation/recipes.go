package generation

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
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
	// OneOf lists groups of slots; at least one slot of each group must
	// be filled (upstream's required_any).
	OneOf [][]string
}

// Slots returns every slot the recipe accepts, required first.
func (r RecipeSlotSet) Slots() []string {
	return append(slices.Clone(r.Required), r.Optional...)
}

// Group returns the OneOf group a slot is in, or nil.
func (r RecipeSlotSet) Group(slot string) []string {
	for _, g := range r.OneOf {
		if slices.Contains(g, slot) {
			return g
		}
	}
	return nil
}

// Video capabilities: what a video set can start from. A set without
// CapText needs a start image. A set with CapAudio needs an audio track
// (speech-to-video), sent to sd-server as ref_audios.
const (
	CapText       = "text"
	CapStartImage = "start_image"
	CapEndImage   = "end_image"
	CapAudio      = "audio"
)

// RecipeSizeMultiple returns a video recipe's size multiple, or 0 for an
// image recipe or an unknown one.
func RecipeSizeMultiple(recipe string) int {
	if r, ok := ActiveRecipes().Get(recipe); ok && r.Kind == KindVideo {
		return r.SizeMultiple
	}
	return 0
}

// SlotLabels names each slot for people, for any recipe. A recipe's own
// labels (recipeLabels, then upstream's) come first. See SlotLabel.
var SlotLabels = map[string]string{
	"model":                      "Checkpoint (one file)",
	"diffusion_model":            "Diffusion model",
	"high_noise_diffusion_model": "High-noise expert",
	"llm":                        "Text model",
	"llm_vision":                 "Vision projector (for edits)",
	"clip_l":                     "CLIP-L text encoder",
	"t5xxl":                      "T5-XXL text encoder",
	"clip_vision":                "CLIP vision encoder",
	"tokenizer":                  "Tokenizer (tokenizer.json)",
	"embeddings_connectors":      "Embeddings connectors",
	"vae":                        "VAE",
	"tae":                        "Tiny autoencoder (TAE)",
	"audio_vae":                  "Audio VAE",
	"audio_encoder":              "Audio encoder",
	"lora":                       "LoRA",
	"high_noise_lora":            "High-noise LoRA",
	"lm":                         "Language model (5 Hz LM)",
	"dit":                        "DiT (music model)",
	"embedding":                  "Text embedding model",
	UpscalerSlot:                 "ESRGAN upscaler",
	// Slots that toolchest has no sd-server flag for. They are named in
	// the reason a recipe cannot be used.
	"clip_g":                 "CLIP-G text encoder",
	"control_net":            "ControlNet",
	"ip_adapter":             "IP-Adapter",
	"photo_maker":            "PhotoMaker",
	"pulid_weights":          "PuLID weights",
	"motion_module":          "motion module",
	"uncond_diffusion_model": "unconditional diffusion model",
}

// SlotLabel returns a slot's name for people: toolchest's label for the
// recipe, else upstream's, else the shared one, else the slot ID.
func SlotLabel(recipe, slot string) string {
	if l := vupscaleSlotLabel(recipe, slot); l != "" {
		return l
	}
	if r, ok := ActiveRecipes().Get(recipe); ok {
		return r.slotLabel(slot)
	}
	if l := recipeLabels[recipe][slot]; l != "" {
		return l
	}
	if l := SlotLabels[slot]; l != "" {
		return l
	}
	return slot
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

// Recipes returns the image recipe IDs a custom set may use: the
// catalog's recipes first, in catalog order, then the rest A–Z.
func Recipes() []string { return ActiveRecipes().IDs(KindSD) }

// VideoRecipeIDs is Recipes for video sets.
func VideoRecipeIDs() []string { return ActiveRecipes().IDs(KindVideo) }

// AceSlots are the four files every ACE-Step set needs.
var AceSlots = []string{"lm", "dit", "embedding", "vae"}

// UpscalerSlot is the one slot of an upscaler set: an ESRGAN model file
// (.pth or .safetensors).
const UpscalerSlot = "upscaler"

// VUpscaleScripts are the DGL scripts a video upscaler set can run, kept
// in the set's Recipe.
var VUpscaleScripts = []string{"relayvsr", "seedvr2"}

// VUpscaleSlots are the two files every video upscaler set needs.
var VUpscaleSlots = []string{"dit", "vae"}

// VUpscaleScales are the scales a video upscaler script can make: RelayVSR
// files are made for ×2 or ×4; SeedVR2's scale is a script setting.
func VUpscaleScales(script string) []int {
	switch script {
	case "relayvsr":
		return []int{2, 4}
	case "seedvr2":
		return []int{2, 3, 4}
	}
	return nil
}

// relayvsrScaleName finds the scale in a RelayVSR file name: "x2" or "x4"
// between separators, as in relayvsr_x2_lightx2v_int8_convrot.safetensors
// (which matches _x2_, not lightx2v).
var relayvsrScaleName = regexp.MustCompile(`(^|[_.-])x([24])([_.-]|$)`)

// RelayVSRFileScale returns the scale a RelayVSR file name says it is for,
// or 0 when the name does not say.
func RelayVSRFileScale(name string) int {
	m := relayvsrScaleName.FindStringSubmatch(strings.ToLower(filepath.Base(name)))
	if m == nil {
		return 0
	}
	return int(m[2][0] - '0')
}

// vupscaleSlotLabel names a video upscaler set's slots for people.
func vupscaleSlotLabel(script, slot string) string {
	names := map[string][2]string{
		"relayvsr": {"RelayVSR diffusion model", "RelayVSR VAE (same scale)"},
		"seedvr2":  {"SeedVR2 diffusion model", "SeedVR2 VAE"},
	}
	n, ok := names[script]
	if !ok {
		return ""
	}
	switch slot {
	case "dit":
		return n[0]
	case "vae":
		return n[1]
	}
	return ""
}

// CapEdit is an image set that can edit images.
const CapEdit = "edit"

// RecipeCapabilities returns the capabilities a custom set gets from its
// recipe and files. A video recipe gives what its model can start from:
// start_image from init_image, end_image from end_image, and text when it
// has no init_image or the overlay says so (recipeTextStart). An image
// recipe edits when it takes reference images (ref_images); one with a
// vision projector slot (llm_vision) edits only when that file is given.
// Ming-Image makes images with alpha.
func RecipeCapabilities(recipe string, comps []Component) []string {
	r, ok := ActiveRecipes().Get(recipe)
	if !ok {
		return nil
	}
	if r.Kind == KindVideo {
		var caps []string
		if recipeTextStart[r.ID] || !r.HasFeature("init_image") {
			caps = append(caps, CapText)
		}
		if r.HasFeature("init_image") {
			caps = append(caps, CapStartImage)
		}
		if r.HasFeature("end_image") {
			caps = append(caps, CapEndImage)
		}
		if r.HasFeature("ref_audios") && recipeAudioInput[r.ID] {
			caps = append(caps, CapAudio)
		}
		return caps
	}
	var caps []string
	if recipeAlpha[r.ID] {
		caps = append(caps, "alpha")
	}
	if r.HasFeature("ref_images") {
		vision := !slices.Contains(r.Slots.Slots(), "llm_vision") ||
			slices.ContainsFunc(comps, func(c Component) bool { return c.Slot == "llm_vision" })
		if vision {
			caps = append(caps, CapEdit)
		}
	}
	return caps
}

// ValidateComponents checks that a set's components fill its slots: every
// required slot (and one of each OneOf group) is present, no slot is
// unknown or repeated. It does not look at the files.
func ValidateComponents(kind BackendKind, recipe string, comps []Component) error {
	var slots RecipeSlotSet
	switch kind {
	case KindSD, KindVideo:
		r, ok := ActiveRecipes().Get(recipe)
		switch {
		case !ok && kind == KindVideo:
			return fmt.Errorf("unknown video recipe %q", recipe)
		case !ok:
			return fmt.Errorf("unknown recipe %q", recipe)
		case r.Kind == KindVideo && kind == KindSD:
			return fmt.Errorf("recipe %s is a video recipe; make the set a video set", recipe)
		case r.Kind == KindSD && kind == KindVideo:
			return fmt.Errorf("recipe %s is an image recipe; make the set an image set", recipe)
		case r.Unavailable != "":
			return fmt.Errorf("recipe %s cannot be used: %s", recipe, r.Unavailable)
		}
		slots = r.Slots
	case KindAce:
		slots = RecipeSlotSet{Required: AceSlots}
	case KindUpscale:
		slots = RecipeSlotSet{Required: []string{UpscalerSlot}}
	case KindVUpscale:
		if !slices.Contains(VUpscaleScripts, recipe) {
			return fmt.Errorf("unknown video upscaler script %q: use relayvsr or seedvr2", recipe)
		}
		slots = RecipeSlotSet{Required: VUpscaleSlots}
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
	for _, g := range slots.OneOf {
		if !slices.ContainsFunc(g, func(s string) bool { return have[s] }) {
			return fmt.Errorf("%s needs one of: %s", describeSet(kind, recipe), strings.Join(g, ", "))
		}
	}
	return nil
}

func describeSet(kind BackendKind, recipe string) string {
	switch kind {
	case KindAce:
		return "an ACE-Step set"
	case KindUpscale:
		return "an upscaler set"
	case KindVUpscale:
		return "a video upscaler set"
	case KindVideo:
		return "video recipe " + recipe
	}
	return "recipe " + recipe
}

// CheckFiles checks that every component path is an absolute path to an
// existing regular file. A missing file's error names the models folder,
// and in a container also says that host folders are not visible.
func CheckFiles(comps []Component, modelsDir string, inContainer bool) error {
	for _, c := range comps {
		if !strings.HasPrefix(c.Path, "/") {
			return fmt.Errorf("%s: %q is not an absolute path", c.Slot, c.Path)
		}
		info, err := os.Stat(c.Path)
		if errors.Is(err, os.ErrNotExist) {
			msg := fmt.Sprintf("%s: %s does not exist. The models folder is %s.", c.Slot, c.Path, modelsDir)
			if inContainer {
				msg += " Toolchest runs in a container and cannot see host folders. Put the file under the models folder, or mount its folder there."
			}
			return errors.New(msg)
		}
		if err != nil {
			return fmt.Errorf("%s: %w", c.Slot, err)
		}
		if !info.Mode().IsRegular() {
			return fmt.Errorf("%s: %s is not a regular file", c.Slot, c.Path)
		}
	}
	return nil
}

// CheckFileFormat reads the start of a file and refuses one that cannot
// be what its slot needs: the tokenizer slot takes a JSON file, every
// other slot a GGUF, safetensors or PyTorch model file. It does not check
// that the model is the right one for the slot.
func CheckFileFormat(slot, path string) error {
	f, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("%s: %w", slot, err)
	}
	defer f.Close()
	head := make([]byte, 16)
	n, _ := io.ReadFull(f, head)
	head = head[:n]
	info, err := f.Stat()
	if err != nil {
		return fmt.Errorf("%s: %w", slot, err)
	}
	if slot == "tokenizer" {
		if trimmed := bytes.TrimLeft(head, " \t\r\n"); len(trimmed) > 0 && trimmed[0] == '{' {
			return nil
		}
		return fmt.Errorf("%s: %s is not a tokenizer JSON file", slot, path)
	}
	if isModelFile(head, info.Size()) {
		return nil
	}
	return fmt.Errorf("%s: %s is not a model file (GGUF, safetensors or PyTorch)", slot, path)
}

// isModelFile reports whether a file's first bytes are those of a GGUF,
// safetensors or PyTorch file. A safetensors file starts with the length
// of its JSON header (little-endian), then the header's "{". A PyTorch
// file is a zip archive, or an older pickle.
func isModelFile(head []byte, size int64) bool {
	switch {
	case bytes.HasPrefix(head, []byte("GGUF")):
		return true
	case bytes.HasPrefix(head, []byte("PK\x03\x04")):
		return true
	case len(head) > 0 && head[0] == 0x80:
		return true
	case len(head) > 8 && head[8] == '{':
		n := binary.LittleEndian.Uint64(head[:8])
		return n >= 2 && n < uint64(size)
	}
	return false
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
