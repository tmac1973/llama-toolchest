package generation

import (
	"slices"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/builder"
)

// Every slot a video recipe can take must have an sd-server flag, or a
// set using it could not be launched.
func TestVideoRecipeSlotsHaveFlags(t *testing.T) {
	for id, r := range VideoRecipes {
		for _, slot := range r.Slots.Slots() {
			if sdSlotFlags[slot] == "" {
				t.Errorf("%s: slot %q has no sd-server flag", id, slot)
			}
		}
		if len(r.Caps) == 0 {
			t.Errorf("%s: no capabilities", id)
		}
	}
}

// Image and video recipes are kept apart: an image recipe makes no video
// set, nor a video recipe an image set.
func TestValidateComponentsKeepsKindsApart(t *testing.T) {
	ti2v := []Component{{Slot: "diffusion_model"}, {Slot: "t5xxl"}, {Slot: "vae"}}
	if err := ValidateComponents(KindVideo, "wan22-ti2v-5b", ti2v); err != nil {
		t.Errorf("valid TI2V set: %v", err)
	}
	if err := ValidateComponents(KindSD, "wan22-ti2v-5b", ti2v); err == nil || !strings.Contains(err.Error(), "video recipe") {
		t.Errorf("video recipe for an image set: %v", err)
	}
	ming := []Component{{Slot: "diffusion_model"}, {Slot: "llm"}, {Slot: "tokenizer"}, {Slot: "vae"}}
	if err := ValidateComponents(KindVideo, "ming-image", ming); err == nil || !strings.Contains(err.Error(), "image recipe") {
		t.Errorf("image recipe for a video set: %v", err)
	}
	pair := []Component{{Slot: "diffusion_model"}, {Slot: "t5xxl"}, {Slot: "vae"}}
	if err := ValidateComponents(KindVideo, "wan22-i2v", pair); err == nil || !strings.Contains(err.Error(), "high_noise_diffusion_model") {
		t.Errorf("A14B pair without its high-noise expert: %v", err)
	}
}

func TestVideoRecipeCapabilities(t *testing.T) {
	for recipe, want := range map[string][]string{
		"wan22-ti2v-5b":   {CapText, CapStartImage},
		"wan22-i2v":       {CapStartImage, CapEndImage},
		"ltx23-distilled": {CapText, CapStartImage, CapEndImage},
	} {
		if got := RecipeCapabilities(recipe, nil); !slices.Equal(got, want) {
			t.Errorf("%s = %v, want %v", recipe, got, want)
		}
	}
	got := RecipeCapabilities("wan22-ti2v-5b", nil)
	got[0] = "changed"
	if VideoRecipes["wan22-ti2v-5b"].Caps[0] != CapText {
		t.Error("RecipeCapabilities hands out the table's own slice")
	}
}

// Video recipe IDs list the catalog's video recipes first, then the rest,
// and never an image recipe.
func TestVideoRecipeIDs(t *testing.T) {
	got := VideoRecipeIDs()
	if len(got) != len(VideoRecipes) || got[0] != "wan22-ti2v-5b" {
		t.Errorf("VideoRecipeIDs = %v, want every video recipe, the catalog's first", got)
	}
	for _, id := range Recipes() {
		if _, video := VideoRecipes[id]; video {
			t.Errorf("Recipes lists video recipe %s", id)
		}
	}
}

// A video set launches sd-server like an image set, with a flag for each
// of its files, including the second expert of a Wan2.2 A14B pair.
func TestLaunchSpecVideo(t *testing.T) {
	set := ModelSet{ID: "pair", Kind: KindVideo, Recipe: "wan22-i2v", Complete: true, Components: []Component{
		{Slot: "diffusion_model", Path: "/m/low.gguf"},
		{Slot: "high_noise_diffusion_model", Path: "/m/high.gguf"},
		{Slot: "t5xxl", Path: "/m/umt5.gguf"},
		{Slot: "vae", Path: "/m/vae.safetensors"},
	}}
	inst := Instance{ID: "pair", Port: 8090, ReservedMiB: 20480}
	build := builder.BuildResult{ID: "sd-x-rocm", Profile: "rocm", BinaryPath: "/b/sd-x-rocm/sd-server"}
	cfg, err := LaunchSpec(inst, set, build, "0.0.0.0", t.TempDir(), LaunchOptions{})
	if err != nil {
		t.Fatal(err)
	}
	args := strings.Join(cfg.Args, " ")
	for _, want := range []string{"--diffusion-model /m/low.gguf", "--high-noise-diffusion-model /m/high.gguf", "--t5xxl /m/umt5.gguf", "--max-vram 20.0"} {
		if !strings.Contains(args, want) {
			t.Errorf("args %q lack %q", args, want)
		}
	}
	if cfg.HealthURL != "http://127.0.0.1:8090/v1/models" {
		t.Errorf("health URL %q", cfg.HealthURL)
	}
}

// A video catalog set needs a test video and a ceiling at least as big.
func TestCatalogVideoSets(t *testing.T) {
	found := false
	for _, e := range Catalog() {
		if e.Kind != KindVideo {
			if e.DefaultFrames != 0 || e.MaxFrames != 0 {
				t.Errorf("%s: video fields on a %s set", e.ID, e.Kind)
			}
			continue
		}
		found = true
		if e.DefaultFrames <= 0 || e.DefaultFPS <= 0 || e.DefaultWidth <= 0 || e.DefaultHeight <= 0 {
			t.Errorf("%s: needs a test video size, length and frame rate", e.ID)
		}
		if e.MaxFrames < e.DefaultFrames || e.MaxWidth < e.DefaultWidth || e.MaxHeight < e.DefaultHeight {
			t.Errorf("%s: ceiling %dx%d %d frames is smaller than the test video", e.ID, e.MaxWidth, e.MaxHeight, e.MaxFrames)
		}
		if _, ok := VideoRecipes[e.Recipe]; !ok {
			t.Errorf("%s: recipe %s is not a video recipe", e.ID, e.Recipe)
		}
		if !slices.Equal(e.Capabilities, VideoRecipes[e.Recipe].Caps) {
			t.Errorf("%s: capabilities %v differ from its recipe's %v", e.ID, e.Capabilities, VideoRecipes[e.Recipe].Caps)
		}
	}
	if !found {
		t.Error("no video set in the catalog")
	}
}

// Every catalog file comes from a repo checked to be ungated, so a
// one-click download never needs a Hugging Face token. A new repo is
// checked by hand and added here.
func TestCatalogReposAllowed(t *testing.T) {
	allowed := []string{
		"black-forest-labs/FLUX.2-small-decoder",
		"city96/t5-v1_1-xxl-encoder-gguf",
		"Comfy-Org/Ming-Image",
		"Comfy-Org/Qwen-Image-2.1",
		"Comfy-Org/Qwen-Image_ComfyUI",
		"Comfy-Org/z_image_turbo",
		"inclusionAI/Ming-Image-0.1-Design",
		"leejet/FLUX.2-klein-4B-GGUF",
		"leejet/flux-3-image-klein-GGUF",
		"leejet/Z-Image-Turbo-GGUF",
		"mradermacher/Qwen2.5-VL-7B-Instruct-GGUF",
		"Qwen/Qwen3-VL-4B-Instruct-GGUF",
		"Qwen/Qwen3-VL-8B-Instruct-GGUF",
		"Serveurperso/ACE-Step-1.5-GGUF",
		"silveroxides/Chroma1-Flash-GGUF",
		"silveroxides/Chroma-GGUF",
		"stabilityai/stable-diffusion-xl-base-1.0",
		"unsloth/Qwen3-4B-GGUF",
		"unsloth/Qwen3-4B-Instruct-2507-GGUF",
		"unsloth/Qwen-Image-2.1-Turbo-GGUF",
		"unsloth/Qwen-Image-Edit-2511-GGUF",
		"voltaire321/Ming-Image-0.1-Design-GGUF",
		// Video sets.
		"QuantStack/Wan2.2-TI2V-5B-GGUF",
		"city96/umt5-xxl-encoder-gguf",
		"Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
		"QuantStack/Wan2.2-I2V-A14B-GGUF",
		"Comfy-Org/Wan_2.1_ComfyUI_repackaged",
	}
	for _, e := range Catalog() {
		for _, c := range e.Components {
			if !slices.Contains(allowed, c.Repo) {
				t.Errorf("%s/%s: repo %s is not on the checked list", e.ID, c.Slot, c.Repo)
			}
		}
	}
}

// The Wan2.2 A14B pair shares its text model with TI2V 5B, so the second
// set downloads it once, and it has both experts.
func TestCatalogWanSetsShareText(t *testing.T) {
	ti2v, _ := LookupCatalog("wan22-ti2v-5b")
	pair, ok := LookupCatalog("wan22-i2v-a14b")
	if !ok {
		t.Fatal("no wan22-i2v-a14b in the catalog")
	}
	t5 := func(e CatalogEntry) Component {
		for _, c := range e.Components {
			if c.Slot == "t5xxl" {
				return c
			}
		}
		return Component{}
	}
	if t5(ti2v) != t5(pair) || t5(pair).Repo == "" {
		t.Errorf("text models differ: %+v vs %+v", t5(ti2v), t5(pair))
	}
	slots := map[string]bool{}
	for _, c := range pair.Components {
		slots[c.Slot] = true
	}
	if !slots["diffusion_model"] || !slots["high_noise_diffusion_model"] {
		t.Errorf("slots %v, want both experts", slots)
	}
}
