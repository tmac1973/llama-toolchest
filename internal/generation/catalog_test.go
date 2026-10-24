package generation

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func TestCatalogEntriesAreValid(t *testing.T) {
	seen := map[string]bool{}
	for _, e := range Catalog() {
		if seen[e.ID] {
			t.Errorf("catalog ID %q repeated", e.ID)
		}
		seen[e.ID] = true
		if err := ValidateComponents(e.Kind, e.Recipe, e.Components); err != nil {
			t.Errorf("%s: %v", e.ID, err)
		}
		for _, c := range e.Components {
			if c.Repo == "" || c.File == "" || c.Bytes <= 0 {
				t.Errorf("%s/%s: repo, file and size are all needed: %+v", e.ID, c.Slot, c)
			}
		}
		if e.Kind == KindSD && (e.DefaultWidth == 0 || e.DefaultHeight == 0) {
			t.Errorf("%s: an image set needs a default test size", e.ID)
		}
		if _, err := ValidateExtraArgs(strings.Join(e.LaunchArgs, " ")); err != nil {
			t.Errorf("%s: launch args: %v", e.ID, err)
		}
	}
}

// Ming-Image is the one freely licensed model with an alpha channel, and
// Haruspex uses it, so it leads the image sets.
func TestCatalogMingImageFirst(t *testing.T) {
	for _, e := range Catalog() {
		if e.Kind != KindSD {
			continue
		}
		if e.ID != "ming-image" || !slices.Contains(e.Capabilities, "alpha") {
			t.Fatalf("first image set = %s %v, want ming-image with alpha", e.ID, e.Capabilities)
		}
		if e.Components[0].Repo != "voltaire321/Ming-Image-0.1-Design-GGUF" {
			t.Errorf("Ming-Image DiT comes from %s", e.Components[0].Repo)
		}
		return
	}
	t.Fatal("no image sets in the catalog")
}

// The two ACE-Step sets differ only in their DiT, so a second download
// fetches one file.
func TestCatalogAceSetsShareFiles(t *testing.T) {
	turbo, _ := LookupCatalog("ace-step-turbo")
	xl, _ := LookupCatalog("ace-step-xl-sft")
	differ := 0
	for i := range turbo.Components {
		if turbo.Components[i].File != xl.Components[i].File {
			differ++
		}
	}
	if differ != 1 {
		t.Errorf("ACE-Step sets differ in %d files, want 1 (the DiT)", differ)
	}
}

func TestValidateComponents(t *testing.T) {
	ok := []Component{{Slot: "diffusion_model"}, {Slot: "llm"}, {Slot: "tokenizer"}, {Slot: "vae"}}
	if err := ValidateComponents(KindSD, "ming-image", ok); err != nil {
		t.Errorf("valid Ming-Image set: %v", err)
	}
	for name, tc := range map[string]struct {
		kind   BackendKind
		recipe string
		comps  []Component
		want   string
	}{
		"missing required": {KindSD, "ming-image", ok[:2], `needs a "tokenizer"`},
		"no vae or tae":    {KindSD, "ming-image", ok[:3], "needs one of: vae, tae"},
		"unknown slot":     {KindSD, "sdxl-checkpoint", []Component{{Slot: "model"}, {Slot: "llm"}}, `slot "llm" is not used`},
		"repeated slot":    {KindSD, "sdxl-checkpoint", []Component{{Slot: "model"}, {Slot: "model"}}, "given twice"},
		"unknown recipe":   {KindSD, "dalle", nil, "unknown recipe"},
		"ace incomplete":   {KindAce, "", []Component{{Slot: "lm"}, {Slot: "dit"}}, `needs a "embedding"`},
	} {
		err := ValidateComponents(tc.kind, tc.recipe, tc.comps)
		if err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: %v, want %q", name, err, tc.want)
		}
	}
}

func TestCheckFiles(t *testing.T) {
	f := filepath.Join(t.TempDir(), "m.gguf")
	if err := os.WriteFile(f, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := CheckFiles([]Component{{Slot: "model", Path: f}}); err != nil {
		t.Errorf("existing file: %v", err)
	}
	for _, p := range []string{"relative/m.gguf", f + ".missing", filepath.Dir(f)} {
		if err := CheckFiles([]Component{{Slot: "model", Path: p}}); err == nil {
			t.Errorf("%q accepted", p)
		}
	}
}

func TestValidateExtraArgs(t *testing.T) {
	got, err := ValidateExtraArgs(`--sigmas qwen-image-2.1-turbo --prompt-prefix "a b" --x='c d'`)
	want := []string{"--sigmas", "qwen-image-2.1-turbo", "--prompt-prefix", "a b", "--x=c d"}
	if err != nil || !slices.Equal(got, want) {
		t.Errorf("ValidateExtraArgs = %q, %v; want %q", got, err, want)
	}
	for _, bad := range []string{"--max-vram 4", "--listen-port=9000", "--models /x", `--x "unclosed`} {
		if _, err := ValidateExtraArgs(bad); err == nil {
			t.Errorf("%q accepted", bad)
		}
	}
	if got, err := ValidateExtraArgs("  "); err != nil || len(got) != 0 {
		t.Errorf("blank = %q, %v", got, err)
	}
}

func TestSlug(t *testing.T) {
	for in, want := range map[string]string{
		"Qwen Turbo (local)": "qwen-turbo-local",
		"  ming--image  ":    "ming-image",
		"ACE_Step 1.5":       "ace-step-1-5",
	} {
		if got := Slug(in); got != want {
			t.Errorf("Slug(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestRecipeCapabilities(t *testing.T) {
	if got := RecipeCapabilities("ming-image", nil); !slices.Equal(got, []string{"alpha"}) {
		t.Errorf("ming-image = %v", got)
	}
	if got := RecipeCapabilities("qwen-image-21-turbo", []Component{{Slot: "llm"}}); got != nil {
		t.Errorf("qwen without vision weights = %v, want none", got)
	}
	if got := RecipeCapabilities("qwen-image-21-turbo", []Component{{Slot: "llm_vision"}}); !slices.Equal(got, []string{"edit"}) {
		t.Errorf("qwen with vision weights = %v, want edit", got)
	}
	if got := RecipeCapabilities("sdxl-checkpoint", nil); got != nil {
		t.Errorf("sdxl = %v", got)
	}
}

func TestRecipesOrder(t *testing.T) {
	got := Recipes()
	if len(got) != len(RecipeSlots) || got[0] != "ming-image" {
		t.Errorf("Recipes = %v, want every recipe, ming-image first", got)
	}
}
