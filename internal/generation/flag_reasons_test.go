package generation

import (
	"slices"
	"testing"
)

// Every flag a catalog set runs with has a reason, so a set's Details never
// shows a flag without one.
func TestCatalogFlagsHaveReasons(t *testing.T) {
	for _, e := range Catalog() {
		for _, f := range SplitFlags(e.LaunchArgs) {
			if FlagReason(f[0], f[1], e) == "" {
				t.Errorf("%s: %s %s has no reason (add it to FlagReasons or the entry's FlagNotes)", e.ID, f[0], f[1])
			}
		}
	}
}

func TestToolchestFlagsHaveReasons(t *testing.T) {
	for _, f := range ToolchestFlags {
		if FlagReasons[f] == "" {
			t.Errorf("%s has no reason", f)
		}
	}
}

// A FlagNotes key must name a flag the entry runs with, so a note is not
// left behind when its flag is removed.
func TestFlagNotesMatchLaunchArgs(t *testing.T) {
	for _, e := range Catalog() {
		var keys []string
		for _, f := range SplitFlags(e.LaunchArgs) {
			keys = append(keys, f[0], f[0]+" "+f[1])
		}
		for k := range e.FlagNotes {
			if !slices.Contains(keys, k) {
				t.Errorf("%s: FlagNotes has %q, which is not in its launch args", e.ID, k)
			}
		}
	}
}

func TestFlagReasonOrder(t *testing.T) {
	e := CatalogEntry{FlagNotes: map[string]string{"--steps": "set note", "--cfg-scale 1": "set value note"}}
	for _, tc := range []struct{ flag, value, want string }{
		{"--steps", "4", "set note"},
		{"--cfg-scale", "1", "set value note"},
		{"--cfg-scale", "4", FlagReasons["--cfg-scale"]},
		{"--backend", "te=cpu", FlagReasons["--backend te=cpu"]},
		{"--backend", "vae=cpu", FlagReasons["--backend"]},
		{"--no-such-flag", "", ""},
	} {
		if got := FlagReason(tc.flag, tc.value, e); got != tc.want {
			t.Errorf("FlagReason(%s %s) = %q, want %q", tc.flag, tc.value, got, tc.want)
		}
	}
	// A custom set has no catalog entry: it gets the shared reasons.
	if FlagReason("--steps", "4", CatalogEntry{}) != FlagReasons["--steps"] {
		t.Error("custom set did not get the shared --steps reason")
	}
}

func TestSplitFlags(t *testing.T) {
	got := SplitFlags([]string{"--steps", "4", "--diffusion-fa", "--negative-prompt", "a, b", "--vae-tiling"})
	want := [][2]string{{"--steps", "4"}, {"--diffusion-fa", ""}, {"--negative-prompt", "a, b"}, {"--vae-tiling", ""}}
	if !slices.Equal(got, want) {
		t.Errorf("got %q, want %q", got, want)
	}
}

// Every slot a set can have gets a label other than its ID.
func TestSlotLabels(t *testing.T) {
	check := func(recipe, slot string) {
		if SlotLabel(recipe, slot) == slot {
			t.Errorf("recipe %q slot %s has no label", recipe, slot)
		}
	}
	for id, r := range RecipeSlots {
		for _, s := range r.Slots() {
			check(id, s)
		}
	}
	for id, r := range VideoRecipes {
		for _, s := range r.Slots.Slots() {
			check(id, s)
		}
	}
	for _, s := range AceSlots {
		check("", s)
	}
	check("", UpscalerSlot)
	if SlotLabel("wan22-i2v", "diffusion_model") != "Low-noise expert" || SlotLabel("flux3-klein", "diffusion_model") != "Diffusion model" {
		t.Error("recipe labels do not take precedence over the shared ones")
	}
	for recipe, labels := range RecipeSlotLabels {
		r, ok := RecipeSlots[recipe]
		if !ok {
			r = VideoRecipes[recipe].Slots
		}
		for slot := range labels {
			if !slices.Contains(r.Slots(), slot) {
				t.Errorf("RecipeSlotLabels[%s] labels %s, which the recipe does not have", recipe, slot)
			}
		}
	}
}

func TestVideoRecipeSizeMultiples(t *testing.T) {
	for id, r := range VideoRecipes {
		if r.SizeMultiple <= 0 {
			t.Errorf("%s has no size multiple", id)
		}
	}
	// A catalog entry agrees with its recipe.
	for _, e := range Catalog() {
		if e.SizeMultiple > 0 && e.SizeMultiple != RecipeSizeMultiple(e.Recipe) {
			t.Errorf("%s: SizeMultiple %d, recipe %s says %d", e.ID, e.SizeMultiple, e.Recipe, RecipeSizeMultiple(e.Recipe))
		}
	}
}

// A custom video set rounds sizes like a catalog set with the same recipe.
func TestCustomSetSizeMultiple(t *testing.T) {
	custom := ModelSet{ID: "mine", Kind: KindVideo, Recipe: "wan22-i2v"}
	if n := custom.SizeMultiple(); n != 16 {
		t.Errorf("SizeMultiple = %d, want 16", n)
	}
	if w, h := custom.MadeSize(830, 470); w != 832 || h != 480 {
		t.Errorf("MadeSize(830, 470) = %dx%d, want 832x480", w, h)
	}
	image := ModelSet{ID: "img", Kind: KindSD, Recipe: "flux3-klein"}
	if w, h := image.MadeSize(1000, 1000); w != 1000 || h != 1000 {
		t.Errorf("an image set's size changed: %dx%d", w, h)
	}
}

func TestUpscaleScale(t *testing.T) {
	if n := (ModelSet{Kind: KindUpscale, Scale: 2}).UpscaleScale(); n != 2 {
		t.Errorf("custom set with Scale 2: %d", n)
	}
	if n := (ModelSet{Kind: KindUpscale}).UpscaleScale(); n != 0 {
		t.Errorf("custom set saved before Scale existed: %d, want 0", n)
	}
	if n := (ModelSet{Kind: KindUpscale, CatalogID: "real-esrgan-x4plus", Scale: 2}).UpscaleScale(); n != 4 {
		t.Errorf("catalog set: %d, want the catalog's 4", n)
	}
}
