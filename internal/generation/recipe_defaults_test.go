package generation

import (
	"slices"
	"strings"
	"testing"
)

// Every defaults and context key of the snapshot is turned into a flag
// or a size, or skipped on purpose. A new upstream key fails here at the
// next snapshot refresh.
func TestRecipeKeysAreKnown(t *testing.T) {
	book := SnapshotRecipes()
	for _, id := range append(book.IDs(KindSD), book.IDs(KindVideo)...) {
		r, _ := book.Get(id)
		if keys := unknownRecipeKeys(r); len(keys) > 0 {
			t.Errorf("%s: unknown recipe keys %v; add them to recipeFlags or skippedDefaults", id, keys)
		}
	}
	for _, r := range append(book.Unavailable(KindSD), book.Unavailable(KindVideo)...) {
		if keys := unknownRecipeKeys(r); len(keys) > 0 {
			t.Errorf("%s: unknown recipe keys %v", r.ID, keys)
		}
	}
}

func TestRecipeLaunchDefaults(t *testing.T) {
	book := SnapshotRecipes()
	get := func(id string) *Recipe {
		r, ok := book.Get(id)
		if !ok {
			t.Fatalf("no recipe %s", id)
		}
		return r
	}
	contains := func(args []string, words ...string) bool {
		for i := range args {
			if i+len(words) <= len(args) && slices.Equal(args[i:i+len(words)], words) {
				return true
			}
		}
		return false
	}

	flux := get("flux1-dev")
	if got, want := flux.LaunchDefaults(), []string{"--steps", "20", "--cfg-scale", "1", "--guidance", "3.5", "--sampling-method", "euler"}; !slices.Equal(got, want) {
		t.Errorf("flux1-dev = %v, want %v", got, want)
	}
	if w, h, fr, fps := flux.SizeDefaults(); w != "1024" || h != "1024" || fr != "" || fps != "" {
		t.Errorf("flux1-dev size = %s %s %s %s", w, h, fr, fps)
	}
	if args := get("qwen-image-edit-2511").LaunchDefaults(); !contains(args, "--model-args", "qwen_image_zero_cond_t=true") || !contains(args, "--flow-shift", "3") {
		t.Errorf("qwen-image-edit-2511 = %v", args)
	}
	if args := get("pid").LaunchDefaults(); !contains(args, "--vae-format", "flux") {
		t.Errorf("pid = %v", args)
	}
	wan := get("wan22-t2v")
	args := wan.LaunchDefaults()
	if !contains(args, "--high-noise-steps", "-1") || !contains(args, "--moe-boundary", "0.875") || strings.Contains(strings.Join(args, " "), "high-noise-flow-shift") {
		t.Errorf("wan22-t2v = %v", args)
	}
	if _, _, fr, fps := wan.SizeDefaults(); fr != "33" || fps != "16" {
		t.Errorf("wan22-t2v frames %s fps %s", fr, fps)
	}
	if args := get("qwen-image-21-turbo").LaunchDefaults(); !contains(args, "--sigmas", "1,0.978453,0.95418,0.926626,0.89508,0.845148,0.704534,0.414568,0") {
		t.Errorf("qwen-image-21-turbo = %v", args)
	}
}

// Every available recipe's flags pass the check a saved set's flags get.
func TestRecipeLaunchDefaultsValidate(t *testing.T) {
	book := SnapshotRecipes()
	for _, id := range append(book.IDs(KindSD), book.IDs(KindVideo)...) {
		r, _ := book.Get(id)
		want := r.LaunchDefaults()
		got, err := ValidateExtraArgs(JoinArgs(want))
		if err != nil || !slices.Equal(got, want) {
			t.Errorf("%s: %v → %v, %v", id, want, got, err)
		}
	}
}
