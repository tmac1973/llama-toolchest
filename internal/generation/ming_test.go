package generation

import (
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/builder"
)

func TestMingBuildProblem(t *testing.T) {
	for count, broken := range map[int]bool{0: false, 1106: false, 1115: false, 1116: true, 1132: true, 1141: true, 1142: false, 1200: false} {
		b := builder.BuildResult{ID: "sd-x", CommitCount: count}
		got := MingBuildProblem("ming-image", b)
		if (got != "") != broken {
			t.Errorf("commit count %d: %q, want broken=%v", count, got, broken)
		}
		if MingBuildProblem("z-image-turbo", b) != "" {
			t.Errorf("commit count %d: a Z-Image set was refused", count)
		}
	}
}

// A Ming-Image instance does not start on a build that takes it for
// Z-Image; the refusal says which builds work.
func TestCheckRecipeForBuildMing(t *testing.T) {
	b := fixtureBuild(t, "sd-test-ming", "ming-image")
	set := ModelSet{Name: "Ming", Kind: KindSD, Recipe: "ming-image"}
	if err := CheckRecipeForBuild(set, b); err != nil {
		t.Errorf("build with no commit count: %v", err)
	}
	b.CommitCount = 1132
	if err := CheckRecipeForBuild(set, b); err == nil || !strings.Contains(err.Error(), "master-1142 or newer") {
		t.Errorf("master-1132: %v", err)
	}
	b.CommitCount = 1142
	if err := CheckRecipeForBuild(set, b); err != nil {
		t.Errorf("master-1142: %v", err)
	}
}

// Both Ming-Image versions share the VAE, so a second one downloads only
// its own DiT, text model and tokenizer.
func TestCatalogMingVersionsShareVAE(t *testing.T) {
	var vaes []Component
	for _, e := range Catalog() {
		if e.Recipe != "ming-image" {
			continue
		}
		for _, c := range e.Components {
			if c.Slot == "vae" {
				vaes = append(vaes, c)
			}
		}
	}
	if len(vaes) != 2 || vaes[0] != vaes[1] {
		t.Errorf("Ming-Image VAEs = %v, want the same file in both sets", vaes)
	}
}
