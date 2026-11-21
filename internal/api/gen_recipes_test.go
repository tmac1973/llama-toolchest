package api

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/builder"
	"github.com/tmac1973/llama-toolchest/internal/generation"
)

// A custom set whose recipe the active list does not know, as after a
// newer stable-diffusion.cpp dropped it.
func addGoneRecipeSet(t *testing.T, s *Server) generation.ModelSet {
	t.Helper()
	dir := t.TempDir()
	var comps []generation.Component
	for _, slot := range []string{"diffusion_model", "vae"} {
		p := filepath.Join(dir, slot+".gguf")
		if err := os.WriteFile(p, fakeGGUF, 0o644); err != nil {
			t.Fatal(err)
		}
		comps = append(comps, generation.Component{Slot: slot, Path: p})
	}
	set := generation.ModelSet{ID: "gone", Name: "Gone", Kind: generation.KindSD, Recipe: "gone-recipe", Complete: true, Components: comps}
	if err := s.gen.AddModelSet(set); err != nil {
		t.Fatal(err)
	}
	return set
}

// Copy and edit keep a recipe the active list does not know, show the
// set's own slots and say why.
func TestGenSetFormKeepsUnknownRecipe(t *testing.T) {
	s := genServer(t)
	addGoneRecipeSet(t, s)

	for _, query := range []string{"copy=gone", "edit=gone"} {
		r := htmxReq("GET", "/?"+query, nil)
		rec := httptest.NewRecorder()
		s.handleGenSetForm(rec, r)
		if rec.Code != http.StatusOK {
			t.Fatalf("%s: HTTP %d %s", query, rec.Code, rec.Body)
		}
		body := rec.Body.String()
		for _, want := range []string{"gone-recipe (not in the recipe list)", "does not know recipe gone-recipe", `name="slot_diffusion_model"`, `name="slot_vae"`, `name="recipe_slots"`} {
			if !strings.Contains(body, want) {
				t.Errorf("%s: form lacks %q", query, want)
			}
		}
	}

	// A redraw keeps it, from the hidden slot list.
	f := s.genSetFormFrom(func(k string) string {
		return map[string]string{"kind": "sd", "recipe": "gone-recipe", "recipe_slots": "diffusion_model,vae"}[k]
	})
	if !f.UnknownRecipe || f.Recipe != "gone-recipe" || len(f.Slots) != 2 {
		t.Errorf("redraw = %+v", f)
	}
	// A new set with an unknown recipe falls back to the kind's first.
	f = s.genSetFormFrom(func(k string) string { return map[string]string{"kind": "sd", "recipe": "gone-recipe"}[k] })
	if f.UnknownRecipe || f.Recipe != generation.Recipes()[0] {
		t.Errorf("new set = recipe %q, unknown %v", f.Recipe, f.UnknownRecipe)
	}
	// A known recipe of the other kind is not kept.
	f = s.genSetFormFrom(func(k string) string {
		return map[string]string{"kind": "video", "recipe": "z-image-turbo", "recipe_slots": "diffusion_model,llm,vae"}[k]
	})
	if f.UnknownRecipe || f.Recipe != generation.VideoRecipeIDs()[0] {
		t.Errorf("kind switch = recipe %q, unknown %v", f.Recipe, f.UnknownRecipe)
	}
}

// Details warns about a recipe the active list does not know.
func TestGenDetailsRecipeWarning(t *testing.T) {
	s := genServer(t)
	set := addGoneRecipeSet(t, s)
	d, err := generation.DetailsForSet(set, generation.ModelsDir(s.cfg))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(d.RecipeWarning, "does not know recipe gone-recipe") {
		t.Errorf("warning = %q", d.RecipeWarning)
	}
	ok := generation.ModelSet{ID: "z", Kind: generation.KindSD, Recipe: "z-image-turbo"}
	if d, _ := generation.DetailsForSet(ok, generation.ModelsDir(s.cfg)); d.RecipeWarning != "" {
		t.Errorf("known recipe warns: %q", d.RecipeWarning)
	}
}

// With no sd build the active list is the snapshot.
func TestRefreshRecipesNoBuild(t *testing.T) {
	s := genServer(t)
	s.refreshRecipes()
	t.Cleanup(func() { generation.SetActiveRecipes(generation.SnapshotRecipes()) })
	if got := generation.ActiveRecipes().Source; !strings.HasPrefix(got, "snapshot") {
		t.Errorf("source %q", got)
	}
}

// The newest sd build's recipe files become the active list; a build
// without them falls back to the snapshot.
func TestRefreshRecipesNewestBuild(t *testing.T) {
	s := genServer(t)
	dir := s.cfg.DataDir
	newer := filepath.Join(dir, "builds", "sd-master-2000-aaaaaaa-rocm")
	if err := os.MkdirAll(filepath.Join(newer, "recipes"), 0o755); err != nil {
		t.Fatal(err)
	}
	for _, f := range []string{"flux1-dev.json", "wan22-t2v.json"} {
		data, err := os.ReadFile(filepath.Join("..", "generation", "testdata", "recipes", f))
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(newer, "recipes", f), data, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	records := `[
	  {"id":"sd-master-1000-bbbbbbb-rocm","project":"stable-diffusion.cpp","profile":"rocm","git_ref":"master-1000-bbbbbbb","status":"success","commit_count":1000,"binary_path":"` + filepath.Join(dir, "builds", "sd-master-1000-bbbbbbb-rocm", "sd-server") + `"},
	  {"id":"sd-master-2000-aaaaaaa-rocm","project":"stable-diffusion.cpp","profile":"rocm","git_ref":"master-2000-aaaaaaa","status":"success","commit_count":2000,"binary_path":"` + filepath.Join(newer, "sd-server") + `"}
	]`
	if err := os.MkdirAll(filepath.Join(dir, "config"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "config", "builds.json"), []byte(records), 0o644); err != nil {
		t.Fatal(err)
	}
	s.builder = builder.NewBuilder(dir)
	t.Cleanup(func() { generation.SetActiveRecipes(generation.SnapshotRecipes()) })

	s.refreshRecipes()
	book := generation.ActiveRecipes()
	if book.Source != "sd-master-2000-aaaaaaa-rocm" || book.Len() != 2 {
		t.Fatalf("active list from %q with %d recipes", book.Source, book.Len())
	}
	if got := generation.Recipes(); len(got) != 1 || got[0] != "flux1-dev" {
		t.Errorf("Recipes = %v", got)
	}

	// The newer build deleted: the older one, which has no recipe files,
	// gives the snapshot.
	if err := s.builder.Delete("sd-master-2000-aaaaaaa-rocm"); err != nil {
		t.Fatal(err)
	}
	s.refreshRecipes()
	if got := generation.ActiveRecipes().Source; !strings.HasPrefix(got, "snapshot") {
		t.Errorf("after delete, source %q", got)
	}
}
