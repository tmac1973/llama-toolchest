package generation

import (
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/builder"
)

// legacyImageCaps are the capabilities the old hand-kept RecipeCapabilities
// gave, without and with an llm_vision file.
var legacyImageCaps = map[string][2][]string{
	"ming-image":           {{"alpha"}, {"alpha"}},
	"flux2-klein-4b":       {{CapEdit}, {CapEdit}},
	"flux3-klein":          {{CapEdit}, {CapEdit}},
	"qwen-image-21":        {nil, {CapEdit}},
	"qwen-image-21-turbo":  {nil, {CapEdit}},
	"qwen-image-edit-2511": {nil, {CapEdit}},
}

// sameSlotSet compares two slot sets, ignoring the order of the slots.
func sameSlotSet(a, b RecipeSlotSet) []string {
	var diffs []string
	sorted := func(s []string) []string { s = slices.Clone(s); slices.Sort(s); return s }
	if !slices.Equal(sorted(a.Required), sorted(b.Required)) {
		diffs = append(diffs, "required "+strings.Join(a.Required, ",")+" vs "+strings.Join(b.Required, ","))
	}
	if !slices.Equal(sorted(a.Optional), sorted(b.Optional)) {
		diffs = append(diffs, "optional "+strings.Join(a.Optional, ",")+" vs "+strings.Join(b.Optional, ","))
	}
	if len(a.OneOf) != len(b.OneOf) {
		diffs = append(diffs, "one-of groups differ")
	} else {
		for i := range a.OneOf {
			if !slices.Equal(sorted(a.OneOf[i]), sorted(b.OneOf[i])) {
				diffs = append(diffs, "one-of "+strings.Join(a.OneOf[i], ",")+" vs "+strings.Join(b.OneOf[i], ","))
			}
		}
	}
	return diffs
}

// The snapshot gives the recipes toolchest knew before the same slots,
// size multiples and capabilities as the old tables.
func TestSnapshotMatchesLegacyTables(t *testing.T) {
	book := SnapshotRecipes()
	check := func(id string, kind BackendKind, want RecipeSlotSet) *Recipe {
		r, ok := book.Get(id)
		if !ok {
			t.Errorf("%s: not in the snapshot", id)
			return nil
		}
		if r.Kind != kind || r.Unavailable != "" {
			t.Errorf("%s: kind %s, unavailable %q", id, r.Kind, r.Unavailable)
		}
		for _, d := range sameSlotSet(r.Slots, want) {
			t.Errorf("%s: %s (snapshot vs old table)", id, d)
		}
		return r
	}
	for id, want := range legacyImageRecipes {
		check(id, KindSD, want)
		caps := legacyImageCaps[id]
		if got := RecipeCapabilities(id, nil); !slices.Equal(got, caps[0]) {
			t.Errorf("%s: capabilities %v, old table %v", id, got, caps[0])
		}
		if got := RecipeCapabilities(id, []Component{{Slot: "llm_vision"}}); !slices.Equal(got, caps[1]) {
			t.Errorf("%s with llm_vision: capabilities %v, old table %v", id, got, caps[1])
		}
	}
	for id, want := range legacyVideoRecipes {
		r := check(id, KindVideo, want.Slots)
		if r == nil {
			continue
		}
		if r.SizeMultiple != want.SizeMultiple {
			t.Errorf("%s: size multiple %d, old table %d", id, r.SizeMultiple, want.SizeMultiple)
		}
		if got := RecipeCapabilities(id, nil); !slices.Equal(got, want.Caps) {
			t.Errorf("%s: capabilities %v, old table %v", id, got, want.Caps)
		}
	}
}

// availableRecipes are the snapshot's recipes of a kind that toolchest
// can run.
func availableRecipes(kind BackendKind) []*Recipe {
	var out []*Recipe
	for _, id := range SnapshotRecipes().IDs(kind) {
		r, _ := SnapshotRecipes().Get(id)
		out = append(out, r)
	}
	return out
}

func TestParseRecipeFixtures(t *testing.T) {
	book, err := LoadRecipes("testdata/recipes")
	if err != nil {
		t.Fatal(err)
	}
	for _, skipped := range []string{"future-schema", "not-the-file-name", "wrong-name"} {
		if _, ok := book.Get(skipped); ok {
			t.Errorf("%s was read; want it skipped", skipped)
		}
	}
	flux, ok := book.Get("flux1-dev")
	if !ok {
		t.Fatal("flux1-dev not read")
	}
	if flux.Kind != KindSD || flux.Name != "FLUX.1 dev" || flux.Description == "" || flux.Unavailable != "" {
		t.Errorf("flux1-dev = %+v", flux)
	}
	if !slices.Equal(flux.Slots.Required, []string{"diffusion_model", "clip_l", "t5xxl"}) ||
		!slices.Equal(flux.Slots.Optional, []string{"vae", "tae"}) ||
		len(flux.Slots.OneOf) != 1 || !slices.Equal(flux.Slots.OneOf[0], []string{"vae", "tae"}) {
		t.Errorf("flux1-dev slots = %+v", flux.Slots)
	}
	if !flux.HasFeature("distilled_guidance") || flux.HasFeature("ref_images") {
		t.Errorf("flux1-dev features = %v", flux.Features)
	}
	wan, _ := book.Get("wan22-t2v")
	if wan.Kind != KindVideo || wan.SizeMultiple != 16 {
		t.Errorf("wan22-t2v kind %s, size multiple %d", wan.Kind, wan.SizeMultiple)
	}
	pid, _ := book.Get("pid")
	if pid.Unavailable != "" || pid.UpstreamLabels["llm"] != "Gemma 2 2B ELM Text Encoder" {
		t.Errorf("pid = %+v", pid)
	}
	// An optional slot sd-server has no flag for is left out; the recipe
	// stays usable.
	extra, _ := book.Get("optional-extra")
	if extra.Unavailable != "" || slices.Contains(extra.Slots.Slots(), "clip_g") {
		t.Errorf("optional-extra = %+v", extra)
	}
}

func TestParseRecipesEmpty(t *testing.T) {
	if _, err := LoadRecipes(t.TempDir()); err == nil {
		t.Error("an empty folder gave a recipe book")
	}
}

// The whole snapshot parses: no file is skipped.
func TestSnapshotParses(t *testing.T) {
	names, err := fsGlobSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	if got := SnapshotRecipes().Len(); got != len(names) || got < 100 {
		t.Errorf("snapshot has %d recipes from %d files", got, len(names))
	}
	if !strings.HasPrefix(SnapshotRecipes().Source, "snapshot master-") {
		t.Errorf("snapshot source %q", SnapshotRecipes().Source)
	}
}

func TestRecipeAvailability(t *testing.T) {
	book := SnapshotRecipes()
	for id, want := range map[string]string{
		"animatediff-sd15":  "needs the motion module file",
		"flux1-canny":       "needs a control image input",
		"sd35-large":        "needs the CLIP-G text encoder file",
		"wan21-vace":        "needs control video frames",
		"minimax-h3-ref2va": "needs a reference video input",
	} {
		r, ok := book.Get(id)
		if !ok {
			t.Errorf("%s not in the snapshot", id)
			continue
		}
		if !strings.HasPrefix(r.Unavailable, want) {
			t.Errorf("%s: unavailable %q, want it to start with %q", id, r.Unavailable, want)
		}
		if slices.Contains(book.IDs(r.Kind), id) {
			t.Errorf("%s is listed as usable", id)
		}
		if !slices.ContainsFunc(book.Unavailable(r.Kind), func(u *Recipe) bool { return u.ID == id }) {
			t.Errorf("%s is not in the unavailable list", id)
		}
	}
	// Wan2.2 S2V's audio is sent (recipeAudioInput), so it is usable.
	s2v, ok := book.Get("wan22-s2v")
	if !ok || s2v.Unavailable != "" || !slices.Contains(book.IDs(KindVideo), "wan22-s2v") {
		t.Errorf("wan22-s2v: %v, unavailable %q; want it usable", ok, s2v.Unavailable)
	}
	if ok && (!slices.Equal(s2v.Slots.Required, []string{"diffusion_model", "t5xxl", "audio_encoder"}) ||
		len(s2v.Slots.OneOf) != 1 || !slices.Equal(s2v.Slots.OneOf[0], []string{"vae", "tae"})) {
		t.Errorf("wan22-s2v slots = %+v", s2v.Slots)
	}
	err := ValidateComponents(KindSD, "sd35-large", []Component{{Slot: "diffusion_model"}})
	if err == nil || !strings.Contains(err.Error(), "cannot be used") {
		t.Errorf("sd35-large set: %v", err)
	}
}

func TestRecipeOverlay(t *testing.T) {
	book := SnapshotRecipes()
	wan, _ := book.Get("wan22-t2v")
	for _, s := range []string{"lora", "high_noise_lora"} {
		if !slices.Contains(wan.Slots.Optional, s) {
			t.Errorf("wan22-t2v lacks slot %s", s)
		}
	}
	for recipe, want := range map[string][]string{
		"ltx23-dev":  {CapText, CapStartImage, CapEndImage},
		"wan22-t2v":  {CapText},
		"wan21-i2v":  {CapStartImage},
		"ming-image": {"alpha"},
		"flux1-dev":  nil,
	} {
		if got := RecipeCapabilities(recipe, nil); !slices.Equal(got, want) {
			t.Errorf("%s capabilities %v, want %v", recipe, got, want)
		}
	}
	// A recipe with a vision projector slot edits only with that file.
	if got := RecipeCapabilities("krea2-turbo-edit", nil); got != nil {
		t.Errorf("krea2-turbo-edit without llm_vision: %v", got)
	}
	if got := RecipeCapabilities("krea2-turbo-edit", []Component{{Slot: "llm_vision"}}); !slices.Equal(got, []string{CapEdit}) {
		t.Errorf("krea2-turbo-edit with llm_vision: %v", got)
	}
	if SlotLabel("pid", "llm") != "Gemma 2 2B ELM Text Encoder" {
		t.Errorf("upstream label not used: %q", SlotLabel("pid", "llm"))
	}
	if SlotLabel("qwen-image-edit-2511", "llm") != "Qwen2.5-VL 7B text model" {
		t.Errorf("toolchest label not first: %q", SlotLabel("qwen-image-edit-2511", "llm"))
	}
}

// Every catalog recipe is in the snapshot, usable, and of the entry's kind.
func TestCatalogRecipesInSnapshot(t *testing.T) {
	for _, e := range Catalog() {
		if e.Kind != KindSD && e.Kind != KindVideo {
			continue
		}
		r, ok := SnapshotRecipes().Get(e.Recipe)
		switch {
		case !ok:
			t.Errorf("%s: recipe %s is not in the snapshot", e.ID, e.Recipe)
		case r.Unavailable != "":
			t.Errorf("%s: recipe %s cannot be used: %s", e.ID, e.Recipe, r.Unavailable)
		case r.Kind != e.Kind:
			t.Errorf("%s: recipe %s is a %s recipe", e.ID, e.Recipe, r.Kind)
		}
	}
}

func TestSetActiveRecipes(t *testing.T) {
	book, err := LoadRecipes("testdata/recipes")
	if err != nil {
		t.Fatal(err)
	}
	SetActiveRecipes(book)
	t.Cleanup(func() { SetActiveRecipes(SnapshotRecipes()) })
	if got := Recipes(); !slices.Contains(got, "flux1-dev") || slices.Contains(got, "z-image-turbo") {
		t.Errorf("Recipes with the fixture book = %v", got)
	}
}

func fsGlobSnapshot() ([]string, error) { return fs.Glob(snapshotFS, "sdrecipes/*.json") }

// fixtureBuild is an sd build whose folder holds the given fixture
// recipes.
func fixtureBuild(t *testing.T, id string, recipes ...string) builder.BuildResult {
	t.Helper()
	dir := filepath.Join(t.TempDir(), id)
	if err := os.MkdirAll(filepath.Join(dir, "recipes"), 0o755); err != nil {
		t.Fatal(err)
	}
	for _, r := range recipes {
		data, err := os.ReadFile(filepath.Join("testdata", "recipes", r+".json"))
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, "recipes", r+".json"), data, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	t.Cleanup(func() { ForgetBuildRecipes(id) })
	return builder.BuildResult{ID: id, Profile: "rocm", BinaryPath: filepath.Join(dir, "sd-server")}
}

func TestRecipesForBuild(t *testing.T) {
	b := fixtureBuild(t, "sd-test-two", "flux1-dev", "wan22-t2v")
	book := RecipesForBuild(b)
	if book.Source != b.ID || book.Len() != 2 {
		t.Fatalf("book from %q with %d recipes", book.Source, book.Len())
	}
	// Cached: a new file is not seen until the build is forgotten.
	extra, _ := os.ReadFile(filepath.Join("testdata", "recipes", "pid.json"))
	if err := os.WriteFile(filepath.Join(filepath.Dir(b.BinaryPath), "recipes", "pid.json"), extra, 0o644); err != nil {
		t.Fatal(err)
	}
	if RecipesForBuild(b).Len() != 2 {
		t.Error("the cache was not used")
	}
	ForgetBuildRecipes(b.ID)
	if RecipesForBuild(b).Len() != 3 {
		t.Error("ForgetBuildRecipes did not make the folder be read again")
	}
	// A build without recipe files uses the snapshot.
	old := builder.BuildResult{ID: "sd-test-old", Profile: "rocm", BinaryPath: filepath.Join(t.TempDir(), "sd-server")}
	t.Cleanup(func() { ForgetBuildRecipes(old.ID) })
	if RecipesForBuild(old) != SnapshotRecipes() {
		t.Error("a build without recipes did not get the snapshot")
	}
}

func TestCheckRecipeForBuild(t *testing.T) {
	b := fixtureBuild(t, "sd-test-check", "flux1-dev", "flux1-canny", "wan22-t2v")
	ok := ModelSet{Name: "Flux", Kind: KindSD, Recipe: "flux1-dev"}
	if err := CheckRecipeForBuild(ok, b); err != nil {
		t.Errorf("known recipe: %v", err)
	}
	for set, want := range map[*ModelSet]string{
		{Name: "Z", Kind: KindSD, Recipe: "z-image-turbo"}:         "does not know recipe z-image-turbo",
		{Name: "Canny", Kind: KindSD, Recipe: "flux1-canny"}:       "cannot be used",
		{Name: "Wan", Kind: KindSD, Recipe: "wan22-t2v"}:           "is a video recipe",
		{Name: "Flux video", Kind: KindVideo, Recipe: "flux1-dev"}: "is an image recipe",
	} {
		if err := CheckRecipeForBuild(*set, b); err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("%s: %v, want %q", set.Name, err, want)
		}
	}
	if err := CheckRecipeForBuild(ModelSet{Kind: KindAce}, b); err != nil {
		t.Errorf("ace set: %v", err)
	}
	// An instance start on that build is refused.
	set := ModelSet{ID: "z", Name: "Z", Kind: KindSD, Recipe: "z-image-turbo", Complete: true,
		Components: []Component{{Slot: "diffusion_model", Path: "/m/z.gguf"}, {Slot: "llm", Path: "/m/q.gguf"}, {Slot: "vae", Path: "/m/v.safetensors"}}}
	if _, err := LaunchSpec(Instance{ID: "z", Port: 8090}, set, b, "", t.TempDir(), LaunchOptions{}); err == nil || !strings.Contains(err.Error(), "does not know recipe") {
		t.Errorf("LaunchSpec: %v", err)
	}
}

func TestRecipeWarning(t *testing.T) {
	if w := RecipeWarning(KindSD, "z-image-turbo"); w != "" {
		t.Errorf("known recipe: %q", w)
	}
	if w := RecipeWarning(KindSD, "gone"); !strings.Contains(w, "does not know recipe gone") || !strings.Contains(w, "snapshot") {
		t.Errorf("unknown recipe: %q", w)
	}
	if w := RecipeWarning(KindSD, "sd35-large"); !strings.Contains(w, "CLIP-G") {
		t.Errorf("unavailable recipe: %q", w)
	}
	if w := RecipeWarning(KindAce, ""); w != "" {
		t.Errorf("ace: %q", w)
	}
}

func TestRecipeFixedFPS(t *testing.T) {
	book := SnapshotRecipes()
	for id, want := range map[string]int{"wan22-s2v": 16, "minimax-h3-fl2va": 24, "wan22-i2v": 0} {
		r, ok := book.Get(id)
		if !ok {
			t.Fatalf("snapshot has no %s", id)
		}
		if r.FixedFPS != want {
			t.Errorf("%s FixedFPS = %d, want %d", id, r.FixedFPS, want)
		}
	}
	if got := (ModelSet{Recipe: "wan22-s2v"}).FixedFPS(); got != 16 {
		t.Errorf("a wan22-s2v set's FixedFPS() = %d, want 16", got)
	}
}
