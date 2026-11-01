package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/generation"
)

// setFormFields renders the form's changing part for a query.
func setFormFields(t *testing.T, s *Server, query string) string {
	t.Helper()
	rec := httptest.NewRecorder()
	s.handleGenSetFormFields(rec, htmxReq("GET", "/?"+query, nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("%s: HTTP %d %s", query, rec.Code, rec.Body)
	}
	return rec.Body.String()
}

var slotNameRE = regexp.MustCompile(`name="slot_([a-z0-9_]+)"`)

func slotNames(body string) []string {
	var out []string
	for _, m := range slotNameRE.FindAllStringSubmatch(body, -1) {
		out = append(out, m[1])
	}
	return out
}

// The form shows one field per slot of the chosen recipe, offers only the
// kind's recipes, and the fields of the kind.
func TestGenSetFormFields(t *testing.T) {
	s := genServer(t)

	body := setFormFields(t, s, "kind=video&recipe=wan22-i2v")
	if got, want := slotNames(body), []string{"diffusion_model", "high_noise_diffusion_model", "t5xxl", "vae", "tae", "lora", "high_noise_lora"}; !slices.Equal(got, want) {
		t.Errorf("video slots = %v, want %v", got, want)
	}
	for _, want := range []string{"Low-noise expert", `value="wan22-i2v" selected`, `name="max_frames"`, `name="frames"`, "(one of: vae, tae)", "(required)"} {
		if !strings.Contains(body, want) {
			t.Errorf("video form lacks %q", want)
		}
	}
	if strings.Contains(body, `value="flux3-klein"`) || strings.Contains(body, `name="scale"`) {
		t.Error("the video form offers an image recipe or the upscale factor")
	}

	body = setFormFields(t, s, "kind=upscale")
	if got := slotNames(body); !slices.Equal(got, []string{"upscaler"}) || strings.Contains(body, `name="recipe"`) || !strings.Contains(body, `name="scale"`) {
		t.Errorf("upscaler form: slots %v, recipe select %v, scale %v", got, strings.Contains(body, `name="recipe"`), strings.Contains(body, `name="scale"`))
	}

	body = setFormFields(t, s, "kind=ace")
	if got := slotNames(body); !slices.Equal(got, generation.AceSlots) || !strings.Contains(body, `name="stem_tasks"`) {
		t.Errorf("music form: slots %v", got)
	}

	// A typed file stays when the recipe changes to one with the same slot.
	body = setFormFields(t, s, "kind=sd&recipe=flux3-klein&slot_vae=%2Fm%2Fvae.safetensors&slot_llm=%2Fm%2Fq.gguf")
	body = setFormFields(t, s, "kind=sd&recipe=z-image-turbo&slot_vae=%2Fm%2Fvae.safetensors&slot_llm=%2Fm%2Fq.gguf")
	if !strings.Contains(body, `name="slot_vae" value="/m/vae.safetensors"`) || !strings.Contains(body, `name="slot_llm" value="/m/q.gguf"`) {
		t.Error("typed files were lost on a recipe change")
	}
	// The tokenizer slot suggests tokenizer files.
	if body := setFormFields(t, s, "kind=sd&recipe=ming-image"); !strings.Contains(body, `name="slot_tokenizer" value="" list="gen-tokenizer-file-list"`) {
		t.Error("the tokenizer slot does not use the tokenizer list")
	}
	// A recipe of another kind falls back to the kind's first recipe.
	if body := setFormFields(t, s, "kind=sd&recipe=wan22-i2v"); !strings.Contains(body, `value="`+generation.Recipes()[0]+`" selected`) {
		t.Error("a video recipe was kept for an image set")
	}
}

func TestGenSetFormDialog(t *testing.T) {
	s := genServer(t)
	rec := httptest.NewRecorder()
	s.handleGenSetForm(rec, htmxReq("GET", "/", nil))
	body := rec.Body.String()
	for _, want := range []string{`hx-post="/api/gen/models/sets"`, `id="gen-set-form-fields"`, `hx-get="/api/gen/models/files" hx-trigger="load" hx-target="this"`, `name="slot_diffusion_model"`, s.cfg.ModelsPath()} {
		if !strings.Contains(body, want) {
			t.Errorf("form lacks %q", want)
		}
	}
	rec = httptest.NewRecorder()
	s.handleGenModels(rec, htmxReq("GET", "/", nil))
	if !strings.Contains(rec.Body.String(), "#gen-set-form-body-models") || strings.Contains(rec.Body.String(), `name="files"`) {
		t.Error("the set list has no Add a custom set button, or still has the old form")
	}
}

// The slot_<slot> fields make the same set as the older files list, and
// the two cannot be mixed.
func TestGenCreateSetFromSlotFields(t *testing.T) {
	s := genServer(t)
	dir := t.TempDir()
	paths := map[string]string{}
	for _, slot := range []string{"diffusion_model", "llm", "vae"} {
		paths[slot] = filepath.Join(dir, slot+".gguf")
		if err := os.WriteFile(paths[slot], fakeGGUF, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	form := url.Values{"name": {"From Slots"}, "kind": {"sd"}, "recipe": {"z-image-turbo"}, "slot_tae": {""}}
	for slot, p := range paths {
		form.Set("slot_"+slot, p)
	}
	rec := httptest.NewRecorder()
	s.handleGenCreateSet(rec, htmxReq("POST", "/", form))
	if rec.Code != http.StatusOK {
		t.Fatalf("HTTP %d %s", rec.Code, rec.Body)
	}
	set, err := s.gen.ModelSet("from-slots")
	if err != nil {
		t.Fatal(err)
	}
	var got []string
	for _, c := range set.Components {
		got = append(got, c.Slot+"="+c.Path)
		if c.Bytes != int64(len(fakeGGUF)) {
			t.Errorf("%s size %d", c.Slot, c.Bytes)
		}
	}
	if want := []string{"diffusion_model=" + paths["diffusion_model"], "llm=" + paths["llm"], "vae=" + paths["vae"]}; !slices.Equal(got, want) {
		t.Errorf("components = %v, want %v (the empty tae field is skipped)", got, want)
	}

	form.Set("name", "Both")
	form.Set("files", "vae = "+paths["vae"])
	rec = httptest.NewRecorder()
	s.handleGenCreateSet(rec, htmxReq("POST", "/", form))
	if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), "not both") {
		t.Errorf("slot fields and files list: HTTP %d %s", rec.Code, rec.Body)
	}
}

func TestGenCreateSetFileChecks(t *testing.T) {
	s := genServer(t)
	s.inContainerFn = func() bool { return true }
	text := filepath.Join(t.TempDir(), "notes.gguf")
	if err := os.WriteFile(text, []byte("not a model"), 0o644); err != nil {
		t.Fatal(err)
	}
	for name, tc := range map[string]struct{ path, want string }{
		"host path": {"/home/someone/llm/x4.pth", "cannot see host folders"},
		"text file": {text, "is not a model file"},
	} {
		form := url.Values{"name": {name}, "kind": {"upscale"}, "slot_upscaler": {tc.path}}
		rec := httptest.NewRecorder()
		s.handleGenCreateSet(rec, htmxReq("POST", "/", form))
		if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), tc.want) {
			t.Errorf("%s: HTTP %d %s, want 400 with %q", name, rec.Code, rec.Body, tc.want)
		}
	}
}

func TestGenModelFiles(t *testing.T) {
	s := genServer(t)
	root := generation.ModelsDir(s.cfg)
	for _, name := range []string{"Repo--a/model.gguf", "Repo--a/tokenizer.json", "Repo--a/config.json"} {
		p := filepath.Join(root, name)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	rec := httptest.NewRecorder()
	s.handleGenModelFiles(rec, httptest.NewRequest("GET", "/", nil))
	var got struct {
		Models, Tokenizers []string
		Truncated          bool
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatalf("%v: %s", err, rec.Body)
	}
	if len(got.Models) != 1 || len(got.Tokenizers) != 1 || got.Truncated {
		t.Errorf("JSON = %+v", got)
	}

	rec = httptest.NewRecorder()
	s.handleGenModelFiles(rec, htmxReq("GET", "/", nil))
	body := rec.Body.String()
	if !strings.Contains(body, `<datalist id="gen-model-file-list"><option value="`+filepath.Join(root, "Repo--a/model.gguf")+`">`) ||
		!strings.Contains(body, `<datalist id="gen-tokenizer-file-list"><option value="`) || strings.Contains(body, "config.json") {
		t.Errorf("datalists = %s", body)
	}
}
