package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/generation"
	"github.com/tmac1973/llama-toolchest/internal/huggingface"
)

// copyForm asks for the custom set form filled from a set or catalog
// entry.
func copyForm(t *testing.T, s *Server, query string) (genSetForm, *httptest.ResponseRecorder) {
	t.Helper()
	r := htmxReq("GET", "/?"+query, nil)
	rec := httptest.NewRecorder()
	s.handleGenSetForm(rec, r)
	if rec.Code != http.StatusOK {
		return genSetForm{}, rec
	}
	f, _, err := s.genSetCopyForm(r, s.gen)
	if err != nil {
		t.Fatal(err)
	}
	return f, rec
}

// formValues is what the browser sends for a form drawn from f.
func formValues(f genSetForm) url.Values {
	v := url.Values{"name": {f.Name}, "kind": {string(f.Kind)}, "recipe": {f.Recipe}, "launch_args": {f.LaunchArgs},
		"width": {f.Width}, "height": {f.Height}, "frames": {f.Frames}, "fps": {f.FPS},
		"max_frames": {f.MaxFrames}, "max_width": {f.MaxWidth}, "max_height": {f.MaxHeight}, "scale": {f.Scale},
		"tab": {f.Tab}, "video_sizes": {f.VideoSizes}, "license": {f.License}}
	if f.StemTasks {
		v.Set("stem_tasks", "1")
	}
	for _, sf := range f.Slots {
		v.Set("slot_"+sf.Slot, sf.Value)
	}
	return v
}

// writeCatalogFiles puts a catalog entry's files where the downloader
// saves them; only the slots named, or all when none are.
func writeCatalogFiles(t *testing.T, s *Server, e generation.CatalogEntry, slots ...string) {
	t.Helper()
	for _, c := range e.Components {
		if len(slots) > 0 && !slices.Contains(slots, c.Slot) {
			continue
		}
		p := huggingface.FilePath(generation.ModelsDir(s.cfg), c.Repo, c.File)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, fakeGGUF, 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

// A catalog entry not fully downloaded: the files on disk fill their
// slots, the others are empty with the catalog file name as the hint, and
// the flags, sizes, limits and licence are the entry's.
func TestGenCopyCatalogEntry(t *testing.T) {
	s := genServer(t)
	e, _ := generation.LookupCatalog("wan22-i2v-a14b-lightning")
	writeCatalogFiles(t, s, e, "vae")

	f, rec := copyForm(t, s, "copy_catalog="+e.ID+"&tab=catalog")
	if rec.Code != http.StatusOK {
		t.Fatalf("HTTP %d %s", rec.Code, rec.Body)
	}
	body := rec.Body.String()
	if f.Name != e.Name+" custom" || f.CopiedFrom != e.Name || f.Tab != "catalog" || f.Kind != generation.KindVideo || f.Recipe != "wan22-i2v" {
		t.Errorf("form = %+v", f)
	}
	for _, sf := range f.Slots {
		var c generation.Component
		for _, ec := range e.Components {
			if ec.Slot == sf.Slot {
				c = ec
			}
		}
		switch {
		case c.Slot == "":
			if sf.Value != "" {
				t.Errorf("slot %s the entry lacks has %q", sf.Slot, sf.Value)
			}
		case c.Slot == "vae":
			if sf.Value != huggingface.FilePath(generation.ModelsDir(s.cfg), c.Repo, c.File) {
				t.Errorf("vae = %q, want the downloaded file", sf.Value)
			}
		default:
			if sf.Value != "" || sf.Hint != filepath.Base(c.File) {
				t.Errorf("slot %s: value %q hint %q, want empty with hint %s", sf.Slot, sf.Value, sf.Hint, filepath.Base(c.File))
			}
		}
	}
	if args, _ := generation.ValidateExtraArgs(f.LaunchArgs); !slices.Equal(args, e.LaunchArgs) {
		t.Errorf("launch args = %s", f.LaunchArgs)
	}
	if f.Frames != "33" || f.FPS != "16" || f.MaxFrames != "81" || f.MaxWidth != "1280" || f.MaxHeight != "720" || f.License != e.License {
		t.Errorf("sizes and limits = %+v", f)
	}
	var sizes []generation.VideoSize
	if err := json.Unmarshal([]byte(f.VideoSizes), &sizes); err != nil || !slices.Equal(sizes, e.VideoSizes) {
		t.Errorf("video sizes = %s (%v)", f.VideoSizes, err)
	}
	for _, want := range []string{"Copied from <strong>" + e.Name, `hx-post="/api/gen/models/sets?view=catalog"`, `name="video_sizes"`, `name="tab" value="catalog"`,
		`placeholder="` + filepath.Base(e.Components[0].File) + `"`} {
		if !strings.Contains(body, want) {
			t.Errorf("form lacks %q", want)
		}
	}
}

// Copy a downloaded catalog set and save it unchanged: the custom set runs
// and advertises the same as the source.
func TestGenCopyThenSaveMatchesSource(t *testing.T) {
	s := genServer(t)
	e, _ := generation.LookupCatalog("wan22-i2v-a14b-lightning")
	writeCatalogFiles(t, s, e)
	src := generation.ModelSet{ID: e.ID, Name: e.Name, Kind: e.Kind, Recipe: e.Recipe, CatalogID: e.ID, Complete: true}
	for _, c := range e.Components {
		c.Path = huggingface.FilePath(generation.ModelsDir(s.cfg), c.Repo, c.File)
		src.Components = append(src.Components, c)
	}
	if err := s.gen.AddModelSet(src); err != nil {
		t.Fatal(err)
	}

	f, rec := copyForm(t, s, "copy="+src.ID)
	if rec.Code != http.StatusOK {
		t.Fatalf("form: HTTP %d %s", rec.Code, rec.Body)
	}
	for _, sf := range f.Slots {
		if slices.ContainsFunc(src.Components, func(c generation.Component) bool { return c.Slot == sf.Slot }) && sf.Value == "" {
			t.Errorf("slot %s was not filled from the set", sf.Slot)
		}
	}
	rec = httptest.NewRecorder()
	s.handleGenCreateSet(rec, htmxReq("POST", "/", formValues(f)))
	if rec.Code != http.StatusOK {
		t.Fatalf("save: HTTP %d %s", rec.Code, rec.Body)
	}
	got, err := s.gen.ModelSet(generation.Slug(f.Name))
	if err != nil {
		t.Fatal(err)
	}
	if got.CatalogID != "" || !slices.Equal(got.RunArgs(), src.RunArgs()) || !slices.Equal(got.VideoSizes, e.VideoSizes) ||
		got.MaxFrames != e.MaxFrames || got.MaxWidth != e.MaxWidth || got.MaxHeight != e.MaxHeight ||
		got.DefaultFrames != e.DefaultFrames || got.DefaultFPS != e.DefaultFPS || got.License != e.License ||
		got.SizeMultiple() != src.SizeMultiple() || !slices.Equal(got.Capabilities, src.CurrentCapabilities()) {
		t.Errorf("copy = %+v\nsource entry = %+v", got, e)
	}
	if len(got.Loras()) != 2 {
		t.Errorf("the copy has %d LoRAs, want 2", len(got.Loras()))
	}
	if !slices.Equal(generation.VideoPresets(got), generation.VideoPresets(src)) {
		t.Errorf("Measure sizes = %v, want %v", generation.VideoPresets(got), generation.VideoPresets(src))
	}

	// A second copy gets a free name.
	f2, _ := copyForm(t, s, "copy="+src.ID)
	if f2.Name != e.Name+" custom 2" {
		t.Errorf("second copy's name = %q", f2.Name)
	}
}

// A copy of a custom upscaler keeps its factor; the save from the
// Download Models tab answers with a note.
func TestGenCopyCustomSetAndCatalogNote(t *testing.T) {
	s := genServer(t)
	file := filepath.Join(t.TempDir(), "x2.pth")
	if err := os.WriteFile(file, fakeGGUF, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := s.gen.AddModelSet(generation.ModelSet{ID: "mine", Name: "Mine", Kind: generation.KindUpscale, Scale: 2, Complete: true,
		Components: []generation.Component{{Slot: generation.UpscalerSlot, Path: file}}}); err != nil {
		t.Fatal(err)
	}
	f, _ := copyForm(t, s, "copy=mine")
	if f.Scale != "2" || f.Slots[0].Value != file || f.Tab != "models" {
		t.Errorf("form = %+v", f)
	}

	form := formValues(f)
	form.Set("name", "From Catalog Tab")
	rec := httptest.NewRecorder()
	s.handleGenCreateSet(rec, htmxReq("POST", "/?view=catalog", form))
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), "<strong>From Catalog Tab</strong> is added.") || strings.Contains(rec.Body.String(), "<table") {
		t.Errorf("catalog tab save: HTTP %d %s", rec.Code, rec.Body)
	}
}

func TestGenCopyErrors(t *testing.T) {
	s := genServer(t)
	for query, code := range map[string]int{
		"copy=nope":                         http.StatusNotFound,
		"copy_catalog=nope":                 http.StatusNotFound,
		"copy=x&copy_catalog=z-image-turbo": http.StatusBadRequest,
	} {
		rec := httptest.NewRecorder()
		s.handleGenSetForm(rec, htmxReq("GET", "/?"+query, nil))
		if rec.Code != code {
			t.Errorf("%s: HTTP %d, want %d", query, rec.Code, code)
		}
	}
}

// The video sizes a copy keeps: the first takes the form's test size, and
// none may pass the ceiling.
func TestGenCopyVideoSizes(t *testing.T) {
	set := generation.ModelSet{DefaultFrames: 17, DefaultWidth: 640, DefaultHeight: 384, MaxFrames: 81, MaxWidth: 1280, MaxHeight: 720}
	req := func(sizes string) *http.Request {
		return htmxReq("POST", "/", url.Values{"video_sizes": {sizes}})
	}
	got, err := formVideoSizes(req(`[{"label":"Short","frames":33,"width":832,"height":480},{"label":"Full","frames":81,"width":1280,"height":720}]`), set)
	if err != nil || got[0] != (generation.VideoSize{Label: "Short", Frames: 17, Width: 640, Height: 384}) || got[1].Frames != 81 {
		t.Errorf("sizes = %v, %v", got, err)
	}
	if _, err := formVideoSizes(req(`[{"label":"Short","frames":33,"width":832,"height":480},{"label":"Huge","frames":121,"width":1920,"height":1080}]`), set); err == nil {
		t.Error("a size past the ceiling was kept")
	}
	if got, err := formVideoSizes(req(""), set); got != nil || err != nil {
		t.Errorf("no field: %v, %v", got, err)
	}
}

// The Details dialog's copy button asks for the right source and opens
// the form dialog on the same tab.
func TestGenDetailsCopyButton(t *testing.T) {
	s := genServer(t)
	rec := httptest.NewRecorder()
	s.handleGenCatalogDetails(rec, withID(htmxReq("GET", "/?tab=catalog", nil), "z-image-turbo"))
	body := rec.Body.String()
	if !strings.Contains(body, `hx-get="/api/gen/models/sets/form?copy_catalog=z-image-turbo&amp;tab=catalog"`) ||
		!strings.Contains(body, `hx-target="#gen-set-form-body-catalog"`) || !strings.Contains(body, "gen-catalog-details-dialog") {
		t.Errorf("catalog details copy button:\n%s", body)
	}

	if err := s.gen.AddModelSet(generation.ModelSet{ID: "mine", Name: "Mine", Kind: generation.KindUpscale, Complete: true,
		Components: []generation.Component{{Slot: generation.UpscalerSlot, Path: "/m/x.pth"}}}); err != nil {
		t.Fatal(err)
	}
	rec = httptest.NewRecorder()
	s.handleGenSetDetails(rec, withID(htmxReq("GET", "/?tab=models", nil), "mine"))
	body = rec.Body.String()
	if !strings.Contains(body, `hx-get="/api/gen/models/sets/form?copy=mine&amp;tab=models"`) || !strings.Contains(body, `hx-target="#gen-set-form-body-models"`) {
		t.Errorf("set details copy button:\n%s", body)
	}
}
