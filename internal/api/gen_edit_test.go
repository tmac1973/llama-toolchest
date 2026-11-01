package api

import (
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/generation"
)

// editServer has a custom image set "mine" with three model files, and
// returns the folder they are in.
func editServer(t *testing.T) (*Server, string) {
	t.Helper()
	s := genServer(t)
	dir := t.TempDir()
	var comps []generation.Component
	for _, slot := range []string{"diffusion_model", "llm", "vae"} {
		p := filepath.Join(dir, slot+".gguf")
		if err := os.WriteFile(p, fakeGGUF, 0o644); err != nil {
			t.Fatal(err)
		}
		comps = append(comps, generation.Component{Slot: slot, Path: p})
	}
	if err := s.gen.AddModelSet(generation.ModelSet{ID: "mine", Name: "Mine", Kind: generation.KindSD, Recipe: "z-image-turbo",
		Components: comps, LaunchArgs: []string{"--steps", "9"}, DefaultWidth: 1024, DefaultHeight: 1024,
		Created: time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC), Complete: true}); err != nil {
		t.Fatal(err)
	}
	return s, dir
}

// setEditForm is the edit form of a set, as the browser would send it back.
func setEditForm(t *testing.T, s *Server, id string) url.Values {
	t.Helper()
	f, code, err := s.genSetEditForm(htmxReq("GET", "/?edit="+id, nil), s.gen)
	if err != nil {
		t.Fatalf("edit form: %d %v", code, err)
	}
	return formValues(f)
}

func putSet(s *Server, id string, form url.Values) *httptest.ResponseRecorder {
	rec := httptest.NewRecorder()
	s.handleGenUpdateSet(rec, withID(htmxReq("PUT", "/", form), id))
	return rec
}

func TestGenEditCustomSet(t *testing.T) {
	s, dir := editServer(t)
	before, _ := s.gen.ModelSet("mine")

	form := setEditForm(t, s, "mine")
	if form.Get("name") != "Mine" || form.Get("slot_vae") != filepath.Join(dir, "vae.gguf") || form.Get("launch_args") != "--steps 9" {
		t.Errorf("edit form = %v", form)
	}
	form.Set("launch_args", "--steps 12 --cfg-scale 1")
	form.Set("width", "768")
	// A name, backend or recipe sent with the edit is ignored.
	form.Set("name", "Renamed")
	form.Set("kind", "video")
	form.Set("recipe", "wan22-i2v")
	if rec := putSet(s, "mine", form); rec.Code != http.StatusOK {
		t.Fatalf("HTTP %d %s", rec.Code, rec.Body)
	}
	after, err := s.gen.ModelSet("mine")
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(after.RunArgs(), []string{"--steps", "12", "--cfg-scale", "1"}) || after.DefaultWidth != 768 {
		t.Errorf("edit not saved: %+v", after)
	}
	if after.ID != "mine" || after.Name != "Mine" || after.Kind != generation.KindSD || after.Recipe != "z-image-turbo" || !after.Created.Equal(before.Created) {
		t.Errorf("identity changed: %+v", after)
	}
	if _, err := s.gen.ModelSet("renamed"); err == nil {
		t.Error("the edit made a new set")
	}

	// The format check runs on an edit too.
	text := filepath.Join(dir, "notes.gguf")
	if err := os.WriteFile(text, []byte("not a model"), 0o644); err != nil {
		t.Fatal(err)
	}
	form = setEditForm(t, s, "mine")
	form.Set("slot_vae", text)
	if rec := putSet(s, "mine", form); rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), "is not a model file") {
		t.Errorf("text file: HTTP %d %s", rec.Code, rec.Body)
	}
}

// An edit is refused while an instance of the set runs or measures.
func TestGenEditRefusedWhileInUse(t *testing.T) {
	s, _ := editServer(t)
	addInstance(t, s, generation.Instance{ID: "inst", Name: "My Instance", ModelSetID: "mine", Port: 8090, Enabled: true})
	form := setEditForm(t, s, "mine")
	form.Set("launch_args", "--steps 1")

	s.genMgr.SetMeasuring("inst", true)
	rec := putSet(s, "mine", form)
	s.genMgr.SetMeasuring("inst", false)
	if rec.Code != http.StatusConflict || !strings.Contains(rec.Body.String(), "stop My Instance first") {
		t.Errorf("while measuring: HTTP %d %s", rec.Code, rec.Body)
	}
	if set, _ := s.gen.ModelSet("mine"); !slices.Equal(set.LaunchArgs, []string{"--steps", "9"}) {
		t.Errorf("the refused edit changed the set: %v", set.LaunchArgs)
	}

	// Stopped, the edit saves.
	if rec := putSet(s, "mine", form); rec.Code != http.StatusOK {
		t.Errorf("stopped: HTTP %d %s", rec.Code, rec.Body)
	}
}

func TestGenEditCatalogSetRefused(t *testing.T) {
	s := genServer(t)
	if err := s.gen.AddModelSet(generation.ModelSet{ID: "z-image-turbo", Name: "Z", Kind: generation.KindSD, Recipe: "z-image-turbo", CatalogID: "z-image-turbo"}); err != nil {
		t.Fatal(err)
	}
	rec := httptest.NewRecorder()
	s.handleGenSetForm(rec, htmxReq("GET", "/?edit=z-image-turbo", nil))
	if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), "Copy the set") {
		t.Errorf("form: HTTP %d %s", rec.Code, rec.Body)
	}
	if rec := putSet(s, "z-image-turbo", url.Values{"launch_args": {"--steps 1"}}); rec.Code != http.StatusBadRequest {
		t.Errorf("PUT: HTTP %d %s", rec.Code, rec.Body)
	}
	if rec := putSet(s, "nope", url.Values{}); rec.Code != http.StatusNotFound {
		t.Errorf("unknown set: HTTP %d", rec.Code)
	}
}

// After an edit to its files, an instance's measurement is out of date;
// after an edit to the test size only, it is not. The reservation stays.
func TestGenEditMarksMeasurementStale(t *testing.T) {
	s, dir := editServer(t)
	set, _ := s.gen.ModelSet("mine")
	inst := generation.Instance{ID: "inst", ModelSetID: "mine", Port: 8090, Enabled: true, ReservedMiB: 9000,
		MeasuredPeakMiB: 8000, MeasuredArgs: runArgs(set, generation.Instance{}), MeasuredFiles: generation.FilesKey(set)}
	addInstance(t, s, inst)
	stale := func() bool {
		set, _ := s.gen.ModelSet("mine")
		inst, _ := s.gen.Instance("inst")
		return measuredStale(inst, set)
	}
	if stale() {
		t.Fatal("stale before any edit")
	}

	form := setEditForm(t, s, "mine")
	form.Set("width", "768")
	putSet(s, "mine", form)
	if stale() {
		t.Error("a test size change marked the measurement stale")
	}

	other := filepath.Join(dir, "other-vae.gguf")
	if err := os.WriteFile(other, fakeGGUF, 0o644); err != nil {
		t.Fatal(err)
	}
	form = setEditForm(t, s, "mine")
	form.Set("slot_vae", other)
	if rec := putSet(s, "mine", form); rec.Code != http.StatusOK {
		t.Fatalf("HTTP %d %s", rec.Code, rec.Body)
	}
	if !stale() {
		t.Error("a file change did not mark the measurement stale")
	}
	if got, _ := s.gen.Instance("inst"); got.ReservedMiB != 9000 {
		t.Errorf("reservation = %d, want it kept", got.ReservedMiB)
	}

	// A measurement from before files were recorded is not marked by them.
	old := inst
	old.MeasuredFiles = nil
	cur, _ := s.gen.ModelSet("mine")
	if measuredStale(old, cur) {
		t.Error("an older measurement was marked stale by a file change")
	}
}

// An upscaler set has no flags, so its measurement stores none (nil
// after a save and load); a file change must still mark it stale, and a
// flag added later too.
func TestGenEditUpscalerMeasurementStale(t *testing.T) {
	set := generation.ModelSet{ID: "up", Kind: generation.KindUpscale, Components: []generation.Component{{Slot: generation.UpscalerSlot, Path: "/m/x4.pth"}}}
	inst := generation.Instance{ID: "up", ModelSetID: "up", MeasuredPeakMiB: 846, MeasuredArgs: nil, MeasuredFiles: generation.FilesKey(set)}
	if measuredStale(inst, set) {
		t.Fatal("stale before any change")
	}
	moved := set
	moved.Components = []generation.Component{{Slot: generation.UpscalerSlot, Path: "/m/../m/x4.pth"}}
	if !measuredStale(inst, moved) {
		t.Error("a file change was not seen on a set without flags")
	}
	flagged := set
	flagged.LaunchArgs = []string{"--threads", "8"}
	if !measuredStale(inst, flagged) {
		t.Error("a flag added to a set measured without flags was not seen")
	}
}

func TestGenEditFormAndButtons(t *testing.T) {
	s, _ := editServer(t)
	rec := httptest.NewRecorder()
	s.handleGenSetForm(rec, htmxReq("GET", "/?edit=mine", nil))
	body := rec.Body.String()
	for _, want := range []string{`hx-put="/api/gen/models/sets/mine"`, `name="name" required value="Mine" placeholder="e.g. qwen-turbo-local" readonly`,
		`<input type="hidden" name="kind" value="sd">`, `<input type="hidden" name="recipe" value="z-image-turbo">`, "Save changes", "Editing <strong>Mine</strong>"} {
		if !strings.Contains(body, want) {
			t.Errorf("edit form lacks %q", want)
		}
	}
	if strings.Count(body, " disabled>") != 2 {
		t.Error("the backend and recipe selects are not locked")
	}

	if err := s.gen.AddModelSet(generation.ModelSet{ID: "z-image-turbo", Name: "Z", Kind: generation.KindSD, CatalogID: "z-image-turbo"}); err != nil {
		t.Fatal(err)
	}
	rec = httptest.NewRecorder()
	s.handleGenModels(rec, htmxReq("GET", "/", nil))
	body = rec.Body.String()
	if !strings.Contains(body, "/api/gen/models/sets/form?edit=mine") || strings.Contains(body, "form?edit=z-image-turbo") {
		t.Error("Edit should show on custom sets only")
	}
	rec = httptest.NewRecorder()
	s.handleGenSetDetails(rec, withID(htmxReq("GET", "/?tab=models", nil), "mine"))
	if !strings.Contains(rec.Body.String(), "/api/gen/models/sets/form?edit=mine") {
		t.Error("the custom set's Details has no Edit button")
	}
}
