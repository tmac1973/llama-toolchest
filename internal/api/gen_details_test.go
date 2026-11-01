package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/go-chi/chi/v5"
	"github.com/tmac1973/llama-toolchest/internal/generation"
	"github.com/tmac1973/llama-toolchest/web"
)

// A catalog entry's Details: JSON for API clients, the dialog body for
// htmx.
func TestGenCatalogDetails(t *testing.T) {
	s := genServer(t)

	rec := httptest.NewRecorder()
	s.handleGenCatalogDetails(rec, withID(httptest.NewRequest("GET", "/", nil), "z-image-turbo"))
	if rec.Code != http.StatusOK {
		t.Fatalf("JSON: HTTP %d %s", rec.Code, rec.Body)
	}
	var d struct {
		Program    string `json:"program"`
		Recipe     string `json:"recipe"`
		Command    string `json:"command"`
		Slots      []struct{ Slot, Label, Path string }
		Flags      []struct{ Flag, Value, Reason string }
		FixedFlags []struct{ Flag string } `json:"fixed_flags"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &d); err != nil {
		t.Fatalf("%v: %s", err, rec.Body)
	}
	if d.Program != "sd-server" || d.Recipe != "z-image-turbo" || len(d.Slots) != 3 || len(d.Flags) == 0 || len(d.FixedFlags) != 3 ||
		!strings.Contains(d.Command, "--max-vram <vram limit>") {
		t.Errorf("details = %+v", d)
	}
	for _, f := range d.Flags {
		if f.Reason == "" {
			t.Errorf("flag %s has no reason in the JSON", f.Flag)
		}
	}

	rec = httptest.NewRecorder()
	s.handleGenCatalogDetails(rec, withID(htmxReq("GET", "/", nil), "z-image-turbo"))
	body := rec.Body.String()
	if rec.Code != http.StatusOK {
		t.Fatalf("HTML: HTTP %d %s", rec.Code, body)
	}
	for _, want := range []string{">Settings<", ">Added by toolchest<", "Not downloaded yet", "--backend", "Runs the text model on the CPU", "Copy command", "Catalog set."} {
		if !strings.Contains(body, want) {
			t.Errorf("dialog lacks %q", want)
		}
	}
}

func TestGenSetDetailsCustom(t *testing.T) {
	s := genServer(t)
	if err := s.gen.AddModelSet(generation.ModelSet{ID: "mine", Name: "Mine", Kind: generation.KindUpscale, Complete: true,
		Components: []generation.Component{{Slot: generation.UpscalerSlot, Path: "/m/x4.pth"}}}); err != nil {
		t.Fatal(err)
	}
	rec := httptest.NewRecorder()
	s.handleGenSetDetails(rec, withID(htmxReq("GET", "/", nil), "mine"))
	body := rec.Body.String()
	if rec.Code != http.StatusOK {
		t.Fatalf("HTTP %d %s", rec.Code, body)
	}
	for _, want := range []string{"Custom set, made from your own files.", "ESRGAN upscaler", "/m/x4.pth", "--hires-upscalers-dir", "No settings of its own."} {
		if !strings.Contains(body, want) {
			t.Errorf("dialog lacks %q", want)
		}
	}
}

func TestGenDetailsNotFound(t *testing.T) {
	s := genServer(t)
	for name, h := range map[string]http.HandlerFunc{"set": s.handleGenSetDetails, "catalog": s.handleGenCatalogDetails} {
		for _, req := range []*http.Request{httptest.NewRequest("GET", "/", nil), htmxReq("GET", "/", nil)} {
			rec := httptest.NewRecorder()
			h(rec, withID(req, "no-such-set"))
			if rec.Code != http.StatusNotFound {
				t.Errorf("%s: HTTP %d, want 404", name, rec.Code)
			}
		}
	}
}

// The routes are wired up, and the rows have a Details button for the
// dialog on their page.
func TestGenDetailsRoutesAndButtons(t *testing.T) {
	s := genServer(t)
	router := s.buildRouter()
	for _, path := range []string{"/api/gen/models/catalog/z-image-turbo/details", "/api/gen/models/sets/no-such-set/details"} {
		rctx := chi.NewRouteContext()
		if !router.Match(rctx, "GET", path) {
			t.Errorf("no route for GET %s", path)
		}
	}

	if err := s.gen.AddModelSet(generation.ModelSet{ID: "mine", Name: "Mine", Kind: generation.KindUpscale, Complete: true}); err != nil {
		t.Fatal(err)
	}
	rec := httptest.NewRecorder()
	s.handleGenModels(rec, htmxReq("GET", "/", nil))
	if body := rec.Body.String(); !strings.Contains(body, `/api/gen/models/sets/mine/details?tab=models`) || !strings.Contains(body, "gen-set-details-dialog") {
		t.Error("the set list has no Details button for the Models tab dialog")
	}
	rec = httptest.NewRecorder()
	s.handleGenCatalog(rec, htmxReq("GET", "/", nil))
	if body := rec.Body.String(); !strings.Contains(body, `/api/gen/models/catalog/z-image-turbo/details?tab=catalog`) || !strings.Contains(body, "gen-catalog-details-dialog") {
		t.Error("the catalog has no Details button for the Download Models tab dialog")
	}

	for page, ids := range map[string][]string{
		"templates/models.html":        {`id="gen-set-details-dialog"`, `id="gen-set-details-body"`, `id="gen-set-details-name"`},
		"templates/models_browse.html": {`id="gen-catalog-details-dialog"`, `id="gen-catalog-details-body"`, `id="gen-catalog-details-name"`},
	} {
		text, err := web.Templates.ReadFile(page)
		if err != nil {
			t.Fatal(err)
		}
		for _, id := range ids {
			if !strings.Contains(string(text), id) {
				t.Errorf("%s lacks %s", page, id)
			}
		}
	}
}

// A set with no flags of its own gives an empty list in the JSON, not
// null (found in the 2026-11-01 field test with ACE-Step turbo).
func TestGenDetailsEmptyFlagsJSON(t *testing.T) {
	s := genServer(t)
	rec := httptest.NewRecorder()
	s.handleGenCatalogDetails(rec, withID(httptest.NewRequest("GET", "/", nil), "ace-step-turbo"))
	if !strings.Contains(rec.Body.String(), `"flags":[]`) {
		t.Errorf("flags not an empty list: %s", rec.Body)
	}
}
