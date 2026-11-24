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
)

func TestVUpscaleSetForm(t *testing.T) {
	s := genServer(t)
	body := setFormFields(t, s, "kind=vupscale")
	if got := slotNames(body); !slices.Equal(got, []string{"dit", "vae"}) {
		t.Errorf("slots = %v", got)
	}
	for _, want := range []string{">Script", `value="relayvsr"`, `value="seedvr2"`, `name="scale"`, "×4", "Must match the RelayVSR files"} {
		if !strings.Contains(body, want) {
			t.Errorf("relayvsr form lacks %q", want)
		}
	}
	if strings.Contains(body, `name="frames"`) || strings.Contains(body, `name="width"`) {
		t.Error("the video upscaler form shows video test fields")
	}
	body = setFormFields(t, s, "kind=vupscale&recipe=seedvr2&scale=3")
	if !strings.Contains(body, `<option value="3" selected>×3</option>`) || strings.Contains(body, "Must match the RelayVSR files") {
		t.Errorf("seedvr2 form: %s", body)
	}
}

func TestVUpscaleCreateSet(t *testing.T) {
	s := genServer(t)
	dir := t.TempDir()
	file := func(name string) string {
		p := filepath.Join(dir, name)
		if err := os.WriteFile(p, fakeGGUF, 0o644); err != nil {
			t.Fatal(err)
		}
		return p
	}
	x4dit, x4vae := file("relayvsr_x4_int8_convrot.gguf"), file("relayvsr_flash_vae_x4_fp32.gguf")
	sdit, svae := file("seedvr2_7b_sharp.gguf"), file("seedvr2_vae.gguf")
	create := func(v url.Values) *httptest.ResponseRecorder {
		rec := httptest.NewRecorder()
		s.handleGenCreateSet(rec, htmxReq("POST", "/", v))
		return rec
	}

	rec := create(url.Values{"name": {"Relay four"}, "kind": {"vupscale"}, "recipe": {"relayvsr"}, "scale": {"4"}, "slot_dit": {x4dit}, "slot_vae": {x4vae}})
	if rec.Code != http.StatusOK {
		t.Fatalf("relayvsr ×4: %d %s", rec.Code, rec.Body)
	}
	set, err := s.gen.ModelSet("relay-four")
	if err != nil || set.Recipe != "relayvsr" || set.Scale != 4 || set.UpscaleScale() != 4 {
		t.Errorf("stored set = %+v, %v", set, err)
	}

	rec = create(url.Values{"name": {"Relay wrong"}, "kind": {"vupscale"}, "recipe": {"relayvsr"}, "scale": {"2"}, "slot_dit": {x4dit}, "slot_vae": {x4vae}})
	if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), "the RelayVSR files are for ×4; set the scale to 4") {
		t.Errorf("wrong scale: %d %s", rec.Code, rec.Body)
	}

	rec = create(url.Values{"name": {"Seed sharp"}, "kind": {"vupscale"}, "recipe": {"seedvr2"}, "scale": {"3"}, "slot_dit": {sdit}, "slot_vae": {svae}})
	if rec.Code != http.StatusOK {
		t.Errorf("seedvr2 ×3: %d %s", rec.Code, rec.Body)
	}

	rec = create(url.Values{"name": {"Seed eight"}, "kind": {"vupscale"}, "recipe": {"seedvr2"}, "scale": {"8"}, "slot_dit": {sdit}, "slot_vae": {svae}})
	if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), "must be one of [2 3 4]") {
		t.Errorf("seedvr2 ×8: %d %s", rec.Code, rec.Body)
	}
}

func TestVUpscaleKindNames(t *testing.T) {
	if genKindText("vupscale") != "video upscale" || genKindLabel("vupscale") != "vupscale" {
		t.Error("kind names")
	}
	if got := catalogTags("vupscale", nil); len(got) != 1 || got[0].Label != "video" {
		t.Errorf("catalog tags = %+v", got)
	}
}
