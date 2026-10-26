package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/generation"
)

// addVideoSet adds a complete video set to an instanceServer.
func addVideoSet(t *testing.T, s *Server, id string) {
	t.Helper()
	if err := s.gen.AddModelSet(generation.ModelSet{ID: id, Name: id, Kind: generation.KindVideo, Recipe: "wan22-ti2v-5b", Complete: true,
		Capabilities: []string{generation.CapText, generation.CapStartImage},
		Components:   []generation.Component{{Slot: "diffusion_model", Path: "/m/ti2v.gguf"}}}); err != nil {
		t.Fatal(err)
	}
}

// Image jobs never go to a video instance, even when it is the only one
// running.
func TestPickInstanceSkipsVideoForImages(t *testing.T) {
	s := instanceServer(t)
	addVideoSet(t, s, "ti2v")
	runFakeInstance(t, s, "ti2v", "ti2v", 0, newFakeBackend(t, 0))
	if inst, pe := s.pickInstance("", generation.KindSD); pe == nil {
		t.Errorf("picked %s for an image job", inst.ID)
	}
	if inst, pe := s.pickInstance("ti2v", generation.KindSD); pe == nil {
		t.Errorf("picked video instance %s for an image job by name", inst.ID)
	}
}

// /v1/models lists a video instance as kind video, with no API routes
// yet.
func TestV1ModelsVideoKind(t *testing.T) {
	s := instanceServer(t)
	addVideoSet(t, s, "ti2v")
	addInstance(t, s, generation.Instance{ID: "ti2v", ModelSetID: "ti2v", Port: 8090, Enabled: true, ReservedMiB: 20480})
	rec := serve(s, httptest.NewRequest("GET", "/v1/models/ti2v", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("HTTP %d: %s", rec.Code, rec.Body)
	}
	var got struct {
		Meta struct {
			Kind         string   `json:"kind"`
			API          []string `json:"api"`
			Capabilities []string `json:"capabilities"`
		} `json:"meta"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Meta.Kind != "video" || len(got.Meta.API) != 0 || len(got.Meta.Capabilities) != 2 {
		t.Errorf("meta = %+v, want kind video, no api, two capabilities", got.Meta)
	}
}

// A custom video set gets the video test defaults and its recipe's
// capabilities, and no ceiling.
func TestGenCreateVideoSet(t *testing.T) {
	s := genServer(t)
	dir := t.TempDir()
	var lines []string
	for slot, name := range map[string]string{"diffusion_model": "ti2v.gguf", "t5xxl": "umt5.gguf", "vae": "vae.safetensors"} {
		p := filepath.Join(dir, name)
		if err := os.WriteFile(p, []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
		lines = append(lines, slot+" = "+p)
	}
	form := url.Values{"name": {"my ti2v"}, "kind": {"video"}, "recipe": {"wan22-ti2v-5b"}, "files": {strings.Join(lines, "\n")}}
	rec := httptest.NewRecorder()
	s.handleGenCreateSet(rec, htmxReq("POST", "/", form))
	if rec.Code != http.StatusOK {
		t.Fatalf("HTTP %d: %s", rec.Code, rec.Body)
	}
	set, err := s.gen.ModelSet("my-ti2v")
	if err != nil {
		t.Fatal(err)
	}
	if set.Kind != generation.KindVideo || set.DefaultWidth != 832 || set.DefaultHeight != 480 || set.DefaultFrames != 33 || set.DefaultFPS != 24 {
		t.Errorf("set = %+v, want a video set with the 832x480, 33-frame, 24 fps test defaults", set)
	}
	if set.MaxFrames != 0 || set.MaxWidth != 0 {
		t.Errorf("a custom set got a ceiling: %+v", set)
	}
	if len(set.Capabilities) != 2 {
		t.Errorf("capabilities = %v, want the recipe's two", set.Capabilities)
	}
	if !strings.Contains(rec.Body.String(), "my ti2v") || !strings.Contains(rec.Body.String(), "Video") {
		t.Error("the section does not show the new set as Video")
	}

	form.Set("name", "wrong kind")
	form.Set("kind", "sd")
	rec = httptest.NewRecorder()
	s.handleGenCreateSet(rec, htmxReq("POST", "/", form))
	if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), "video recipe") {
		t.Errorf("video recipe for an image set: HTTP %d %s", rec.Code, rec.Body)
	}
}

// A catalog download copies the video fields into the set.
func TestGenCatalogDownloadVideoFields(t *testing.T) {
	s := genServer(t)
	rec := httptest.NewRecorder()
	s.handleGenCatalogDownload(rec, withID(htmxReq("POST", "/", nil), "wan22-ti2v-5b"))
	if rec.Code != http.StatusOK {
		t.Fatalf("download: %d %s", rec.Code, rec.Body)
	}
	set, err := s.gen.ModelSet("wan22-ti2v-5b")
	if err != nil {
		t.Fatal(err)
	}
	if set.Kind != generation.KindVideo || set.DefaultFrames != 33 || set.DefaultFPS != 24 || set.MaxFrames != 121 || set.MaxWidth != 1280 || set.MaxHeight != 704 {
		t.Errorf("set = %+v, want the catalog's video fields", set)
	}
}
