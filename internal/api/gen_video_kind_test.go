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
	"github.com/tmac1973/llama-toolchest/internal/testutil"
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

// /v1/models lists a video instance as kind video, with the video
// routes.
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
	if got.Meta.Kind != "video" || !slices.Equal(got.Meta.API, []string{"video.jobs", "video.sync"}) || len(got.Meta.Capabilities) != 2 {
		t.Errorf("meta = %+v, want kind video, the video routes, two capabilities", got.Meta)
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
		if err := os.WriteFile(p, fakeGGUF, 0o644); err != nil {
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

// A custom video set can have a ceiling, and /v1/models shows it with the
// recipe's size multiple, as for a catalog set.
func TestGenCreateVideoSetCeiling(t *testing.T) {
	s := genServer(t)
	dir := t.TempDir()
	var lines []string
	for slot, name := range map[string]string{"diffusion_model": "ti2v.gguf", "t5xxl": "umt5.gguf", "vae": "vae.safetensors"} {
		p := filepath.Join(dir, name)
		if err := os.WriteFile(p, fakeGGUF, 0o644); err != nil {
			t.Fatal(err)
		}
		lines = append(lines, slot+" = "+p)
	}
	post := func(name string, extra url.Values) *httptest.ResponseRecorder {
		form := url.Values{"name": {name}, "kind": {"video"}, "recipe": {"wan22-ti2v-5b"}, "files": {strings.Join(lines, "\n")}}
		for k, v := range extra {
			form[k] = v
		}
		rec := httptest.NewRecorder()
		s.handleGenCreateSet(rec, htmxReq("POST", "/", form))
		return rec
	}

	if rec := post("capped", url.Values{"max_frames": {"81"}, "max_width": {"1280"}, "max_height": {"704"}}); rec.Code != http.StatusOK {
		t.Fatalf("HTTP %d: %s", rec.Code, rec.Body)
	}
	addInstance(t, s, generation.Instance{ID: "capped", ModelSetID: "capped", Port: 8090, Enabled: true, ReservedMiB: 20480})
	rec := serve(s, httptest.NewRequest("GET", "/v1/models/capped", nil))
	var got struct {
		Meta struct {
			Video map[string]any `json:"video"`
		} `json:"meta"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	v := got.Meta.Video
	if v["max_frames"] != 81.0 || v["max_size"] != "1280x704" || v["size_multiple"] != 32.0 {
		t.Errorf("meta.video = %v, want max_frames 81, max_size 1280x704, size_multiple 32", v)
	}

	for name, extra := range map[string]url.Values{
		"half a size":   {"max_width": {"1280"}},
		"not 8":         {"max_width": {"1281"}, "max_height": {"704"}},
		"short ceiling": {"max_frames": {"17"}},
		"small ceiling": {"max_width": {"640"}, "max_height": {"360"}},
	} {
		if rec := post("bad "+name, extra); rec.Code != http.StatusBadRequest {
			t.Errorf("%s: HTTP %d %s, want 400", name, rec.Code, rec.Body)
		}
	}
	// A portrait test size fits a landscape ceiling.
	if rec := post("portrait", url.Values{"width": {"480"}, "height": {"832"}, "max_width": {"1280"}, "max_height": {"704"}}); rec.Code != http.StatusOK {
		t.Errorf("portrait test size: HTTP %d %s", rec.Code, rec.Body)
	}
}

// A custom upscaler set keeps its upscale factor (4 when not given), and
// /v1/models shows it.
func TestGenCreateUpscalerSetScale(t *testing.T) {
	s := genServer(t)
	file := filepath.Join(t.TempDir(), "x2.pth")
	if err := os.WriteFile(file, fakeGGUF, 0o644); err != nil {
		t.Fatal(err)
	}
	for name, tc := range map[string]struct {
		scale string
		code  int
		want  float64
	}{
		"x2":      {"2", http.StatusOK, 2},
		"default": {"", http.StatusOK, 4},
		"too big": {"9", http.StatusBadRequest, 0},
	} {
		form := url.Values{"name": {name}, "kind": {"upscale"}, "files": {"upscaler = " + file}, "scale": {tc.scale}}
		rec := httptest.NewRecorder()
		s.handleGenCreateSet(rec, htmxReq("POST", "/", form))
		if rec.Code != tc.code {
			t.Errorf("%s: HTTP %d %s, want %d", name, rec.Code, rec.Body, tc.code)
			continue
		}
		if tc.code != http.StatusOK {
			continue
		}
		id := generation.Slug(name)
		addInstance(t, s, generation.Instance{ID: id, ModelSetID: id, Port: 8090 + len(name), Enabled: true, ReservedMiB: 1024})
		var got struct {
			Meta struct {
				Scale float64 `json:"scale"`
			} `json:"meta"`
		}
		if err := json.Unmarshal(serve(s, httptest.NewRequest("GET", "/v1/models/"+id, nil)).Body.Bytes(), &got); err != nil {
			t.Fatal(err)
		}
		if got.Meta.Scale != tc.want {
			t.Errorf("%s: meta.scale = %v, want %v", name, got.Meta.Scale, tc.want)
		}
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
	// Let the downloads end before the test ends, as in
	// TestGenCatalogCardIsSeparateFromSets. The fake's files are not the
	// catalog's sizes, so they fail and the set stays incomplete.
	testutil.Eventually(t, func() bool {
		return len(s.downloader.ListActive()) == 0
	}, "the catalog downloads never ended")
	if set, _ := s.gen.ModelSet("wan22-ti2v-5b"); set.Complete {
		t.Error("files of the wrong size completed the set")
	}
}
