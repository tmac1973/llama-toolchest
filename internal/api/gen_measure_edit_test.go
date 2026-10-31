package api

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"image"
	"image/png"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"slices"
	"strconv"
	"sync"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/generation"
)

// A set that can edit is measured with an image and then an edit of that
// image, since an edit can need more memory, with as many references as
// the model takes; other image sets only make the image.
func TestMeasureEditsWhenTheSetCanEdit(t *testing.T) {
	var (
		mu       sync.Mutex
		paths    []string
		editImg  string
		editSize string
		editRefs int
	)
	fake := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		paths = append(paths, r.URL.Path)
		switch r.URL.Path {
		case "/v1/images/generations":
			w.Write([]byte(`{"data":[{"b64_json":"` + base64.StdEncoding.EncodeToString([]byte("PNGDATA")) + `"}]}`))
		case "/v1/images/edits":
			if err := r.ParseMultipartForm(1 << 20); err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			f, _, err := r.FormFile("image[]")
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			b, _ := io.ReadAll(f)
			editImg, editSize, editRefs = string(b), r.FormValue("size"), len(r.MultipartForm.File["image[]"])
			w.Write([]byte(`{"data":[{"b64_json":"eA=="}]}`))
		default:
			http.NotFound(w, r)
		}
	}))
	defer fake.Close()
	u, _ := url.Parse(fake.URL)
	port, _ := strconv.Atoi(u.Port())
	inst := generation.Instance{ID: "x", Port: port}
	s := &Server{}

	for _, tc := range []struct {
		recipe string
		caps   []string
		want   []string
		refs   int
	}{
		{"", nil, []string{"/v1/images/generations"}, 0},
		{"", []string{"edit"}, []string{"/v1/images/generations", "/v1/images/edits"}, 1},
		{"flux2-klein-4b", []string{"edit"}, []string{"/v1/images/generations", "/v1/images/edits"}, 4},
	} {
		paths, editImg, editSize, editRefs = nil, "", "", 0
		set := generation.ModelSet{Kind: generation.KindSD, Recipe: tc.recipe, Capabilities: tc.caps, DefaultWidth: 1024, DefaultHeight: 768}
		if err := s.runMeasureGeneration(context.Background(), inst, set, generation.VideoSize{}); err != nil {
			t.Fatalf("caps %v: %v", tc.caps, err)
		}
		if !slices.Equal(paths, tc.want) {
			t.Errorf("caps %v: requests %v, want %v", tc.caps, paths, tc.want)
		}
		if len(tc.caps) > 0 && (editImg != "PNGDATA" || editSize != "1024x768") {
			t.Errorf("edit got image %q at %q, want the generated image at 1024x768", editImg, editSize)
		}
		if editRefs != tc.refs {
			t.Errorf("recipe %q: the test edit had %d references, want %d", tc.recipe, editRefs, tc.refs)
		}
	}
}

// An upscaler set is measured with one upscale of a 1024x1024 PNG.
func TestMeasureUpscaler(t *testing.T) {
	var paths []string
	var size image.Point
	fake := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		paths = append(paths, r.URL.Path)
		var req struct {
			Image string `json:"image"`
		}
		json.NewDecoder(r.Body).Decode(&req)
		raw, _ := base64.StdEncoding.DecodeString(req.Image)
		if cfg, err := png.DecodeConfig(bytes.NewReader(raw)); err == nil {
			size = image.Pt(cfg.Width, cfg.Height)
		}
		w.Write([]byte(`{"images":[{"index":0,"b64_json":"eA=="}]}`))
	}))
	defer fake.Close()
	u, _ := url.Parse(fake.URL)
	port, _ := strconv.Atoi(u.Port())
	s := &Server{}
	set := generation.ModelSet{Kind: generation.KindUpscale}
	if err := s.runMeasureGeneration(context.Background(), generation.Instance{ID: "x", Port: port}, set, generation.VideoSize{}); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(paths, []string{"/sdcpp/v1/upscale"}) || size != image.Pt(1024, 1024) {
		t.Errorf("requests %v with a %v image, want one upscale of a 1024x1024 PNG", paths, size)
	}
}
