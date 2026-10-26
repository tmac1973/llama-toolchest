package api

import (
	"context"
	"encoding/base64"
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
// image, since an edit can need more memory; other image sets only make
// the image.
func TestMeasureEditsWhenTheSetCanEdit(t *testing.T) {
	var (
		mu       sync.Mutex
		paths    []string
		editImg  string
		editSize string
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
			editImg, editSize = string(b), r.FormValue("size")
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
		caps []string
		want []string
	}{
		{nil, []string{"/v1/images/generations"}},
		{[]string{"edit"}, []string{"/v1/images/generations", "/v1/images/edits"}},
	} {
		paths, editImg, editSize = nil, "", ""
		set := generation.ModelSet{Kind: generation.KindSD, Capabilities: tc.caps, DefaultWidth: 1024, DefaultHeight: 768}
		if err := s.runMeasureGeneration(context.Background(), inst, set, generation.VideoSize{}); err != nil {
			t.Fatalf("caps %v: %v", tc.caps, err)
		}
		if !slices.Equal(paths, tc.want) {
			t.Errorf("caps %v: requests %v, want %v", tc.caps, paths, tc.want)
		}
		if len(tc.caps) > 0 && (editImg != "PNGDATA" || editSize != "1024x768") {
			t.Errorf("edit got image %q at %q, want the generated image at 1024x768", editImg, editSize)
		}
	}
}
