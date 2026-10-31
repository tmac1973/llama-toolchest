package api

import (
	"bytes"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/generation"
)

func upscaleReq(body string) *http.Request {
	r := httptest.NewRequest("POST", "/v1/images/upscale", strings.NewReader(body))
	r.Header.Set("Content-Type", "application/json")
	return r
}

// An upscale goes unchanged to the upscaler instance's /sdcpp/v1/upscale,
// never to an image instance, and image routes never pick the upscaler.
func TestImageUpscale(t *testing.T) {
	s := vramServer(t)
	if err := s.gen.AddModelSet(generation.ModelSet{ID: "esrgan", Name: "esrgan", Kind: generation.KindUpscale, Complete: true}); err != nil {
		t.Fatal(err)
	}
	image, upscaler := newFakeBackend(t, 0), newFakeBackend(t, 0)
	runFakeInstance(t, s, "ming", "ming-image", 0, image)
	runFakeInstance(t, s, "esrgan", "esrgan", 0, upscaler)

	body := `{"image":"aW1n","repeats":2}`
	rec := httptest.NewRecorder()
	s.handleGenUpscale(rec, upscaleReq(body))
	if rec.Code != http.StatusOK {
		t.Fatalf("upscale: %d %s", rec.Code, rec.Body.String())
	}
	if len(upscaler.bodies) != 1 || upscaler.bodies[0] != "/sdcpp/v1/upscale "+body || len(image.bodies) != 0 {
		t.Errorf("upscaler got %q, image instance got %q", upscaler.bodies, image.bodies)
	}

	rec = httptest.NewRecorder()
	s.handleGenUpscale(rec, upscaleReq(`{"model":"ming","image":"aW1n"}`))
	if rec.Code != http.StatusNotFound || !strings.Contains(errCode(t, rec), "model_not_found") {
		t.Errorf("upscale on an image instance: %d %s", rec.Code, rec.Body.String())
	}
	rec = httptest.NewRecorder()
	s.handleGenImages(rec, imageReq(`{"model":"esrgan","prompt":"x"}`))
	if rec.Code != http.StatusNotFound {
		t.Errorf("generation on the upscaler: %d %s", rec.Code, rec.Body.String())
	}

	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)
	mw.WriteField("model", "esrgan")
	mw.Close()
	r := httptest.NewRequest("POST", "/v1/images/upscale", &buf)
	r.Header.Set("Content-Type", mw.FormDataContentType())
	rec = httptest.NewRecorder()
	s.handleGenUpscale(rec, r)
	if rec.Code != http.StatusBadRequest || !strings.Contains(errCode(t, rec), "takes JSON") {
		t.Errorf("multipart: %d %s", rec.Code, rec.Body.String())
	}
}
