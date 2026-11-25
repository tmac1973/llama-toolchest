package api

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/builder"
	"github.com/tmac1973/llama-toolchest/internal/generation"
	"github.com/tmac1973/llama-toolchest/internal/process"
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
	useSDBuild(t, s, "--steps N\n  --hires-upscalers-dir DIR\n") // before master-1116: no DGL
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

// useSDBuild makes s's only stable-diffusion.cpp build a fake sd-server
// whose --help prints help.
func useSDBuild(t *testing.T, s *Server, help string) {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("shell-script fake")
	}
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "help.txt"), []byte(help), 0o644)
	bin := filepath.Join(dir, "sd-server")
	os.WriteFile(bin, []byte("#!/bin/sh\ncat \"$(dirname \"$0\")/help.txt\"\nexit 1\n"), 0o755)
	records := fmt.Sprintf(`[{"id":"sd-master-1-abc-vulkan","project":"stable-diffusion.cpp","profile":"vulkan","git_ref":"master-1-abc","status":"success","binary_path":%q,"commit_count":1}]`, bin)
	if err := os.WriteFile(filepath.Join(s.cfg.DataDir, "config", "builds.json"), []byte(records), 0o644); err != nil {
		t.Fatal(err)
	}
	s.builder = builder.NewBuilder(s.cfg.DataDir)
}

func testPNG(t *testing.T, w, h int) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := range h {
		for x := range w {
			img.Set(x, y, color.RGBA{uint8(x), uint8(y), 99, 255})
		}
	}
	var b bytes.Buffer
	if err := png.Encode(&b, img); err != nil {
		t.Fatal(err)
	}
	return b.Bytes()
}

// dglUpscaleServer is an instanceServer whose sd build has DGL, with a
// running Real-ESRGAN ×4 instance "esr" whose sd-server is the fake f.
func dglUpscaleServer(t *testing.T, f *fakeDGL) *Server {
	t.Helper()
	fastPoll(t)
	s := instanceServer(t)
	useSDBuild(t, s, "--steps N\n  --dgl-dir DIR  Web UI script directory\n")
	if err := s.gen.AddModelSet(generation.ModelSet{ID: "esrgan", Name: "esrgan", Kind: generation.KindUpscale, Scale: 4, Complete: true,
		Components: []generation.Component{{Slot: generation.UpscalerSlot, Path: "/m/esr/RealESRGAN_x4plus.safetensors"}}}); err != nil {
		t.Fatal(err)
	}
	u, _ := url.Parse(f.srv.URL)
	port, _ := strconv.Atoi(u.Port())
	addInstance(t, s, generation.Instance{ID: "esr", ModelSetID: "esrgan", Port: port, Enabled: true, ReservedMiB: 1024})
	f.workDir = generation.InstanceDir(s.cfg.DataDir, "esr")
	bin := filepath.Join(t.TempDir(), "fake")
	os.WriteFile(bin, []byte("#!/bin/sh\nexec sleep 30\n"), 0o755)
	if err := s.genMgr.Start("esr", generation.KindUpscale, process.SupervisorConfig{
		Name: "esr", BinaryPath: bin, Env: os.Environ(), HealthURL: f.srv.URL + "/v1/models", HealthInterval: 20 * time.Millisecond,
	}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.genMgr.Stop("esr") })
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := s.genMgr.WaitRunning(ctx, "esr"); err != nil {
		t.Fatal(err)
	}
	return s
}

// On a build with DGL (master-1119 removed /sdcpp/v1/upscale), an upscale
// runs the ESRGAN script and answers as the old route did; the run's
// files do not stay in the instance folder.
func TestImageUpscaleDGL(t *testing.T) {
	f := newFakeDGL(t)
	f.output = testPNG(t, 64, 48)
	s := dglUpscaleServer(t, f)

	in := base64.StdEncoding.EncodeToString(testPNG(t, 16, 12))
	rec := httptest.NewRecorder()
	s.handleGenUpscale(rec, upscaleReq(`{"image":"data:image/png;base64,`+in+`","tile_size":64}`))
	if rec.Code != http.StatusOK {
		t.Fatalf("upscale: %d %s", rec.Code, rec.Body.String())
	}
	var out upscaleAnswer
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	if out.Width != 64 || out.Height != 48 || out.Scale != 4 || out.Repeats != 1 || out.OutputFormat != "png" ||
		out.Upscaler != "RealESRGAN_x4plus" || len(out.Images) != 1 {
		t.Errorf("answer = %+v", out)
	}
	if raw, _ := base64.StdEncoding.DecodeString(out.Images[0].B64JSON); !bytes.Equal(raw, f.output) {
		t.Error("the answer's image is not sd-server's output")
	}
	if len(f.sources) != 1 {
		t.Fatalf("%d scripts sent, want 1", len(f.sources))
	}
	src := f.sources[0]
	for _, want := range []string{`upscaler_path = "/m/esr/RealESRGAN_x4plus.safetensors"`, "tile_size = 64", "repeats = 1", `input_path = "input/upscale-`} {
		if !strings.Contains(src, want) {
			t.Errorf("script lacks %q:\n%s", want, src)
		}
	}
	for _, d := range []string{"input", "output"} {
		if left, _ := os.ReadDir(filepath.Join(f.workDir, d)); len(left) != 0 {
			t.Errorf("%s/ keeps %d files after the upscale", d, len(left))
		}
	}

	// JPEG out, made by toolchest from the PNG.
	rec = httptest.NewRecorder()
	s.handleGenUpscale(rec, upscaleReq(`{"image":"`+in+`","output_format":"jpeg","output_compression":80}`))
	if rec.Code != http.StatusOK {
		t.Fatalf("jpeg upscale: %d %s", rec.Code, rec.Body.String())
	}
	json.Unmarshal(rec.Body.Bytes(), &out)
	raw, _ := base64.StdEncoding.DecodeString(out.Images[0].B64JSON)
	if _, err := jpeg.DecodeConfig(bytes.NewReader(raw)); err != nil || out.OutputFormat != "jpeg" {
		t.Errorf("jpeg answer: format %q, decode error %v", out.OutputFormat, err)
	}
}

// The DGL path checks a request before it waits for the GPU, as
// sd-server's old route did, and names what to change.
func TestImageUpscaleDGLRefuses(t *testing.T) {
	f := newFakeDGL(t)
	s := dglUpscaleServer(t, f)
	small := base64.StdEncoding.EncodeToString(testPNG(t, 16, 12))
	big := base64.StdEncoding.EncodeToString(testPNG(t, 600, 10))
	for _, tc := range []struct{ body, want string }{
		{`{"image":"` + small + `","output_format":"webp"}`, "use png or jpeg"},
		{`{"image":"` + small + `","output_format":"bmp"}`, "png or jpeg"},
		{`{"image":"` + small + `","tile_size":16}`, "tile_size"},
		{`{"repeats":2}`, "image is required"},
		{`{"image":"!!"}`, "base64"},
		{`{"image":"aW1n"}`, "PNG, JPEG or GIF"},
		{`{"image":"` + big + `","repeats":2}`, "9600x160"},
	} {
		rec := httptest.NewRecorder()
		s.handleGenUpscale(rec, upscaleReq(tc.body))
		if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), tc.want) {
			t.Errorf("%s: %d %s, want 400 with %q", tc.body, rec.Code, rec.Body.String(), tc.want)
		}
	}
	if len(f.sources) != 0 {
		t.Errorf("%d scripts reached sd-server", len(f.sources))
	}

	// An out-of-memory job says what to change for an image.
	f.fail, f.failCode = "cannot make enough memory available", "out_of_memory"
	rec := httptest.NewRecorder()
	s.handleGenUpscale(rec, upscaleReq(`{"image":"`+small+`"}`))
	if rec.Code != http.StatusBadGateway || !strings.Contains(rec.Body.String(), "smaller image, fewer repeats") {
		t.Errorf("out of memory: %d %s", rec.Code, rec.Body.String())
	}
}

// On a build with DGL, Measure upscales its 1024x1024 test image once with
// the ESRGAN script.
func TestMeasureUpscalerDGL(t *testing.T) {
	f := newFakeDGL(t)
	f.output = testPNG(t, 8, 8)
	s := dglUpscaleServer(t, f)
	inst, _ := s.gen.Instance("esr")
	set, _ := s.gen.ModelSet("esrgan")
	if err := s.runMeasureGeneration(context.Background(), inst, set, generation.VideoSize{}, 0); err != nil {
		t.Fatal(err)
	}
	if len(f.sources) != 1 || !strings.Contains(f.sources[0], "repeats = 1") {
		t.Errorf("scripts sent: %q", f.sources)
	}
}
