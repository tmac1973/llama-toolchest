package api

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
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

	"github.com/tmac1973/llama-toolchest/internal/generation"
	"github.com/tmac1973/llama-toolchest/internal/process"
	"github.com/tmac1973/llama-toolchest/internal/testutil"
	"github.com/tmac1973/llama-toolchest/internal/videofile"
)

// vupscaleServer is an instanceServer with a running RelayVSR ×2 instance
// "vup" whose sd-server is the fake f.
func vupscaleServer(t *testing.T, f *fakeDGL, mutate ...func(*generation.Instance)) *Server {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("shell-script fake")
	}
	fastPoll(t)
	s := instanceServer(t)
	if err := s.gen.AddModelSet(generation.ModelSet{ID: "relay2", Name: "RelayVSR ×2", Kind: generation.KindVUpscale, Recipe: "relayvsr", Scale: 2, Complete: true,
		Components: []generation.Component{{Slot: "dit", Path: "/m/dit.safetensors"}, {Slot: "vae", Path: "/m/vae.safetensors"}}}); err != nil {
		t.Fatal(err)
	}
	u, _ := url.Parse(f.srv.URL)
	port, _ := strconv.Atoi(u.Port())
	inst := generation.Instance{ID: "vup", ModelSetID: "relay2", Port: port, Enabled: true, ReservedMiB: 16384}
	for _, m := range mutate {
		m(&inst)
	}
	addInstance(t, s, inst)
	f.workDir = generation.InstanceDir(s.cfg.DataDir, "vup")
	bin := filepath.Join(t.TempDir(), "fake")
	os.WriteFile(bin, []byte("#!/bin/sh\nexec sleep 30\n"), 0o755)
	if err := s.genMgr.Start("vup", generation.KindVUpscale, process.SupervisorConfig{
		Name: "vup", BinaryPath: bin, Env: os.Environ(), HealthURL: f.srv.URL + "/v1/models", HealthInterval: 20 * time.Millisecond,
	}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.genMgr.Stop("vup") })
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := s.genMgr.WaitRunning(ctx, "vup"); err != nil {
		t.Fatal(err)
	}
	return s
}

func testClip(t *testing.T, w, h, frames int) []byte {
	t.Helper()
	var b bytes.Buffer
	if err := videofile.WriteRGBAVI(&b, w, h, frames, 16, func(int, []byte) {}); err != nil {
		t.Fatal(err)
	}
	return b.Bytes()
}

func fixture(t *testing.T, name string) []byte {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("..", "videofile", "testdata", name))
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func multipartReq(t *testing.T, fields map[string]string, video []byte) *http.Request {
	t.Helper()
	var body bytes.Buffer
	mw := multipart.NewWriter(&body)
	if video != nil {
		fw, _ := mw.CreateFormFile("video", "clip.bin")
		fw.Write(video)
	}
	for k, v := range fields {
		mw.WriteField(k, v)
	}
	mw.Close()
	r := httptest.NewRequest("POST", "/v1/videos/upscale/jobs", &body)
	r.Header.Set("Content-Type", mw.FormDataContentType())
	return r
}

func jsonVUpscaleReq(fields map[string]any) *http.Request {
	b, _ := json.Marshal(fields)
	r := httptest.NewRequest("POST", "/v1/videos/upscale/jobs", bytes.NewReader(b))
	r.Header.Set("Content-Type", "application/json")
	return r
}

func waitVUpscaleJob(t *testing.T, s *Server, id string) vupscaleJob {
	t.Helper()
	var j vupscaleJob
	testutil.Eventually(t, func() bool {
		j, _ = s.genVUpscaleJobs.view(id)
		return j.Finished != nil
	}, "the video upscale job did not finish")
	return j
}

func createVUpscaleJobID(t *testing.T, s *Server, r *http.Request) string {
	t.Helper()
	rec := serve(s, r)
	if rec.Code != http.StatusAccepted {
		t.Fatalf("create: HTTP %d %s", rec.Code, rec.Body)
	}
	var out struct {
		ID string `json:"id"`
	}
	json.Unmarshal(rec.Body.Bytes(), &out)
	return out.ID
}

func dirEmpty(t *testing.T, dir string) bool {
	t.Helper()
	entries, _ := os.ReadDir(dir)
	for _, e := range entries {
		if !e.IsDir() {
			return false
		}
	}
	return true
}

func TestVUpscaleJobMultipartWebM(t *testing.T) {
	f := newFakeDGL(t)
	f.result = `{"outputs":[{"name":"width","value":"128"},{"name":"height","value":"96"},{"name":"frames","value":"9"}]}`
	s := vupscaleServer(t, f)
	id := createVUpscaleJobID(t, s, multipartReq(t, map[string]string{"model": "vup", "seed": "7"}, fixture(t, "vp8-key-64x48-9f.webm")))
	j := waitVUpscaleJob(t, s, id)
	if j.Status != jobSucceeded || j.Video == nil {
		t.Fatalf("job = %+v", j)
	}
	if j.Video.Width != 128 || j.Video.Height != 96 || j.Video.Frames != 9 || j.Video.MimeType != "video/webm" || j.Video.Format != "webm" {
		t.Errorf("video = %+v", *j.Video)
	}
	if j.Input.Format != "webm" || j.Input.Frames != 9 || j.Input.FPS == nil || *j.Input.FPS != 16 || j.AudioDropped {
		t.Errorf("input = %+v, audio dropped %v", j.Input, j.AudioDropped)
	}
	if j.Request.Seed != 7 || j.Request.WindowSize != 16 || j.Request.OutputFormat != "webm" {
		t.Errorf("request = %+v", j.Request)
	}
	src := f.sources[0]
	for _, want := range []string{`input_path = "input/` + id + `.webm"`, `output_path = "output/` + id + `.webm"`, "seed = 7", "window_size = 16", "video::load(input_path)"} {
		if !strings.Contains(src, want) {
			t.Errorf("script lacks %q", want)
		}
	}
	rec := serve(s, httptest.NewRequest("GET", j.Video.URL, nil))
	if rec.Code != http.StatusOK || rec.Body.String() != "upscaled video" || rec.Header().Get("Content-Type") != "video/webm" {
		t.Errorf("download: HTTP %d %q %s", rec.Code, rec.Body, rec.Header().Get("Content-Type"))
	}
	work := generation.InstanceDir(s.cfg.DataDir, "vup")
	if !dirEmpty(t, filepath.Join(work, "input")) || !dirEmpty(t, filepath.Join(work, "output")) || !dirEmpty(t, s.vupscaleIncomingDir()) {
		t.Error("work files were left behind")
	}
}

func TestVUpscaleJobJSONAudioDropped(t *testing.T) {
	f := newFakeDGL(t)
	f.result = `{"outputs":[{"name":"width","value":"128"},{"name":"height","value":"96"},{"name":"frames","value":"8"}]}`
	s := vupscaleServer(t, f)
	clip := base64.StdEncoding.EncodeToString(fixture(t, "mjpeg-pcm-64x48-8f.avi"))
	id := createVUpscaleJobID(t, s, jsonVUpscaleReq(map[string]any{"model": "relay2", "video": "data:video/x-msvideo;base64," + clip, "window_size": 8}))
	j := waitVUpscaleJob(t, s, id)
	if j.Status != jobSucceeded || !j.AudioDropped || !j.Input.HasAudio || j.Request.WindowSize != 8 {
		t.Errorf("job = %+v", j)
	}

	id = createVUpscaleJobID(t, s, jsonVUpscaleReq(map[string]any{"model": "vup", "video": clip, "output_format": "avi"}))
	j = waitVUpscaleJob(t, s, id)
	if j.Status != jobSucceeded || j.AudioDropped || j.Video.MimeType != "video/x-msvideo" {
		t.Errorf("avi job = %+v", j)
	}
	if src := f.sources[len(f.sources)-1]; !strings.Contains(src, "audio = clip.audio") {
		t.Error("the avi script does not keep the audio")
	}
}

func TestVUpscaleJobRefusals(t *testing.T) {
	f := newFakeDGL(t)
	s := vupscaleServer(t, f)
	if err := s.gen.AddModelSet(generation.ModelSet{ID: "img", Name: "Image", Kind: generation.KindSD, Recipe: "sdxl-checkpoint", Complete: true,
		Components: []generation.Component{{Slot: "model", Path: "/m/x.safetensors"}}}); err != nil {
		t.Fatal(err)
	}
	addInstance(t, s, generation.Instance{ID: "imgi", ModelSetID: "img", Port: 1, Enabled: true, ReservedMiB: 1024})
	webm := fixture(t, "vp8-key-64x48-9f.webm")
	for name, tc := range map[string]struct {
		r    *http.Request
		code int
		want string
	}{
		"mp4":           {multipartReq(t, map[string]string{"model": "vup"}, fixture(t, "tiny.mp4")), 400, "MP4 is not supported"},
		"interframe":    {multipartReq(t, map[string]string{"model": "vup"}, fixture(t, "vp8-interframe.webm")), 400, "independent VP8 frames"},
		"no video":      {multipartReq(t, map[string]string{"model": "vup"}, nil), 400, "has no video"},
		"unknown field": {multipartReq(t, map[string]string{"model": "vup", "prompt": "x"}, webm), 400, "unknown field prompt"},
		"format":        {multipartReq(t, map[string]string{"model": "vup", "output_format": "mp4"}, webm), 400, "webm or avi"},
		"window":        {multipartReq(t, map[string]string{"model": "vup", "window_size": "1"}, webm), 400, "between 2 and 32"},
		"one frame":     {multipartReq(t, map[string]string{"model": "vup"}, fixture(t, "still-64x48.webp")), 400, "at least 2 frames"},
		"wrong kind":    {multipartReq(t, map[string]string{"model": "img"}, webm), 400, "not a video upscaler"},
		"unknown model": {multipartReq(t, map[string]string{"model": "nope"}, webm), 404, "model not found"},
		"too large":     {multipartReq(t, map[string]string{"model": "vup"}, testClip(t, 1280, 720, 100)), 400, "sd.cpp reads at most 256 MiB"},
	} {
		rec := serve(s, tc.r)
		if rec.Code != tc.code || !strings.Contains(rec.Body.String(), tc.want) {
			t.Errorf("%s: HTTP %d %s, want %d with %q", name, rec.Code, rec.Body, tc.code, tc.want)
		}
	}
	if len(f.sources) != 0 {
		t.Errorf("refused requests reached sd-server: %d", len(f.sources))
	}
	work := generation.InstanceDir(s.cfg.DataDir, "vup")
	if !dirEmpty(t, filepath.Join(work, "input")) || !dirEmpty(t, s.vupscaleIncomingDir()) {
		t.Error("refused uploads were left behind")
	}
}

func TestCheckVUpscaleClip(t *testing.T) {
	info := func(w, h, frames int) videofile.Info { return videofile.Info{Width: w, Height: h, Frames: frames} }
	tests := []struct {
		name   string
		script string
		scale  int
		info   videofile.Info
		window int
		want   string
	}{
		{"fits", "relayvsr", 2, info(832, 480, 81), 16, ""},
		{"decoded", "relayvsr", 2, info(1280, 720, 120), 16, "fits at most 97 frames at 1280x720"},
		// At scale 4 or less the input limit always comes first; 5 reaches the output one.
		{"output", "relayvsr", 5, info(640, 360, 300), 2, "writes at most 4096 MiB"},
		{"relay window", "relayvsr", 4, info(1280, 720, 20), 32, "use window_size 24 or less"},
		{"seed window", "seedvr2", 2, info(1280, 720, 20), 9, "use window_size 5"},
		{"seed window fits", "seedvr2", 2, info(640, 360, 20), 1, ""},
	}
	for _, tt := range tests {
		e := checkVUpscaleClip(tt.script, tt.scale, tt.info, tt.window)
		switch {
		case tt.want == "" && e != nil:
			t.Errorf("%s: %v", tt.name, e)
		case tt.want != "" && (e == nil || !strings.Contains(e.msg, tt.want)):
			t.Errorf("%s: %v, want %q", tt.name, e, tt.want)
		}
	}
}

func TestVUpscaleJobFails(t *testing.T) {
	f := newFakeDGL(t)
	f.fail = "load_vae: cannot open file"
	s := vupscaleServer(t, f)
	id := createVUpscaleJobID(t, s, multipartReq(t, map[string]string{"model": "vup"}, fixture(t, "vp8-key-64x48-9f.webm")))
	j := waitVUpscaleJob(t, s, id)
	if j.Status != jobFailed || !strings.Contains(j.Error, "sd.cpp: load_vae: cannot open file") {
		t.Errorf("job = %+v", j)
	}
	if !dirEmpty(t, filepath.Join(generation.InstanceDir(s.cfg.DataDir, "vup"), "input")) {
		t.Error("the clip was left behind")
	}
}

func TestVUpscaleJobCancelStopsInstance(t *testing.T) {
	f := newFakeDGL(t)
	f.polls = 1 << 30
	s := vupscaleServer(t, f)
	id := createVUpscaleJobID(t, s, multipartReq(t, map[string]string{"model": "vup"}, fixture(t, "vp8-key-64x48-9f.webm")))
	testutil.Eventually(t, func() bool {
		f.mu.Lock()
		defer f.mu.Unlock()
		return len(f.sources) == 1
	}, "the job did not reach sd-server")
	rec := serve(s, httptest.NewRequest("DELETE", "/v1/videos/upscale/jobs/"+id, nil))
	if rec.Code != http.StatusNoContent {
		t.Fatalf("delete: HTTP %d %s", rec.Code, rec.Body)
	}
	j := waitVUpscaleJob(t, s, id)
	if j.Status != jobCancelled || !j.CancelRequested {
		t.Errorf("job = %+v", j)
	}
	testutil.Eventually(t, func() bool { return !s.genMgr.IsRunning("vup") }, "the instance was not stopped")
	if !dirEmpty(t, filepath.Join(generation.InstanceDir(s.cfg.DataDir, "vup"), "input")) {
		t.Error("the clip was left behind")
	}
	// The record stays, as for video jobs.
	if rec := serve(s, httptest.NewRequest("GET", "/v1/videos/upscale/jobs/"+id, nil)); rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), `"status":"cancelled"`) {
		t.Errorf("get after cancel: HTTP %d %s", rec.Code, rec.Body)
	}
}

// From master-1132 sd-server cancels the DGL job itself, so the instance
// keeps running.
func TestVUpscaleJobCancelKeepsInstance(t *testing.T) {
	f := newFakeDGL(t)
	f.polls, f.cancelable = 1<<30, true
	s := vupscaleServer(t, f)
	id := createVUpscaleJobID(t, s, multipartReq(t, map[string]string{"model": "vup"}, fixture(t, "vp8-key-64x48-9f.webm")))
	testutil.Eventually(t, func() bool {
		f.mu.Lock()
		defer f.mu.Unlock()
		return len(f.sources) == 1
	}, "the job did not reach sd-server")
	if rec := serve(s, httptest.NewRequest("DELETE", "/v1/videos/upscale/jobs/"+id, nil)); rec.Code != http.StatusNoContent {
		t.Fatalf("delete: HTTP %d %s", rec.Code, rec.Body)
	}
	if j := waitVUpscaleJob(t, s, id); j.Status != jobCancelled {
		t.Errorf("job = %+v", j)
	}
	f.mu.Lock()
	n := f.cancelled
	f.mu.Unlock()
	if n != 1 {
		t.Errorf("cancel posts = %d, want 1", n)
	}
	if !s.genMgr.IsRunning("vup") {
		t.Error("the instance was stopped although sd-server cancelled the job")
	}
}

func TestVUpscaleRestore(t *testing.T) {
	dir, out := t.TempDir(), t.TempDir()
	clip := filepath.Join(t.TempDir(), "job.webm")
	os.WriteFile(clip, []byte("x"), 0o644)
	var m jobStore[vupscaleJob]
	if err := m.open(dir, out, time.Now()); err != nil {
		t.Fatal(err)
	}
	m.add(&vupscaleJob{genJob: genJob{ID: "job", Status: jobRunning, Submitted: time.Now()}, input: clip})
	var m2 jobStore[vupscaleJob]
	if err := m2.open(dir, out, time.Now()); err != nil {
		t.Fatal(err)
	}
	j, ok := m2.view("job")
	if !ok || j.Status != jobFailed || j.Error != jobRestartedError {
		t.Errorf("restored = %+v", j)
	}
	if _, err := os.Stat(clip); err == nil {
		t.Error("the restored job's clip was not deleted")
	}
}

func TestVUpscaleV1Models(t *testing.T) {
	f := newFakeDGL(t)
	s := vupscaleServer(t, f)
	rec := serve(s, httptest.NewRequest("GET", "/v1/models?kind=vupscale", nil))
	var out struct {
		Data []struct {
			ID   string
			Meta map[string]any
		}
	}
	json.Unmarshal(rec.Body.Bytes(), &out)
	if len(out.Data) != 1 || out.Data[0].ID != "vup" {
		t.Fatalf("models = %s", rec.Body)
	}
	meta := out.Data[0].Meta
	v, _ := meta["vupscale"].(map[string]any)
	if meta["kind"] != "vupscale" || v["scale"] != float64(2) || v["script"] != "relayvsr" || v["default_window_size"] != float64(16) {
		t.Errorf("meta = %v", meta)
	}
	if api, _ := meta["api"].([]any); len(api) != 1 || api[0] != "video.upscale.jobs" {
		t.Errorf("api = %v", meta["api"])
	}
	if rec := serve(s, httptest.NewRequest("GET", "/v1/models?kind=upscale", nil)); strings.Contains(rec.Body.String(), `"vup"`) {
		t.Error("?kind=upscale lists the video upscaler")
	}
}
