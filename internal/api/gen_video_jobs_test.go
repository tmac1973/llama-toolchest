package api

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
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
)

// videoServer is an instanceServer with a running TI2V-like video instance
// "ti2v" whose sd-server is the fake f.
func videoServer(t *testing.T, f *fakeVidServer, caps ...string) *Server {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("shell-script fake")
	}
	fastVidPoll(t)
	s := instanceServer(t)
	if len(caps) == 0 {
		caps = []string{generation.CapText, generation.CapStartImage}
	}
	if err := s.gen.AddModelSet(generation.ModelSet{ID: "ti2v", Name: "Wan2.2 TI2V 5B", Kind: generation.KindVideo, Recipe: "wan22-ti2v-5b", Complete: true,
		Capabilities: caps, DefaultWidth: 832, DefaultHeight: 480, DefaultFrames: 33, DefaultFPS: 24, MaxFrames: 121, MaxWidth: 1280, MaxHeight: 704,
		Components: []generation.Component{{Slot: "diffusion_model", Path: "/m/ti2v.gguf"}}}); err != nil {
		t.Fatal(err)
	}
	u, _ := url.Parse(f.srv.URL)
	port, _ := strconv.Atoi(u.Port())
	// Measured at the model's limit, so only the model's limits apply.
	addInstance(t, s, generation.Instance{ID: "ti2v", ModelSetID: "ti2v", Port: port, Enabled: true, ReservedMiB: 16384,
		MeasuredSuggestedMiB: 16384, MeasuredFrames: 121, MeasuredWidth: 1280, MeasuredHeight: 704})
	bin := filepath.Join(t.TempDir(), "fake")
	os.WriteFile(bin, []byte("#!/bin/sh\nexec sleep 30\n"), 0o755)
	if err := s.genMgr.Start("ti2v", generation.KindVideo, process.SupervisorConfig{
		Name: "ti2v", BinaryPath: bin, Env: os.Environ(), HealthURL: f.srv.URL + "/v1/models", HealthInterval: 20 * time.Millisecond,
	}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.genMgr.Stop("ti2v") })
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := s.genMgr.WaitRunning(ctx, "ti2v"); err != nil {
		t.Fatal(err)
	}
	return s
}

func videoReq(target, body string) *http.Request {
	r := httptest.NewRequest("POST", target, strings.NewReader(body))
	r.Header.Set("Content-Type", "application/json")
	return r
}

// waitVideoJob polls a job until it has finished.
func waitVideoJob(t *testing.T, s *Server, id string) videoJob {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		if j, _ := s.genVideoJobs.view(id); j.Finished != nil {
			return j
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("the video job did not finish")
	return videoJob{}
}

func createVideoJobID(t *testing.T, s *Server, r *http.Request) string {
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

func TestVideoJobSucceeds(t *testing.T) {
	f := newFakeVidServer(t, 2, "completed")
	s := videoServer(t, f)
	var progressSeen bool
	s.genVideoJobs.onUpdate = func(j videoJob) {
		if j.Progress != nil && j.Phase == imagePhaseGenerating {
			progressSeen = true
		}
	}
	id := createVideoJobID(t, s, videoReq("/v1/videos/jobs", `{"prompt":"a calm sea"}`))
	j := waitVideoJob(t, s, id)
	if j.Status != jobSucceeded || j.Video == nil {
		t.Fatalf("job = %+v", j)
	}
	if j.Video.Frames != 33 || j.Video.FPS != 24 || j.Video.Width != 832 || j.Video.Height != 480 || j.Video.MimeType != "video/webm" || j.Video.Bytes != 8 {
		t.Errorf("video = %+v", *j.Video)
	}
	if !progressSeen {
		t.Error("no progress while generating")
	}
	if !strings.HasPrefix(j.path, s.videoDir()) {
		t.Errorf("saved at %s, outside %s", j.path, s.videoDir())
	}
	rec := serve(s, httptest.NewRequest("GET", j.Video.URL, nil))
	if rec.Code != http.StatusOK || rec.Body.String() != "WEBMDATA" || rec.Header().Get("Content-Type") != "video/webm" {
		t.Errorf("download: HTTP %d %q %s", rec.Code, rec.Body, rec.Header().Get("Content-Type"))
	}
	if f.body["width"] != float64(832) || f.body["video_frames"] != float64(33) || f.body["fps"] != float64(24) || f.body["seed"] != float64(-1) {
		t.Errorf("sd-server got %v, want the set's defaults and a random seed", f.body)
	}
}

func TestVideoJobMultipart(t *testing.T) {
	f := newFakeVidServer(t, 0, "completed")
	s := videoServer(t, f)
	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)
	mw.WriteField("prompt", "a calm sea")
	mw.WriteField("frames", "17")
	mw.WriteField("extra", `{"sample_params":{"sample_steps":8}}`)
	part, _ := mw.CreateFormFile("start_image", "photo.png")
	part.Write([]byte("PNGDATA"))
	mw.Close()
	r := httptest.NewRequest("POST", "/v1/videos/jobs", &buf)
	r.Header.Set("Content-Type", mw.FormDataContentType())
	j := waitVideoJob(t, s, createVideoJobID(t, s, r))
	if j.Status != jobSucceeded || !j.Request.StartImage {
		t.Fatalf("job = %+v", j)
	}
	if f.body["init_image"] != base64.StdEncoding.EncodeToString([]byte("PNGDATA")) || f.body["video_frames"] != float64(17) || f.body["sample_params"] == nil {
		t.Errorf("sd-server got %v", f.body)
	}
}

func TestVideoJobChecks(t *testing.T) {
	f := newFakeVidServer(t, 0, "completed")
	img := base64.StdEncoding.EncodeToString([]byte("PNG"))
	for name, tc := range map[string]struct {
		body string
		caps []string
		code string
	}{
		"end without start":    {`{"prompt":"p","end_image":"` + img + `"}`, []string{generation.CapStartImage, generation.CapEndImage, generation.CapText}, "start_image_required"},
		"start not supported":  {`{"prompt":"p","start_image":"` + img + `"}`, []string{generation.CapText}, "input_not_supported"},
		"end not supported":    {`{"prompt":"p","start_image":"` + img + `","end_image":"` + img + `"}`, nil, "input_not_supported"},
		"no image, needs one":  {`{"prompt":"p"}`, []string{generation.CapStartImage}, "start_image_required"},
		"too many frames":      {`{"prompt":"p","frames":122}`, nil, "over_limit"},
		"too large":            {`{"prompt":"p","size":"1920x1080"}`, nil, "over_limit"},
		"not a multiple of 8":  {`{"prompt":"p","size":"836x472"}`, nil, "bad_size"},
		"zero frames":          {`{"prompt":"p","frames":0}`, nil, "bad_value"},
		"fps too high":         {`{"prompt":"p","fps":300}`, nil, "bad_value"},
		"too narrow":           {`{"prompt":"p","size":"32x480"}`, nil, "bad_value"},
		"frames and seconds":   {`{"prompt":"p","frames":33,"seconds":2}`, nil, "bad_value"},
		"extra sets own field": {`{"prompt":"p","extra":{"video_frames":9}}`, nil, "bad_value"},
		"avi":                  {`{"prompt":"p","output_format":"avi"}`, nil, "bad_value"},
		"no prompt":            {`{"prompt":"  "}`, nil, "bad_request"},
		"size not WxH":         {`{"prompt":"p","size":"large"}`, nil, "bad_value"},
	} {
		t.Run(name, func(t *testing.T) {
			s := videoServer(t, f, tc.caps...)
			rec := serve(s, videoReq("/v1/videos/jobs", tc.body))
			if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), `"`+tc.code+`"`) {
				t.Errorf("HTTP %d %s, want 400 %s", rec.Code, rec.Body, tc.code)
			}
			if h := s.genQueue.History(0); len(h) != 0 {
				t.Errorf("a refused request was queued: %+v", h)
			}
		})
	}
}

func TestVideoJobLimitsAndConversions(t *testing.T) {
	f := newFakeVidServer(t, 0, "completed")
	s := videoServer(t, f)
	j := waitVideoJob(t, s, createVideoJobID(t, s, videoReq("/v1/videos/jobs", `{"prompt":"p","seconds":2}`)))
	if j.Request.Frames != 48 || f.body["video_frames"] != float64(48) {
		t.Errorf("2 s at 24 fps = %d frames (sent %v), want 48", j.Request.Frames, f.body["video_frames"])
	}
	j = waitVideoJob(t, s, createVideoJobID(t, s, videoReq("/v1/videos/jobs", `{"prompt":"p","size":"704x1280","frames":121}`)))
	if j.Status != jobSucceeded {
		t.Errorf("portrait at the ceiling: %+v", j)
	}
}

// Deleting a running job stops it in sd-server; the job shows cancelled
// once sd-server has stopped, and keeps no file.
func TestVideoJobCancelRunning(t *testing.T) {
	f := newFakeVidServer(t, 0, "hang")
	s := videoServer(t, f)
	id := createVideoJobID(t, s, videoReq("/v1/videos/jobs", `{"prompt":"p"}`))
	deadline := time.Now().Add(5 * time.Second)
	for {
		if j, _ := s.genVideoJobs.view(id); j.Progress != nil || time.Now().After(deadline) {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	rec := serve(s, httptest.NewRequest("DELETE", "/v1/videos/jobs/"+id, nil))
	if rec.Code != http.StatusNoContent {
		t.Fatalf("DELETE: HTTP %d", rec.Code)
	}
	j := waitVideoJob(t, s, id)
	if j.Status != jobCancelled || j.Video != nil || j.path != "" {
		t.Errorf("job = %+v", j)
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if !f.cancelled {
		t.Error("sd-server was not told to cancel")
	}
}

func TestVideoGenerateWaits(t *testing.T) {
	f := newFakeVidServer(t, 1, "completed")
	s := videoServer(t, f)
	rec := serve(s, videoReq("/v1/videos/generations", `{"prompt":"p"}`))
	if rec.Code != http.StatusOK {
		t.Fatalf("HTTP %d %s", rec.Code, rec.Body)
	}
	var got videoJob
	json.Unmarshal(rec.Body.Bytes(), &got)
	if got.Status != jobSucceeded || got.Video == nil {
		t.Fatalf("answer = %s", rec.Body)
	}
	again := serve(s, httptest.NewRequest("GET", "/v1/videos/jobs/"+got.ID, nil))
	if strings.TrimSpace(again.Body.String()) != strings.TrimSpace(rec.Body.String()) {
		t.Errorf("waiting answer differs from GET:\n%s\n%s", rec.Body, again.Body)
	}
}

// A client that leaves the waiting route cancels its job in sd-server.
func TestVideoGenerateClientLeaves(t *testing.T) {
	f := newFakeVidServer(t, 0, "hang")
	s := videoServer(t, f)
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		serve(s, videoReq("/v1/videos/generations", `{"prompt":"p"}`).WithContext(ctx))
		close(done)
	}()
	time.Sleep(100 * time.Millisecond)
	cancel()
	<-done
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		f.mu.Lock()
		c := f.cancelled
		f.mu.Unlock()
		if c {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Error("leaving the waiting route did not cancel the job in sd-server")
}

// A DELETE from elsewhere while the waiting route waits makes it answer
// 409 with the cancelled job.
func TestVideoGenerateCancelledElsewhere(t *testing.T) {
	f := newFakeVidServer(t, 0, "hang")
	s := videoServer(t, f)
	go func() {
		deadline := time.Now().Add(5 * time.Second)
		for time.Now().Before(deadline) {
			s.genVideoJobs.mu.Lock()
			ids := append([]string(nil), s.genVideoJobs.order...)
			s.genVideoJobs.mu.Unlock()
			if len(ids) > 0 {
				s.cancelVideoJob(ids[0])
				return
			}
			time.Sleep(10 * time.Millisecond)
		}
	}()
	rec := serve(s, videoReq("/v1/videos/generations", `{"prompt":"p"}`))
	if rec.Code != http.StatusConflict || !strings.Contains(rec.Body.String(), `"cancelled"`) {
		t.Errorf("HTTP %d %s, want 409 with the cancelled job", rec.Code, rec.Body)
	}
}

func TestVideoJobsPrune(t *testing.T) {
	s := instanceServer(t)
	s.genVideoJobs.max = videoJobMax
	dir := t.TempDir()
	old := time.Now().Add(-time.Hour)
	for n := 0; n < videoJobMax+1; n++ {
		p := filepath.Join(dir, fmt.Sprintf("%d.webm", n))
		os.WriteFile(p, []byte("v"), 0o644)
		fin := old.Add(time.Duration(n) * time.Second)
		s.genVideoJobs.add(&videoJob{ID: fmt.Sprint(n), Status: jobSucceeded, Finished: &fin, path: p})
	}
	s.genVideoJobs.prune(time.Now())
	if _, ok := s.genVideoJobs.view("0"); ok {
		t.Error("the oldest job was kept")
	}
	if _, err := os.Stat(filepath.Join(dir, "0.webm")); !os.IsNotExist(err) {
		t.Error("the oldest job's file was kept")
	}
	if _, ok := s.genVideoJobs.view(fmt.Sprint(videoJobMax)); !ok {
		t.Error("the newest job was dropped")
	}
}

// An image instance is never picked for a video job.
func TestPickInstanceVideoOnly(t *testing.T) {
	s := instanceServer(t)
	runFakeInstance(t, s, "ming", "ming-image", 0, newFakeBackend(t, 0))
	if inst, pe := s.pickInstance("", generation.KindVideo); pe == nil {
		t.Errorf("picked image instance %s for a video job", inst.ID)
	}
	rec := serve(s, videoReq("/v1/videos/jobs", `{"prompt":"p","model":"ming"}`))
	if rec.Code < 400 {
		t.Errorf("a video job went to an image instance: HTTP %d", rec.Code)
	}
	io.Copy(io.Discard, rec.Body)
}

// sd-server rounds a size up to the set's multiple without saying so: the
// job reports the size made, and meta.video gives the multiple. LTX-2.3
// made 1280x720 as 1280x736 on asmodean.
func TestVideoJobReportsMadeSize(t *testing.T) {
	f := newFakeVidServer(t, 0, "completed")
	s := videoServer(t, f)
	set, _ := s.gen.ModelSet("ti2v")
	set.CatalogID = "ltx23-distilled"
	if err := s.gen.UpdateModelSet(set); err != nil {
		t.Fatal(err)
	}
	id := createVideoJobID(t, s, videoReq("/v1/videos/jobs", `{"prompt":"a calm sea","size":"1000x600","frames":33}`))
	j := waitVideoJob(t, s, id)
	if j.Status != jobSucceeded || j.Video == nil {
		t.Fatalf("job = %+v", j)
	}
	if f.body["width"] != float64(1000) || f.body["height"] != float64(600) {
		t.Errorf("sd-server got %vx%v, want the size asked for", f.body["width"], f.body["height"])
	}
	if j.Video.Width != 1024 || j.Video.Height != 608 {
		t.Errorf("video size %dx%d, want 1024x608", j.Video.Width, j.Video.Height)
	}
	rec := serve(s, httptest.NewRequest("GET", "/v1/models/ti2v", nil))
	var got struct {
		Meta struct {
			Video map[string]any `json:"video"`
		} `json:"meta"`
	}
	json.Unmarshal(rec.Body.Bytes(), &got)
	if got.Meta.Video["size_multiple"] != float64(32) {
		t.Errorf("meta.video.size_multiple = %v, want 32", got.Meta.Video["size_multiple"])
	}
}

func TestV1ModelsVideoMeta(t *testing.T) {
	f := newFakeVidServer(t, 0, "completed")
	s := videoServer(t, f)
	rec := serve(s, httptest.NewRequest("GET", "/v1/models/ti2v", nil))
	var got struct {
		Meta struct {
			Video map[string]any `json:"video"`
		} `json:"meta"`
	}
	json.Unmarshal(rec.Body.Bytes(), &got)
	v := got.Meta.Video
	if v["default_size"] != "832x480" || v["default_frames"] != float64(33) || v["max_frames"] != float64(121) || v["max_size"] != "1280x704" {
		t.Errorf("meta.video = %v", v)
	}
	if _, ok := v["size_multiple"]; ok {
		t.Errorf("a custom set has no known size multiple, got %v", v["size_multiple"])
	}
}

// A set with LoRAs sends them with every request, each by its file name,
// and a request may not name its own; a set without LoRAs sends none and
// passes extra.lora through.
func TestBuildVideoRequestLoras(t *testing.T) {
	inst := generation.Instance{ID: "fast", MeasuredSuggestedMiB: 1, MeasuredFrames: 81, MeasuredWidth: 1280, MeasuredHeight: 720}
	set := generation.ModelSet{ID: "fast", Name: "Fast", Kind: generation.KindVideo, Recipe: "wan22-i2v",
		Capabilities: []string{generation.CapStartImage}, DefaultWidth: 832, DefaultHeight: 480, DefaultFrames: 33, DefaultFPS: 16,
		Components: []generation.Component{
			{Slot: "diffusion_model", Path: "/m/low.gguf"},
			{Slot: "high_noise_lora", Path: "/m/loras/HIGH.safetensors"},
			{Slot: "lora", Path: "/m/loras/LOW.safetensors"},
		}}
	in := videoInput{Prompt: "p", StartImage: "UE5H"}
	_, vid, verr := buildVideoRequest(in, inst, set)
	if verr != nil {
		t.Fatal(verr)
	}
	body, _ := vid.body()
	var got struct {
		Lora []map[string]any `json:"lora"`
	}
	json.Unmarshal(body, &got)
	want := []map[string]any{
		{"path": "HIGH.safetensors", "multiplier": float64(1), "is_high_noise": true},
		{"path": "LOW.safetensors", "multiplier": float64(1), "is_high_noise": false},
	}
	if fmt.Sprint(got.Lora) != fmt.Sprint(want) {
		t.Errorf("lora = %v, want %v", got.Lora, want)
	}

	in.Extra = map[string]any{"lora": []any{}}
	if _, _, verr := buildVideoRequest(in, inst, set); verr == nil || verr.code != "bad_value" {
		t.Errorf("extra.lora on a set with LoRAs: %+v, want bad_value", verr)
	}

	set.Components = set.Components[:1]
	_, vid, verr = buildVideoRequest(in, inst, set)
	if verr != nil {
		t.Fatal(verr)
	}
	body, _ = vid.body()
	var plain map[string]any
	json.Unmarshal(body, &plain)
	if l, ok := plain["lora"].([]any); !ok || len(l) != 0 {
		t.Errorf("a set without LoRAs: lora = %v, want the request's own empty list", plain["lora"])
	}
}
