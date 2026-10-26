package api

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"image/png"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/generation"
)

// fakeVidServer is a stand-in for sd-server's native video API. A job
// reports generating for steps polls (step 1..steps of total 20), then
// ends with end: "completed", "failed" or "hang" (stays generating until
// cancelled).
type fakeVidServer struct {
	srv       *httptest.Server
	mu        sync.Mutex
	steps     int
	end       string
	submitErr int
	polls     int
	body      map[string]any
	cancelled bool
}

func newFakeVidServer(t *testing.T, steps int, end string) *fakeVidServer {
	t.Helper()
	f := &fakeVidServer{steps: steps, end: end}
	f.srv = httptest.NewServer(http.HandlerFunc(f.serve))
	t.Cleanup(f.srv.Close)
	return f
}

func (f *fakeVidServer) serve(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	defer f.mu.Unlock()
	switch {
	case r.URL.Path == "/v1/models":
		w.Write([]byte(`{"data":[]}`))
	case r.Method == "POST" && r.URL.Path == "/sdcpp/v1/vid_gen":
		if f.submitErr != 0 {
			w.WriteHeader(f.submitErr)
			w.Write([]byte(`{"error":"video_frames must be at least 1"}`))
			return
		}
		json.NewDecoder(r.Body).Decode(&f.body)
		w.WriteHeader(http.StatusAccepted)
		w.Write([]byte(`{"id":"job_1","kind":"vid_gen","status":"queued"}`))
	case r.Method == "POST" && r.URL.Path == "/sdcpp/v1/jobs/job_1/cancel":
		f.cancelled = true
		w.Write([]byte(`{"id":"job_1","status":"generating","cancel_requested":true}`))
	case r.Method == "GET" && r.URL.Path == "/sdcpp/v1/jobs/job_1":
		f.polls++
		switch {
		case f.cancelled:
			w.Write([]byte(`{"status":"cancelled"}`))
		case f.polls <= f.steps || f.end == "hang":
			step := min(f.polls, 20)
			json.NewEncoder(w).Encode(map[string]any{"status": "generating",
				"preview": map[string]any{"pass": 1, "step": step, "total_steps": 20, "b64_json": "UFJFVklFVw=="}})
		case f.end == "failed":
			w.Write([]byte(`{"status":"failed","error":"out of memory"}`))
		default:
			json.NewEncoder(w).Encode(map[string]any{"status": "completed", "result": map[string]any{
				"b64_json": base64.StdEncoding.EncodeToString([]byte("WEBMDATA")), "mime_type": "video/webm",
				"output_format": "webm", "fps": 24, "frame_count": 33}})
		}
	default:
		http.NotFound(w, r)
	}
}

func fastVidPoll(t *testing.T) {
	t.Helper()
	old := vidGenPoll
	vidGenPoll = 5 * time.Millisecond
	t.Cleanup(func() { vidGenPoll = old })
}

func TestRunVidGenCompletes(t *testing.T) {
	fastVidPoll(t)
	f := newFakeVidServer(t, 2, "completed")
	var seen []vidProgress
	res, err := runVidGen(context.Background(), f.srv.URL, vidGenRequest{Prompt: "p", Width: 832, Height: 480, Frames: 33, FPS: 24, OutputFormat: "webm",
		Extra: map[string]any{"sample_params": map[string]any{"sample_steps": 12}}}, func(p vidProgress) { seen = append(seen, p) })
	if err != nil {
		t.Fatal(err)
	}
	if string(res.Data) != "WEBMDATA" || res.MimeType != "video/webm" || res.Frames != 33 || res.FPS != 24 || res.Format != "webm" {
		t.Errorf("result = %+v", res)
	}
	if len(seen) != 2 || seen[1].Step != 2 || seen[1].TotalSteps != 20 {
		t.Errorf("progress = %+v, want steps 1 and 2 of 20", seen)
	}
	if f.body["preview"] != "proj" || f.body["video_frames"] != float64(33) || f.body["sample_params"] == nil {
		t.Errorf("body = %v, want proj previews, 33 frames and the extra field", f.body)
	}
	if _, ok := f.body["init_image"]; ok {
		t.Error("an empty start image was sent")
	}
}

func TestRunVidGenFails(t *testing.T) {
	fastVidPoll(t)
	f := newFakeVidServer(t, 1, "failed")
	_, err := runVidGen(context.Background(), f.srv.URL, vidGenRequest{Prompt: "p"}, nil)
	if err == nil || !strings.Contains(err.Error(), "out of memory") {
		t.Errorf("err = %v, want sd-server's reason", err)
	}
}

func TestRunVidGenRefusedSubmit(t *testing.T) {
	f := newFakeVidServer(t, 0, "completed")
	f.submitErr = http.StatusBadRequest
	_, err := runVidGen(context.Background(), f.srv.URL, vidGenRequest{Prompt: "p"}, nil)
	if err == nil || !strings.Contains(err.Error(), "video_frames must be at least 1") {
		t.Errorf("err = %v, want sd-server's error text", err)
	}
}

// Ending the context cancels the job in sd-server, and runVidGen returns
// only once sd-server reports it stopped.
func TestRunVidGenCancelsOnContextEnd(t *testing.T) {
	fastVidPoll(t)
	f := newFakeVidServer(t, 0, "hang")
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		_, err := runVidGen(ctx, f.srv.URL, vidGenRequest{Prompt: "p"}, func(vidProgress) { cancel() })
		done <- err
	}()
	select {
	case err := <-done:
		if err != context.Canceled {
			t.Errorf("err = %v, want context.Canceled", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("runVidGen did not return after cancelling")
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if !f.cancelled {
		t.Error("the job was not cancelled in sd-server")
	}
}

// Measure makes the heaviest video the set supports: start and end
// pictures when it takes them, none when it only takes a prompt.
func TestMeasureVideoUsesHeaviestMode(t *testing.T) {
	fastVidPoll(t)
	for _, tc := range []struct {
		caps               []string
		wantStart, wantEnd bool
	}{
		{[]string{generation.CapStartImage, generation.CapEndImage}, true, true},
		{[]string{generation.CapText, generation.CapStartImage}, true, false},
		{[]string{generation.CapText}, false, false},
	} {
		f := newFakeVidServer(t, 0, "completed")
		set := generation.ModelSet{Kind: generation.KindVideo, Capabilities: tc.caps, DefaultWidth: 832, DefaultHeight: 480, DefaultFrames: 33, DefaultFPS: 16}
		if err := runMeasureVideo(context.Background(), f.srv.URL, set, generation.VideoSize{}); err != nil {
			t.Fatalf("%v: %v", tc.caps, err)
		}
		_, hasStart := f.body["init_image"]
		_, hasEnd := f.body["end_image"]
		if hasStart != tc.wantStart || hasEnd != tc.wantEnd {
			t.Errorf("%v: start %v end %v, want %v %v", tc.caps, hasStart, hasEnd, tc.wantStart, tc.wantEnd)
		}
		if f.body["width"] != float64(832) || f.body["video_frames"] != float64(33) || f.body["fps"] != float64(16) {
			t.Errorf("%v: body %v, want the set's test video", tc.caps, f.body)
		}
	}
	if err := runMeasureVideo(context.Background(), "http://127.0.0.1:1", generation.ModelSet{Kind: generation.KindVideo}, generation.VideoSize{}); err == nil {
		t.Error("a set with no way to make a video was measured")
	}
}

func TestMeasureTimeout(t *testing.T) {
	if measureTimeout(generation.KindVideo) != 2*time.Hour || measureTimeout(generation.KindSD) != 10*time.Minute || measureTimeout(generation.KindAce) != 10*time.Minute {
		t.Error("want 2 hours for video, 10 minutes for image and music")
	}
}

func TestMeasurePicture(t *testing.T) {
	start, end := measurePicture(832, 480, false), measurePicture(832, 480, true)
	img, err := png.Decode(bytes.NewReader(start))
	if err != nil {
		t.Fatal(err)
	}
	if b := img.Bounds(); b.Dx() != 832 || b.Dy() != 480 {
		t.Errorf("size %v", b)
	}
	if bytes.Equal(start, end) {
		t.Error("the start and end pictures are the same")
	}
}
