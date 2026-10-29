package api

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"path/filepath"
	"syscall"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/generation"
)

// vidGenPoll is how often a video job's status is read from sd-server.
var vidGenPoll = time.Second

// vidCancelWait caps how long a cancelled video job may take to stop in
// sd-server before the client gives up waiting.
const vidCancelWait = time.Minute

// vidGenRequest is the part of sd-server's POST /sdcpp/v1/vid_gen body
// toolchest sets. CFG, flow shift and sampler are left out: the set's
// launch settings give them, and sd-server uses its startup defaults for
// anything a request does not name. Steps are left out too, except by
// Measure.
type vidGenRequest struct {
	Prompt         string `json:"prompt"`
	NegativePrompt string `json:"negative_prompt,omitempty"`
	Width          int    `json:"width"`
	Height         int    `json:"height"`
	Seed           int64  `json:"seed"`
	Frames         int    `json:"video_frames"`
	FPS            int    `json:"fps"`
	// InitImage and EndImage are base64 images, or empty.
	InitImage    string `json:"init_image,omitempty"`
	EndImage     string `json:"end_image,omitempty"`
	OutputFormat string `json:"output_format"`
	// Lora is the set's LoRA files, by name within --lora-model-dir.
	Lora []vidLora `json:"lora,omitempty"`
	// Steps and HighNoiseSteps, when above zero, replace the set's
	// sampling steps for this video (sample_params.sample_steps and
	// high_noise_sample_params.sample_steps, sd.cpp master-1015).
	Steps          int `json:"-"`
	HighNoiseSteps int `json:"-"`
	// Extra is merged into the body last, at the top level.
	Extra map[string]any `json:"-"`
}

// vidLora is one LoRA in a request to sd-server. IsHighNoise puts it on
// the high-noise expert only; otherwise it goes on the low-noise one.
type vidLora struct {
	Path        string  `json:"path"`
	Multiplier  float64 `json:"multiplier"`
	IsHighNoise bool    `json:"is_high_noise"`
}

// setLoras returns the LoRAs a set's requests apply: each LoRA file by its
// name, since sd-server is given the folder they are in, at full strength.
func setLoras(set generation.ModelSet) []vidLora {
	var out []vidLora
	for _, c := range set.Loras() {
		out = append(out, vidLora{Path: filepath.Base(c.Path), Multiplier: 1, IsHighNoise: generation.LoraSlots[c.Slot]})
	}
	return out
}

// body is the JSON sent to sd-server. It always asks for the cheapest
// preview ("proj") at every step: that is what makes sd-server report
// which step a job is on.
func (r vidGenRequest) body() ([]byte, error) {
	raw, err := json.Marshal(r)
	if err != nil {
		return nil, err
	}
	m := map[string]any{}
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil, err
	}
	m["preview"], m["preview_interval"] = "proj", 1
	for k, v := range r.Extra {
		m[k] = v
	}
	setSampleSteps(m, "sample_params", r.Steps)
	setSampleSteps(m, "high_noise_sample_params", r.HighNoiseSteps)
	return json.Marshal(m)
}

// setSampleSteps sets m[key].sample_steps when steps is above zero,
// keeping any other fields an extra settings object already put there.
func setSampleSteps(m map[string]any, key string, steps int) {
	if steps <= 0 {
		return
	}
	params, _ := m[key].(map[string]any)
	if params == nil {
		params = map[string]any{}
	}
	params["sample_steps"] = steps
	m[key] = params
}

// vidProgress is where a generating video job is: the sampling pass (a
// Wan2.2 A14B pair has two) and the step within it.
type vidProgress struct {
	Pass       int `json:"pass"`
	Step       int `json:"step"`
	TotalSteps int `json:"total_steps"`
}

// vidResult is a finished video.
type vidResult struct {
	Data     []byte
	MimeType string
	Format   string
	FPS      int
	Frames   int
}

// vidJobStatus is what toolchest reads from GET /sdcpp/v1/jobs/{id}. The
// preview picture (preview.b64_json) is left out, so toolchest never holds
// one.
type vidJobStatus struct {
	Status  string       `json:"status"`
	Error   vidJobError  `json:"error"`
	Preview *vidProgress `json:"preview"`
	Result  *struct {
		B64JSON      string `json:"b64_json"`
		MimeType     string `json:"mime_type"`
		OutputFormat string `json:"output_format"`
		FPS          int    `json:"fps"`
		FrameCount   int    `json:"frame_count"`
	} `json:"result"`
}

// vidJobError is a job's error. sd-server master-1015 sends an object,
// {"code": "generation_failed", "message": "…"}, or null; older builds
// sent a plain string. Both are read as text.
type vidJobError string

func (e *vidJobError) UnmarshalJSON(b []byte) error {
	var text string
	if json.Unmarshal(b, &text) == nil {
		*e = vidJobError(text)
		return nil
	}
	var obj struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	}
	if err := json.Unmarshal(b, &obj); err != nil {
		return err
	}
	*e = vidJobError(firstNonEmpty(obj.Message, obj.Code))
	return nil
}

// runVidGen makes one video through sd-server's native async API at base
// (http://127.0.0.1:port): it submits the job, polls it, and returns the
// decoded file. onProgress, when set, hears each new step. When ctx ends
// first, the job is cancelled in sd-server and runVidGen waits (up to
// vidCancelWait) for it to stop, so the GPU is free when it returns
// ctx's error.
func runVidGen(ctx context.Context, base string, req vidGenRequest, onProgress func(vidProgress)) (vidResult, error) {
	body, err := req.body()
	if err != nil {
		return vidResult{}, err
	}
	out, _, err := postRaw(ctx, base+"/sdcpp/v1/vid_gen", body)
	if err != nil {
		return vidResult{}, err
	}
	var job struct {
		ID string `json:"id"`
	}
	if err := json.Unmarshal(out, &job); err != nil || job.ID == "" {
		return vidResult{}, errors.New("sd-server returned no video job ID")
	}
	jobURL := base + "/sdcpp/v1/jobs/" + job.ID
	var last vidProgress
	for {
		select {
		case <-ctx.Done():
			cancelVidGen(context.WithoutCancel(ctx), jobURL)
			return vidResult{}, ctx.Err()
		case <-time.After(vidGenPoll):
		}
		st, err := getVidJob(ctx, jobURL)
		if err != nil {
			if ctx.Err() != nil {
				continue // cancelled while polling; handled above
			}
			return vidResult{}, err
		}
		switch st.Status {
		case "completed":
			return decodeVidResult(st)
		case "failed":
			if st.Error == "" {
				st.Error = "no reason given"
			}
			return vidResult{}, fmt.Errorf("sd-server video job failed: %s", st.Error)
		case "cancelled":
			return vidResult{}, errors.New("sd-server cancelled the video job")
		}
		if st.Preview != nil && *st.Preview != last && onProgress != nil {
			last = *st.Preview
			onProgress(last)
		}
	}
}

// cancelVidGen asks sd-server to stop a video job, then polls until the
// job has stopped or vidCancelWait has passed. A generating job stops at
// sd-server's next cancellation checkpoint.
func cancelVidGen(ctx context.Context, jobURL string) {
	ctx, cancel := context.WithTimeout(ctx, vidCancelWait)
	defer cancel()
	if _, _, err := postRaw(ctx, jobURL+"/cancel", nil); err != nil {
		return
	}
	for {
		st, err := getVidJob(ctx, jobURL)
		if err != nil || (st.Status != "queued" && st.Status != "generating") {
			return
		}
		select {
		case <-ctx.Done():
			return
		case <-time.After(vidGenPoll):
		}
	}
}

func getVidJob(ctx context.Context, url string) (vidJobStatus, error) {
	out, _, err := getBody(ctx, url)
	if err != nil {
		return vidJobStatus{}, err
	}
	var st vidJobStatus
	if err := json.Unmarshal(out, &st); err != nil {
		return vidJobStatus{}, fmt.Errorf("reading the video job status: %w", err)
	}
	return st, nil
}

func decodeVidResult(st vidJobStatus) (vidResult, error) {
	if st.Result == nil || st.Result.B64JSON == "" {
		return vidResult{}, errors.New("sd-server finished the video job without a video")
	}
	data, err := base64.StdEncoding.DecodeString(st.Result.B64JSON)
	if err != nil {
		return vidResult{}, fmt.Errorf("decoding the video: %w", err)
	}
	return vidResult{Data: data, MimeType: st.Result.MimeType, Format: st.Result.OutputFormat,
		FPS: st.Result.FPS, Frames: st.Result.FrameCount}, nil
}

// postRaw POSTs a JSON body (nil for none) through genHTTPClient.
func postRaw(ctx context.Context, url string, body []byte) ([]byte, string, error) {
	var b any
	if body != nil {
		b = json.RawMessage(body)
	}
	return postJSON(ctx, url, b)
}

// measurePicture draws the start (or end) picture Measure gives a video
// set that takes one: a sky-to-sea gradient with a sun at one third of the
// width, or at two thirds for the end picture, so a start-and-end video
// has something to move between. Nothing ships with the program.
func measurePicture(w, h int, end bool) []byte {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	horizon := h * 3 / 5
	for y := 0; y < h; y++ {
		var c color.RGBA
		if y < horizon {
			t := y * 255 / max(horizon, 1)
			c = color.RGBA{uint8(250 - t/4), uint8(170 + t/4), uint8(120 + t/2), 255}
		} else {
			t := (y - horizon) * 255 / max(h-horizon, 1)
			c = color.RGBA{uint8(40 - t/8), uint8(80 - t/6), uint8(140 - t/4), 255}
		}
		for x := 0; x < w; x++ {
			img.SetRGBA(x, y, c)
		}
	}
	cx, cy, r := w/3, horizon-h/8, max(h/10, 2)
	if end {
		cx = w * 2 / 3
	}
	sun := color.RGBA{255, 230, 150, 255}
	for y := cy - r; y <= cy+r; y++ {
		for x := cx - r; x <= cx+r; x++ {
			if (x-cx)*(x-cx)+(y-cy)*(y-cy) <= r*r && image.Pt(x, y).In(img.Rect) {
				img.SetRGBA(x, y, sun)
			}
		}
	}
	var buf bytes.Buffer
	png.Encode(&buf, img)
	return buf.Bytes()
}

// backendGone reports whether err means the backend's port no longer
// answers: the process died mid-job. The supervisor may already have
// restarted it by the time the error is seen, so the process state alone
// does not tell.
func backendGone(err error) bool {
	return errors.Is(err, syscall.ECONNREFUSED)
}
