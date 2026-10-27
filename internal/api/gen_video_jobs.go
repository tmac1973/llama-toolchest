package api

// Video jobs: /v1/videos/jobs takes a video request, checks it against the
// set's abilities and ceiling, queues it on the instance's GPU like image
// and music jobs, and answers at once with a job ID. Clients poll the job
// for its status, phase and progress, then download the video.
// /v1/videos/generations takes the same request and answers once the job
// has finished. Unlike image jobs, a running video job can be cancelled in
// sd-server itself.

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"mime"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/tmac1973/llama-toolchest/internal/generation"
)

// videoJobMax is how many finished video jobs, and their files, are kept:
// videos are large, so fewer than images and music.
const videoJobMax = 50

// videoMaxFPS is sd-server's highest frame rate.
const videoMaxFPS = 240

// videoReservedExtra are sd-server request fields a video request's
// extra object may not set: they have fields of their own, or toolchest
// sets them itself.
var videoReservedExtra = []string{"prompt", "negative_prompt", "width", "height", "seed", "video_frames", "fps",
	"init_image", "end_image", "output_format", "preview", "preview_interval"}

// videoJobRequest is a video request as the job shows it back: without the
// images, only whether they were given.
type videoJobRequest struct {
	Model          string         `json:"model,omitempty"`
	Prompt         string         `json:"prompt"`
	NegativePrompt string         `json:"negative_prompt,omitempty"`
	Size           string         `json:"size"`
	Frames         int            `json:"frames"`
	FPS            int            `json:"fps"`
	Seed           int64          `json:"seed"`
	OutputFormat   string         `json:"output_format"`
	StartImage     bool           `json:"start_image,omitempty"`
	EndImage       bool           `json:"end_image,omitempty"`
	Extra          map[string]any `json:"extra,omitempty"`
}

// videoJobVideo is a finished job's video.
type videoJobVideo struct {
	URL      string `json:"url"`
	MimeType string `json:"mime_type"`
	Format   string `json:"format"`
	Frames   int    `json:"frames"`
	FPS      int    `json:"fps"`
	Width    int    `json:"width"`
	Height   int    `json:"height"`
	Bytes    int64  `json:"bytes"`
}

// videoJob is one video request.
type videoJob struct {
	genJob
	Progress        *vidProgress    `json:"progress,omitempty"`
	CancelRequested bool            `json:"cancel_requested,omitempty"`
	Request         videoJobRequest `json:"request"`
	Video           *videoJobVideo  `json:"video,omitempty"`

	// Not shown to clients.
	vid  vidGenRequest // dropped once sent
	path string
	// done is closed when the job has finished, for the waiting route.
	done chan struct{}
}

func (j videoJob) state() jobState {
	var files []string
	if j.path != "" {
		files = []string{j.path}
	}
	return j.jobState(files)
}

// videoJobPrivate is the part of a video job its record keeps that clients
// do not see.
type videoJobPrivate struct {
	Path string `json:"path,omitempty"`
}

func (j videoJob) private() any { return videoJobPrivate{Path: j.path} }

func (j videoJob) restored(raw json.RawMessage, now time.Time) (videoJob, error) {
	var p videoJobPrivate
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &p); err != nil {
			return j, err
		}
	}
	j.path, j.Progress = p.Path, nil
	j.restore(now)
	j.done = make(chan struct{})
	close(j.done)
	return j, nil
}

func (j videoJob) withPosition(n int) videoJob {
	j.Position = n
	return j
}

func (s *Server) videoDir() string {
	return filepath.Join(s.cfg.DataDir, "gen-output", "videos")
}

// videoInput is a parsed video request before the set's defaults are
// applied. Pointers tell a field left out from one set to zero.
type videoInput struct {
	Model          string         `json:"model"`
	Prompt         string         `json:"prompt"`
	NegativePrompt string         `json:"negative_prompt"`
	Size           string         `json:"size"`
	Frames         *int           `json:"frames"`
	Seconds        *float64       `json:"seconds"`
	FPS            *int           `json:"fps"`
	Seed           *int64         `json:"seed"`
	StartImage     string         `json:"start_image"`
	EndImage       string         `json:"end_image"`
	OutputFormat   string         `json:"output_format"`
	Extra          map[string]any `json:"extra"`
}

// readVideoInput reads a JSON body or a multipart form with the same field
// names; in a form, start_image and end_image are files and extra is JSON
// text.
func readVideoInput(r *http.Request) (videoInput, *requestError) {
	var in videoInput
	r.Body = http.MaxBytesReader(nil, r.Body, maxImageRequestBytes)
	tooLarge := func(err error) bool {
		var mbe *http.MaxBytesError
		return errors.As(err, &mbe)
	}
	if mt, _, _ := mime.ParseMediaType(r.Header.Get("Content-Type")); mt == "multipart/form-data" {
		if err := r.ParseMultipartForm(32 << 20); err != nil {
			if tooLarge(err) {
				return in, tooLargeError()
			}
			return in, badRequest("bad_request", "reading the form: %v", err)
		}
		f := r.FormValue
		in.Model, in.Prompt, in.NegativePrompt, in.Size, in.OutputFormat = f("model"), f("prompt"), f("negative_prompt"), f("size"), f("output_format")
		for name, dst := range map[string]any{"frames": &in.Frames, "seconds": &in.Seconds, "fps": &in.FPS, "seed": &in.Seed} {
			if v := strings.TrimSpace(f(name)); v != "" {
				if err := json.Unmarshal([]byte(v), dst); err != nil {
					return in, badRequest("bad_value", "%s must be a number", name)
				}
			}
		}
		if v := strings.TrimSpace(f("extra")); v != "" {
			if err := json.Unmarshal([]byte(v), &in.Extra); err != nil {
				return in, badRequest("bad_value", "extra must be a JSON object")
			}
		}
		for name, dst := range map[string]*string{"start_image": &in.StartImage, "end_image": &in.EndImage} {
			file, _, err := r.FormFile(name)
			if errors.Is(err, http.ErrMissingFile) {
				continue
			}
			if err != nil {
				return in, badRequest("bad_request", "reading %s: %v", name, err)
			}
			data, err := io.ReadAll(file)
			file.Close()
			if err != nil {
				return in, badRequest("bad_request", "reading %s: %v", name, err)
			}
			*dst = base64.StdEncoding.EncodeToString(data)
		}
		return in, nil
	}
	body, err := io.ReadAll(r.Body)
	if err != nil {
		if tooLarge(err) {
			return in, tooLargeError()
		}
		return in, badRequest("bad_request", "reading the request: %v", err)
	}
	if err := json.Unmarshal(body, &in); err != nil {
		return in, badRequest("bad_request", "the body is not valid JSON: %v", err)
	}
	in.StartImage, in.EndImage = stripDataURL(in.StartImage), stripDataURL(in.EndImage)
	return in, nil
}

// stripDataURL turns a data: URL into its base64 payload; anything else is
// returned as it is.
func stripDataURL(s string) string {
	if strings.HasPrefix(s, "data:") {
		if _, payload, ok := strings.Cut(s, ","); ok {
			return payload
		}
	}
	return s
}

// buildVideoRequest checks a request against the set and the instance's
// measured size, and fills in its defaults. The checks run in a fixed
// order, after seconds has become frames and before sd-server's 4n+1
// rounding (which only lowers the count); the first failure is returned.
func buildVideoRequest(in videoInput, inst generation.Instance, set generation.ModelSet) (videoJobRequest, vidGenRequest, *requestError) {
	in.Prompt = strings.TrimSpace(in.Prompt)
	if in.Prompt == "" {
		return videoJobRequest{}, vidGenRequest{}, badRequest("bad_request", "prompt is required")
	}
	format := in.OutputFormat
	if format == "" {
		format = "webm"
	}
	if format != "webm" && format != "webp" {
		return videoJobRequest{}, vidGenRequest{}, badRequest("bad_value", "output_format must be webm or webp")
	}
	if in.Frames != nil && in.Seconds != nil {
		return videoJobRequest{}, vidGenRequest{}, badRequest("bad_value", "give frames or seconds, not both")
	}
	for k := range in.Extra {
		if slices.Contains(videoReservedExtra, k) {
			return videoJobRequest{}, vidGenRequest{}, badRequest("bad_value", "extra may not set %q; use the request field instead", k)
		}
	}
	loras := setLoras(set)
	if _, ok := in.Extra["lora"]; ok && len(loras) > 0 {
		return videoJobRequest{}, vidGenRequest{}, badRequest("bad_value", `extra may not set "lora"; %s sends its own LoRAs`, set.Name)
	}

	w, h := set.DefaultWidth, set.DefaultHeight
	if in.Size != "" {
		ws, hs, ok := strings.Cut(in.Size, "x")
		var err1, err2 error
		w, err1 = strconv.Atoi(ws)
		h, err2 = strconv.Atoi(hs)
		if !ok || err1 != nil || err2 != nil {
			return videoJobRequest{}, vidGenRequest{}, badRequest("bad_value", "size must be WIDTHxHEIGHT, e.g. 832x480")
		}
	}
	fps := set.DefaultFPS
	if in.FPS != nil {
		fps = *in.FPS
	}
	frames := set.DefaultFrames
	switch {
	case in.Frames != nil:
		frames = *in.Frames
	case in.Seconds != nil:
		frames = int(math.Round(*in.Seconds * float64(fps)))
	}
	seed := int64(-1)
	if in.Seed != nil {
		seed = *in.Seed
	}
	switch {
	case frames < 1:
		return videoJobRequest{}, vidGenRequest{}, badRequest("bad_value", "the video needs at least 1 frame")
	case fps < 1 || fps > videoMaxFPS:
		return videoJobRequest{}, vidGenRequest{}, badRequest("bad_value", "fps must be from 1 to %d", videoMaxFPS)
	case w < 64 || h < 64:
		return videoJobRequest{}, vidGenRequest{}, badRequest("bad_value", "width and height must be at least 64")
	}

	can := func(c string) bool { return slices.Contains(set.Capabilities, c) }
	switch {
	case in.StartImage == "" && !can(generation.CapText):
		return videoJobRequest{}, vidGenRequest{}, badRequest("start_image_required", "%s needs a start_image", set.Name)
	case in.StartImage != "" && !can(generation.CapStartImage):
		return videoJobRequest{}, vidGenRequest{}, badRequest("input_not_supported", "%s cannot start from an image", set.Name)
	case in.EndImage != "" && !can(generation.CapEndImage):
		return videoJobRequest{}, vidGenRequest{}, badRequest("input_not_supported", "%s cannot end on an image", set.Name)
	case in.EndImage != "" && in.StartImage == "":
		return videoJobRequest{}, vidGenRequest{}, badRequest("start_image_required", "an end_image needs a start_image too")
	}

	// The ceiling holds in either orientation, so a portrait video of the
	// same size is allowed. A zero field (custom sets) is not checked.
	long, short := max(w, h), min(w, h)
	maxLong, maxShort := max(set.MaxWidth, set.MaxHeight), min(set.MaxWidth, set.MaxHeight)
	if (set.MaxFrames > 0 && frames > set.MaxFrames) || (maxShort > 0 && (long > maxLong || short > maxShort)) {
		return videoJobRequest{}, vidGenRequest{}, badRequest("over_limit", "%s makes at most %d frames (%.0f s at %d fps) and %dx%d",
			set.Name, set.MaxFrames, float64(set.MaxFrames)/float64(max(set.DefaultFPS, 1)), set.DefaultFPS, set.MaxWidth, set.MaxHeight)
	}
	if verr := checkMeasuredSize(inst, set, frames, w, h); verr != nil {
		return videoJobRequest{}, vidGenRequest{}, verr
	}
	if w%8 != 0 || h%8 != 0 {
		return videoJobRequest{}, vidGenRequest{}, badRequest("bad_size", "width and height must be multiples of 8")
	}

	shown := videoJobRequest{
		Model: in.Model, Prompt: in.Prompt, NegativePrompt: in.NegativePrompt, Size: fmt.Sprintf("%dx%d", w, h),
		Frames: frames, FPS: fps, Seed: seed, OutputFormat: format,
		StartImage: in.StartImage != "", EndImage: in.EndImage != "", Extra: in.Extra,
	}
	vid := vidGenRequest{
		Prompt: in.Prompt, NegativePrompt: in.NegativePrompt, Width: w, Height: h, Seed: seed,
		Frames: frames, FPS: fps, InitImage: in.StartImage, EndImage: in.EndImage, OutputFormat: format, Lora: loras, Extra: in.Extra,
	}
	return shown, vid, nil
}

// checkMeasuredSize refuses a request with more video (frames × width ×
// height) than the instance was measured for: its reservation covers that
// much and no more, and sd-server can go past its memory limit when it
// decodes a larger video.
func checkMeasuredSize(inst generation.Instance, set generation.ModelSet, frames, w, h int) *requestError {
	limit, measured := generation.VideoLimit(inst, set)
	if generation.PixelFrames(frames, w, h) <= generation.PixelFrames(limit.Frames, limit.Width, limit.Height) {
		return nil
	}
	first := fmt.Sprintf("%s is measured for videos up to %d frames at %dx%d (%s).", inst.Name, limit.Frames, limit.Width, limit.Height, limit.Label)
	if !measured {
		first = fmt.Sprintf("%s is only allowed its test size, %d frames at %dx%d, because it has not been measured at a larger size (or its reservation was set below what that measurement needs).",
			inst.Name, limit.Frames, limit.Width, limit.Height)
	}
	how := "a custom size"
	if p, ok := generation.CoveringPreset(generation.VideoPresets(set), frames, w, h); ok {
		how = p.Label
	}
	return badRequest("over_measured_size", "%s This request is %d frames at %dx%d, which needs more GPU memory than the instance's reservation covers. "+
		"To allow it, measure the instance at a larger size on the Server page (Measure, then %s).", first, frames, w, h, how)
}

// submitVideoJob queues a video request on the instance's GPU and runs it
// in the background. It fails at once when the GPU's queue is full.
func (s *Server) submitVideoJob(inst generation.Instance, req videoJobRequest, vid vidGenRequest) (*videoJob, error) {
	ticket, err := s.genQueue.Submit(inst.GPU, inst.ID, generation.KindVideo)
	if err != nil {
		return nil, err
	}
	ctx, cancel := context.WithCancel(context.Background())
	job := &videoJob{
		genJob: genJob{
			ID: newGenJobID(), InstanceID: inst.ID, Status: jobQueued, Phase: imagePhaseQueued,
			Submitted: ticket.Info().Submitted, cancel: cancel, ticket: ticket,
		},
		Request: req, vid: vid, done: make(chan struct{}),
	}
	if started := ticket.Info().Started; !started.IsZero() {
		job.Status, job.Phase, job.Started = jobRunning, imagePhaseGenerating, &started
		if !s.genMgr.IsRunning(inst.ID) {
			job.Phase = imagePhaseStarting
		}
	}
	s.genVideoJobs.add(job)
	go s.runVideoJob(ctx, job.ID, inst, ticket)
	return job, nil
}

// runVideoJob waits for the GPU, has sd-server make the video and saves
// it. A cancelled job stops in sd-server; it shows cancelled only once
// sd-server has stopped, so the GPU is free by then.
func (s *Server) runVideoJob(ctx context.Context, id string, inst generation.Instance, ticket *generation.Ticket) {
	finish := func(status, errMsg string, video *videoJobVideo, path string) {
		now := time.Now()
		var done chan struct{}
		s.genVideoJobs.update(id, func(j *videoJob) {
			j.vid = vidGenRequest{}
			if j.CancelRequested && status != jobSucceeded {
				status, errMsg = jobCancelled, ""
			}
			j.Status, j.Phase, j.Error, j.Finished, j.Progress = status, "", errMsg, &now, nil
			j.Video, j.path, done = video, path, j.done
		})
		if done != nil {
			close(done)
		}
		if status == jobFailed {
			slog.Warn("video job failed", "job", id, "instance", inst.ID, "error", errMsg)
		}
	}
	if err := ticket.Wait(ctx); err != nil {
		finish(jobCancelled, "", nil, "")
		return
	}
	defer ticket.Done()
	if !s.genMgr.IsRunning(inst.ID) {
		startJob(&s.genVideoJobs, id, imagePhaseStarting)
	}
	if err := s.genReady(ctx, inst); err != nil {
		if ctx.Err() != nil {
			finish(jobCancelled, "", nil, "")
			return
		}
		finish(jobFailed, "starting the backend: "+err.Error(), nil, "")
		return
	}

	var vid vidGenRequest
	now := time.Now()
	s.genVideoJobs.update(id, func(j *videoJob) {
		vid = j.vid
		j.start(imagePhaseGenerating, now)
	})
	base := inst.BaseURL()
	res, err := runVidGen(ctx, base, vid, func(p vidProgress) {
		s.genVideoJobs.update(id, func(j *videoJob) { j.Progress = &p })
	})
	if err != nil {
		switch {
		case ctx.Err() != nil:
			finish(jobCancelled, "", nil, "")
		case !s.genMgr.IsRunning(inst.ID) || backendGone(err):
			finish(jobFailed, "the backend stopped while making the video; its log on the Server page says why", nil, "")
		default:
			finish(jobFailed, "generating: "+err.Error(), nil, "")
		}
		return
	}

	s.genVideoJobs.update(id, func(j *videoJob) { j.Phase, j.Progress = imagePhaseSaving, nil })
	if err := os.MkdirAll(s.videoDir(), 0o755); err != nil {
		finish(jobFailed, "saving: "+err.Error(), nil, "")
		return
	}
	format := res.Format
	if format == "" {
		format = vid.OutputFormat
	}
	path := filepath.Join(s.videoDir(), id+"."+format)
	if err := os.WriteFile(path, res.Data, 0o644); err != nil {
		finish(jobFailed, "saving: "+err.Error(), nil, "")
		return
	}
	// sd-server does not say what size it made; it rounds the request up.
	width, height := vid.Width, vid.Height
	if set, err := s.gen.ModelSet(inst.ModelSetID); err == nil {
		width, height = set.MadeSize(width, height)
	}
	finish(jobSucceeded, "", &videoJobVideo{
		URL: "/v1/videos/jobs/" + id + "/video", MimeType: res.MimeType, Format: format,
		Frames: res.Frames, FPS: res.FPS, Width: width, Height: height, Bytes: int64(len(res.Data)),
	}, path)
}

// cancelVideoJob asks a job to stop: a queued one leaves the queue at
// once; a running one is stopped in sd-server and shows cancelled once it
// has.
func (s *Server) cancelVideoJob(id string) bool {
	var cancel context.CancelFunc
	found := s.genVideoJobs.update(id, func(j *videoJob) {
		if j.Finished != nil {
			return
		}
		j.CancelRequested, cancel = true, j.cancel
	})
	if cancel != nil {
		cancel()
	}
	return found
}

// --- /v1/videos -------------------------------------------------------------

// createVideoJob reads, checks and queues a video request, writing the
// error answer itself when it cannot.
func (s *Server) createVideoJob(w http.ResponseWriter, r *http.Request) (*videoJob, bool) {
	if !s.genEnabled() {
		writeNotEnabled(w, "video")
		return nil, false
	}
	in, verr := readVideoInput(r)
	if verr != nil {
		writeRequestError(w, verr)
		return nil, false
	}
	inst, pe := s.pickInstance(in.Model, generation.KindVideo)
	if pe != nil {
		writeRequestError(w, pe)
		return nil, false
	}
	set, err := s.gen.ModelSet(inst.ModelSetID)
	if err != nil {
		writeOpenAIError(w, http.StatusInternalServerError, "model_set_missing", err.Error())
		return nil, false
	}
	req, vid, verr := buildVideoRequest(in, inst, set)
	if verr != nil {
		writeRequestError(w, verr)
		return nil, false
	}
	job, err := s.submitVideoJob(inst, req, vid)
	if err != nil {
		writeQueueError(w, err)
		return nil, false
	}
	return job, true
}

func (s *Server) handleVideoJobCreate(w http.ResponseWriter, r *http.Request) {
	job, ok := s.createVideoJob(w, r)
	if !ok {
		return
	}
	view, _ := s.genVideoJobs.view(job.ID)
	writeJobAccepted(w, view.ID, view.Status, view.Phase, view.Position)
}

// handleVideoGenerate is the waiting route: it answers with the finished
// job, as GET on the job would show it. A client that leaves first cancels
// the job, since nobody else knows its ID.
func (s *Server) handleVideoGenerate(w http.ResponseWriter, r *http.Request) {
	job, ok := s.createVideoJob(w, r)
	if !ok {
		return
	}
	select {
	case <-job.done:
	case <-r.Context().Done():
		s.cancelVideoJob(job.ID)
		return
	}
	view, _ := s.genVideoJobs.view(job.ID)
	status := http.StatusOK
	switch view.Status {
	case jobFailed:
		status = http.StatusInternalServerError
	case jobCancelled:
		status = http.StatusConflict
	}
	respondJSONStatus(w, status, view)
}

// videoJobRoutes is what the shared job routes need from video jobs.
func (s *Server) videoJobRoutes() jobRoutes[videoJob] {
	return jobRoutes[videoJob]{kind: "video", store: &s.genVideoJobs, enabled: s.genEnabled, cancel: s.cancelVideoJob}
}

func (s *Server) handleVideoJobGet(w http.ResponseWriter, r *http.Request) {
	if j, ok := s.videoJobRoutes().jobOr404(w, chi.URLParam(r, "id")); ok {
		respondJSON(w, j)
	}
}

func (s *Server) handleVideoJobVideo(w http.ResponseWriter, r *http.Request) {
	j, ok := s.videoJobRoutes().jobOr404(w, chi.URLParam(r, "id"))
	if !ok {
		return
	}
	if j.Status != jobSucceeded || j.Video == nil {
		writeOpenAIError(w, http.StatusConflict, "not_ready", "the job is "+j.Status)
		return
	}
	w.Header().Set("Content-Type", j.Video.MimeType)
	http.ServeFile(w, r, j.path)
}

func (s *Server) handleVideoJobList(w http.ResponseWriter, r *http.Request) {
	s.videoJobRoutes().list(w, r)
}

func (s *Server) handleVideoJobDelete(w http.ResponseWriter, r *http.Request) {
	s.videoJobRoutes().delete(w, r)
}
