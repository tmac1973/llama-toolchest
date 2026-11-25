package api

// Video upscale jobs: /v1/videos/upscale/jobs takes a clip, checks it
// against what sd.cpp's DGL video::load reads and the set's limits before
// any GPU work, queues it on a video upscaler instance's GPU and answers at
// once with a job ID. The job runs a DGL script (RelayVSR or SeedVR2) on
// the instance's sd-server. Clients poll the job, then download the
// video. sd-server cannot cancel a DGL job, so cancelling a running job
// stops the instance.

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"math/rand/v2"
	"mime"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/tmac1973/llama-toolchest/internal/generation"
	"github.com/tmac1973/llama-toolchest/internal/videofile"
)

// Limits of video upscale requests.
const (
	// maxVUpscaleUploadBytes is the largest clip a multipart request may
	// send. JSON bodies hold base64, so they keep maxImageRequestBytes.
	maxVUpscaleUploadBytes = 512 << 20
	// vupscaleMaxDecoded is video::load's limit on decoded frames
	// (sd.cpp src/dgl/ops/video.cpp).
	vupscaleMaxDecoded = 256 << 20
	// vupscaleMaxOutput is the largest joined output (RelayVSR guide).
	vupscaleMaxOutput = 4 << 30
	// relayvsrMaxWindow is DGL's general tensor limit, which each upscaled
	// RelayVSR window (U8 frames) must fit.
	relayvsrMaxWindow = 1 << 30
	// seedvr2MaxWindow is DGL's 256 MiB limit on intermediate tensors,
	// which each SeedVR2 window's F32 pixels must fit (SeedVR2 guide).
	seedvr2MaxWindow = 256 << 20
	// vupscaleFieldMax is the largest text field of a multipart request.
	vupscaleFieldMax = 4 << 10
)

// vupscaleJobRequest is a request as the job shows it back.
type vupscaleJobRequest struct {
	Model        string `json:"model,omitempty"`
	OutputFormat string `json:"output_format"`
	Seed         int64  `json:"seed"`
	WindowSize   int    `json:"window_size"`
}

// vupscaleJobInput describes the clip a job was given.
type vupscaleJobInput struct {
	Format   string   `json:"format"`
	Width    int      `json:"width"`
	Height   int      `json:"height"`
	Frames   int      `json:"frames"`
	FPS      *float64 `json:"fps"`
	HasAudio bool     `json:"has_audio"`
}

// vupscaleJobVideo is a finished job's video.
type vupscaleJobVideo struct {
	URL      string `json:"url"`
	MimeType string `json:"mime_type"`
	Format   string `json:"format"`
	Frames   int    `json:"frames"`
	Width    int    `json:"width"`
	Height   int    `json:"height"`
	Bytes    int64  `json:"bytes"`
}

// vupscaleJobProgress is a running job's progress.
type vupscaleJobProgress struct {
	Stage     string `json:"stage,omitempty"`
	Detail    string `json:"detail,omitempty"`
	Completed *int   `json:"completed,omitempty"`
	Total     *int   `json:"total,omitempty"`
	ElapsedMS int64  `json:"elapsed_ms"`
}

// vupscaleJob is one video upscale request.
type vupscaleJob struct {
	genJob
	Progress        *vupscaleJobProgress `json:"progress,omitempty"`
	CancelRequested bool                 `json:"cancel_requested,omitempty"`
	Request         vupscaleJobRequest   `json:"request"`
	Input           vupscaleJobInput     `json:"input"`
	// AudioDropped is true when the clip had sound and the output is webm,
	// which has none.
	AudioDropped bool              `json:"audio_dropped,omitempty"`
	Video        *vupscaleJobVideo `json:"video,omitempty"`

	// Not shown to clients.
	script generation.VUpscaleJob // dropped once sent
	input  string                 // the clip, in the instance's input folder
	output string                 // where sd-server writes the result
	path   string                 // the finished video
}

func (j vupscaleJob) state() jobState {
	var files []string
	if j.path != "" {
		files = []string{j.path}
	}
	return j.jobState(files)
}

// vupscaleJobPrivate is the part of a job its record keeps that clients do
// not see.
type vupscaleJobPrivate struct {
	Input  string `json:"input,omitempty"`
	Output string `json:"output,omitempty"`
	Path   string `json:"path,omitempty"`
}

func (j vupscaleJob) private() any {
	return vupscaleJobPrivate{Input: j.input, Output: j.output, Path: j.path}
}

func (j vupscaleJob) restored(raw json.RawMessage, now time.Time) (vupscaleJob, error) {
	var p vupscaleJobPrivate
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &p); err != nil {
			return j, err
		}
	}
	j.input, j.output, j.path, j.Progress = p.Input, p.Output, p.Path, nil
	if j.Finished == nil {
		// Its sd-server stopped with toolchest: the clip and any partial
		// result are of no use now.
		removeFiles(p.Input, p.Output)
		j.input, j.output = "", ""
	}
	j.restore(now)
	return j, nil
}

func (j vupscaleJob) withPosition(n int) vupscaleJob {
	j.Position = n
	return j
}

func removeFiles(paths ...string) {
	for _, p := range paths {
		if p != "" {
			os.Remove(p)
		}
	}
}

// vupscaleDir holds finished upscaled videos; its incoming folder holds
// uploads until they are checked.
func (s *Server) vupscaleDir() string {
	return filepath.Join(s.cfg.DataDir, "gen-output", "vupscale")
}

func (s *Server) vupscaleIncomingDir() string {
	return filepath.Join(s.vupscaleDir(), "incoming")
}

// vupscaleInput is a parsed request; the clip is in a file.
type vupscaleInput struct {
	Model        string
	OutputFormat string
	Seed         *int64
	WindowSize   *int
	// clip is the uploaded file, in the incoming folder.
	clip string
}

// readVUpscaleInput reads a JSON body (the video as base64 or a data URL)
// or a multipart form (the video as a file). The clip is written to
// <incoming>/<id>.part, which the caller removes on any error.
func readVUpscaleInput(r *http.Request, incoming, id string) (vupscaleInput, *requestError) {
	in := vupscaleInput{clip: filepath.Join(incoming, id+".part")}
	if err := os.MkdirAll(incoming, 0o755); err != nil {
		return in, &requestError{http.StatusInternalServerError, "server_error", err.Error()}
	}
	set := func(name, value string) *requestError {
		switch name {
		case "model":
			in.Model = value
		case "output_format":
			in.OutputFormat = value
		case "seed":
			n, err := strconv.ParseInt(strings.TrimSpace(value), 10, 64)
			if err != nil {
				return badRequest("invalid_seed", "seed must be a whole number")
			}
			in.Seed = &n
		case "window_size":
			n, err := strconv.Atoi(strings.TrimSpace(value))
			if err != nil {
				return badRequest("invalid_window_size", "window_size must be a whole number")
			}
			in.WindowSize = &n
		default:
			return badRequest("unknown_field", "unknown field %s", name)
		}
		return nil
	}
	gotVideo := false
	if mt, _, _ := mime.ParseMediaType(r.Header.Get("Content-Type")); mt == "multipart/form-data" {
		mr, err := r.MultipartReader()
		if err != nil {
			return in, badRequest("bad_request", "reading the form: %v", err)
		}
		for {
			part, err := mr.NextPart()
			if errors.Is(err, io.EOF) {
				break
			}
			if err != nil {
				return in, badRequest("bad_request", "reading the form: %v", err)
			}
			name := part.FormName()
			if name == "video" {
				if rerr := saveClip(in.clip, part, maxVUpscaleUploadBytes); rerr != nil {
					return in, rerr
				}
				gotVideo = true
				continue
			}
			value, err := io.ReadAll(io.LimitReader(part, vupscaleFieldMax+1))
			if err != nil {
				return in, badRequest("bad_request", "reading the form: %v", err)
			}
			if len(value) > vupscaleFieldMax {
				return in, badRequest("bad_request", "field %s is too long", name)
			}
			if rerr := set(name, string(value)); rerr != nil {
				return in, rerr
			}
		}
	} else {
		body, rerr := readImageBody(r)
		if rerr != nil {
			if rerr.status == http.StatusRequestEntityTooLarge {
				rerr.msg = fmt.Sprintf("the request is larger than %d MiB; send the video as a multipart file (up to %d MiB) or split it", maxImageRequestBytes>>20, maxVUpscaleUploadBytes>>20)
			}
			return in, rerr
		}
		var fields map[string]json.RawMessage
		if err := json.Unmarshal(body, &fields); err != nil {
			return in, badRequest("invalid_json", "the request must be JSON or a multipart form: %v", err)
		}
		for name, raw := range fields {
			if name == "video" {
				var b64 string
				if err := json.Unmarshal(raw, &b64); err != nil {
					return in, badRequest("invalid_video", "video must be base64 or a data URL")
				}
				data, err := base64.StdEncoding.DecodeString(stripDataURL(b64))
				if err != nil {
					return in, badRequest("invalid_video", "video is not valid base64")
				}
				if err := os.WriteFile(in.clip, data, 0o644); err != nil {
					return in, &requestError{http.StatusInternalServerError, "server_error", err.Error()}
				}
				gotVideo = len(data) > 0
				continue
			}
			var value string
			if err := json.Unmarshal(raw, &value); err != nil {
				var n json.Number
				if err := json.Unmarshal(raw, &n); err != nil {
					return in, badRequest("bad_request", "field %s must be a string or a number", name)
				}
				value = n.String()
			}
			if rerr := set(name, value); rerr != nil {
				return in, rerr
			}
		}
	}
	if !gotVideo {
		return in, badRequest("missing_video", "the request has no video")
	}
	return in, nil
}

// saveClip copies an upload to path, refusing more than limit bytes.
func saveClip(path string, src io.Reader, limit int64) *requestError {
	f, err := os.Create(path)
	if err != nil {
		return &requestError{http.StatusInternalServerError, "server_error", err.Error()}
	}
	n, err := io.Copy(f, io.LimitReader(src, limit+1))
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		return badRequest("bad_request", "reading the video: %v", err)
	}
	if n > limit {
		return &requestError{http.StatusRequestEntityTooLarge, "too_large",
			fmt.Sprintf("the video is larger than %d MiB; split it", limit>>20)}
	}
	return nil
}

// probeClip reads the clip's Info.
func probeClip(path string) (videofile.Info, error) {
	f, err := os.Open(path)
	if err != nil {
		return videofile.Info{}, err
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return videofile.Info{}, err
	}
	return videofile.Probe(f, st.Size())
}

// mib writes a byte count in MiB, rounded up.
func mib(n int64) int64 { return (n + 1<<20 - 1) >> 20 }

// checkVUpscaleClip refuses a clip that sd.cpp's DGL limits would refuse
// on the GPU: the decoded input, the joined output and one window's
// tensor. It does not check the measured ceiling.
func checkVUpscaleClip(script string, scale int, info videofile.Info, window int) *requestError {
	w, h, frames := int64(info.Width), int64(info.Height), int64(info.Frames)
	if decoded := info.DecodedBytes(); decoded > vupscaleMaxDecoded {
		return badRequest("clip_too_large",
			"the clip decodes to %d MiB; sd.cpp reads at most %d MiB (3 × width × height × frames). This clip fits at most %d frames at %dx%d",
			mib(decoded), vupscaleMaxDecoded>>20, vupscaleMaxDecoded/(3*w*h), w, h)
	}
	ow, oh := w*int64(scale), h*int64(scale)
	if out := 3 * ow * oh * frames; out > vupscaleMaxOutput {
		return badRequest("clip_too_large",
			"the upscaled clip would be %d MiB; sd.cpp writes at most %d MiB (3 × width × height × frames at %dx%d). This clip fits at most %d frames",
			mib(out), vupscaleMaxOutput>>20, ow, oh, vupscaleMaxOutput/(3*ow*oh))
	}
	switch script {
	case "seedvr2":
		padded := int64((window-1+3)/4*4 + 1)
		if window == 1 {
			padded = 1
		}
		if need := 4 * 3 * ow * oh * padded; need > seedvr2MaxWindow {
			return badRequest("window_too_large",
				"the window is too large for sd.cpp's %d MiB tensor limit at %dx%d; use window_size %s",
				seedvr2MaxWindow>>20, ow, oh, largestSeedWindow(ow, oh))
		}
	case "relayvsr":
		if need := 3 * ow * oh * int64(window); need > relayvsrMaxWindow {
			most := relayvsrMaxWindow / (3 * ow * oh)
			hint := fmt.Sprintf("use window_size %d or less", most)
			if most < 2 {
				hint = "this frame size is too large for RelayVSR"
			}
			return badRequest("window_too_large",
				"the window is too large for sd.cpp's %d MiB tensor limit at %dx%d; %s",
				relayvsrMaxWindow>>20, ow, oh, hint)
		}
		if info.Frames < 2 {
			return badRequest("too_few_frames", "RelayVSR needs at least 2 frames")
		}
	}
	return nil
}

// largestSeedWindow is the largest SeedVR2 window that fits the tensor
// limit at an output size, as text.
func largestSeedWindow(ow, oh int64) string {
	best := 0
	for _, n := range []int{1, 5, 9, 13, 17, 21, 25, 29, 33} {
		if 4*3*ow*oh*int64(n) <= seedvr2MaxWindow {
			best = n
		}
	}
	if best == 0 {
		return "1 with a smaller clip; this frame size is too large for SeedVR2"
	}
	return strconv.Itoa(best)
}

// vupscaleMime is the MIME type of an output format.
func vupscaleMime(format string) string {
	if format == "avi" {
		return "video/x-msvideo"
	}
	return "video/webm"
}

// createVUpscaleJob reads, checks and queues a request, writing the error
// answer itself when it cannot.
func (s *Server) createVUpscaleJob(w http.ResponseWriter, r *http.Request) (*vupscaleJob, bool) {
	if !s.genEnabled() {
		writeNotEnabled(w, "video upscale")
		return nil, false
	}
	id := newGenJobID()
	in, rerr := readVUpscaleInput(r, s.vupscaleIncomingDir(), id)
	fail := func(e *requestError) (*vupscaleJob, bool) {
		os.Remove(in.clip)
		writeRequestError(w, e)
		return nil, false
	}
	if rerr != nil {
		return fail(rerr)
	}
	inst, pe := s.pickInstance(in.Model, generation.KindVUpscale)
	if pe != nil {
		if pe.code == "model_not_found" && in.Model != "" {
			if kind := s.genModelKind(in.Model); kind != "" {
				pe = badRequest("wrong_model_kind", "model %s is not a video upscaler (its kind is %s); video upscaling needs a vupscale set", in.Model, kind)
			}
		}
		return fail(pe)
	}
	set, err := s.gen.ModelSet(inst.ModelSetID)
	if err != nil {
		return fail(&requestError{http.StatusInternalServerError, "model_set_missing", err.Error()})
	}
	if b := s.resolveBackendBuild(string(generation.KindVUpscale), inst.BuildID); b != nil && !generation.BuildHasDGL(*b) {
		return fail(&requestError{http.StatusServiceUnavailable, "backend_unavailable",
			fmt.Sprintf("video upscaler sets need stable-diffusion.cpp %s or newer (DGL); build %s is older. Build a newer stable-diffusion.cpp on the Builds page, or pick a newer build for instance %s",
				generation.VUpscaleMinBuild, b.ID, inst.ID)})
	}

	info, err := probeClip(in.clip)
	if err != nil {
		if errors.Is(err, videofile.ErrMP4) || errors.Is(err, videofile.ErrUnknownFormat) || errors.Is(err, videofile.ErrCodec) {
			return fail(badRequest("unsupported_video", "%s", err.Error()))
		}
		return fail(badRequest("unsupported_video", "reading the video: %v", err))
	}
	req := vupscaleJobRequest{Model: in.Model, OutputFormat: in.OutputFormat}
	if req.OutputFormat == "" {
		req.OutputFormat = "webm"
	}
	if req.OutputFormat != "webm" && req.OutputFormat != "avi" {
		return fail(badRequest("invalid_output_format", "output_format must be webm or avi"))
	}
	req.WindowSize = vupscaleDefaultWindow(set, inst)
	if in.WindowSize != nil {
		req.WindowSize = *in.WindowSize
	}
	if err := generation.WindowSizeOK(set.Recipe, req.WindowSize); err != nil {
		return fail(badRequest("invalid_window_size", "%s", err.Error()))
	}
	scale := set.UpscaleScale()
	if e := checkVUpscaleClip(set.Recipe, scale, info, req.WindowSize); e != nil {
		return fail(e)
	}
	if e := checkVUpscaleCeiling(inst, info, req.WindowSize); e != nil {
		return fail(e)
	}
	req.Seed = rand.Int64N(math.MaxInt32)
	if in.Seed != nil {
		req.Seed = *in.Seed
	}

	inDir, outDir, err := generation.VUpscaleWorkDirs(s.cfg.DataDir, inst.ID)
	if err != nil {
		return fail(&requestError{http.StatusInternalServerError, "server_error", err.Error()})
	}
	clip := filepath.Join(inDir, id+"."+info.Format)
	if err := os.Rename(in.clip, clip); err != nil {
		return fail(&requestError{http.StatusInternalServerError, "server_error", err.Error()})
	}
	input := vupscaleJobInput{Format: info.Format, Width: info.Width, Height: info.Height, Frames: info.Frames, HasAudio: info.HasAudio}
	if fps := info.FPS(); fps > 0 {
		v := math.Round(fps*1000) / 1000
		input.FPS = &v
	}
	script := generation.VUpscaleJob{
		InputPath: "input/" + filepath.Base(clip), OutputPath: "output/" + id + "." + req.OutputFormat,
		OutputFormat: req.OutputFormat, Seed: req.Seed, WindowSize: req.WindowSize,
	}
	if info.FPSNum == 0 {
		// video::load needs a rate for a clip without one (a still WebP).
		script.FPS = 24
	}
	job, err := s.submitVUpscaleJob(id, inst, req, input, script, clip, filepath.Join(outDir, id+"."+req.OutputFormat))
	if err != nil {
		os.Remove(clip)
		writeQueueError(w, err)
		return nil, false
	}
	return job, true
}

// vupscaleDefaultWindow is the window a request without window_size gets:
// the script's default, or the measured window when that is smaller.
func vupscaleDefaultWindow(set generation.ModelSet, inst generation.Instance) int {
	n := generation.DefaultWindowSize(set.Recipe)
	if inst.MeasuredWindow > 0 && inst.MeasuredWindow < n {
		n = inst.MeasuredWindow
	}
	return n
}

// checkVUpscaleCeiling refuses a clip larger than the instance was
// measured with: each side, the frame count and the window on its own.
// Without a measurement there is no ceiling, as for video sets.
func checkVUpscaleCeiling(inst generation.Instance, info videofile.Info, window int) *requestError {
	if inst.MeasuredWindow == 0 || inst.MeasuredFrames == 0 {
		return nil
	}
	if info.Width <= inst.MeasuredWidth && info.Height <= inst.MeasuredHeight && info.Frames <= inst.MeasuredFrames && window <= inst.MeasuredWindow {
		return nil
	}
	return badRequest("over_measured_size",
		"this instance was measured with %dx%d, %d frames and window %d; this clip is %dx%d, %d frames, window %d. "+
			"Measure it again at the larger size on the Server page, or send a smaller clip",
		inst.MeasuredWidth, inst.MeasuredHeight, inst.MeasuredFrames, inst.MeasuredWindow, info.Width, info.Height, info.Frames, window)
}

// genModelKind names the kind of the set or instance model is, or "" when
// there is none.
func (s *Server) genModelKind(model string) string {
	for _, inst := range s.gen.Instances() {
		if inst.ID == model || inst.ModelSetID == model {
			if set, err := s.gen.ModelSet(inst.ModelSetID); err == nil {
				return genKindLabel(set.Kind)
			}
		}
	}
	if set, err := s.gen.ModelSet(model); err == nil {
		return genKindLabel(set.Kind)
	}
	return ""
}

// submitVUpscaleJob queues a job on the instance's GPU and runs it in the
// background. It fails at once when the GPU's queue is full.
func (s *Server) submitVUpscaleJob(id string, inst generation.Instance, req vupscaleJobRequest, input vupscaleJobInput,
	script generation.VUpscaleJob, clip, output string) (*vupscaleJob, error) {
	ticket, err := s.genQueue.Submit(inst.GPU, inst.ID, generation.KindVUpscale)
	if err != nil {
		return nil, err
	}
	ctx, cancel := context.WithCancel(context.Background())
	job := &vupscaleJob{
		genJob: genJob{
			ID: id, InstanceID: inst.ID, Status: jobQueued, Phase: imagePhaseQueued,
			Submitted: ticket.Info().Submitted, cancel: cancel, ticket: ticket,
		},
		Request: req, Input: input, AudioDropped: input.HasAudio && req.OutputFormat == "webm",
		script: script, input: clip, output: output,
	}
	if started := ticket.Info().Started; !started.IsZero() {
		job.Status, job.Phase, job.Started = jobRunning, imagePhaseGenerating, &started
		if !s.genMgr.IsRunning(inst.ID) {
			job.Phase = imagePhaseStarting
		}
	}
	s.genVUpscaleJobs.add(job)
	go s.runVUpscaleJob(ctx, id, inst, ticket)
	return job, nil
}

// runVUpscaleJob waits for the GPU, has sd-server run the script and keeps
// the result. When the job is cancelled while sd-server runs it, the
// instance is stopped before the GPU is given back, since sd-server cannot
// cancel a DGL job.
func (s *Server) runVUpscaleJob(ctx context.Context, id string, inst generation.Instance, ticket *generation.Ticket) {
	var clip, output string
	s.genVUpscaleJobs.update(id, func(j *vupscaleJob) { clip, output = j.input, j.output })
	finish := func(status, errMsg string, video *vupscaleJobVideo, path string) {
		now := time.Now()
		s.genVUpscaleJobs.update(id, func(j *vupscaleJob) {
			j.script = generation.VUpscaleJob{}
			// A cancelled job stays cancelled, whatever error its end
			// brought (a stopped backend, a context error).
			if j.CancelRequested && status != jobSucceeded {
				status, errMsg = jobCancelled, ""
			}
			j.Status, j.Phase, j.Error, j.Finished, j.Progress = status, "", errMsg, &now, nil
			j.Video, j.path, j.input, j.output = video, path, "", ""
		})
		removeFiles(clip)
		if status != jobSucceeded {
			removeFiles(output)
		}
		if status == jobFailed {
			slog.Warn("video upscale job failed", "job", id, "instance", inst.ID, "error", errMsg)
		}
	}
	if err := ticket.Wait(ctx); err != nil {
		finish(jobCancelled, "", nil, "")
		return
	}
	defer ticket.Done()
	if !s.genMgr.IsRunning(inst.ID) {
		startJob(&s.genVUpscaleJobs, id, imagePhaseStarting)
	}
	if err := s.genReady(ctx, inst); err != nil {
		if ctx.Err() != nil {
			finish(jobCancelled, "", nil, "")
			return
		}
		finish(jobFailed, "starting the backend: "+err.Error(), nil, "")
		return
	}
	set, err := s.gen.ModelSet(inst.ModelSetID)
	if err != nil {
		finish(jobFailed, err.Error(), nil, "")
		return
	}
	var script generation.VUpscaleJob
	var input vupscaleJobInput
	now := time.Now()
	s.genVUpscaleJobs.update(id, func(j *vupscaleJob) {
		script, input = j.script, j.Input
		j.start(imagePhaseGenerating, now)
	})
	source, err := generation.RenderVUpscale(set, script)
	if err != nil {
		finish(jobFailed, err.Error(), nil, "")
		return
	}
	res, err := runDGLJob(ctx, inst.BaseURL(), source, func(p dglProgress) {
		s.genVUpscaleJobs.update(id, func(j *vupscaleJob) {
			j.Progress = &vupscaleJobProgress{Stage: p.Stage, Detail: p.Detail, Completed: p.Completed, Total: p.Total, ElapsedMS: p.ElapsedMS}
		})
	})
	if err != nil {
		switch {
		case ctx.Err() != nil:
			// sd-server is still running the script. Stop it before the
			// ticket frees the GPU for the next job.
			slog.Info("stopping the instance to cancel a video upscale job; sd-server cannot cancel DGL jobs", "instance", inst.ID, "job", id)
			if err := s.genMgr.Stop(inst.ID); err != nil {
				slog.Warn("could not stop the instance after a cancelled video upscale job", "instance", inst.ID, "error", err)
			}
			finish(jobCancelled, "", nil, "")
		case !s.genMgr.IsRunning(inst.ID) || backendGone(err):
			finish(jobFailed, "the backend stopped while upscaling the video; its log on the Server page says why", nil, "")
		default:
			finish(jobFailed, "upscaling: "+err.Error(), nil, "")
		}
		return
	}

	s.genVUpscaleJobs.update(id, func(j *vupscaleJob) { j.Phase, j.Progress = imagePhaseSaving, nil })
	if err := os.MkdirAll(s.vupscaleDir(), 0o755); err != nil {
		finish(jobFailed, "saving: "+err.Error(), nil, "")
		return
	}
	path := filepath.Join(s.vupscaleDir(), id+"."+script.OutputFormat)
	if err := os.Rename(output, path); err != nil {
		finish(jobFailed, "saving: sd-server's output is missing: "+err.Error(), nil, "")
		return
	}
	st, err := os.Stat(path)
	if err != nil {
		finish(jobFailed, "saving: "+err.Error(), nil, "")
		return
	}
	scale := set.UpscaleScale()
	if res.Width != input.Width*scale || res.Height != input.Height*scale || res.Frames != input.Frames {
		slog.Warn("the upscaled video's size differs from the expected one",
			"job", id, "got", fmt.Sprintf("%dx%d, %d frames", res.Width, res.Height, res.Frames),
			"want", fmt.Sprintf("%dx%d, %d frames", input.Width*scale, input.Height*scale, input.Frames))
	}
	last := generation.LastVideo{Frames: res.Frames, Width: res.Width, Height: res.Height, Inputs: "a " + input.Format + " clip",
		Seconds: int(time.Since(now).Round(time.Second).Seconds()), At: time.Now()}
	if err := s.gen.EditInstance(inst.ID, func(i *generation.Instance) { i.LastVideo = last }); err != nil {
		slog.Warn("failed to record the upscale's time on its instance", "instance", inst.ID, "error", err)
	}
	finish(jobSucceeded, "", &vupscaleJobVideo{
		URL: "/v1/videos/upscale/jobs/" + id + "/video", MimeType: vupscaleMime(script.OutputFormat), Format: script.OutputFormat,
		Frames: res.Frames, Width: res.Width, Height: res.Height, Bytes: st.Size(),
	}, path)
}

// cancelVUpscaleJob asks a job to stop: a queued one leaves the queue at
// once; a running one stops waiting, and its runner stops the instance.
// The job shows cancelled from now on.
func (s *Server) cancelVUpscaleJob(id string) bool {
	var cancel context.CancelFunc
	found := s.genVUpscaleJobs.update(id, func(j *vupscaleJob) {
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

// --- /v1/videos/upscale -------------------------------------------------------

func (s *Server) vupscaleJobRoutes() jobRoutes[vupscaleJob] {
	return jobRoutes[vupscaleJob]{kind: "video upscale", store: &s.genVUpscaleJobs, enabled: s.genEnabled, cancel: s.cancelVUpscaleJob}
}

func (s *Server) handleVUpscaleJobCreate(w http.ResponseWriter, r *http.Request) {
	job, ok := s.createVUpscaleJob(w, r)
	if !ok {
		return
	}
	view, _ := s.genVUpscaleJobs.view(job.ID)
	writeJobAccepted(w, view.ID, view.Status, view.Phase, view.Position)
}

func (s *Server) handleVUpscaleJobGet(w http.ResponseWriter, r *http.Request) {
	if j, ok := s.vupscaleJobRoutes().jobOr404(w, chi.URLParam(r, "id")); ok {
		respondJSON(w, j)
	}
}

func (s *Server) handleVUpscaleJobVideo(w http.ResponseWriter, r *http.Request) {
	j, ok := s.vupscaleJobRoutes().jobOr404(w, chi.URLParam(r, "id"))
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

func (s *Server) handleVUpscaleJobList(w http.ResponseWriter, r *http.Request) {
	s.vupscaleJobRoutes().list(w, r)
}

func (s *Server) handleVUpscaleJobDelete(w http.ResponseWriter, r *http.Request) {
	s.vupscaleJobRoutes().delete(w, r)
}
