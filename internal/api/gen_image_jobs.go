package api

// Async image jobs: /v1/images/jobs takes the same JSON as
// /v1/images/generations, or the same multipart form as /v1/images/edits
// for an edit, queues it on the instance's GPU like the synchronous routes
// do, and answers at once with a job ID. Clients poll the job for its
// status and phase, then download each image. The API mirrors
// /v1/music/jobs.

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/tmac1973/llama-toolchest/internal/generation"
)

// Image job phases: what a job is doing while it is queued or running.
// A finished job has none.
const (
	imagePhaseQueued = "queued"
	// imagePhaseStarting: the job's backend was stopped or still
	// starting; with eviction on, room is made for it first (gen_evict.go).
	imagePhaseStarting   = "starting"
	imagePhaseGenerating = "generating"
	imagePhaseSaving     = "saving"
)

// imageJobRequest is the part of an image job's request that the job
// shows back to clients. The rest of the body (init_image, an
// <sd_cpp_extra_args> block, an edit's images) can be megabytes and is
// never shown.
type imageJobRequest struct {
	Model  string `json:"model,omitempty"`
	Prompt string `json:"prompt"`
	Size   string `json:"size,omitempty"`
	N      int    `json:"n,omitempty"`
	// Edit: the job came as a multipart form and goes to sd-server's
	// /v1/images/edits.
	Edit bool `json:"edit,omitempty"`
}

// extraArgsRE matches sd-server's <sd_cpp_extra_args> block in a prompt.
var extraArgsRE = regexp.MustCompile(`(?s)<sd_cpp_extra_args>.*?</sd_cpp_extra_args>`)

// imageJobImage is one finished image of a job.
type imageJobImage struct {
	URL string `json:"url"`
}

// imageJob is one image request.
type imageJob struct {
	genJob
	Request imageJobRequest `json:"request"`
	Images  []imageJobImage `json:"images,omitempty"`

	// Not shown to clients.
	body        []byte // the request as sent, until it is forwarded
	contentType string // body's Content-Type, with a form's boundary
	paths       []string
}

func (j imageJob) state() jobState { return j.jobState(j.paths) }

// imageJobPrivate is the part of an image job its record keeps that
// clients do not see.
type imageJobPrivate struct {
	Paths []string `json:"paths,omitempty"`
}

func (j imageJob) private() any { return imageJobPrivate{Paths: j.paths} }

func (j imageJob) restored(raw json.RawMessage, now time.Time) (imageJob, error) {
	var p imageJobPrivate
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &p); err != nil {
			return j, err
		}
	}
	j.paths = p.Paths
	j.restore(now)
	return j, nil
}

func (j imageJob) withPosition(n int) imageJob {
	j.Position = n
	return j
}

func (s *Server) imageDir() string {
	return filepath.Join(s.cfg.DataDir, "gen-output", "images")
}

// submitImageJob queues an image request on the instance's GPU and runs
// it in the background. It fails at once when the GPU's queue is full.
func (s *Server) submitImageJob(inst generation.Instance, req imageJobRequest, body []byte, contentType string) (*imageJob, error) {
	ticket, err := s.genQueue.Submit(inst.GPU, inst.ID, generation.KindSD)
	if err != nil {
		return nil, err
	}
	ctx, cancel := context.WithCancel(context.Background())
	job := &imageJob{
		genJob: genJob{
			ID: newGenJobID(), InstanceID: inst.ID, Status: jobQueued, Phase: imagePhaseQueued,
			Submitted: ticket.Info().Submitted, cancel: cancel, ticket: ticket,
		},
		Request: req, body: body, contentType: contentType,
	}
	// A free GPU starts the job at once; say so in the first answer.
	if started := ticket.Info().Started; !started.IsZero() {
		job.Status, job.Phase, job.Started = jobRunning, imagePhaseGenerating, &started
		if !s.genMgr.IsRunning(inst.ID) {
			job.Phase = imagePhaseStarting
		}
	}
	s.genImageJobs.add(job)
	go s.runImageJob(ctx, job.ID, inst, ticket)
	return job, nil
}

// runImageJob waits for the GPU, forwards the request to sd-server and
// saves the images. sd-server cannot be interrupted, so a job cancelled
// while it runs keeps the GPU until sd-server answers, and the answer is
// thrown away.
func (s *Server) runImageJob(ctx context.Context, id string, inst generation.Instance, ticket *generation.Ticket) {
	finish := func(status, errMsg string, images []imageJobImage, paths []string) {
		now := time.Now()
		cancelled := false
		found := s.genImageJobs.update(id, func(j *imageJob) {
			j.body = nil
			if j.Status == jobCancelled {
				cancelled = true
				return
			}
			j.Status, j.Phase, j.Error, j.Finished = status, "", errMsg, &now
			j.Images, j.paths = images, paths
		})
		// A job deleted after it was cancelled is gone from the store.
		if cancelled || !found {
			for _, p := range paths {
				os.Remove(p)
			}
		}
		if status == jobFailed {
			slog.Warn("image job failed", "job", id, "instance", inst.ID, "error", errMsg)
		}
	}
	if err := ticket.Wait(ctx); err != nil {
		finish(jobCancelled, "", nil, nil)
		return
	}
	defer ticket.Done()
	if !s.genMgr.IsRunning(inst.ID) {
		startJob(&s.genImageJobs, id, imagePhaseStarting)
	}
	if err := s.genReady(ctx, inst); err != nil {
		finish(jobFailed, "starting the backend: "+err.Error(), nil, nil)
		return
	}

	var body []byte
	var contentType string
	path := "/v1/images/generations"
	cancelled := false
	now := time.Now()
	s.genImageJobs.update(id, func(j *imageJob) {
		if !j.start(imagePhaseGenerating, now) {
			cancelled = true
			return
		}
		body, contentType = j.body, j.contentType
		if j.Request.Edit {
			path = "/v1/images/edits"
		}
	})
	if cancelled {
		finish(jobCancelled, "", nil, nil)
		return
	}

	fail := func(err error) {
		if !s.genMgr.IsRunning(inst.ID) {
			finish(jobFailed, "backend stopped", nil, nil)
			return
		}
		finish(jobFailed, "generating: "+err.Error(), nil, nil)
	}
	// Not tied to ctx: once sd-server has the request it runs to the end,
	// and the GPU stays booked until then.
	images, err := generateImages(context.Background(), inst, path, contentType, body)
	if err != nil {
		fail(err)
		return
	}
	if v, _ := s.genImageJobs.view(id); v.Status == jobCancelled {
		finish(jobCancelled, "", nil, nil) // drops the late result
		return
	}

	s.genImageJobs.update(id, func(j *imageJob) {
		if j.Finished == nil {
			j.Phase = imagePhaseSaving
		}
	})
	if err := os.MkdirAll(s.imageDir(), 0o755); err != nil {
		finish(jobFailed, "saving: "+err.Error(), nil, nil)
		return
	}
	var refs []imageJobImage
	var paths []string
	for n, img := range images {
		file := filepath.Join(s.imageDir(), fmt.Sprintf("%s-%d.png", id, n))
		if err := os.WriteFile(file, img, 0o644); err != nil {
			for _, p := range paths {
				os.Remove(p)
			}
			finish(jobFailed, "saving: "+err.Error(), nil, nil)
			return
		}
		paths = append(paths, file)
		refs = append(refs, imageJobImage{URL: fmt.Sprintf("/v1/images/jobs/%s/image/%d", id, n)})
	}
	finish(jobSucceeded, "", refs, paths)
}

// generateImages sends a generation or edit request to sd-server at path
// and decodes the images in its answer.
func generateImages(ctx context.Context, inst generation.Instance, path, contentType string, body []byte) ([][]byte, error) {
	resp, err := forwardImage(ctx, inst, path, contentType, "application/json", body)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading the answer: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, backendErrorMessage(raw))
	}
	var out struct {
		Data []struct {
			B64JSON string `json:"b64_json"`
		} `json:"data"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, fmt.Errorf("reading the answer: %w", err)
	}
	if len(out.Data) == 0 {
		return nil, errors.New("the answer has no images")
	}
	images := make([][]byte, len(out.Data))
	for i, d := range out.Data {
		img, err := base64.StdEncoding.DecodeString(d.B64JSON)
		if err != nil || len(img) == 0 {
			return nil, fmt.Errorf("image %d in the answer is not base64 data", i)
		}
		images[i] = img
	}
	return images, nil
}

// backendErrorMessage is the message of an OpenAI-shaped error answer, or
// the start of the answer as text.
func backendErrorMessage(raw []byte) string {
	var e struct {
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if json.Unmarshal(raw, &e) == nil && e.Error.Message != "" {
		return e.Error.Message
	}
	msg := strings.TrimSpace(string(raw))
	if len(msg) > 300 {
		msg = msg[:300] + "…"
	}
	return msg
}

// --- /v1/images/jobs --------------------------------------------------------

func (s *Server) handleImageJobCreate(w http.ResponseWriter, r *http.Request) {
	if !s.genEnabled() {
		writeNotEnabled(w, "image")
		return
	}
	body, rerr := readImageBody(r)
	if rerr != nil {
		writeRequestError(w, rerr)
		return
	}
	contentType := r.Header.Get("Content-Type")
	req, err := imageJobRequestOf(contentType, body)
	if err != nil {
		openAIError(w, http.StatusBadRequest, "invalid_request_error", "bad_request", err.Error())
		return
	}
	req.Prompt = strings.TrimSpace(extraArgsRE.ReplaceAllString(req.Prompt, ""))
	inst, pe := s.pickInstance(req.Model, generation.KindSD)
	if pe == nil && req.Edit {
		pe = s.editRefusal(inst)
	}
	if pe != nil {
		writeRequestError(w, pe)
		return
	}
	if !req.Edit {
		contentType = "application/json"
	}
	job, err := s.submitImageJob(inst, req, body, contentType)
	if err != nil {
		writeQueueError(w, err)
		return
	}
	view, _ := s.genImageJobs.view(job.ID)
	writeJobAccepted(w, view.ID, view.Status, view.Phase, view.Position)
}

// imageJobRequestOf reads the shown part of an image job's request: a JSON
// generation, or a multipart edit form.
func imageJobRequestOf(contentType string, body []byte) (imageJobRequest, error) {
	var req imageJobRequest
	if !isMultipartType(contentType) {
		if err := json.Unmarshal(body, &req); err != nil {
			return req, fmt.Errorf("the body is not valid JSON: %w", err)
		}
		req.Edit = false
		return req, nil
	}
	f, err := readFormFields(contentType, body, "model", "prompt", "size", "n")
	if err != nil {
		return req, err
	}
	req.Model, req.Prompt, req.Size, req.Edit = f["model"], f["prompt"], f["size"], true
	// sd-server checks n; here it is only shown back.
	req.N, _ = strconv.Atoi(f["n"])
	return req, nil
}

// imageJobRoutes is what the shared job routes need from image jobs.
func (s *Server) imageJobRoutes() jobRoutes[imageJob] {
	return jobRoutes[imageJob]{
		kind: "image", store: &s.genImageJobs, enabled: s.genEnabled,
		cancel: func(id string) bool { return cancelJob(&s.genImageJobs, id) },
	}
}

func (s *Server) handleImageJobGet(w http.ResponseWriter, r *http.Request) {
	if j, ok := s.imageJobRoutes().jobOr404(w, chi.URLParam(r, "id")); ok {
		respondJSON(w, j)
	}
}

func (s *Server) handleImageJobImage(w http.ResponseWriter, r *http.Request) {
	j, ok := s.imageJobRoutes().jobOr404(w, chi.URLParam(r, "id"))
	if !ok {
		return
	}
	if j.Status != jobSucceeded {
		openAIError(w, http.StatusConflict, "invalid_request_error", "not_ready", "the job is "+j.Status)
		return
	}
	n, err := strconv.Atoi(chi.URLParam(r, "n"))
	if err != nil || n < 0 || n >= len(j.paths) {
		openAIError(w, http.StatusNotFound, "invalid_request_error", "not_found",
			fmt.Sprintf("the job has %d image(s), numbered from 0", len(j.paths)))
		return
	}
	w.Header().Set("Content-Type", "image/png")
	http.ServeFile(w, r, j.paths[n])
}

func (s *Server) handleImageJobList(w http.ResponseWriter, r *http.Request) {
	s.imageJobRoutes().list(w, r)
}

func (s *Server) handleImageJobDelete(w http.ResponseWriter, r *http.Request) {
	s.imageJobRoutes().delete(w, r)
}
