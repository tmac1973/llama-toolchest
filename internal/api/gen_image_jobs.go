package api

// Async image jobs: /v1/images/jobs takes the same JSON as
// /v1/images/generations, queues it on the instance's GPU like the
// synchronous route does, and answers at once with a job ID. Clients poll
// the job for its status and phase, then download each image. The API
// mirrors /v1/music/jobs.

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"mime"
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
	imagePhaseQueued     = "queued"
	imagePhaseGenerating = "generating"
	imagePhaseSaving     = "saving"
)

// imageJobRequest is the part of an image job's request that the job
// shows back to clients. The rest of the body (init_image, an
// <sd_cpp_extra_args> block) can be megabytes and is never shown.
type imageJobRequest struct {
	Model  string `json:"model,omitempty"`
	Prompt string `json:"prompt"`
	Size   string `json:"size,omitempty"`
	N      int    `json:"n,omitempty"`
}

// extraArgsRE matches sd-server's <sd_cpp_extra_args> block in a prompt.
var extraArgsRE = regexp.MustCompile(`(?s)<sd_cpp_extra_args>.*?</sd_cpp_extra_args>`)

// imageJobImage is one finished image of a job.
type imageJobImage struct {
	URL string `json:"url"`
}

// imageJob is one image request.
type imageJob struct {
	ID         string          `json:"id"`
	InstanceID string          `json:"model"`
	Status     string          `json:"status"`
	Phase      string          `json:"phase,omitempty"`
	Position   int             `json:"queue_position"`
	Request    imageJobRequest `json:"request"`
	Images     []imageJobImage `json:"images,omitempty"`
	Error      string          `json:"error,omitempty"`
	Submitted  time.Time       `json:"submitted"`
	Started    *time.Time      `json:"started,omitempty"`
	Finished   *time.Time      `json:"finished,omitempty"`

	// Not shown to clients.
	body   []byte // the request as sent, until it is forwarded
	paths  []string
	cancel context.CancelFunc
	ticket *generation.Ticket
}

func (j imageJob) state() jobState {
	return jobState{id: j.ID, status: j.Status, ticket: j.ticket, finished: j.Finished, files: j.paths}
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
func (s *Server) submitImageJob(inst generation.Instance, req imageJobRequest, body []byte) (*imageJob, error) {
	ticket, err := s.genQueue.Submit(inst.GPU, inst.ID, generation.KindSD)
	if err != nil {
		return nil, err
	}
	ctx, cancel := context.WithCancel(context.Background())
	job := &imageJob{
		ID: newGenJobID(), InstanceID: inst.ID, Status: jobQueued, Phase: imagePhaseQueued, Request: req,
		Submitted: ticket.Info().Submitted, body: body, cancel: cancel, ticket: ticket,
	}
	// A free GPU starts the job at once; say so in the first answer.
	if started := ticket.Info().Started; !started.IsZero() {
		job.Status, job.Phase, job.Started = jobRunning, imagePhaseGenerating, &started
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
		s.genImageJobs.update(id, func(j *imageJob) {
			j.body = nil
			if j.Status == jobCancelled {
				cancelled = true
				return
			}
			j.Status, j.Phase, j.Error, j.Finished = status, "", errMsg, &now
			j.Images, j.paths = images, paths
		})
		if cancelled {
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

	var body []byte
	cancelled := false
	now := time.Now()
	s.genImageJobs.update(id, func(j *imageJob) {
		if j.Status == jobCancelled {
			cancelled = true
			return
		}
		body = j.body
		if j.Started == nil {
			j.Started = &now
		}
		j.Status, j.Phase = jobRunning, imagePhaseGenerating
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
	images, err := generateImages(context.Background(), inst, body)
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

// generateImages sends a generation request to sd-server and decodes the
// images in its answer.
func generateImages(ctx context.Context, inst generation.Instance, body []byte) ([][]byte, error) {
	resp, err := forwardImage(ctx, inst, "/v1/images/generations", "application/json", "application/json", body)
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

// cancelImageJob stops a job: a queued one leaves the queue; a running
// one is marked cancelled at once and its result is discarded when
// sd-server answers.
func (s *Server) cancelImageJob(id string) bool {
	var cancel context.CancelFunc
	found := false
	s.genImageJobs.update(id, func(j *imageJob) {
		found = true
		if j.Finished != nil {
			return
		}
		now := time.Now()
		cancel, j.Status, j.Phase, j.Finished = j.cancel, jobCancelled, "", &now
	})
	if cancel != nil {
		cancel()
	}
	return found
}

func (s *Server) imageEnabled() bool {
	return s.cfg.GenerationEnabled && s.gen != nil
}

// --- /v1/images/jobs --------------------------------------------------------

func (s *Server) handleImageJobCreate(w http.ResponseWriter, r *http.Request) {
	if !s.imageEnabled() {
		openAIError(w, http.StatusNotFound, "invalid_request_error", "not_found", "image generation is not enabled on this server")
		return
	}
	if mt, _, _ := mime.ParseMediaType(r.Header.Get("Content-Type")); mt == "multipart/form-data" {
		openAIError(w, http.StatusBadRequest, "invalid_request_error", "bad_request", "image jobs take JSON")
		return
	}
	body, err := io.ReadAll(io.LimitReader(r.Body, maxImageRequestBytes+1))
	if err != nil {
		openAIError(w, http.StatusBadRequest, "invalid_request_error", "bad_request", "reading the request: "+err.Error())
		return
	}
	if len(body) > maxImageRequestBytes {
		openAIError(w, http.StatusRequestEntityTooLarge, "invalid_request_error", "too_large", "the request is larger than 128 MiB")
		return
	}
	var req imageJobRequest
	if err := json.Unmarshal(body, &req); err != nil {
		openAIError(w, http.StatusBadRequest, "invalid_request_error", "bad_request", "the body is not valid JSON: "+err.Error())
		return
	}
	req.Prompt = strings.TrimSpace(extraArgsRE.ReplaceAllString(req.Prompt, ""))
	inst, pe := s.pickInstance(req.Model, generation.KindSD)
	if pe != nil {
		errType := "invalid_request_error"
		if pe.status >= 500 {
			errType = "server_error"
		}
		openAIError(w, pe.status, errType, pe.code, pe.msg)
		return
	}
	job, err := s.submitImageJob(inst, req, body)
	if errors.Is(err, generation.ErrQueueFull) {
		w.Header().Set("Retry-After", "30")
		openAIError(w, http.StatusTooManyRequests, "server_error", "queue_full", err.Error())
		return
	}
	if err != nil {
		openAIError(w, http.StatusInternalServerError, "server_error", "queue_error", err.Error())
		return
	}
	view, _ := s.genImageJobs.view(job.ID)
	respondJSONStatus(w, http.StatusAccepted, map[string]any{"id": view.ID, "status": view.Status, "phase": view.Phase, "queue_position": view.Position})
}

func (s *Server) imageJobOr404(w http.ResponseWriter, id string) (imageJob, bool) {
	if !s.imageEnabled() {
		openAIError(w, http.StatusNotFound, "invalid_request_error", "not_found", "image generation is not enabled on this server")
		return imageJob{}, false
	}
	j, ok := s.genImageJobs.view(id)
	if !ok {
		openAIError(w, http.StatusNotFound, "invalid_request_error", "job_not_found", "no image job "+id)
	}
	return j, ok
}

func (s *Server) handleImageJobGet(w http.ResponseWriter, r *http.Request) {
	if j, ok := s.imageJobOr404(w, chi.URLParam(r, "id")); ok {
		respondJSON(w, j)
	}
}

func (s *Server) handleImageJobImage(w http.ResponseWriter, r *http.Request) {
	j, ok := s.imageJobOr404(w, chi.URLParam(r, "id"))
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

func (s *Server) handleImageJobDelete(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	if _, ok := s.imageJobOr404(w, id); !ok {
		return
	}
	s.cancelImageJob(id)
	w.WriteHeader(http.StatusNoContent)
}
