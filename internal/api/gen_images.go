package api

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/generation"
)

// maxImageRequestBytes bounds an image request body read into memory: a
// prompt, or an edit's source image and mask.
const maxImageRequestBytes = 128 << 20

// openAIError writes an error in the OpenAI API's format.
func openAIError(w http.ResponseWriter, status int, errType, code, message string) {
	respondJSONStatus(w, status, map[string]any{
		"error": map[string]any{"message": message, "type": errType, "code": code},
	})
}

// pickError is why no instance can take a request, with its HTTP status.
type pickError struct {
	status int
	code   string
	msg    string
}

func (e *pickError) Error() string { return e.msg }

// pickInstance chooses the instance of a kind that serves model: the
// instance with that ID, or among the instances of the model set with that
// ID the one whose GPU has the fewest jobs (lowest ID on a tie), or with
// model empty the only one. Only instances that are enabled, running, not
// being measured and have a VRAM reservation are eligible; instances do
// not start on request.
func (s *Server) pickInstance(model string, kind generation.BackendKind) (generation.Instance, *pickError) {
	if s.gen == nil {
		return generation.Instance{}, &pickError{http.StatusNotFound, "model_not_found", "generation backends are not available"}
	}
	sets := map[string]generation.ModelSet{}
	for _, m := range s.gen.ModelSets() {
		sets[m.ID] = m
	}
	var matches []generation.Instance
	for _, inst := range s.gen.Instances() {
		if sets[inst.ModelSetID].Kind != kind {
			continue
		}
		if model == "" || inst.ID == model {
			matches = append(matches, inst)
		}
	}
	// An instance ID wins over a model set of the same name.
	if model != "" && len(matches) == 0 {
		for _, inst := range s.gen.Instances() {
			if sets[inst.ModelSetID].Kind == kind && inst.ModelSetID == model {
				matches = append(matches, inst)
			}
		}
	}
	if len(matches) == 0 {
		return generation.Instance{}, &pickError{http.StatusNotFound, "model_not_found", "model not found: " + model}
	}
	var eligible []generation.Instance
	var reasons []string
	for _, inst := range matches {
		switch {
		case !inst.Enabled:
			reasons = append(reasons, fmt.Sprintf("backend %s is disabled", inst.ID))
		case s.genMgr.IsMeasuring(inst.ID):
			reasons = append(reasons, fmt.Sprintf("backend %s is being measured", inst.ID))
		case inst.ReservedMiB == 0:
			reasons = append(reasons, fmt.Sprintf("backend %s has no VRAM reservation yet", inst.ID))
		case !s.genMgr.IsRunning(inst.ID):
			reasons = append(reasons, fmt.Sprintf("backend %s is not running", inst.ID))
		default:
			eligible = append(eligible, inst)
		}
	}
	if len(eligible) == 0 {
		return generation.Instance{}, &pickError{http.StatusServiceUnavailable, "backend_unavailable", strings.Join(reasons, "; ")}
	}
	if model == "" && len(eligible) > 1 {
		ids := make([]string, len(eligible))
		for i, inst := range eligible {
			ids[i] = inst.ID
		}
		return generation.Instance{}, &pickError{http.StatusBadRequest, "model_required",
			"several backends can serve this request; set model to one of: " + strings.Join(ids, ", ")}
	}
	slices.SortFunc(eligible, func(a, b generation.Instance) int {
		if la, lb := s.genQueue.Load(a.GPU), s.genQueue.Load(b.GPU); la != lb {
			return la - lb
		}
		return strings.Compare(a.ID, b.ID)
	})
	return eligible[0], nil
}

// requestModel finds the "model" field of a JSON or multipart body.
func requestModel(contentType string, body []byte) (string, error) {
	mediaType, params, _ := mime.ParseMediaType(contentType)
	if mediaType == "multipart/form-data" {
		mr := multipart.NewReader(bytes.NewReader(body), params["boundary"])
		for {
			part, err := mr.NextPart()
			if errors.Is(err, io.EOF) {
				return "", nil
			}
			if err != nil {
				return "", fmt.Errorf("reading the form: %w", err)
			}
			if part.FormName() == "model" {
				v, err := io.ReadAll(io.LimitReader(part, 1024))
				return strings.TrimSpace(string(v)), err
			}
		}
	}
	var req struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		return "", fmt.Errorf("the body is not valid JSON: %w", err)
	}
	return req.Model, nil
}

// handleGenImages serves /v1/images/generations and /v1/images/edits: it
// picks the sd-server instance for the request's model, waits its turn in
// that GPU's queue, and forwards the request unchanged.
func (s *Server) handleGenImages(w http.ResponseWriter, r *http.Request) {
	if !s.cfg.GenerationEnabled || s.gen == nil {
		openAIError(w, http.StatusNotFound, "invalid_request_error", "not_found", "image generation is not enabled on this server")
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
	model, err := requestModel(r.Header.Get("Content-Type"), body)
	if err != nil {
		openAIError(w, http.StatusBadRequest, "invalid_request_error", "bad_request", err.Error())
		return
	}
	inst, pe := s.pickInstance(model, generation.KindSD)
	if pe != nil {
		errType := "invalid_request_error"
		if pe.status >= 500 {
			errType = "server_error"
		}
		openAIError(w, pe.status, errType, pe.code, pe.msg)
		return
	}
	if strings.HasSuffix(r.URL.Path, "/edits") {
		set, _ := s.gen.ModelSet(inst.ModelSetID)
		if !slices.Contains(set.Capabilities, "edit") {
			openAIError(w, http.StatusBadRequest, "invalid_request_error", "edits_not_supported",
				fmt.Sprintf("model %s does not support image edits", inst.ID))
			return
		}
	}

	ticket, err := s.genQueue.Submit(inst.GPU, inst.ID, generation.KindSD)
	if errors.Is(err, generation.ErrQueueFull) {
		w.Header().Set("Retry-After", "30")
		openAIError(w, http.StatusTooManyRequests, "server_error", "queue_full", err.Error())
		return
	}
	if err != nil {
		openAIError(w, http.StatusInternalServerError, "server_error", "queue_error", err.Error())
		return
	}
	if err := ticket.Wait(r.Context()); err != nil {
		return // the client went away while waiting
	}
	defer ticket.Done()
	s.forwardToInstance(w, r, inst, body)
}

// forwardToInstance sends a request to an instance and copies the answer
// back. The upstream request is not tied to the client: if the client
// goes away mid-generation the backend still finishes (sd-server cannot be
// cancelled on its OpenAI route), so the GPU's queue slot is held until
// the GPU is really free.
func (s *Server) forwardToInstance(w http.ResponseWriter, r *http.Request, inst generation.Instance, body []byte) {
	url := fmt.Sprintf("http://127.0.0.1:%d%s", inst.Port, r.URL.Path)
	req, err := http.NewRequestWithContext(context.WithoutCancel(r.Context()), r.Method, url, bytes.NewReader(body))
	if err != nil {
		openAIError(w, http.StatusInternalServerError, "server_error", "forward_failed", err.Error())
		return
	}
	req.Header.Set("Content-Type", r.Header.Get("Content-Type"))
	req.Header.Set("Accept", r.Header.Get("Accept"))
	resp, err := genHTTPClient.Do(req)
	if err != nil {
		openAIError(w, http.StatusBadGateway, "server_error", "backend_error", fmt.Sprintf("backend %s: %v", inst.ID, err))
		return
	}
	defer resp.Body.Close()
	for _, h := range []string{"Content-Type", "Content-Length"} {
		if v := resp.Header.Get(h); v != "" {
			w.Header().Set(h, v)
		}
	}
	w.WriteHeader(resp.StatusCode)
	// Errors here are the client having gone; the body is read to the end
	// either way, so the backend's answer is complete before the GPU's
	// next job starts.
	if _, err := io.Copy(w, resp.Body); err != nil {
		io.Copy(io.Discard, resp.Body)
	}
}

// genHTTPClient has no overall timeout: a generation takes as long as it
// takes, like the LLM proxy.
var genHTTPClient = &http.Client{}

// handleGenQueueHistory returns a GPU's finished generation jobs with
// their timestamps, for checking that jobs on one GPU never overlap.
func (s *Server) handleGenQueueHistory(w http.ResponseWriter, r *http.Request) {
	if s.genStore(w) == nil {
		return
	}
	gpu, err := strconv.Atoi(r.URL.Query().Get("gpu"))
	if err != nil {
		http.Error(w, "gpu must be a GPU index", http.StatusBadRequest)
		return
	}
	type entry struct {
		generation.TicketInfo
		Submitted string `json:"submitted"`
		Started   string `json:"started"`
		Finished  string `json:"finished"`
	}
	const ms = "2006-01-02T15:04:05.000Z07:00"
	out := []entry{}
	for _, t := range s.genQueue.History(gpu) {
		out = append(out, entry{TicketInfo: t, Submitted: t.Submitted.Format(ms), Started: t.Started.Format(ms), Finished: t.Finished.Format(ms)})
	}
	respondJSON(w, map[string]any{"gpu": gpu, "jobs": out, "snapshot": s.genQueue.Snapshot(gpu), "now": time.Now().Format(ms)})
}
