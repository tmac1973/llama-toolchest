package api

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"mime"
	"mime/multipart"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/models"
)

// autoLoadTimeout caps how long an auto-load triggered by /v1/chat/completions
// (or any proxied POST with a `model` field) is allowed to take. Cold loads
// of a 35B model can exceed 30 seconds when reading from disk, so we set
// this generously and let the client wait — the alternative is a 400 from
// the upstream router, which is what bug 2 was about in the first place.
const autoLoadTimeout = 90 * time.Second

// autoLoadPollInterval is how often we re-poll the router for a model's
// load state while waiting for an auto-load to complete.
const autoLoadPollInterval = 250 * time.Millisecond

// newProxyHandler creates a reverse proxy to the llama-server router.
// Injects per-model sampling defaults for chat completion requests.
func (s *Server) newProxyHandler() http.Handler {
	target := &url.URL{
		Scheme: "http",
		Host:   fmt.Sprintf("localhost:%d", s.llamaPort()),
	}

	proxy := httputil.NewSingleHostReverseProxy(target)
	proxy.FlushInterval = 50 * time.Millisecond
	proxy.ErrorHandler = func(w http.ResponseWriter, r *http.Request, err error) {
		writeProxyError(w, http.StatusServiceUnavailable, "service_unavailable",
			"llama-server router is not running")
	}

	// Capture timings from chat completion responses — both JSON and SSE streams.
	proxy.ModifyResponse = func(resp *http.Response) error {
		if resp.Request == nil || resp.StatusCode != http.StatusOK {
			return nil
		}
		if !strings.HasSuffix(resp.Request.URL.Path, "/chat/completions") {
			return nil
		}

		ct := resp.Header.Get("Content-Type")
		if strings.Contains(ct, "text/event-stream") {
			// Wrap the body so we can scan SSE chunks for the final
			// timings event without blocking delivery to the client.
			resp.Body = newSSETimingCapture(resp.Body, s.captureTimings)
			return nil
		}

		body, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			return nil
		}
		resp.Body = io.NopCloser(bytes.NewReader(body))

		var result struct {
			Model   string         `json:"model"`
			Timings map[string]any `json:"timings"`
		}
		if json.Unmarshal(body, &result) == nil && result.Timings != nil && result.Model != "" {
			go s.captureTimings(result.Model, result.Timings)
		}
		return nil
	}

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// POSTs can carry a "model" field that would 400 upstream if the
		// named model is enabled in the registry but not currently
		// resident in the router, or that the router would load itself,
		// past the room checks. Find the model, ensure it is loaded, then
		// forward. A JSON body is read whole; a multipart form (a /tts
		// speaker reference, an audio transcription) only up to its
		// "model" field, and the rest streams through.
		if r.Method != http.MethodPost {
			proxy.ServeHTTP(w, r)
			return
		}
		switch {
		case isJSONContentType(r.Header.Get("Content-Type")):
			body, err := io.ReadAll(r.Body)
			r.Body.Close()
			if err == nil {
				end, ok := s.loadForProxy(w, r, body)
				if !ok {
					return
				}
				defer end()
				if strings.HasSuffix(r.URL.Path, "/chat/completions") {
					body = s.injectSamplingDefaults(body)
				}
				r.Body = io.NopCloser(bytes.NewReader(body))
				r.ContentLength = int64(len(body))
			}
		case isMultipartForm(r):
			if model := multipartModel(r, multipartModelLimit); model != "" {
				body, _ := json.Marshal(map[string]string{"model": model})
				end, ok := s.loadForProxy(w, r, body)
				if !ok {
					return
				}
				defer end()
			}
		}
		proxy.ServeHTTP(w, r)
	})
}

// loadForProxy loads and counts the model a proxied request names (see
// loadAndCountRequest). When that fails it answers the request and
// returns false.
func (s *Server) loadForProxy(w http.ResponseWriter, r *http.Request, body []byte) (end func(), ok bool) {
	end, err := s.loadAndCountRequest(r.Context(), body)
	if err == nil {
		return end, true
	}
	if r.Context().Err() != nil {
		return end, false // the client went away
	}
	var fitErr *genLLMFitError
	if errors.As(err, &fitErr) {
		writeProxyError(w, http.StatusInsufficientStorage, "insufficient_vram", err.Error())
	} else {
		writeProxyError(w, http.StatusServiceUnavailable, "auto_load_failed", err.Error())
	}
	return end, false
}

// multipartModelLimit caps how much of a multipart form is held in
// memory while looking for its "model" field. Files sent before the field
// count towards it.
const multipartModelLimit = 64 << 20

// multipartModel returns the "model" field of a multipart form request,
// reading the body only as far as that field. What it read is put back in
// front of the rest, so the body is forwarded unchanged. "" when the form
// has no model field before limit bytes, or cannot be read; the router
// then gets the request as it is.
func multipartModel(r *http.Request, limit int64) string {
	_, params, err := mime.ParseMediaType(r.Header.Get("Content-Type"))
	if err != nil || params["boundary"] == "" {
		return ""
	}
	var read bytes.Buffer
	orig := r.Body
	defer func() {
		r.Body = struct {
			io.Reader
			io.Closer
		}{io.MultiReader(&read, orig), orig}
	}()
	mr := multipart.NewReader(io.TeeReader(io.LimitReader(orig, limit), &read), params["boundary"])
	for {
		part, err := mr.NextPart()
		if err != nil {
			return ""
		}
		if part.FormName() == "model" && part.FileName() == "" {
			value, err := io.ReadAll(io.LimitReader(part, 1024))
			if err != nil {
				return ""
			}
			return strings.TrimSpace(string(value))
		}
		if _, err := io.Copy(io.Discard, part); err != nil {
			return ""
		}
	}
}

// loadAttempts caps how many times a request loads its model again
// after an unload for a generation backend got in between the load and
// the request being counted.
const loadAttempts = 4

// loadAndCountRequest makes sure the model a request names is loaded,
// then counts the request (llmUse) until end is called, so an unload for
// a generation backend waits for its reply. The request is counted only
// once the model is loaded: while it waits for the load — which may wait
// for a generation job's turn on the GPU — it holds nothing a job's
// unload waits for, so neither waits on the other. When an unload got in
// between the load and the count, the model is loaded again.
func (s *Server) loadAndCountRequest(ctx context.Context, body []byte) (end func(), err error) {
	noop := func() {}
	id := s.requestModelID(body)
	for attempt := 0; ; attempt++ {
		if err := s.ensureModelLoadedForRequest(ctx, body); err != nil {
			return noop, err
		}
		if id == "" {
			return noop, nil
		}
		end, err := s.llmUse.begin(ctx, id)
		if err != nil {
			return noop, err
		}
		if attempt+1 >= loadAttempts || s.modelLoaded(id) {
			// After an eviction unloaded it: the conversations' prompt
			// caches, before this request goes on. Restored once counted:
			// a restore is a request to the model, and one reaching the
			// router after an unload loads the model again behind every
			// room check.
			if m, _ := s.registry.Get(id); m != nil {
				s.restorePromptCache(ctx, m, s.registry.RouterName(id))
			}
			return end, nil
		}
		end()
	}
}

// modelLoaded reports whether the router has model id loaded.
func (s *Server) modelLoaded(id string) bool {
	m, err := s.registry.Get(id)
	if err != nil || m == nil {
		return false
	}
	state, _ := s.lookupRouterState(s.registry.RouterName(id), m)
	return state == "loaded"
}

// isJSONContentType returns true for application/json content types,
// tolerating optional charset suffixes.
func isJSONContentType(ct string) bool {
	if ct == "" {
		return false
	}
	if i := strings.Index(ct, ";"); i >= 0 {
		ct = ct[:i]
	}
	return strings.EqualFold(strings.TrimSpace(ct), "application/json")
}

// ensureModelLoadedForRequest inspects a proxied request body for a
// "model" field. If the model is one we know but the router does not
// currently have loaded, it triggers a load and waits for the router to
// report the model ready. Errors here surface as 5xx to the client; missing
// "model" or unknown models are ignored so the upstream router can produce
// its own (more specific) error.
func (s *Server) ensureModelLoadedForRequest(ctx context.Context, body []byte) error {
	if len(body) == 0 {
		return nil
	}
	regModel, _ := s.modelFromBody(body)
	if regModel == nil {
		return nil // unknown — let the upstream return its own error
	}

	routerName := s.registry.RouterName(regModel.ID)

	// Checked first: a stopped router knows no models, and the "not in the
	// router preset" error below would send the user looking in the wrong
	// place.
	if !s.process.IsRunning() {
		return errors.New("the router is not running; start it on the Server page")
	}
	state, knownToRouter := s.lookupRouterState(routerName, regModel)
	if state == "loaded" {
		return nil
	}
	if !knownToRouter {
		// The router doesn't know this model at all — typically because
		// preset.ini was generated before the model was added or before
		// section-collision was fixed. A restart would pick it up; until
		// then there's nothing /models/load can do for us.
		return fmt.Errorf("model %q is not in the router preset; restart the router to pick it up", regModel.PublicName())
	}
	if state != "loading" {
		// With eviction on this stops the generation backends in the way,
		// and keeps their GPUs until the load is done.
		release, err := s.makeRoomForLLM(ctx, regModel)
		if err != nil {
			return err
		}
		defer release()
		if err := s.process.LoadModel(routerName); err != nil {
			return fmt.Errorf("auto-load %s: %w", regModel.PublicName(), err)
		}
		slog.Info("auto-loading model for inference request", "model", regModel.PublicName(), "router_name", routerName)
	}
	return s.waitForModelLoaded(ctx, routerName, regModel)
}

// lookupRouterState returns the router's current state for a model, plus
// whether the router knows about the model at all. The router may report
// the model under its primary ID, public name, or any alias, so we check
// all three before declaring it unknown.
func (s *Server) lookupRouterState(routerName string, m *models.Model) (string, bool) {
	if !s.process.IsRunning() {
		return "", false
	}
	loaded, err := s.process.ListModels()
	if err != nil {
		return "", false
	}
	for _, lm := range loaded {
		if lm.ID == routerName || lm.Model == routerName {
			return lm.Status.Value, true
		}
		for _, a := range lm.Aliases {
			if a == routerName || a == m.ID || a == m.PublicName() {
				return lm.Status.Value, true
			}
		}
	}
	return "", false
}

// waitForModelLoaded polls the router until the named model reports
// "loaded", the request context is cancelled, or autoLoadTimeout elapses.
func (s *Server) waitForModelLoaded(ctx context.Context, routerName string, m *models.Model) error {
	deadline := time.Now().Add(autoLoadTimeout)
	for {
		state, _ := s.lookupRouterState(routerName, m)
		switch state {
		case "loaded":
			return nil
		case "":
			return fmt.Errorf("model %q disappeared from router during load", m.PublicName())
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("timed out after %s waiting for model %q to load", autoLoadTimeout, m.PublicName())
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(autoLoadPollInterval):
		}
	}
}

// writeProxyError writes an OpenAI-shaped error object with the given code.
func writeProxyError(w http.ResponseWriter, status int, code, message string) {
	openAIError(w, status, "server_error", code, message)
}

// sseTimingCapture wraps a streaming SSE response body, passing bytes
// through to the client while scanning for the final chunk that carries
// llama.cpp's `timings` field. When seen, captureFn is invoked once.
type sseTimingCapture struct {
	orig     io.ReadCloser
	capture  func(modelID string, timings map[string]any)
	buf      bytes.Buffer
	model    string
	captured bool
}

func newSSETimingCapture(orig io.ReadCloser, capture func(string, map[string]any)) *sseTimingCapture {
	return &sseTimingCapture{orig: orig, capture: capture}
}

func (s *sseTimingCapture) Read(p []byte) (int, error) {
	n, err := s.orig.Read(p)
	if n > 0 && !s.captured {
		s.buf.Write(p[:n])
		s.scan()
	}
	return n, err
}

func (s *sseTimingCapture) scan() {
	// SSE events are separated by a blank line ("\n\n").
	for {
		data := s.buf.Bytes()
		idx := bytes.Index(data, []byte("\n\n"))
		if idx < 0 {
			return
		}
		event := string(data[:idx])
		s.buf.Next(idx + 2)

		for _, line := range strings.Split(event, "\n") {
			line = strings.TrimPrefix(line, "data: ")
			line = strings.TrimPrefix(line, "data:") // tolerate no-space form
			line = strings.TrimSpace(line)
			if line == "" || line == "[DONE]" {
				continue
			}
			var chunk struct {
				Model   string         `json:"model"`
				Timings map[string]any `json:"timings"`
			}
			if json.Unmarshal([]byte(line), &chunk) != nil {
				continue
			}
			if s.model == "" && chunk.Model != "" {
				s.model = chunk.Model
			}
			if chunk.Timings != nil && s.model != "" {
				go s.capture(s.model, chunk.Timings)
				s.captured = true
				return
			}
		}
	}
}

func (s *sseTimingCapture) Close() error {
	return s.orig.Close()
}

// injectSamplingDefaults reads the model field from the request body,
// looks up per-model sampling config, and merges defaults for any
// parameters the client didn't specify.
func (s *Server) injectSamplingDefaults(body []byte) []byte {
	// Clients name a model by its registry ID, public name (what
	// /v1/models lists) or an alias; all three find its config.
	_, cfg := s.modelFromBody(body)
	if cfg == nil {
		return body
	}

	overrides := cfg.SamplingOverrides()
	if len(overrides) == 0 {
		return body
	}

	var reqMap map[string]any
	if json.Unmarshal(body, &reqMap) != nil {
		return body
	}

	for k, v := range overrides {
		if _, exists := reqMap[k]; !exists {
			reqMap[k] = v
		}
	}

	modified, err := json.Marshal(reqMap)
	if err != nil {
		return body
	}
	return modified
}
