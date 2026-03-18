package api

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"
	"time"
)

// newProxyHandler creates a reverse proxy to the llama-server OpenAI API.
// For chat completion requests, it injects per-model sampling defaults for
// any parameters not already specified by the client.
func (s *Server) newProxyHandler() http.Handler {
	target := &url.URL{
		Scheme: "http",
		Host:   fmt.Sprintf("localhost:%d", s.cfg.LlamaPort),
	}

	proxy := httputil.NewSingleHostReverseProxy(target)

	// Flush immediately for streaming responses (SSE chat completions).
	proxy.FlushInterval = 50 * time.Millisecond

	proxy.ErrorHandler = func(w http.ResponseWriter, r *http.Request, err error) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusServiceUnavailable)
		fmt.Fprintf(w, `{"error":{"message":"llama-server is not running","type":"server_error","code":"service_unavailable"}}`)
	}

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Only inject sampling defaults for chat completion POSTs.
		if r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/chat/completions") {
			s.injectSamplingDefaults(r)
		}
		proxy.ServeHTTP(w, r)
	})
}

// injectSamplingDefaults reads the request body, merges any configured
// sampling parameters that the client didn't set, and replaces the body.
func (s *Server) injectSamplingDefaults(r *http.Request) {
	modelID := s.activeModelID
	if modelID == "" {
		return
	}

	cfg, err := s.registry.GetConfig(modelID)
	if err != nil {
		return
	}

	overrides := cfg.SamplingOverrides()
	if len(overrides) == 0 {
		return
	}

	body, err := io.ReadAll(r.Body)
	r.Body.Close()
	if err != nil {
		r.Body = io.NopCloser(bytes.NewReader(body))
		return
	}

	var reqMap map[string]any
	if err := json.Unmarshal(body, &reqMap); err != nil {
		r.Body = io.NopCloser(bytes.NewReader(body))
		return
	}

	// Only set parameters that the client didn't already provide.
	for k, v := range overrides {
		if _, exists := reqMap[k]; !exists {
			reqMap[k] = v
		}
	}

	modified, err := json.Marshal(reqMap)
	if err != nil {
		r.Body = io.NopCloser(bytes.NewReader(body))
		return
	}

	r.Body = io.NopCloser(bytes.NewReader(modified))
	r.ContentLength = int64(len(modified))
}
