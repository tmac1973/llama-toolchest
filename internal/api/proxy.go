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

// newProxyHandler creates a reverse proxy to the llama-server router.
// Injects per-model sampling defaults for chat completion requests.
func (s *Server) newProxyHandler() http.Handler {
	target := &url.URL{
		Scheme: "http",
		Host:   fmt.Sprintf("localhost:%d", s.cfg.LlamaPort),
	}

	proxy := httputil.NewSingleHostReverseProxy(target)
	proxy.FlushInterval = 50 * time.Millisecond
	proxy.ErrorHandler = func(w http.ResponseWriter, r *http.Request, err error) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusServiceUnavailable)
		json.NewEncoder(w).Encode(map[string]any{
			"error": map[string]any{
				"message": "llama-server router is not running",
				"type":    "server_error",
				"code":    "service_unavailable",
			},
		})
	}

	// Capture timings from non-streaming chat completion responses.
	proxy.ModifyResponse = func(resp *http.Response) error {
		if resp.Request == nil || resp.StatusCode != http.StatusOK {
			return nil
		}
		if !strings.HasSuffix(resp.Request.URL.Path, "/chat/completions") {
			return nil
		}
		ct := resp.Header.Get("Content-Type")
		if strings.Contains(ct, "text/event-stream") {
			return nil // streaming — skip for now
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
		// Inject sampling defaults for chat completions
		if r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/chat/completions") {
			body, err := io.ReadAll(r.Body)
			r.Body.Close()
			if err == nil {
				body = s.injectSamplingDefaults(body)
				r.Body = io.NopCloser(bytes.NewReader(body))
				r.ContentLength = int64(len(body))
			}
		}
		proxy.ServeHTTP(w, r)
	})
}

// injectSamplingDefaults reads the model field from the request body,
// looks up per-model sampling config, and merges defaults for any
// parameters the client didn't specify.
func (s *Server) injectSamplingDefaults(body []byte) []byte {
	var req struct {
		Model string `json:"model"`
	}
	if json.Unmarshal(body, &req) != nil || req.Model == "" {
		return body
	}

	// Look up config by model name (which is the registry ID / alias)
	cfg, err := s.registry.GetConfig(req.Model)
	if err != nil {
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
