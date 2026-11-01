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
	"sync"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/forks"
	"github.com/tmac1973/llama-toolchest/internal/process"
)

// forkProxies holds one reverse proxy per fork instance port, made on
// first use.
type forkProxies struct {
	mu     sync.Mutex
	byPort map[int]*httputil.ReverseProxy
}

func (p *forkProxies) get(port int) *httputil.ReverseProxy {
	p.mu.Lock()
	defer p.mu.Unlock()
	if rp, ok := p.byPort[port]; ok {
		return rp
	}
	if p.byPort == nil {
		p.byPort = map[int]*httputil.ReverseProxy{}
	}
	rp := httputil.NewSingleHostReverseProxy(&url.URL{Scheme: "http", Host: fmt.Sprintf("127.0.0.1:%d", port)})
	// As the router proxy does, so streamed tokens reach the client as
	// they come.
	rp.FlushInterval = 50 * time.Millisecond
	rp.ErrorHandler = func(w http.ResponseWriter, r *http.Request, err error) {
		writeProxyError(w, http.StatusServiceUnavailable, "service_unavailable",
			"the fork instance did not answer; it may have stopped")
	}
	p.byPort[port] = rp
	return rp
}

// forkForModel returns the fork instance a request's model names, while
// forks are on. Only the instance name is matched: it is the one name a
// fork instance answers to.
func (s *Server) forkForModel(model string) (forks.Instance, bool) {
	if model == "" || !s.forksOn() || s.forks == nil {
		return forks.Instance{}, false
	}
	inst, err := s.forks.Instance(model)
	return inst, err == nil
}

// bodyModel reads the "model" field of a JSON request body.
func bodyModel(body []byte) string {
	var req struct {
		Model string `json:"model"`
	}
	if json.Unmarshal(body, &req) != nil {
		return ""
	}
	return req.Model
}

// serveFork sends a request to a fork instance: a 503 that says why when
// it is stopped or still loading, otherwise the request with the fork's
// sampling defaults added to a chat request.
func (s *Server) serveFork(w http.ResponseWriter, r *http.Request, inst forks.Instance, body []byte) {
	switch st := s.forkMgr.Status(inst.ID); st.State {
	case process.StateRunning:
	case process.StateStarting:
		w.Header().Set("Retry-After", "30")
		writeProxyError(w, http.StatusServiceUnavailable, "model_loading",
			fmt.Sprintf("Fork instance %s is loading. Try again in a few minutes.", inst.ID))
		return
	default:
		writeProxyError(w, http.StatusServiceUnavailable, "model_not_running",
			fmt.Sprintf("Fork instance %s is not running. Start it on the Server page.", inst.ID))
		return
	}
	if strings.HasSuffix(r.URL.Path, "/chat/completions") {
		if f, ok := findFork(inst.ForkID); ok {
			body = addSamplingDefaults(body, f.SamplingDefaults)
		}
	}
	r.Body = io.NopCloser(bytes.NewReader(body))
	r.ContentLength = int64(len(body))
	s.forkProxy.get(inst.Port).ServeHTTP(w, r)
}

// addSamplingDefaults adds each default the request does not set itself.
func addSamplingDefaults(body []byte, defaults map[string]any) []byte {
	if len(defaults) == 0 {
		return body
	}
	var req map[string]any
	if json.Unmarshal(body, &req) != nil {
		return body
	}
	changed := false
	for k, v := range defaults {
		if _, ok := req[k]; !ok {
			req[k] = v
			changed = true
		}
	}
	if !changed {
		return body
	}
	out, err := json.Marshal(req)
	if err != nil {
		return body
	}
	return out
}

// forkV1Models lists the fork instances that are starting or running, as
// chat models (meta.kind "text"), with meta.fork naming the fork and
// meta.state "running" or "loading". Empty while forks are off.
func (s *Server) forkV1Models() []map[string]any {
	if !s.forksOn() || s.forks == nil {
		return nil
	}
	var out []map[string]any
	for _, inst := range s.forks.Instances() {
		if obj := s.forkV1Model(inst); obj != nil {
			out = append(out, obj)
		}
	}
	return out
}

// forkV1Model is one fork instance as a /v1/models entry; nil while it is
// stopped.
func (s *Server) forkV1Model(inst forks.Instance) map[string]any {
	st := s.forkMgr.Status(inst.ID)
	state := ""
	switch st.State {
	case process.StateRunning:
		state = "running"
	case process.StateStarting:
		state = "loading"
	default:
		return nil
	}
	return map[string]any{
		"id":       inst.ID,
		"object":   "model",
		"created":  st.StartedAt.Unix(),
		"owned_by": "fork:" + inst.ForkID,
		"meta": map[string]any{
			"kind":  "text",
			"fork":  inst.ForkID,
			"state": state,
		},
	}
}

// forkLoadedModel is a running or loading fork instance as an
// /api/service/loaded-models entry, so a client that reads only that
// list (Haruspex does) sees the fork as a model it can use; ok is false
// while it is stopped. A running instance's capabilities come from its
// own /props.
func (s *Server) forkLoadedModel(inst forks.Instance) (status string, caps map[string]any, ok bool) {
	switch s.forkMgr.Status(inst.ID).State {
	case process.StateRunning:
		return "loaded", forkPropsCapabilities(inst.Port), true
	case process.StateStarting:
		return "loading", nil, true
	}
	return "", nil, false
}

// forkPropsCapabilities reads the context layout and vision support from
// a fork's llama-server /props, in the shape of buildCapabilities. It is
// nil when /props does not answer in time or cannot be read; the client
// then uses its own defaults.
func forkPropsCapabilities(port int) map[string]any {
	client := http.Client{Timeout: 2 * time.Second}
	resp, err := client.Get(fmt.Sprintf("http://127.0.0.1:%d/props", port))
	if err != nil {
		return nil
	}
	defer resp.Body.Close()
	var props struct {
		Settings struct {
			NCtx int `json:"n_ctx"`
		} `json:"default_generation_settings"`
		TotalSlots int `json:"total_slots"`
		Modalities struct {
			Vision bool `json:"vision"`
		} `json:"modalities"`
	}
	if resp.StatusCode != http.StatusOK || json.NewDecoder(resp.Body).Decode(&props) != nil || props.Settings.NCtx <= 0 {
		return nil
	}
	slots := max(props.TotalSlots, 1)
	return map[string]any{
		"schema_version": CapabilitiesSchemaVersion,
		// n_ctx in default_generation_settings is one slot's context.
		"context_size":        props.Settings.NCtx * slots,
		"parallel":            slots,
		"context_shared":      false,
		"context_per_request": props.Settings.NCtx,
		"vision":              props.Modalities.Vision,
		"chat":                true,
	}
}
