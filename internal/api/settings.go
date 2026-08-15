package api

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/config"
	"gopkg.in/yaml.v3"
)

type settingsResponse struct {
	ListenAddr    string `json:"listen_addr"`
	LlamaPort     int    `json:"llama_port"`
	ProxyEndpoint string `json:"proxy_endpoint"`
	HasAPIKey     bool   `json:"has_api_key"`
	HasHFToken    bool   `json:"has_hf_token"`
	DataDir       string `json:"data_dir"`
}

type connectionTestResult struct {
	OK      bool   `json:"ok"`
	Error   string `json:"error,omitempty"`
	Latency string `json:"latency"`
}

func (s *Server) handleGetSettings(w http.ResponseWriter, r *http.Request) {
	resp := settingsResponse{
		ListenAddr:    s.cfg.ListenAddr,
		LlamaPort:     s.cfg.LlamaPort,
		ProxyEndpoint: fmt.Sprintf("http://localhost%s/v1", s.cfg.ListenAddr),
		HasAPIKey:     s.cfg.APIKey != "",
		HasHFToken:    s.cfg.HFToken != "",
		DataDir:       s.cfg.DataDir,
	}

	if isHTMX(r) {
		respondHTML(w)
		w.Write([]byte("<p>Settings saved.</p>"))
		return
	}

	respondJSON(w, resp)
}

func (s *Server) handleUpdateSettings(w http.ResponseWriter, r *http.Request) {
	// Held across the whole mutate-then-persist sequence: the benchmark
	// queue writes cfg.ActiveBuild from its own goroutine, and two
	// interleaved saveConfig calls could otherwise serialize a
	// half-updated struct and lose one side's changes.
	s.cfgMu.Lock()
	defer s.cfgMu.Unlock()

	// Set when this request changed the runtime environment section, so
	// the response can carry the footgun warnings and refreshed preview.
	envTouched := false

	if r.Header.Get("Content-Type") == "application/json" {
		var update struct {
			LlamaPort *int    `json:"llama_port,omitempty"`
			APIKey    *string `json:"api_key,omitempty"`
			HFToken   *string `json:"hf_token,omitempty"`
		}
		if err := json.NewDecoder(r.Body).Decode(&update); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if update.LlamaPort != nil {
			s.cfg.LlamaPort = *update.LlamaPort
		}
		if update.APIKey != nil {
			s.cfg.APIKey = *update.APIKey
		}
		if update.HFToken != nil {
			s.cfg.HFToken = *update.HFToken
		}
	} else {
		r.ParseForm()
		if v := r.FormValue("api_key"); v != "" {
			s.cfg.APIKey = v
		}
		if v := r.FormValue("hf_token"); v != "" {
			s.cfg.HFToken = v
		}
		if r.Form.Has("external_url") {
			s.cfg.ExternalURL = r.FormValue("external_url")
		}
		if r.Form.Has("active_build") {
			s.cfg.ActiveBuild = r.FormValue("active_build")
		}
		if r.Form.Has("models_max") {
			if v, err := strconv.Atoi(r.FormValue("models_max")); err == nil {
				s.cfg.ModelsMax = v
			}
		}
		// Runtime env. Presence of the marker means the form included
		// this section, so a blanked field clears the value rather than
		// being ignored. Fields absent from the form entirely (curated
		// options hidden by the backend filter) keep their saved value —
		// though the filter already keeps any set variable visible, so
		// this is a second line of defense, not the mechanism.
		if r.Form.Has("runtime_env_touched") {
			env := map[string]string{}
			for _, opt := range config.RuntimeEnvOptions() {
				if r.Form.Has("env_" + opt.Name) {
					if v := strings.TrimSpace(r.FormValue("env_" + opt.Name)); v != "" {
						env[opt.Name] = v
					}
				} else if v := strings.TrimSpace(s.cfg.RuntimeEnv[opt.Name]); v != "" {
					env[opt.Name] = v
				}
			}
			extra := s.cfg.RuntimeEnvExtra
			if r.Form.Has("runtime_env_extra") {
				extra = r.FormValue("runtime_env_extra")
			}
			set := config.EnvSet{Curated: env, Extra: extra}
			if err := set.Validate(); err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			s.cfg.RuntimeEnv = env
			s.cfg.RuntimeEnvExtra = extra
			envTouched = true
		}
		if r.Form.Has("auto_start_touched") {
			s.cfg.AutoStart = r.FormValue("auto_start") == "on"
		}
		if r.Form.Has("models_dir") {
			newDir := strings.TrimSpace(r.FormValue("models_dir"))
			if newDir != s.cfg.ModelsDir {
				if err := validateModelsDir(newDir); err != nil {
					http.Error(w, fmt.Sprintf("models_dir: %s", err), http.StatusBadRequest)
					return
				}
				s.cfg.ModelsDir = newDir
			}
		}
	}

	// Persist config
	s.saveConfigLocked()

	if isHTMX(r) {
		respondHTML(w)
		if envTouched {
			// Footgun warnings plus an out-of-band refresh of the
			// effective-environment preview.
			s.renderPartial(w, "runtime_env_status", struct {
				Warnings  []string
				Effective []envLine
			}{
				Warnings:  s.cfg.EnvSet().Warnings(),
				Effective: s.effectiveEnvLines(),
			})
			return
		}
		proxyEndpoint := strings.TrimRight(s.cfg.ExternalURL, "/") + "/v1"
		// Out-of-band swap to update the proxy endpoint display
		fmt.Fprintf(w, `<p>Settings saved.</p><pre id="proxy-endpoint" hx-swap-oob="true">%s</pre>`, proxyEndpoint)
		return
	}

	s.handleGetSettings(w, r)
}

func (s *Server) handleTestConnection(w http.ResponseWriter, r *http.Request) {
	url := fmt.Sprintf("http://localhost:%d/v1/models", s.cfg.LlamaPort)

	client := &http.Client{Timeout: 5 * time.Second}
	start := time.Now()
	resp, err := client.Get(url)
	latency := time.Since(start)

	result := connectionTestResult{
		Latency: latency.Truncate(time.Millisecond).String(),
	}

	if err != nil {
		result.Error = err.Error()
	} else {
		resp.Body.Close()
		if resp.StatusCode == http.StatusOK {
			result.OK = true
		} else {
			result.Error = fmt.Sprintf("HTTP %d", resp.StatusCode)
		}
	}

	if isHTMX(r) {
		respondHTML(w)
		s.renderPartial(w, "connection_result", result)
		return
	}

	respondJSON(w, result)
}

// saveConfigLocked persists cfg. Callers must hold cfgMu.
func (s *Server) saveConfigLocked() {
	// Write back to the same path the config was loaded from, so the next
	// startup actually sees what the user just changed. Old behavior wrote
	// to <DataDir>/config/llama-toolchest.yaml unconditionally, which only
	// matched the read path in container mode — host installs read from
	// ~/.config/... or /etc/..., so saved settings silently vanished on
	// the next restart.
	configPath := s.configPath
	if configPath == "" {
		// Defensive fallback for callers that didn't supply a path
		// (e.g. tests). Same shape as the legacy default.
		configPath = filepath.Join(s.cfg.DataDir, "config", "llama-toolchest.yaml")
	}
	os.MkdirAll(filepath.Dir(configPath), 0o755)

	data, err := yaml.Marshal(s.cfg)
	if err != nil {
		return
	}
	os.WriteFile(configPath, data, 0o644)
}

// validateModelsDir checks that a user-supplied models directory path is
// usable. Empty is allowed — that means "use the default <DataDir>/models".
// Otherwise the path must exist, be a directory, and be readable by us.
func validateModelsDir(path string) error {
	if path == "" {
		return nil
	}
	if !filepath.IsAbs(path) {
		return fmt.Errorf("must be an absolute path")
	}
	info, err := os.Stat(path)
	if err != nil {
		if os.IsNotExist(err) {
			return fmt.Errorf("path does not exist")
		}
		return err
	}
	if !info.IsDir() {
		return fmt.Errorf("not a directory")
	}
	// Check readability — opening the dir is a cheap proxy.
	dh, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("not readable: %w", err)
	}
	dh.Close()
	return nil
}

// saveConfig persists cfg, taking the config lock.
func (s *Server) saveConfig() {
	s.cfgMu.Lock()
	defer s.cfgMu.Unlock()
	s.saveConfigLocked()
}
