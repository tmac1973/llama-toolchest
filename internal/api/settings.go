package api

import (
	"errors"
	"fmt"
	"html"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/atomicfile"
	"github.com/tmac1973/llama-toolchest/internal/config"
	"github.com/tmac1973/llama-toolchest/internal/modelsource"
	"gopkg.in/yaml.v3"
)

type settingsResponse struct {
	ListenAddr    string `json:"listen_addr"`
	LlamaPort     int    `json:"llama_port"`
	ProxyEndpoint string `json:"proxy_endpoint"`
	HasAPIKey     bool   `json:"has_api_key"`
	HasHFToken    bool   `json:"has_hf_token"`
	HasMSToken    bool   `json:"has_ms_token"`
	DefaultSource string `json:"default_model_source"`
	DataDir       string `json:"data_dir"`

	GenerationEnabled bool `json:"generation_enabled"`
	GenPortStart      int  `json:"gen_port_start"`
	GenPortCount      int  `json:"gen_port_count"`
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
		ProxyEndpoint: s.proxyEndpoint(),
		HasAPIKey:     s.cfg.APIKey != "",
		HasHFToken:    s.cfg.HFToken != "",
		HasMSToken:    s.cfg.MSToken != "",
		DefaultSource: s.defaultModelSource(),
		DataDir:       s.cfg.DataDir,

		GenerationEnabled: s.cfg.GenerationEnabled,
	}
	resp.GenPortStart, resp.GenPortCount = s.cfg.GenPorts(s.inContainer())

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
	// the response can carry the variable warnings and refreshed preview.
	envTouched := false
	// Switching generation backends off stops every running instance.
	wasGenerationEnabled := s.cfg.GenerationEnabled

	if isJSONRequest(r) {
		var update struct {
			LlamaPort *int    `json:"llama_port,omitempty"`
			APIKey    *string `json:"api_key,omitempty"`
			HFToken   *string `json:"hf_token,omitempty"`
			MSToken   *string `json:"ms_token,omitempty"`

			DefaultModelSource *string `json:"default_model_source,omitempty"`

			GenerationEnabled *bool `json:"generation_enabled,omitempty"`
			GenPortStart      *int  `json:"gen_port_start,omitempty"`
			GenPortCount      *int  `json:"gen_port_count,omitempty"`
		}
		if !decodeJSONBody(w, r, &update) {
			return
		}
		// Checked before anything is changed, so a rejected range leaves
		// the other fields in this request unsaved too.
		if update.GenPortStart != nil || update.GenPortCount != nil {
			if err := s.applyGenPorts(update.GenPortStart, update.GenPortCount); err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
		}
		if update.GenerationEnabled != nil {
			s.cfg.GenerationEnabled = *update.GenerationEnabled
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
		if update.MSToken != nil {
			s.cfg.MSToken = *update.MSToken
		}
		// Normalized on the way in so an unrecognized value can't be
		// persisted and then silently mean HuggingFace forever after.
		if update.DefaultModelSource != nil {
			s.cfg.DefaultModelSource = modelsource.NormalizeSource(*update.DefaultModelSource)
		}
	} else {
		r.ParseForm()
		if r.Form.Has("gen_port_start") || r.Form.Has("gen_port_count") {
			start, count, err := formGenPorts(r)
			if err == nil {
				err = s.applyGenPorts(start, count)
			}
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
		}
		// A blank secret field means "keep the saved value", so removing
		// one takes the separate clear_* checkbox. A value typed in the
		// field wins over the checkbox.
		formSecret(r, "api_key", &s.cfg.APIKey)
		formSecret(r, "hf_token", &s.cfg.HFToken)
		formSecret(r, "ms_token", &s.cfg.MSToken)
		// A select always posts a value, so presence of the field means
		// the user chose; unlike the token fields there is no "left
		// blank means leave alone" case.
		if r.Form.Has("default_model_source") {
			s.cfg.DefaultModelSource = modelsource.NormalizeSource(r.FormValue("default_model_source"))
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
		if r.Form.Has("generation_touched") {
			s.cfg.GenerationEnabled = r.FormValue("generation_enabled") == "on"
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
	if err := s.saveConfigLocked(); err != nil {
		slog.Error("settings not saved", "error", err)
		if isHTMX(r) {
			// 200, so htmx shows the message where "Settings saved." goes.
			respondHTML(w)
			fmt.Fprintf(w, `<p><mark>Settings not saved: %s</mark></p>`, html.EscapeString(err.Error()))
			return
		}
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	// A token saved here has to reach the live clients, or it would only
	// take effect on the next restart while the page cheerfully reports
	// it as set.
	s.applySourceCredentialsLocked()
	// In the background: each stop can wait for a backend to exit.
	if wasGenerationEnabled && !s.cfg.GenerationEnabled && s.genMgr != nil {
		go s.genMgr.StopAll()
	}

	if isHTMX(r) {
		respondHTML(w)
		if envTouched {
			// Variable warnings plus an out-of-band refresh of the
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
		// Out-of-band swap to update the proxy endpoint display
		fmt.Fprintf(w, `<p>Settings saved.</p><pre id="proxy-endpoint" hx-swap-oob="true">%s</pre>`, html.EscapeString(s.proxyEndpoint()))
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

// applyGenPorts validates and stores a new generation port range. A nil
// value keeps the current one. Inside a container the range is fixed by
// the published ports, so any change is refused. Callers must hold cfgMu.
func (s *Server) applyGenPorts(start, count *int) error {
	if s.inContainer() {
		return errors.New("the generation port range is fixed at 8081–8088 inside a container; change the host side with setup.sh instead")
	}
	newStart, newCount := s.cfg.GenPorts(false)
	if start != nil {
		newStart = *start
	}
	if count != nil {
		newCount = *count
	}
	if err := config.ValidateGenPorts(newStart, newCount, s.cfg.LlamaPort, s.cfg.ListenAddr); err != nil {
		return err
	}
	s.cfg.GenPortStart, s.cfg.GenPortCount = newStart, newCount
	return nil
}

// formGenPorts reads the generation port fields from the settings form.
// A field that is absent keeps its current value.
func formGenPorts(r *http.Request) (start, count *int, err error) {
	for _, f := range []struct {
		name string
		dst  **int
	}{{"gen_port_start", &start}, {"gen_port_count", &count}} {
		if !r.Form.Has(f.name) {
			continue
		}
		v, err := strconv.Atoi(strings.TrimSpace(r.FormValue(f.name)))
		if err != nil {
			return nil, nil, fmt.Errorf("%s: not a number", f.name)
		}
		*f.dst = &v
	}
	return start, count, nil
}

// formSecret applies a secret field from the settings form to dst: a
// typed value replaces it, a blank field keeps it, and the clear_<name>
// checkbox (with the field blank) removes it.
func formSecret(r *http.Request, name string, dst *string) {
	switch v := r.FormValue(name); {
	case v != "":
		*dst = v
	case r.FormValue("clear_"+name) == "1":
		*dst = ""
	}
}

// saveConfigLocked persists cfg. Callers must hold cfgMu.
//
// The file holds the HuggingFace and ModelScope tokens and the API key, so
// it is written readable by its owner only, and replaced atomically so a
// failed write leaves the previous file whole.
func (s *Server) saveConfigLocked() error {
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
	data, err := yaml.Marshal(s.cfg)
	if err != nil {
		return fmt.Errorf("saving settings: %w", err)
	}
	if err := atomicfile.WriteMode(configPath, data, 0o600); err != nil {
		return fmt.Errorf("saving settings to %s: %w", configPath, err)
	}
	return nil
}

// validateModelsDir checks that a user-supplied models directory path is
// usable. Empty is allowed — that means "use the default <DataDir>/models".
// Otherwise the path must exist, be a directory, and be readable by us.
func validateModelsDir(path string) error {
	if path == "" {
		return nil
	}
	if !filepath.IsAbs(path) {
		return errors.New("must be an absolute path")
	}
	info, err := os.Stat(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return errors.New("path does not exist")
		}
		return err
	}
	if !info.IsDir() {
		return errors.New("not a directory")
	}
	// Check readability — opening the dir is a cheap proxy.
	dh, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("not readable: %w", err)
	}
	dh.Close()
	return nil
}

// proxyEndpoint is the /v1 URL clients should use: the external URL that
// all dashboard links are built from. It used to be built from the listen
// address, which gave "http://localhost127.0.0.1:3000/v1" when that
// address included a host.
func (s *Server) proxyEndpoint() string {
	return strings.TrimRight(s.cfg.ExternalURL, "/") + "/v1"
}
