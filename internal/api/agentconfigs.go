package api

import (
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/tmac1973/llama-toolchest/internal/agentconfig"
)

// agentConfigsData feeds the agent_configs partial, the Server tab's
// "Agent configs" dialog.
type agentConfigsData struct {
	// BaseURL is the endpoint the files will point at, as shown in the
	// dialog's URL field.
	BaseURL string
	// LocalOnly is true when BaseURL names this machine only, so an agent
	// elsewhere could not reach it.
	LocalOnly bool
	// APIKey is true when the server requires a key.
	APIKey bool
	KeyEnv string
	// Models are what the files will list; empty when the router is not
	// running, and the dialog then offers nothing to download.
	Models []agentconfig.Model
	Agents []agentConfigRow
}

// agentConfigRow is one agent in the dialog.
type agentConfigRow struct {
	agentconfig.Agent
	// DownloadURL carries the base URL, so the file matches the field.
	DownloadURL string
	// Steps lists where to save the file per OS, in a fixed order.
	Steps []agentConfigStep
}

// ActivateCopy is the button that copies the agent's activation command.
func (r agentConfigRow) ActivateCopy() copyButtonData {
	return copyButtonData{Title: "Copy command", Copy: r.Activate}
}

// KeyExport is the line that sets the key's variable, with a placeholder:
// the key is write-only on the Settings page, and this dialog does not
// reveal it either.
func (d agentConfigsData) KeyExport() string {
	return "export " + d.KeyEnv + "='your API key'"
}

// KeyExportCopy is the button that copies KeyExport.
func (d agentConfigsData) KeyExportCopy() copyButtonData {
	return copyButtonData{Title: "Copy command", Copy: d.KeyExport()}
}

type agentConfigStep struct {
	OS   agentconfig.OS
	Path string
}

// agentInput assembles the generators' input: the served chat models, the
// one loaded now first (it is the agents' default), each with the context
// one conversation gets.
func (s *Server) agentInput(base string) agentconfig.Input {
	in := agentconfig.Input{
		BaseURL:     base,
		APIKey:      s.cfg.APIKey != "",
		GeneratedAt: time.Now(),
	}
	var loaded, rest []agentconfig.Model
	for _, sm := range s.servedChatModels() {
		caps := s.buildCapabilities(sm.Model, sm.Config)
		ctx, _ := caps["context_per_request"].(int)
		vision := sm.Model.HasBuiltinVision ||
			(sm.Config != nil && sm.Config.MmprojPath != "" && !sm.Config.MmprojDisabled)
		m := agentconfig.Model{
			ID:        sm.Model.PublicName(),
			Context:   ctx,
			MaxOutput: agentconfig.MaxOutputFor(ctx),
			Tools:     sm.Model.SupportsTools,
			Vision:    vision,
			Reasoning: sm.Model.EffectiveReasoning(sm.Config).Supported,
		}
		if sm.State == "loaded" {
			loaded = append(loaded, m)
		} else {
			rest = append(rest, m)
		}
	}
	in.Models = append(loaded, rest...)
	return in
}

// agentBaseURL is the endpoint the files point at: the dialog's URL field
// when given, else the configured external URL, else the address the
// browser reached this page on. A missing /v1 is added, so pasting the
// server's address works.
func (s *Server) agentBaseURL(r *http.Request) (string, error) {
	raw := strings.TrimSpace(r.URL.Query().Get("base"))
	if raw == "" {
		raw = s.cfg.ExternalURL
	}
	if raw == "" && r.Host != "" {
		scheme := "http"
		if r.TLS != nil {
			scheme = "https"
		}
		raw = scheme + "://" + r.Host
	}
	u, err := url.Parse(raw)
	if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" {
		return "", fmt.Errorf("not an http(s) URL: %q", raw)
	}
	u.RawQuery, u.Fragment = "", ""
	u.Path = strings.TrimRight(u.Path, "/")
	if !strings.HasSuffix(u.Path, "/v1") {
		u.Path += "/v1"
	}
	return u.String(), nil
}

// isLocalOnly reports a base URL only this machine can reach.
func isLocalOnly(base string) bool {
	u, err := url.Parse(base)
	if err != nil {
		return false
	}
	switch h := u.Hostname(); h {
	case "localhost", "127.0.0.1", "::1", "0.0.0.0":
		return true
	default:
		return strings.HasPrefix(h, "127.")
	}
}

func (s *Server) handleAgentConfigs(w http.ResponseWriter, r *http.Request) {
	base, err := s.agentBaseURL(r)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	in := s.agentInput(base)
	data := agentConfigsData{
		BaseURL:   base,
		LocalOnly: isLocalOnly(base),
		APIKey:    in.APIKey,
		KeyEnv:    agentconfig.KeyEnv,
		Models:    in.Models,
	}
	for _, a := range agentconfig.Agents {
		row := agentConfigRow{
			Agent:       a,
			DownloadURL: "/api/agent-configs/" + a.ID + "?base=" + url.QueryEscape(base),
		}
		for _, os := range []agentconfig.OS{agentconfig.Linux, agentconfig.MacOS, agentconfig.Windows} {
			if p := a.SavePath[os]; p != "" {
				row.Steps = append(row.Steps, agentConfigStep{OS: os, Path: p})
			}
		}
		data.Agents = append(data.Agents, row)
	}
	respondHTML(w)
	s.renderPartial(w, "agent_configs", data)
}

func (s *Server) handleAgentConfigDownload(w http.ResponseWriter, r *http.Request) {
	a, ok := agentconfig.Find(chi.URLParam(r, "agent"))
	if !ok {
		http.Error(w, "unknown agent", http.StatusNotFound)
		return
	}
	base, err := s.agentBaseURL(r)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	in := s.agentInput(base)
	if len(in.Models) == 0 {
		http.Error(w, "the server is not serving any chat models: start it on the Server tab first", http.StatusConflict)
		return
	}
	body, err := a.Generate(in)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	ctype := "application/json"
	if !strings.HasSuffix(a.FileName, ".json") {
		ctype = "text/plain"
	}
	w.Header().Set("Content-Type", ctype+"; charset=utf-8")
	w.Header().Set("Content-Disposition", fmt.Sprintf("attachment; filename=%q", a.FileName))
	w.Header().Set("Cache-Control", "no-store")
	w.Write(body)
}
