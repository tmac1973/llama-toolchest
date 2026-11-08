package api

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/config"
	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/process"
)

// The virtual model name (config.DefaultModelConfig) is a name clients
// send instead of a real model. The proxy puts the real model in its
// place before anything else reads the request (rewriteDefaultModel), so
// fork routing, loading, counting, sampling defaults and the system
// prompt work as for a request that named the model.

// chatPathSuffixes are the request paths the virtual model name applies
// to: the chat and text completion APIs, with or without /v1.
var chatPathSuffixes = []string{"/chat/completions", "/completions", "/messages", "/responses"}

// isChatPath reports whether path ends in one of chatPathSuffixes.
func isChatPath(path string) bool {
	for _, suffix := range chatPathSuffixes {
		if strings.HasSuffix(path, suffix) {
			return true
		}
	}
	return false
}

// noteChatUse records router model id as the chat model used last, for
// the virtual model name after a restart. Forks are not recorded: one
// does not start on request, so after a restart it could only answer 503.
func (s *Server) noteChatUse(id string) {
	if s.defaultState != nil {
		s.defaultState.note(id, false, time.Now())
	}
}

// Reasons a defaultTarget was picked.
const (
	defaultReasonPinned   = "pinned"
	defaultReasonLoaded   = "loaded"
	defaultReasonLastUsed = "last used"
)

// defaultTarget is the real model the virtual name resolves to now.
type defaultTarget struct {
	Name   string // what goes in the request's model field
	ID     string // registry ID, or fork instance ID when Fork
	Fork   bool
	Reason string // defaultReasonPinned, defaultReasonLoaded or defaultReasonLastUsed
}

// defaultChatModel returns registry model id when it can be the virtual
// name's target: a chat model clients can use (enabled, not a helper or
// draft head). Nil otherwise.
func (s *Server) defaultChatModel(id string) *models.Model {
	m, err := s.registry.Get(id)
	if err != nil || m == nil || m.Kind() != models.KindChat {
		return nil
	}
	cfg, _ := s.registry.GetConfig(id)
	if !advertisedToClients(m, cfg) {
		return nil
	}
	return m
}

// forkInstanceExists reports whether fork instance id is in the fork
// store, whether or not forks are switched on. known is false when there
// is no fork store, so nothing can be said.
func (s *Server) forkInstanceExists(id string) (exists, known bool) {
	if s.forks == nil {
		return false, false
	}
	_, err := s.forks.Instance(id)
	return err == nil, true
}

// resolveDefaultModel picks the virtual name's target: the pinned model;
// else the loaded chat model or running fork used last; else the router
// chat model used last before a restart. ok is false when there is none.
// It never loads a model and never writes the config; clearStaleDefaultPin
// does the pin cleanup.
func (s *Server) resolveDefaultModel() (t defaultTarget, ok bool) {
	cfg := s.cfgCopy()
	if pin := cfg.DefaultModel.Pin; pin != "" {
		if id, fork := strings.CutPrefix(pin, config.DefaultModelForkPrefix); fork {
			if exists, _ := s.forkInstanceExists(id); exists && cfg.ForksEnabled {
				return defaultTarget{Name: id, ID: id, Fork: true, Reason: defaultReasonPinned}, true
			}
		} else if m := s.defaultChatModel(pin); m != nil {
			return defaultTarget{Name: m.PublicName(), ID: m.ID, Reason: defaultReasonPinned}, true
		}
	}

	last := s.lastChatModel()
	var best defaultTarget
	var bestAt time.Time
	found := false
	// beats breaks a tie in last-use time (no request since the start):
	// the model used last before the restart wins, then the lower name,
	// so the choice is stable.
	beats := func(t, than defaultTarget) bool {
		tLast, thanLast := !t.Fork && t.ID == last, !than.Fork && than.ID == last
		if tLast != thanLast {
			return tLast
		}
		return t.Name < than.Name
	}
	consider := func(t defaultTarget, at time.Time) {
		if found && (at.Before(bestAt) || at.Equal(bestAt) && !beats(t, best)) {
			return
		}
		best, bestAt, found = t, at, true
	}
	if s.process != nil && s.process.IsRunning() {
		if loaded, err := s.process.ListModels(); err == nil {
			for _, m := range s.registry.List() {
				if s.defaultChatModel(m.ID) == nil {
					continue
				}
				routerName := s.registry.RouterName(m.ID)
				for _, lm := range loaded {
					if lm.Status.Value == "loaded" && routerEntryIs(lm, routerName, m) {
						consider(defaultTarget{Name: m.PublicName(), ID: m.ID, Reason: defaultReasonLoaded}, s.llmUse.lastUse(m.ID))
						break
					}
				}
			}
		}
	}
	if cfg.ForksEnabled && s.forks != nil && s.forkMgr != nil {
		for _, inst := range s.forks.Instances() {
			if s.forkMgr.Status(inst.ID).State == process.StateRunning {
				consider(defaultTarget{Name: inst.ID, ID: inst.ID, Fork: true, Reason: defaultReasonLoaded}, s.forkUse.lastUse(inst.ID))
			}
		}
	}
	if found {
		return best, true
	}

	if m := s.defaultChatModel(last); last != "" && m != nil {
		return defaultTarget{Name: m.PublicName(), ID: m.ID, Reason: defaultReasonLastUsed}, true
	}
	return defaultTarget{}, false
}

// lastChatModel is the registry ID of the router chat model used last,
// from the state file; "" when there is none.
func (s *Server) lastChatModel() string {
	if s.defaultState == nil {
		return ""
	}
	if last := s.defaultState.get(); !last.Fork {
		return last.ID
	}
	return ""
}

// clearStaleDefaultPin clears a pin whose model is gone for good: a model
// pin whose model is deleted, disabled or not a chat model, or a fork pin
// whose instance was removed. A fork pin stays while forks are switched
// off, since the instance works again when they come back on, and while
// there is no fork store, since then nothing can be said.
func (s *Server) clearStaleDefaultPin() {
	pin := s.cfgCopy().DefaultModel.Pin
	if pin == "" || !s.defaultPinStale(pin) {
		return
	}
	s.cfgMu.Lock()
	defer s.cfgMu.Unlock()
	// The user may have changed the pin in the meantime.
	if s.cfg.DefaultModel.Pin != pin {
		return
	}
	s.cfg.DefaultModel.Pin = ""
	if err := s.saveConfigLocked(); err != nil {
		slog.Warn("default model pin cleared but config not saved", "pin", pin, "error", err)
		return
	}
	slog.Info("default model pin cleared: model is gone", "pin", pin)
}

func (s *Server) defaultPinStale(pin string) bool {
	if id, fork := strings.CutPrefix(pin, config.DefaultModelForkPrefix); fork {
		exists, known := s.forkInstanceExists(id)
		return known && !exists
	}
	return s.defaultChatModel(pin) == nil
}

// defaultNameClash says why the virtual name cannot be used: a model or a
// fork instance has the same name. FindByAny matches registry IDs (which
// are also the router names), public names and aliases. A fork instance
// clashes while forks are off too, since switching them on would make it
// shadow the name. "" when there is no clash.
func (s *Server) defaultNameClash(name string) string {
	if m, _ := s.registry.FindByAny(name); m != nil {
		return fmt.Sprintf("the model %s has the name or alias %q", m.PublicName(), name)
	}
	if exists, _ := s.forkInstanceExists(name); exists {
		return fmt.Sprintf("the fork instance %s has the same name", name)
	}
	return ""
}

// defaultModelError is a request to the virtual name that cannot be
// served, with the OpenAI-shaped error to answer.
type defaultModelError struct {
	status  int
	errType string
	code    string
	message string
}

func (e *defaultModelError) write(w http.ResponseWriter) {
	openAIError(w, e.status, e.errType, e.code, e.message)
}

// rewriteDefaultModel puts the virtual name's target in the model field
// of a JSON chat request that names the virtual name or no model. Other
// bodies come back unchanged. Values other than the model keep their
// exact bytes, so a large integer such as a seed is not rounded.
func (s *Server) rewriteDefaultModel(path string, body []byte) ([]byte, *defaultModelError) {
	if !isChatPath(path) {
		return body, nil
	}
	var probe struct {
		Model *string `json:"model"`
	}
	if json.Unmarshal(body, &probe) != nil {
		return body, nil // not JSON, or a model that is not a string: the router answers
	}
	cfg := s.cfgCopy()
	name := cfg.DefaultModelName()
	if probe.Model != nil && *probe.Model != "" && *probe.Model != name {
		return body, nil
	}
	if !cfg.DefaultModelOn() || s.defaultNameClash(name) != "" {
		return body, nil
	}
	var req map[string]json.RawMessage
	if json.Unmarshal(body, &req) != nil || req == nil {
		return body, nil
	}

	s.clearStaleDefaultPin()
	t, ok := s.resolveDefaultModel()
	if !ok {
		return nil, &defaultModelError{http.StatusServiceUnavailable, "server_error", "no_default_model",
			fmt.Sprintf("No model for %q: pin a model for it, or send a request to a model by its name first.", name)}
	}
	model, _ := json.Marshal(t.Name)
	req["model"] = model
	out, err := json.Marshal(req)
	if err != nil {
		return body, nil
	}
	slog.Debug("default model resolved", "name", name, "target", t.Name, "reason", t.Reason)
	return out, nil
}

// refuseDefaultModelOffChat answers a request on a path other than the
// chat ones that names the virtual name, which names a chat model only.
// It reports whether it answered.
func (s *Server) refuseDefaultModelOffChat(w http.ResponseWriter, path, model string) bool {
	if isChatPath(path) || model == "" {
		return false
	}
	cfg := s.cfgCopy()
	name := cfg.DefaultModelName()
	if model != name || !cfg.DefaultModelOn() || s.defaultNameClash(name) != "" {
		return false
	}
	(&defaultModelError{http.StatusBadRequest, "invalid_request_error", "default_model_chat_only",
		fmt.Sprintf("%q names a chat model; send this request to a model by its name.", name)}).write(w)
	return true
}
