package api

import (
	"cmp"
	"fmt"
	"net/http"
	"net/url"
	"slices"
	"strings"

	"github.com/tmac1973/llama-toolchest/internal/config"
)

// defaultModelView is what the Settings page's Default model card
// (default_model_card partial) shows.
type defaultModelView struct {
	On bool
	// Name is the saved name, "" for the default one; Placeholder is
	// config.DefaultModelNameDefault.
	Name        string
	Placeholder string
	// Pin is the saved pin, "" for none. Options are the models it can
	// name: the chat models, then the fork instances.
	Pin     string
	Options []defaultPinOption
	// Target is the model requests to the name go to now, "" for none,
	// and Reason why (defaultReason*).
	Target string
	Reason string
	// Clash says why the name is off: a model or fork instance has it.
	Clash string
}

// defaultPinOption is one entry of the card's pin select.
type defaultPinOption struct {
	Value, Label string
}

// defaultModelInfo builds the card's view. It clears a pin whose model is
// gone first, so the card never offers a pin that no longer works.
func (s *Server) defaultModelInfo() defaultModelView {
	s.clearStaleDefaultPin()
	cfg := s.cfgCopy()
	v := defaultModelView{
		On:          cfg.DefaultModelOn(),
		Name:        cfg.DefaultModel.Name,
		Placeholder: config.DefaultModelNameDefault,
		Pin:         cfg.DefaultModel.Pin,
		Options:     s.defaultPinOptions(&cfg),
		Clash:       s.defaultNameClash(cfg.DefaultModelName()),
	}
	if t, ok := s.resolveDefaultModel(); ok {
		v.Target, v.Reason = t.Name, t.Reason
	}
	return v
}

// defaultPinOptions lists what the pin can name: each chat model clients
// can use, then, while forks are on, each fork instance, each group by
// label. While forks are off a saved fork pin is still listed, so the
// select shows it and saving another field keeps it.
func (s *Server) defaultPinOptions(cfg *config.Config) []defaultPinOption {
	var chat, fork []defaultPinOption
	for _, m := range s.registry.List() {
		if s.defaultChatModel(m.ID) != nil {
			chat = append(chat, defaultPinOption{m.ID, m.PublicName()})
		}
	}
	if cfg.ForksEnabled && s.forks != nil {
		for _, inst := range s.forks.Instances() {
			fork = append(fork, defaultPinOption{config.DefaultModelForkPrefix + inst.ID, inst.ID + " (fork)"})
		}
	} else if id, ok := strings.CutPrefix(cfg.DefaultModel.Pin, config.DefaultModelForkPrefix); ok {
		fork = append(fork, defaultPinOption{cfg.DefaultModel.Pin, id + " (fork, forks are off)"})
	}
	byLabel := func(a, b defaultPinOption) int { return cmp.Compare(a.Label, b.Label) }
	slices.SortFunc(chat, byLabel)
	slices.SortFunc(fork, byLabel)
	return append(chat, fork...)
}

// handleDefaultModelCard renders the Settings page's Default model card,
// which fetches itself again after each save (the settingsSaved event).
func (s *Server) handleDefaultModelCard(w http.ResponseWriter, r *http.Request) {
	respondHTML(w)
	s.renderPartial(w, "default_model_card", s.defaultModelInfo())
}

// defaultModelUpdate is a settings request's change to the virtual model
// name; nil fields stay as they are.
type defaultModelUpdate struct {
	On   *bool
	Name *string
	Pin  *string
}

// formDefaultModel reads the Default model card's form: it is in the
// request when the card's default_model_form marker is.
func formDefaultModel(form url.Values) defaultModelUpdate {
	if !form.Has("default_model_form") {
		return defaultModelUpdate{}
	}
	on := form.Get("default_model_on") == "on"
	name, pin := form.Get("default_model_name"), form.Get("default_model_pin")
	return defaultModelUpdate{On: &on, Name: &name, Pin: &pin}
}

// applyDefaultModelLocked checks and applies a change to the virtual model
// name, with cfgMu held. Nothing changes when it returns an error.
func (s *Server) applyDefaultModelLocked(u defaultModelUpdate) error {
	name := s.cfg.DefaultModel.Name
	if u.Name != nil {
		name = strings.TrimSpace(*u.Name)
		if name != "" {
			if err := config.ValidDefaultModelName(name); err != nil {
				return fmt.Errorf("default model name: %w", err)
			}
		}
		// A name kept as it is passes, so the switch and the pin can
		// still be saved while a model has taken the name.
		check := cmp.Or(name, config.DefaultModelNameDefault)
		if clash := s.defaultNameClash(check); clash != "" && name != s.cfg.DefaultModel.Name {
			return fmt.Errorf("default model name %q: %s", check, clash)
		}
	}
	pin := s.cfg.DefaultModel.Pin
	if u.Pin != nil {
		pin = *u.Pin
		if pin != "" && pin != s.cfg.DefaultModel.Pin && !slices.ContainsFunc(s.defaultPinOptions(s.cfg),
			func(o defaultPinOption) bool { return o.Value == pin }) {
			return fmt.Errorf("default model pin %q: not a chat model or fork instance", pin)
		}
	}
	if u.On != nil {
		s.cfg.DefaultModel.Off = !*u.On
	}
	s.cfg.DefaultModel.Name, s.cfg.DefaultModel.Pin = name, pin
	return nil
}

// defaultTargetIDs are the registry model and the fork instance that
// requests to the virtual name go to now, for the badges on the Models
// and Server pages; "" for none.
func (s *Server) defaultTargetIDs() (modelID, forkID string) {
	cfg := s.cfgCopy()
	if !cfg.DefaultModelOn() || s.defaultNameClash(cfg.DefaultModelName()) != "" {
		return "", ""
	}
	t, ok := s.resolveDefaultModel()
	switch {
	case !ok:
		return "", ""
	case t.Fork:
		return "", t.ID
	}
	return t.ID, ""
}

// defaultBadgeTitle is the tooltip of the badge that marks the virtual
// name's target.
func (s *Server) defaultBadgeTitle() string {
	cfg := s.cfgCopy()
	return fmt.Sprintf("Requests to %q go to this model.", cfg.DefaultModelName())
}
