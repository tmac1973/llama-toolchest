package models

import "strings"

// Reasoning toggle mechanisms — the *how* a client flips a model's thinking
// mode on or off. These are stable string constants because they're part of
// the capabilities API contract (see internal/api/capabilities.go).
const (
	// ReasoningToggleChatTemplateKwargs: the model's chat template branches on
	// a boolean kwarg (Qwen3's `enable_thinking`). Clients pass it through
	// chat_template_kwargs.
	ReasoningToggleChatTemplateKwargs = "chat_template_kwargs"
	// ReasoningToggleReasoningEffort: OpenAI-style `reasoning_effort` request
	// field (gpt-oss and friends).
	ReasoningToggleReasoningEffort = "reasoning_effort"
	// ReasoningToggleNone: no client-facing switch. Either the model has no
	// reasoning mode, or it always reasons and can't be turned off.
	ReasoningToggleNone = "none"
)

// ApplyThinkingOff adds to a chat request payload whatever a model needs
// to skip its reasoning pass, given how it exposes the switch (one of the
// ReasoningToggle* constants, and the kwarg name for chat_template_kwargs).
// A model with no switch (ReasoningToggleNone, or no kwarg) is left as is.
func ApplyThinkingOff(toggle, kwarg string, payload map[string]any) {
	switch toggle {
	case ReasoningToggleChatTemplateKwargs:
		if kwarg != "" {
			payload["chat_template_kwargs"] = map[string]any{kwarg: false}
		}
	case ReasoningToggleReasoningEffort:
		payload["reasoning_effort"] = "none"
	}
}

// ReasoningCapability describes whether and how a model exposes a "thinking" /
// reasoning mode. It's surfaced under capabilities.reasoning so a client can
// decide whether to send a thinking toggle at all, and which mechanism to use,
// instead of assuming Qwen's enable_thinking on every model.
type ReasoningCapability struct {
	Supported      bool   `json:"supported"`       // model exposes a reasoning mode
	DefaultEnabled bool   `json:"default_enabled"` // reasoning is on unless the client turns it off
	Toggle         string `json:"toggle"`          // one of the ReasoningToggle* constants
	Kwarg          string `json:"kwarg"`           // kwarg key when Toggle == chat_template_kwargs; "" otherwise
}

// EffectiveReasoning returns the reasoning capability to report for this model:
// a per-model override when the user set one, otherwise the value detected from
// the chat template. The toggle is normalized to "none" so the contract never
// emits an empty mechanism for older records parsed before detection existed.
func (m *Model) EffectiveReasoning(cfg *ModelConfig) ReasoningCapability {
	if cfg != nil && cfg.ReasoningOverride != nil {
		r := *cfg.ReasoningOverride
		if r.Toggle == "" {
			r.Toggle = ReasoningToggleNone
		}
		return r
	}
	r := m.Reasoning
	if r.Toggle == "" {
		r.Toggle = ReasoningToggleNone
	}
	return r
}

// detectReasoning inspects a chat-template string for the markers of a
// reasoning toggle. It recognizes the two unambiguous mechanisms that show up
// verbatim in templates; anything else reports unsupported. The contract is
// what matters here — a model that reasons via some bespoke mechanism can be
// corrected with a per-model override (ModelConfig.ReasoningOverride).
func detectReasoning(chatTemplate string) ReasoningCapability {
	switch {
	case strings.Contains(chatTemplate, "enable_thinking"):
		// Qwen3-style: the template reads `enable_thinking`, defaulting to on
		// unless the caller passes false.
		return ReasoningCapability{
			Supported:      true,
			DefaultEnabled: true,
			Toggle:         ReasoningToggleChatTemplateKwargs,
			Kwarg:          "enable_thinking",
		}
	case strings.Contains(chatTemplate, "reasoning_effort"):
		// OpenAI-style: a `reasoning_effort` request field gates the depth.
		return ReasoningCapability{
			Supported:      true,
			DefaultEnabled: true,
			Toggle:         ReasoningToggleReasoningEffort,
			Kwarg:          "",
		}
	default:
		return ReasoningCapability{
			Supported:      false,
			DefaultEnabled: false,
			Toggle:         ReasoningToggleNone,
			Kwarg:          "",
		}
	}
}
