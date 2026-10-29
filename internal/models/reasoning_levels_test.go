package models

import (
	"slices"
	"testing"
)

// Turning thinking off for a reasoning_effort model sends the lowest level
// the template accepts once the levels are known, and "none" before.
func TestApplyThinkingOffLevels(t *testing.T) {
	cases := []struct {
		name   string
		toggle string
		levels []string
		want   map[string]any
	}{
		{"levels known", ReasoningToggleReasoningEffort, []string{"high", "max"}, map[string]any{"reasoning_effort": "high"}},
		{"levels not known", ReasoningToggleReasoningEffort, nil, map[string]any{"reasoning_effort": "none"}},
		{"template accepts none", ReasoningToggleReasoningEffort, []string{}, map[string]any{"reasoning_effort": "none"}},
		{"kwarg switch is unchanged", ReasoningToggleChatTemplateKwargs, []string{"low"}, map[string]any{"chat_template_kwargs": map[string]any{"enable_thinking": false}}},
		{"no switch", ReasoningToggleNone, []string{"low"}, map[string]any{}},
	}
	for _, c := range cases {
		payload := map[string]any{}
		ApplyThinkingOff(c.toggle, "enable_thinking", c.levels, payload)
		if len(payload) != len(c.want) {
			t.Errorf("%s: payload %v, want %v", c.name, payload, c.want)
			continue
		}
		for k, v := range c.want {
			if k == "chat_template_kwargs" {
				if kw, _ := payload[k].(map[string]any); kw["enable_thinking"] != false {
					t.Errorf("%s: %v", c.name, payload)
				}
				continue
			}
			if payload[k] != v {
				t.Errorf("%s: %s = %v, want %v", c.name, k, payload[k], v)
			}
		}
	}
}

// The levels are reported as not known (nil) until read, as empty when
// the template accepts none, and are carried on the effective reasoning
// capability, also under a per-model override.
func TestEffortLevels(t *testing.T) {
	m := &Model{Reasoning: ReasoningCapability{Supported: true, Toggle: ReasoningToggleReasoningEffort}}
	if m.EffortLevels() != nil || m.EffectiveReasoning(nil).EffortLevels != nil {
		t.Error("levels reported before they were read")
	}
	m.ReasoningLevelsKnown = true
	if got := m.EffortLevels(); got == nil || len(got) != 0 {
		t.Errorf("template that accepts none: %#v, want an empty list", got)
	}
	m.ReasoningEffortLevels = []string{"low", "medium", "xhigh"}
	override := &ModelConfig{ReasoningOverride: &ReasoningCapability{Supported: true, Toggle: ReasoningToggleChatTemplateKwargs, Kwarg: "think", EffortLevels: []string{"bogus"}}}
	for _, cfg := range []*ModelConfig{nil, override} {
		if got := m.EffectiveReasoning(cfg).EffortLevels; !slices.Equal(got, []string{"low", "medium", "xhigh"}) {
			t.Errorf("cfg %v: levels %v", cfg, got)
		}
	}
	// The returned list is a copy.
	m.EffortLevels()[0] = "changed"
	if m.ReasoningEffortLevels[0] != "low" {
		t.Error("EffortLevels returned the record's own slice")
	}
}

// A saved default effort is a request default like the sampling ones.
func TestReasoningEffortDefault(t *testing.T) {
	cfg := &ModelConfig{ReasoningEffort: "medium"}
	if got := cfg.SamplingOverrides()["reasoning_effort"]; got != "medium" {
		t.Errorf("reasoning_effort default = %v", got)
	}
	if _, ok := (&ModelConfig{}).SamplingOverrides()["reasoning_effort"]; ok {
		t.Error("reasoning_effort added without a saved default")
	}
}

// The levels are saved with the build they came from, and a model record
// handed out is a copy.
func TestSetReasoningLevels(t *testing.T) {
	r, _, _ := newTestRegistry(t)
	m := &Model{ID: "m", ModelID: "org/m", Filename: "m.gguf"}
	if err := r.Add(m); err != nil {
		t.Fatal(err)
	}
	if err := r.SetReasoningLevels("m", []string{"high", "max"}, true, "b1"); err != nil {
		t.Fatal(err)
	}
	got, _ := r.Get("m")
	if !got.ReasoningLevelsKnown || got.ReasoningLevelsBuild != "b1" || !slices.Equal(got.ReasoningEffortLevels, []string{"high", "max"}) {
		t.Errorf("record: %+v", got)
	}
	got.ReasoningEffortLevels[0] = "changed"
	again, _ := r.Get("m")
	if again.ReasoningEffortLevels[0] != "high" {
		t.Error("Get handed out the registry's own slice")
	}
	if err := r.SetReasoningLevels("nobody", nil, false, "b1"); err == nil {
		t.Error("no error for an unknown model")
	}
}
