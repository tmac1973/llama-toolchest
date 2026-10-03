package api

import (
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/models"
)

func TestContextLayoutText(t *testing.T) {
	current := models.Target{Version: 11364}
	tests := []struct {
		name    string
		cfg     models.ModelConfig
		t       models.Target
		text    string
		warning string // "" = none
	}{
		{"blank", models.ModelConfig{ContextSize: 32768}, current, "4 conversations at once (llama.cpp's default), sharing 32,768", ""},
		{"one", models.ModelConfig{ContextSize: 32768, Parallel: 1}, current, "One conversation at a time, with all 32,768", ""},
		{"equal", models.ModelConfig{ContextSize: 32768, Parallel: 4}, current, "each with its own 8,192 tokens (32,768 ÷ 4)", ""},
		{"shared within the pool", models.ModelConfig{ContextSize: 32768, Parallel: 4, SharedContext: true, ContextPerSlot: 8192}, current,
			"sharing 32,768 tokens. Each can use up to 8,192", ""},
		{"shared over the pool", models.ModelConfig{ContextSize: 32768, Parallel: 4, SharedContext: true, ContextPerSlot: 16384}, current,
			"Each can use up to 16,384", "need 65,536 tokens, 2.0× the context"},
		{"limit above the pool", models.ModelConfig{ContextSize: 32768, Parallel: 4, SharedContext: true, ContextPerSlot: 65536}, current,
			"Each can use up to 32,768", "no effect"},
		{"old build", models.ModelConfig{ContextSize: 32768, Parallel: 4, SharedContext: true, ContextPerSlot: 8192}, models.Target{Version: 10661},
			"Each can use up to 32,768", "older than b10662"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			text, warning := contextLayoutText(&tt.cfg, 131072, tt.t)
			if !strings.Contains(text, tt.text) {
				t.Errorf("text %q, want it to contain %q", text, tt.text)
			}
			if tt.warning == "" && warning != "" {
				t.Errorf("unexpected warning %q", warning)
			}
			if tt.warning != "" && !strings.Contains(warning, tt.warning) {
				t.Errorf("warning %q, want it to contain %q", warning, tt.warning)
			}
		})
	}
}

// The sharing controls appear only from two conversations, and the
// limit only with sharing on, so no hidden value is left behind.
func TestSharedContextControlsRender(t *testing.T) {
	has := func(cfg *models.ModelConfig, name string) bool {
		out := renderModelConfigWith(t, cfg, modelConfigPanelData{})
		return strings.Contains(out, `name="`+name+`"`)
	}
	if has(&models.ModelConfig{Parallel: 1}, "shared_context") {
		t.Error("Shared Context shown for one conversation")
	}
	if !has(&models.ModelConfig{Parallel: 4}, "shared_context") {
		t.Error("Shared Context missing for four conversations")
	}
	if has(&models.ModelConfig{Parallel: 4}, "context_per_slot") {
		t.Error("limit shown with sharing off")
	}
	if !has(&models.ModelConfig{Parallel: 4, SharedContext: true}, "context_per_slot") {
		t.Error("limit missing with sharing on")
	}
}
