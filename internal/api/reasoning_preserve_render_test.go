package api

import (
	"bytes"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/models"
)

func renderModelConfigWith(t *testing.T, cfg *models.ModelConfig, data modelConfigPanelData) string {
	t.Helper()
	base := testTemplates(t)
	data.ModelID = "test-id"
	data.Config = cfg
	data.DraftModes = models.DraftModes()
	data.AssistModes = models.AssistModes()
	var buf bytes.Buffer
	if err := base.ExecuteTemplate(&buf, "model_config", data); err != nil {
		t.Fatalf("execute: %v", err)
	}
	return buf.String()
}

// Only models with a reasoning mode get the control, and the stored
// choice comes back selected so saving the form does not reset it.
func TestReasoningPreserveSelector(t *testing.T) {
	cfg := &models.ModelConfig{Enabled: true, ReasoningPreserve: "off"}
	if out := renderModelConfigWith(t, cfg, modelConfigPanelData{}); strings.Contains(out, `name="reasoning_preserve"`) {
		t.Error("selector rendered for a model without a reasoning mode")
	}
	out := renderModelConfigWith(t, cfg, modelConfigPanelData{HasReasoning: true, ReasoningPreserveDefault: "keep"})
	idx := strings.Index(out, `name="reasoning_preserve"`)
	if idx < 0 {
		t.Fatal("selector missing for a reasoning model")
	}
	block := out[idx:]
	if end := strings.Index(block, "</select>"); end > 0 {
		block = block[:end]
	}
	if !strings.Contains(block, `value="off" selected`) {
		t.Errorf("stored \"off\" not preselected:\n%s", block)
	}
	if !strings.Contains(block, "Default (llama.cpp: keep)") {
		t.Errorf("default label should name the build's default:\n%s", block)
	}
}
