package api

import (
	"bytes"
	"html/template"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/web"
)

// renderModelConfig renders the model_config partial with the PLE fields
// set as given, returning the HTML. It uses modelConfigPanelData, the
// type the handler builds, so the two cannot drift apart.
func renderModelConfig(t *testing.T, cfg *models.ModelConfig, hasPLE bool, sizeLabel string) string {
	t.Helper()
	base, err := template.New("").Funcs(testFuncMap).ParseFS(web.Templates,
		"templates/layout.html", "templates/partials/*.html")
	if err != nil {
		t.Fatalf("parse templates: %v", err)
	}
	data := modelConfigPanelData{
		ModelID:      "test-id",
		Config:       cfg,
		DraftModes:   models.DraftModes(),
		AssistModes:  models.AssistModes(),
		DraftParams:  models.SpecDraftParams(cfg.SpecType),
		AssistParams: models.SpecAssistParams(cfg.SpecAssist),
		HasMTP:       cfg.MtpPath != "",

		EffectiveSpecType: cfg.EffectiveSpecType(),
		HasPLE:            hasPLE,
		PLESizeLabel:      sizeLabel,
	}
	var buf bytes.Buffer
	if err := base.ExecuteTemplate(&buf, "model_config", data); err != nil {
		t.Fatalf("execute: %v", err)
	}
	return buf.String()
}

// The control is meaningless for the overwhelming majority of models, so
// it must not appear unless the file actually carries the table.
func TestPLESelectorHiddenWithoutTable(t *testing.T) {
	out := renderModelConfig(t, &models.ModelConfig{Enabled: true, ContextSize: 8192}, false, "")
	if strings.Contains(out, `name="ple_mode"`) {
		t.Error("PLE selector rendered for a model with no per-layer embedding table")
	}
}

func TestPLESelectorShownWithTable(t *testing.T) {
	out := renderModelConfig(t, &models.ModelConfig{Enabled: true, ContextSize: 8192}, true, "28.8 GB")
	if !strings.Contains(out, `name="ple_mode"`) {
		t.Fatalf("PLE selector missing for a model with a table; output=\n%s", out)
	}
	// The size is the whole reason the label is worth rendering — it tells
	// the user how much memory the choice is actually about.
	if !strings.Contains(out, "28.8 GB") {
		t.Error("PLE selector did not render the table size")
	}
	for _, opt := range []string{`value=""`, `value="on"`, `value="off"`} {
		if !strings.Contains(out, opt) {
			t.Errorf("PLE selector missing option %s", opt)
		}
	}
}

// The stored mode must come back selected, or saving the form would
// silently reset it to auto on the next render.
func TestPLESelectorPreservesSelection(t *testing.T) {
	out := renderModelConfig(t, &models.ModelConfig{Enabled: true, ContextSize: 8192, PLEMode: "on"}, true, "28.8 GB")
	idx := strings.Index(out, `name="ple_mode"`)
	if idx < 0 {
		t.Fatal("PLE selector missing")
	}
	block := out[idx:]
	if end := strings.Index(block, "</select>"); end > 0 {
		block = block[:end]
	}
	if !strings.Contains(block, `value="on" selected`) {
		t.Errorf("stored mode \"on\" not preselected; select block=\n%s", block)
	}
}
