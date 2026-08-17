package api

import (
	"bytes"
	"html/template"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/web"
)

// testFuncMap is the REAL template function map from
// Server.templateFuncs, with only the functions that read live server
// state replaced by stubs.
//
// It used to be a hand-copied list, with a comment warning that it had
// to be kept in sync. It was not: adding a template function without
// adding it here failed the whole partial set's parse, which broke a
// dozen unrelated render tests at once with a message pointing at
// neither the new function nor the test. Deriving it means a new
// function is available here the moment it exists.
var testFuncMap = func() template.FuncMap {
	m := (&Server{}).templateFuncs()
	// vramFit is the only one that dereferences server state (the GPU
	// monitor); nothing else needs a live Server.
	m["vramFit"] = func(gb float64) string { return "" }
	return m
}()

// TestSamplingPresetsPartialRenders verifies the model_config partial renders
// with a populated SamplingPresetsJSON and that the embedded JSON survives
// html/template attribute escaping such that the browser will decode it
// back to valid JSON when read via dataset.presets.
func TestSamplingPresetsPartialRenders(t *testing.T) {
	base, err := template.New("").Funcs(testFuncMap).ParseFS(web.Templates,
		"templates/layout.html", "templates/partials/*.html")
	if err != nil {
		t.Fatalf("parse templates: %v", err)
	}

	temp := 0.6
	topP := 0.95
	topK := 20
	cfg := &models.ModelConfig{
		Enabled: true, GPULayers: 999, ContextSize: 8192, Threads: 8,
		SamplingPreset: "thinking",
	}
	presets := []models.SamplingPreset{
		{
			Name: "thinking", Label: "Thinking mode",
			Description: "From README — for enable_thinking=True",
			Source:      "readme",
			Temperature: &temp, TopP: &topP, TopK: &topK,
		},
	}
	data := struct {
		ModelID             string
		Config              *models.ModelConfig
		EffectiveFlags      string
		MaxContext          int
		HasMMProj           bool
		HasBuiltinVision    bool
		IsEmbedding         bool
		DraftCandidates     []models.DraftCandidate
		GPUOptions          []models.GPUOption
		NumGPUs             int
		SamplingPresets     []models.SamplingPreset
		SamplingPresetsJSON string
		HasEmbeddedDefault  bool
	}{
		ModelID:             "test-id",
		Config:              cfg,
		SamplingPresets:     presets,
		SamplingPresetsJSON: `[{"name":"thinking","temperature":0.6}]`,
	}

	var buf bytes.Buffer
	if err := base.ExecuteTemplate(&buf, "model_config", data); err != nil {
		t.Fatalf("execute: %v", err)
	}
	out := buf.String()

	if !strings.Contains(out, `class="sampling-preset-picker"`) {
		t.Errorf("expected preset picker rendered; output=\n%s", out)
	}
	// html/template will HTML-entity-encode the quotes inside attribute
	// values. The browser decodes them when reading dataset.presets, so
	// the wire form should contain &#34; or &quot;, NOT raw double quotes
	// inside the attribute.
	if !strings.Contains(out, `data-presets="`) {
		t.Errorf("expected data-presets attribute; output=\n%s", out)
	}
	if strings.Contains(out, `data-presets="[{"name":`) {
		t.Errorf("attribute contains unescaped JSON quotes — would break HTML parsing")
	}
	if !strings.Contains(out, "Thinking mode") {
		t.Errorf("expected option label; output=\n%s", out)
	}
	// The applied preset persists on the config and its option renders
	// selected, so reopening the panel shows which preset is running.
	if !strings.Contains(out, `value="thinking" title="From README — for enable_thinking=True" selected`) {
		t.Errorf("expected the stored preset's option to render selected; output=\n%s", out)
	}
	if !strings.Contains(out, `name="sampling_preset"`) {
		t.Errorf("expected picker to be a named form field so the selection is saved; output=\n%s", out)
	}
}

// TestSamplingPresetsPartialHidden ensures the picker is omitted when no
// presets are available for the model.
func TestSamplingPresetsPartialHidden(t *testing.T) {
	base, err := template.New("").Funcs(testFuncMap).ParseFS(web.Templates,
		"templates/layout.html", "templates/partials/*.html")
	if err != nil {
		t.Fatalf("parse templates: %v", err)
	}

	data := struct {
		ModelID             string
		Config              *models.ModelConfig
		EffectiveFlags      string
		MaxContext          int
		HasMMProj           bool
		HasBuiltinVision    bool
		IsEmbedding         bool
		DraftCandidates     []models.DraftCandidate
		GPUOptions          []models.GPUOption
		NumGPUs             int
		SamplingPresets     []models.SamplingPreset
		SamplingPresetsJSON string
		HasEmbeddedDefault  bool
	}{
		ModelID: "test-id",
		Config:  &models.ModelConfig{Enabled: true, GPULayers: 999, ContextSize: 8192},
	}

	var buf bytes.Buffer
	if err := base.ExecuteTemplate(&buf, "model_config", data); err != nil {
		t.Fatalf("execute: %v", err)
	}
	if strings.Contains(buf.String(), "sampling-preset-picker") {
		t.Errorf("preset picker should be hidden when no presets")
	}
}
