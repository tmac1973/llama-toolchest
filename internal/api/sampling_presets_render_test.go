package api

import (
	"bytes"
	"html/template"
	"strings"
	"testing"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/benchmark"
	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/web"
)

// testFuncMap mirrors the custom funcs registered in parseTemplates so the
// layout/partial set parses standalone. The values returned here don't matter
// for the partial under test — only that the functions exist.
var testFuncMap = template.FuncMap{
	"divGB":     func(v int64) float64 { return 0 },
	"cssID":     func(s string) string { return s },
	"deref":     func(v interface{}) interface{} { return v },
	"shortSHA":  func(s string) string { return s },
	"divf":      func(a, b interface{}) float64 { return 0 },
	"pctOf":     func(a, b float64) float64 { return 0 },
	"vramFit":   func(gb float64) string { return "" },
	"hasHFRepo": func(s string) bool { return false },
	"version":   func() string { return "test" },
	// Capability-score rendering funcs (bench_eval_display.go). The
	// templates must parse with every custom func present, so these
	// mirror parseTemplates' registration.
	"evalScoreText":  func(e *benchmark.EvalScores) string { return evalScoreText(e) },
	"evalScoreValue": func(e *benchmark.EvalScores) float64 { return evalScoreValue(e) },
	"evalHas":        func(e *benchmark.EvalScores) bool { return evalHas(e) },
	"evalSizeText":   func(p benchmark.Preset) string { return evalSizeText(p) },
	"evalMeaningText": func(p benchmark.Preset) string {
		return evalMeaningText(p)
	},
	"fmtBytes": func(b int64) string { return formatBytes(b) },
	"fmtAge":   func(t time.Time) string { return fmtDurationSince(time.Since(t)) },
	"add":      func(a, b int) int { return a + b },
	"tern": func(cond bool, a, b int) int {
		if cond {
			return a
		}
		return b
	},
	// Note: this map must stay in sync with parseTemplates' funcMap —
	// any template referencing a missing func fails the whole partial
	// set's parse (and renderPartial then silently renders nothing).
}

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
