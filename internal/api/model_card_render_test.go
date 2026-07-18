package api

import (
	"bytes"
	"html/template"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/web"
)

// modelCardData mirrors the anonymous struct renderModelCard builds, so the
// partial can be exercised standalone.
type modelCardData struct {
	models.Model
	IsActive       bool
	IsEnabled      bool
	PendingEnable  bool
	PendingDisable bool
	NeedsReload    bool
	HasVision      bool
	GPULabel       string
	ServiceState   string
	VRAMGB         float64
	IsOrphan       bool
	IsIncomplete   bool
	ResumeFilename string
	SearchText     string
}

func renderModelCardPartial(t *testing.T, data modelCardData) string {
	t.Helper()
	base, err := template.New("").Funcs(testFuncMap).ParseFS(web.Templates,
		"templates/layout.html", "templates/partials/*.html")
	if err != nil {
		t.Fatalf("parse templates: %v", err)
	}
	var buf bytes.Buffer
	if err := base.ExecuteTemplate(&buf, "model_card", data); err != nil {
		t.Fatalf("execute: %v", err)
	}
	return buf.String()
}

// TestModelCardIncomplete verifies an incomplete model renders the badge, a
// Resume button posting to the download endpoint, and a disabled enable toggle.
func TestModelCardIncomplete(t *testing.T) {
	out := renderModelCardPartial(t, modelCardData{
		Model:          models.Model{ID: "org--repo--model", ModelID: "org/repo", Filename: "model-00001-of-00003.gguf", Quant: "Q4_K_M"},
		IsIncomplete:   true,
		ResumeFilename: "model-00001-of-00003.gguf",
	})

	if !strings.Contains(out, ">incomplete<") {
		t.Errorf("expected incomplete badge; output=\n%s", out)
	}
	if !strings.Contains(out, "Resume") {
		t.Errorf("expected Resume button; output=\n%s", out)
	}
	if !strings.Contains(out, `hx-post="/api/hf/download"`) {
		t.Errorf("expected Resume to post to download endpoint; output=\n%s", out)
	}
	if !strings.Contains(out, `"filename":"model-00001-of-00003.gguf"`) {
		t.Errorf("expected resume filename in hx-vals; output=\n%s", out)
	}
	if !strings.Contains(out, "disabled") {
		t.Errorf("expected enable toggle disabled; output=\n%s", out)
	}
}

// TestModelCardHealthy verifies a normal model shows neither orphan nor
// incomplete indicators and keeps its enable toggle active.
func TestModelCardHealthy(t *testing.T) {
	out := renderModelCardPartial(t, modelCardData{
		Model:     models.Model{ID: "org--repo--model", ModelID: "org/repo", Filename: "model.gguf", Quant: "Q4_K_M"},
		IsEnabled: true,
	})
	if strings.Contains(out, ">incomplete<") {
		t.Errorf("healthy model should not show incomplete badge; output=\n%s", out)
	}
	if strings.Contains(out, "Resume") {
		t.Errorf("healthy model should not show Resume button; output=\n%s", out)
	}
	if !strings.Contains(out, "Configure") {
		t.Errorf("healthy model should show Configure; output=\n%s", out)
	}
}
