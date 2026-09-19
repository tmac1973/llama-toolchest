package api

import (
	"bytes"
	"html/template"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/web"
)

const testReadOnlyReason = "models.json was written by a newer version of llama-toolchest (schema 99; this version reads up to 1)."

// The config panel says why nothing will save before the user edits
// anything, and says nothing when the registry is writable.
func TestModelConfigShowsReadOnlyReason(t *testing.T) {
	base, err := template.New("").Funcs(testFuncMap).ParseFS(web.Templates,
		"templates/layout.html", "templates/partials/*.html")
	if err != nil {
		t.Fatalf("parse templates: %v", err)
	}
	render := func(reason string) string {
		cfg := &models.ModelConfig{Enabled: true, ContextSize: 8192}
		data := modelConfigPanelData{
			ModelID:        "test-id",
			Config:         cfg,
			DraftModes:     models.DraftModes(),
			AssistModes:    models.AssistModes(),
			ReadOnlyReason: reason,
		}
		var buf bytes.Buffer
		if err := base.ExecuteTemplate(&buf, "model_config", data); err != nil {
			t.Fatalf("execute model_config: %v", err)
		}
		return buf.String()
	}

	out := render(testReadOnlyReason)
	if !strings.Contains(out, `role="alert"`) || !strings.Contains(out, "schema 99") {
		t.Errorf("read-only panel does not show the reason as an alert")
	}
	if !strings.Contains(out, "The file on disk has not been changed") {
		t.Errorf("read-only banner has no explanatory tooltip")
	}
	if strings.Contains(render(""), "The file on disk has not been changed") {
		t.Errorf("writable panel shows the read-only banner")
	}
}

func TestBenchmarksPageShowsReadOnlyReason(t *testing.T) {
	base, err := template.New("").Funcs(testFuncMap).ParseFS(web.Templates,
		"templates/layout.html", "templates/partials/*.html")
	if err != nil {
		t.Fatalf("parse base: %v", err)
	}
	page, err := template.Must(base.Clone()).ParseFS(web.Templates, "templates/benchmarks.html")
	if err != nil {
		t.Fatalf("parse benchmarks.html: %v", err)
	}
	render := func(reason string) string {
		var buf bytes.Buffer
		err := page.ExecuteTemplate(&buf, "layout", benchmarksPageData{
			pageData:       pageData{Title: "Benchmarks", Nav: "benchmarks"},
			ReadOnlyReason: reason,
		})
		if err != nil {
			t.Fatalf("execute benchmarks page: %v", err)
		}
		return buf.String()
	}

	reason := "benchmarks.json could not be parsed (unexpected end of JSON input)."
	if out := render(reason); !strings.Contains(out, "unexpected end of JSON input") {
		t.Errorf("read-only Benchmarks page does not show the reason")
	}
	if strings.Contains(render(""), "New benchmark jobs are refused") {
		t.Errorf("writable Benchmarks page shows the read-only banner")
	}
}
