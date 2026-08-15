package api

import (
	"bytes"
	"html/template"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/backup"
	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/web"
)

// The restore report partial is the handler's only user-visible output;
// a parse/execute error turns every restore into a blank box.
func TestRestoreReportPartialRenders(t *testing.T) {
	base, err := template.New("").Funcs(testFuncMap).ParseFS(web.Templates,
		"templates/layout.html", "templates/partials/*.html")
	if err != nil {
		t.Fatalf("parse templates: %v", err)
	}

	rep := backup.Report{
		Applied:             []string{"settings: models_max", "model config: org/a-GGUF Q4_K_M"},
		AppliedModelConfigs: 1,
		Notes:               []string{"these changes take effect on the next server restart"},
		Warnings:            []string{"org/a-GGUF Q4_K_M: custom tensor split imported verbatim"},
		Skipped:             []backup.SkippedItem{{Item: "flag preset BAD", Reason: "invalid name"}},
		NotSelected:         []string{"runtime env"},
		Missing: []backup.MissingModel{
			{ModelID: "org/x-GGUF", Quant: "Q8_0", Filename: "x-Q8_0.gguf", Pending: true},
			{ModelID: "org/y-GGUF", Quant: "Q4_K_M", Filename: "y-Q4_K_M.gguf", Pending: true},
		},
	}
	var buf bytes.Buffer
	if err := base.ExecuteTemplate(&buf, "restore_report", rep); err != nil {
		t.Fatalf("execute: %v", err)
	}
	out := buf.String()
	for _, want := range []string{
		"settings: models_max",
		"next server restart",
		"custom tensor split",
		"invalid name",
		"runtime env",          // not-selected line
		"held as pending",      // pending annotation
		"org/x-GGUF",           // missing identity
		"/api/hf/download",     // per-row download wiring
		`"inline": "1"`,        // inline response mode
		"Download all missing", // bulk action (>1 missing)
		"restore-dl-",          // per-row status container
	} {
		if !strings.Contains(out, want) {
			t.Errorf("report partial missing %q", want)
		}
	}

	// Error-only shape renders as a banner and nothing else.
	buf.Reset()
	if err := base.ExecuteTemplate(&buf, "restore_report", backup.Report{Error: "backup version 9 is not supported"}); err != nil {
		t.Fatalf("execute error shape: %v", err)
	}
	if !strings.Contains(buf.String(), "version 9") || strings.Contains(buf.String(), "Applied") {
		t.Errorf("error shape wrong: %s", buf.String())
	}
}

// Ghost cards must render for pending configs — including with an empty
// registry (fresh target server right after a restore).
func TestPendingGhostCardRenders(t *testing.T) {
	s := &Server{}
	rec := httptest.NewRecorder()
	s.renderPendingCard(rec, &models.PendingConfig{
		ModelID: "org/x-GGUF", Quant: "Q8_0", Filename: "x-Q8_0.gguf",
		SavedAt: time.Date(2026, 8, 15, 0, 0, 0, 0, time.UTC),
	})
	out := rec.Body.String()
	for _, want := range []string{
		"org/x-GGUF",
		"config waiting",
		"imported 2026-08-15",
		"/api/hf/download",
		`"inline": "1"`,
		"/api/backup/pending/discard",
		"pending-dl-",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("ghost card missing %q:\n%s", want, out)
		}
	}
}

// Discard's contract: 204 + HX-Trigger so #model-list refreshes itself,
// and the entry actually gone; 404 for an unknown identity.
func TestDiscardPendingHandler(t *testing.T) {
	dir := t.TempDir()
	reg := models.NewRegistry(dir, dir)
	reg.SetPendingConfig(models.PendingConfig{ModelID: "org/x-GGUF", Quant: "Q8_0", Filename: "x.gguf"})
	s := &Server{registry: reg}

	req := httptest.NewRequest("POST", "/api/backup/pending/discard",
		strings.NewReader("model_id=org/x-GGUF&quant=Q8_0"))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	rec := httptest.NewRecorder()
	s.handleDiscardPending(rec, req)

	if rec.Code != 204 || rec.Header().Get("HX-Trigger") != "modelsChanged" {
		t.Errorf("want 204 + HX-Trigger modelsChanged, got %d %q", rec.Code, rec.Header().Get("HX-Trigger"))
	}
	if len(reg.PendingConfigs()) != 0 {
		t.Error("entry survived discard")
	}

	rec2 := httptest.NewRecorder()
	req2 := httptest.NewRequest("POST", "/api/backup/pending/discard",
		strings.NewReader("model_id=org/x-GGUF&quant=Q8_0"))
	req2.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	s.handleDiscardPending(rec2, req2)
	if rec2.Code != 404 {
		t.Errorf("unknown identity should 404, got %d", rec2.Code)
	}
}
