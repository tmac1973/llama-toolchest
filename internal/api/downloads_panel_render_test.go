package api

import (
	"bytes"
	"html/template"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/web"
)

func renderDownloadsPanel(t *testing.T, rows []downloadRow) string {
	t.Helper()
	base, err := template.New("").Funcs(testFuncMap).ParseFS(web.Templates,
		"templates/layout.html", "templates/partials/*.html")
	if err != nil {
		t.Fatalf("parse templates: %v", err)
	}
	var buf bytes.Buffer
	if err := base.ExecuteTemplate(&buf, "downloads_panel", struct{ Rows []downloadRow }{rows}); err != nil {
		t.Fatalf("execute: %v", err)
	}
	return buf.String()
}

func TestDownloadsPanelActiveRow(t *testing.T) {
	out := renderDownloadsPanel(t, []downloadRow{{
		ID: "org--M-GGUF--m.gguf", ModelID: "org/M-GGUF", Filename: "m.gguf",
		Active: true, Pct: 42, DownGB: 4.2, TotalGB: 10.0, SpeedMB: 55.5,
	}})
	if !strings.Contains(out, ">Pause</button>") {
		t.Errorf("active row must offer Pause; output=\n%s", out)
	}
	if strings.Contains(out, ">Resume</button>") || strings.Contains(out, ">Discard</button>") {
		t.Errorf("active row must not offer Resume/Discard; output=\n%s", out)
	}
	if !strings.Contains(out, `hx-delete="/api/hf/download/org--M-GGUF--m.gguf"`) {
		t.Errorf("Pause must target the cancel route; output=\n%s", out)
	}
	if !strings.Contains(out, "dl-row-active") || !strings.Contains(out, "<progress") {
		t.Errorf("active row must render progress and the active marker class; output=\n%s", out)
	}
}

func TestDownloadsPanelPausedRow(t *testing.T) {
	out := renderDownloadsPanel(t, []downloadRow{{
		ID: "org--M-GGUF--m.gguf", ModelID: "org/M-GGUF", Filename: "m.gguf",
		OnDiskGB: 3.5, PartCount: 2,
	}})
	if !strings.Contains(out, ">Resume</button>") || !strings.Contains(out, ">Discard</button>") {
		t.Errorf("paused row must offer Resume and Discard; output=\n%s", out)
	}
	if strings.Contains(out, ">Pause</button>") {
		t.Errorf("paused row must not offer Pause; output=\n%s", out)
	}
	if strings.Contains(out, "dl-row-active") {
		t.Errorf("paused row must not carry the active marker class")
	}
	if !strings.Contains(out, "3.5 GiB on disk, 2 partial file(s)") {
		t.Errorf("paused row must show on-disk size; output=\n%s", out)
	}
	if !strings.Contains(out, "hx-confirm") {
		t.Errorf("Discard must keep its confirmation; output=\n%s", out)
	}
}

func TestDownloadsPanelEmpty(t *testing.T) {
	if out := strings.TrimSpace(renderDownloadsPanel(t, nil)); out != "" {
		t.Errorf("empty row set must render nothing (card hidden), got:\n%s", out)
	}
}
