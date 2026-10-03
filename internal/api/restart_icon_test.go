package api

import (
	"bytes"
	"html/template"
	"net/http/httptest"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/monitor"
	"github.com/tmac1973/llama-toolchest/internal/process"
	"github.com/tmac1973/llama-toolchest/web"
)

// The card always carries the slot the restart icon goes in, so a config
// change can fill it in without re-rendering the list.
func TestModelCardRestartSlot(t *testing.T) {
	m := models.Model{ID: "org--r-GGUF--m-Q4_K_M", ModelID: "org/r-GGUF", Quant: "Q4_K_M"}
	slot := `id="restart-` + domID(m.ID) + `"`

	plain := renderModelCardPartial(t, modelCardData{Model: m})
	if !strings.Contains(plain, slot) || strings.Contains(plain, "Restart server to apply config changes") {
		t.Errorf("without a pending change: slot present, no icon; got\n%s", plain)
	}
	pending := renderModelCardPartial(t, modelCardData{Model: m, NeedsReload: true})
	if !strings.Contains(pending, slot) || !strings.Contains(pending, "Restart server to apply config changes") {
		t.Errorf("with a pending change: slot and icon; got\n%s", pending)
	}
}

// The Models page fills the slot when a response says a restart is needed.
func TestModelsPageListensForRestartNeeded(t *testing.T) {
	page, err := web.Templates.ReadFile("templates/models.html")
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{`addEventListener('restartNeeded'`, `id="restart-icon-template"`, `'restart-' + detail.dom`} {
		if !strings.Contains(string(page), want) {
			t.Errorf("models.html lacks %q", want)
		}
	}
}

// A config change with the server running marks the model and tells the
// page so — the path Autotune's "Use these settings" takes.
func TestConfigChangeSignalsRestartNeeded(t *testing.T) {
	s, proc := evalStopServer(t)
	m := &models.Model{ID: "org--r-GGUF--m-Q4_K_M", ModelID: "org/r-GGUF", Filename: "m-Q4_K_M.gguf"}
	if err := s.registry.Add(m); err != nil {
		t.Fatal(err)
	}
	s.dirtyModels = map[string]bool{}
	s.monitor = monitor.New(time.Hour) // never polled: no GPUs
	cfg, _ := s.registry.GetConfig(m.ID)

	// Stopped: nothing to restart, nothing signalled.
	w := httptest.NewRecorder()
	r := httptest.NewRequest("POST", "/", nil)
	r.Header.Set("HX-Request", "true")
	s.afterConfigChange(w, r, m.ID, cfg)
	if strings.Contains(w.Header().Get("HX-Trigger"), "restartNeeded") || s.isDirty(m.ID) {
		t.Error("signalled a restart with the server stopped")
	}

	if err := proc.Start(process.RouterConfig{
		BinaryPath: filepath.Join(s.builderDataDirForTest(t), "llama-server"),
		Host:       "127.0.0.1", Port: s.cfg.LlamaPort, ModelsMax: 1,
	}); err != nil {
		t.Fatalf("start: %v", err)
	}
	waitRouterRunning(t, proc)

	w = httptest.NewRecorder()
	s.afterConfigChange(w, r, m.ID, cfg)
	trig := w.Header().Get("HX-Trigger")
	if !s.isDirty(m.ID) || !strings.Contains(trig, `"restartNeeded":{"dom":"`+domID(m.ID)+`"}`) {
		t.Errorf("dirty=%v, HX-Trigger=%s", s.isDirty(m.ID), trig)
	}
}

// The Configure panel's "Restart required" shows only while a change
// waits for a restart; it used to be there on every panel.
func TestConfigPanelRestartLabel(t *testing.T) {
	base, err := template.New("").Funcs(testFuncMap).ParseFS(web.Templates,
		"templates/layout.html", "templates/partials/*.html")
	if err != nil {
		t.Fatal(err)
	}
	cfg := &models.ModelConfig{GPULayers: 999, ContextSize: 8192}
	render := func(pending bool) string {
		var buf bytes.Buffer
		data := modelConfigPanelData{ModelID: "test-id", Config: cfg, DraftModes: models.DraftModes(),
			AssistModes: models.AssistModes(), NeedsRestart: pending}
		if err := base.ExecuteTemplate(&buf, "model_config", data); err != nil {
			t.Fatal(err)
		}
		return buf.String()
	}
	label := regexp.MustCompile(`id="mc-restart-test-id"[^>]*>([^<]*)</span>`)
	for pending, want := range map[bool]string{false: "", true: "Restart required"} {
		m := label.FindStringSubmatch(render(pending))
		if m == nil || m[1] != want {
			t.Errorf("pending=%v: label %q, want %q", pending, m, want)
		}
	}

	page, _ := web.Templates.ReadFile("templates/models.html")
	if !strings.Contains(string(page), `'mc-restart-' + detail.dom`) {
		t.Error("the restartNeeded listener does not update the panel's label")
	}
}
