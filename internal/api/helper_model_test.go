package api

import (
	"context"
	"errors"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/huggingface"
	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/modelsource"
	"github.com/tmac1973/llama-toolchest/internal/process"
)

func newHelperServer(t *testing.T) *Server {
	t.Helper()
	s := newProfileServer(t, "")
	dir := t.TempDir()
	s.downloader = huggingface.NewDownloader(dir, dir, "")
	s.configPath = filepath.Join(dir, "llama-toolchest.yaml")
	return s
}

func addDefaultHelper(t *testing.T, s *Server) string {
	t.Helper()
	m := &models.Model{ID: "unsloth--Qwen3.5-4B-GGUF--Qwen3.5-4B-Q4_K_M", ModelID: defaultHelperRepo,
		Filename: "Qwen3.5-4B-Q4_K_M.gguf", Quant: defaultHelperQuant, ContextLength: 262144}
	if err := s.registry.Add(m); err != nil {
		t.Fatal(err)
	}
	return m.ID
}

func TestPickHelperFileChoosesTheRecommendedQuant(t *testing.T) {
	f, ok := pickHelperFile(&modelsource.Detail{Files: []modelsource.File{
		{Filename: "mmproj-Q4_K_M.gguf", Quant: "Q4_K_M"},
		{Filename: "Qwen3.5-4B-Q8_0.gguf", Quant: "Q8_0"},
		{Filename: "Qwen3.5-4B-Q4_K_M.gguf", Quant: "Q4_K_M", Size: 2 << 30},
	}})
	if !ok || f.Filename != "Qwen3.5-4B-Q4_K_M.gguf" {
		t.Errorf("picked %+v, %v", f, ok)
	}
	if _, ok := pickHelperFile(&modelsource.Detail{Files: []modelsource.File{{Filename: "x-Q8_0.gguf", Quant: "Q8_0"}}}); ok {
		t.Error("picked a file without the recommended quant")
	}
}

// A model becomes the helper by being downloaded as one, and an
// already-installed recommended model is adopted at startup.
func TestHelperModelIsTheOneDownloadedAsSuch(t *testing.T) {
	s := newHelperServer(t)
	if m := s.helperModel(); m != nil {
		t.Fatalf("helper without one installed = %v", m.ID)
	}
	id := addDefaultHelper(t, s)
	if m := s.helperModel(); m != nil {
		t.Fatalf("an ordinary model was taken as the helper: %v", m.ID)
	}

	// Installed before helper models had a role: adopted at startup.
	s.adoptExistingHelper()
	m := s.helperModel()
	if m == nil || m.ID != id {
		t.Fatalf("recommended model not adopted: %v", m)
	}
	if !m.HelperRole {
		t.Error("the adopted model is not marked as a helper")
	}
	// A second start changes nothing.
	s.adoptExistingHelper()
	if len(s.registry.ListHelpers()) != 1 {
		t.Errorf("helpers = %d, want 1", len(s.registry.ListHelpers()))
	}
}

// A finished download claims the helper role only when it is the file the
// Settings panel asked for.
func TestClaimDownloadedHelper(t *testing.T) {
	s := newHelperServer(t)
	id := addDefaultHelper(t, s)
	m, _ := s.registry.Get(id)

	s.claimDownloadedHelper(m)
	if len(s.registry.ListHelpers()) != 0 {
		t.Fatal("a download nobody asked for became the helper")
	}

	s.cfg.PendingHelper = m.ModelID + "|" + m.Filename
	s.claimDownloadedHelper(m)
	if h := s.helperModel(); h == nil || h.ID != id {
		t.Fatalf("the requested download did not become the helper: %v", h)
	}
	if s.cfg.PendingHelper != "" {
		t.Error("the pending marker was not cleared")
	}
	if cfg, _ := s.registry.GetConfig(id); cfg.ContextSize < models.HelperContextSmall {
		t.Errorf("the helper was left on ordinary defaults: context %d", cfg.ContextSize)
	}
}

// Helper models are the app's own: not in the chat list, not offered to
// clients, and not benchmark targets.
func TestHelperModelIsNotOfferedElsewhere(t *testing.T) {
	s := newHelperServer(t)
	id := addDefaultHelper(t, s)
	if err := s.registry.SetHelperRole(id, true); err != nil {
		t.Fatal(err)
	}
	for _, m := range s.registry.ListServing() {
		if m.ID == id {
			t.Error("the helper model is in the serving list")
		}
	}
	if len(s.registry.ListHelpers()) != 1 {
		t.Error("the helper model is not in the helper list")
	}
}

// The helper's settings are not the user's: they are recomputed before
// every use, and a hand edit is corrected.
func TestEnsureHelperConfigAppliesFixedSettings(t *testing.T) {
	s := newHelperServer(t)
	id := addDefaultHelper(t, s)
	cfg, _ := s.registry.GetConfig(id)
	edited := *cfg
	edited.Enabled, edited.ContextSize, edited.GPULayers, edited.SpecType = false, 8192, 5, "draft-mtp"
	s.registry.SetConfig(id, &edited)

	changed, err := s.ensureHelperConfig(id)
	if err != nil || !changed {
		t.Fatalf("ensureHelperConfig = %v, %v; want a change", changed, err)
	}
	got, _ := s.registry.GetConfig(id)
	if !got.Enabled || got.GPULayers != 999 || got.SpecType != "" || got.ContextSize < models.HelperContextSmall {
		t.Errorf("helper config = %+v", got)
	}
	if changed, _ := s.ensureHelperConfig(id); changed {
		t.Error("a second call changed it again")
	}
}

// The Settings panel offers a download when there is no helper, and its
// managed settings plus Remove when there is.
func TestHelperPanel(t *testing.T) {
	s := newHelperServer(t)
	panel := func() string {
		rec := httptest.NewRecorder()
		s.handleHelperPanel(rec, httptest.NewRequest("GET", "/api/helper-model/panel", nil))
		return rec.Body.String()
	}
	out := panel()
	if !strings.Contains(out, "No helper model is installed") || !strings.Contains(out, "/api/helper-model/download") {
		t.Errorf("empty panel lacks the explanation or the download button:\n%s", out)
	}

	id := addDefaultHelper(t, s)
	if err := s.registry.SetHelperRole(id, true); err != nil {
		t.Fatal(err)
	}
	out = panel()
	for _, want := range []string{"is installed", "Settings are managed by llama-toolchest", "tokens of context", "/api/helper-model/remove"} {
		if !strings.Contains(out, want) {
			t.Errorf("panel missing %q:\n%s", want, out)
		}
	}
	if strings.Contains(out, "/api/helper-model/download") {
		t.Error("download button shown although a helper is installed")
	}

	// Removing it frees the disk and brings the download offer back.
	rec := httptest.NewRecorder()
	s.handleRemoveHelperModel(rec, httptest.NewRequest("POST", "/api/helper-model/remove", nil))
	if !strings.Contains(rec.Body.String(), "Removed") || len(s.registry.ListHelpers()) != 0 {
		t.Errorf("remove failed:\n%s", rec.Body.String())
	}
	if !strings.Contains(panel(), "/api/helper-model/download") {
		t.Error("the download offer did not come back")
	}
}

// The Models page lists helper models in a section of their own, with
// their managed settings and nothing to configure.
func TestHelperModelsSectionOnModelsPage(t *testing.T) {
	s := newHelperServer(t)
	id := addDefaultHelper(t, s)
	list := func() string {
		req := httptest.NewRequest("GET", "/api/models/helpers", nil)
		req.Header.Set("HX-Request", "true")
		rec := httptest.NewRecorder()
		s.handleListHelperModels(rec, req)
		return rec.Body.String()
	}
	if out := list(); strings.TrimSpace(out) != "" {
		t.Errorf("a section was rendered with no helper models:\n%s", out)
	}
	if err := s.registry.SetHelperRole(id, true); err != nil {
		t.Fatal(err)
	}
	out := list()
	for _, want := range []string{"Helper models", "Managed settings", "tokens of context", "/api/models/helpers/remove"} {
		if !strings.Contains(out, want) {
			t.Errorf("helper section missing %q:\n%s", want, out)
		}
	}
	for _, gone := range []string{"/config", "autoconfig"} {
		if strings.Contains(out, gone) {
			t.Errorf("helper section offers %q", gone)
		}
	}
}

// The wait loop: a router that is not answering yet is waited for; one
// that answers without the model is restarted once and then given
// another chance; and a model that never appears is reported plainly.
func TestWaitForRouterModel(t *testing.T) {
	base := func() routerWait {
		return routerWait{
			alive: func() bool { return true }, name: "helper",
			settle: 5 * time.Millisecond, timeout: 2 * time.Second, interval: time.Millisecond,
			restart: func() error { return nil },
		}
	}

	t.Run("waits for a router that is still starting", func(t *testing.T) {
		calls := 0
		w := base()
		w.answers = func() (int, error) {
			calls++
			if calls < 5 {
				return 0, errors.New("connection refused")
			}
			return 4, nil
		}
		w.knows = func() bool { return calls >= 5 }
		if err := waitForRouterModel(context.Background(), w); err != nil {
			t.Fatalf("err = %v", err)
		}
	})

	t.Run("restarts once when the model is missing", func(t *testing.T) {
		restarts := 0
		w := base()
		w.answers = func() (int, error) { return 3, nil }
		w.knows = func() bool { return restarts > 0 }
		w.restart = func() error { restarts++; return nil }
		if err := waitForRouterModel(context.Background(), w); err != nil {
			t.Fatalf("err = %v", err)
		}
		if restarts != 1 {
			t.Errorf("restarts = %d, want 1", restarts)
		}
	})

	t.Run("reports a model that never appears", func(t *testing.T) {
		restarts := 0
		w := base()
		w.answers = func() (int, error) { return 3, nil }
		w.knows = func() bool { return false }
		w.restart = func() error { restarts++; return nil }
		err := waitForRouterModel(context.Background(), w)
		if err == nil || !strings.Contains(err.Error(), "still not in its list") {
			t.Errorf("err = %v", err)
		}
		if restarts != 1 {
			t.Errorf("restarts = %d, want exactly 1", restarts)
		}
	})

	t.Run("gives up when the server stops", func(t *testing.T) {
		w := base()
		w.alive = func() bool { return false }
		w.answers = func() (int, error) { return 0, errors.New("no") }
		w.knows = func() bool { return false }
		if err := waitForRouterModel(context.Background(), w); err == nil || !strings.Contains(err.Error(), "server stopped") {
			t.Errorf("err = %v", err)
		}
	})
}

// A router that has just been started reports "starting" until its first
// health check passes. Treating that as stopped ended a run that was
// about to work.
func TestRouterAlive(t *testing.T) {
	for state, want := range map[string]bool{
		process.StateRunning: true, process.StateStarting: true,
		process.StateStopped: false, process.StateFailed: false, "": false,
	} {
		if got := routerAlive(state); got != want {
			t.Errorf("routerAlive(%q) = %v, want %v", state, got, want)
		}
	}
}
