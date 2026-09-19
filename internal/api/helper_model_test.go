package api

import (
	"context"
	"errors"
	"net/http/httptest"
	"net/url"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/huggingface"
	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/modelsource"
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

func TestHelperModelResolution(t *testing.T) {
	s := newHelperServer(t)
	if m, _ := s.helperModel(); m != nil {
		t.Fatalf("helper without any installed = %v", m.ID)
	}
	def := addDefaultHelper(t, s)
	if m, chosen := s.helperModel(); m == nil || m.ID != def || chosen {
		t.Fatalf("default not used when installed and nothing chosen: %v %v", m, chosen)
	}
	s.cfg.HelperModelID = profTestID
	if m, chosen := s.helperModel(); m == nil || m.ID != profTestID || !chosen {
		t.Fatalf("chosen model not used: %v %v", m, chosen)
	}
	s.cfg.HelperModelID = "gone"
	if m, chosen := s.helperModel(); m == nil || m.ID != def || chosen {
		t.Errorf("an uninstalled choice should fall back to the default: %v %v", m, chosen)
	}
}

func TestEnsureHelperConfigRaisesContextAndEnables(t *testing.T) {
	s := newHelperServer(t)
	id := addDefaultHelper(t, s)
	cfg, _ := s.registry.GetConfig(id)
	next := *cfg
	next.Enabled, next.ContextSize = false, 8192
	s.registry.SetConfig(id, &next)

	changed, err := s.ensureHelperConfig(id)
	if err != nil || !changed {
		t.Fatalf("ensureHelperConfig = %v, %v; want a change", changed, err)
	}
	cfg, _ = s.registry.GetConfig(id)
	if !cfg.Enabled || cfg.ContextSize != helperMinContext {
		t.Errorf("helper config = enabled %v, ctx %d", cfg.Enabled, cfg.ContextSize)
	}
	if changed, _ := s.ensureHelperConfig(id); changed {
		t.Error("a second call changed it again")
	}
}

func TestHelperPanelAndSave(t *testing.T) {
	s := newHelperServer(t)
	rec := httptest.NewRecorder()
	s.handleHelperPanel(rec, httptest.NewRequest("GET", "/api/helper-model/panel", nil))
	out := rec.Body.String()
	if !strings.Contains(out, "No helper model is installed yet") || !strings.Contains(out, "/api/helper-model/download") {
		t.Errorf("empty panel lacks the explanation or the download button:\n%s", out)
	}

	addDefaultHelper(t, s)
	req := httptest.NewRequest("PUT", "/api/helper-model", strings.NewReader(url.Values{"helper_model_id": {profTestID}}.Encode()))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	rec = httptest.NewRecorder()
	s.handleSetHelperModel(rec, req)
	out = rec.Body.String()
	if s.cfg.HelperModelID != profTestID || !strings.Contains(out, "Autoconfigure will use") {
		t.Errorf("save did not take: id %q\n%s", s.cfg.HelperModelID, out)
	}
	if !strings.Contains(out, "context was raised") {
		t.Errorf("the context change is not reported:\n%s", out)
	}
	if strings.Contains(out, "/api/helper-model/download") {
		t.Error("download button shown although the recommended model is installed")
	}
}

// The wait loop: a router that is not answering yet is waited for; one
// that answers without the model is restarted once and then given
// another chance; and a model that never appears is reported plainly.
func TestWaitForRouterModel(t *testing.T) {
	base := func() routerWait {
		return routerWait{
			running: func() bool { return true }, name: "helper",
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
		w.running = func() bool { return false }
		w.answers = func() (int, error) { return 0, errors.New("no") }
		w.knows = func() bool { return false }
		if err := waitForRouterModel(context.Background(), w); err == nil || !strings.Contains(err.Error(), "server stopped") {
			t.Errorf("err = %v", err)
		}
	})
}
