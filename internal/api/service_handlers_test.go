package api

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/process"
)

// svcPost calls a service handler with a plain (non-HTMX) request.
func svcPost(h http.HandlerFunc, path string) *httptest.ResponseRecorder {
	rec := httptest.NewRecorder()
	h(rec, httptest.NewRequest(http.MethodPost, path, nil))
	return rec
}

// svcStatus decodes the JSON body of a service status answer.
func svcStatus(t *testing.T, rec *httptest.ResponseRecorder) serviceStatusResponse {
	t.Helper()
	var out serviceStatusResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("response is not a service status: %v\n%s", err, rec.Body.String())
	}
	return out
}

// svcAddModel registers a model with a saved config of the given context
// size.
func svcAddModel(t *testing.T, s *Server, id string, ctx int) {
	t.Helper()
	if err := s.registry.Add(&models.Model{
		ID: id, ModelID: "org/" + id + "-GGUF", Quant: "Q4_K_M", Filename: id + ".gguf",
		FilePath: filepath.Join(t.TempDir(), id+".gguf"),
	}); err != nil {
		t.Fatal(err)
	}
	cfg := models.DefaultConfig()
	cfg.Enabled = true
	cfg.ContextSize = ctx
	if err := s.registry.SetConfig(id, &cfg); err != nil {
		t.Fatal(err)
	}
}

// Start and restart refuse with 409 while a benchmark job drives the
// router, whether the queue reports the job or the job environment still
// owns the router. Restarting under a running cell would make it measure
// a config other than the one it records. The router is not started.
func TestSvcStartAndRestartRefusedWhileJobOwnsRouter(t *testing.T) {
	t.Run("job running on the queue", func(t *testing.T) {
		s, _ := bjhServer(t)
		bjhStartBusyJob(t, s)
		for name, h := range map[string]http.HandlerFunc{"start": s.handleServiceStart, "restart": s.handleServiceRestart} {
			rec := svcPost(h, "/api/service/"+name)
			if rec.Code != http.StatusConflict || !strings.Contains(rec.Body.String(), "cancel it first") {
				t.Errorf("%s: status %d, body %q; want 409 asking to cancel the job", name, rec.Code, rec.Body.String())
			}
		}
		if s.process.IsRunning() {
			t.Error("the router was started while a job was running")
		}
	})
	t.Run("job environment owns the router", func(t *testing.T) {
		s := newTestServer(t)
		env := newJobEnv(s)
		env.mu.Lock()
		env.ownsRouter = true
		env.mu.Unlock()
		s.env = env
		for name, h := range map[string]http.HandlerFunc{"start": s.handleServiceStart, "restart": s.handleServiceRestart} {
			if rec := svcPost(h, "/api/service/"+name); rec.Code != http.StatusConflict {
				t.Errorf("%s: status %d, want 409", name, rec.Code)
			}
		}
	})
}

// With no compiled build there is nothing to launch: start and restart
// fail and say what to do, instead of answering with a status that
// claims the router is starting.
func TestSvcStartAndRestartWithoutBuildFail(t *testing.T) {
	s := newTestServer(t)
	for name, c := range map[string]struct {
		h    http.HandlerFunc
		want int
	}{
		"start":   {s.handleServiceStart, http.StatusInternalServerError},
		"restart": {s.handleServiceRestart, http.StatusBadRequest},
	} {
		rec := svcPost(c.h, "/api/service/"+name)
		if rec.Code != c.want {
			t.Errorf("%s: status %d, want %d", name, rec.Code, c.want)
		}
		if !strings.Contains(rec.Body.String(), "build llama.cpp first") {
			t.Errorf("%s: message %q does not say to build first", name, rec.Body.String())
		}
	}
	if s.process.IsRunning() {
		t.Error("the router reports running with no build")
	}
}

// Start launches the router on the saved build, records which build is
// serving and the configs it launched with, clears the "restart needed"
// marks, and answers with the status and the engine name. Starting a
// router that is already running does not start a second one.
func TestSvcStartLaunchesRouterOnce(t *testing.T) {
	s, proc := evalStopServer(t)
	svcAddModel(t, s, "m1", 8192)
	s.markDirty("m1")

	rec := svcPost(s.handleServiceStart, "/api/service/start")
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	st := svcStatus(t, rec)
	if st.Engine != serviceEngine || st.PID == 0 {
		t.Errorf("status = %+v, want engine %q and a PID", st, serviceEngine)
	}
	waitRouterRunning(t, proc)
	if got := s.runningBuild(); got != "b1" {
		t.Errorf("running build = %q, want b1", got)
	}
	if live, ok := s.runningConfigFor("m1"); !ok || live.ContextSize != 8192 {
		t.Errorf("launch config of m1 = %+v, %v; want the saved config", live, ok)
	}
	if s.isDirty("m1") {
		t.Error("m1 is still marked as needing a restart after the router started")
	}

	pid := proc.PID()
	rec = svcPost(s.handleServiceStart, "/api/service/start")
	if rec.Code != http.StatusOK {
		t.Fatalf("second start: status %d: %s", rec.Code, rec.Body.String())
	}
	if proc.PID() != pid {
		t.Errorf("a second start replaced the running router (PID %d → %d)", pid, proc.PID())
	}
}

// Restart stops the router and starts it again through the same path as
// start, so a build selected after the first start is used. Restarting
// the old process with its old arguments would ignore the new choice.
func TestSvcRestartUsesNewlySelectedBuild(t *testing.T) {
	s, proc := evalStopServer(t)
	s.builder = testBuilder(t, testBuild("b1", fakeLlamaServerDir(t)), testBuild("b2", fakeLlamaServerDir(t)))
	s.cfg.ActiveBuild = "b1"
	if rec := svcPost(s.handleServiceStart, "/api/service/start"); rec.Code != http.StatusOK {
		t.Fatalf("start: status %d: %s", rec.Code, rec.Body.String())
	}
	waitRouterRunning(t, proc)
	firstPID := proc.PID()

	s.withConfig(func() { s.cfg.ActiveBuild = "b2" })
	rec := svcPost(s.handleServiceRestart, "/api/service/restart")
	if rec.Code != http.StatusOK {
		t.Fatalf("restart: status %d: %s", rec.Code, rec.Body.String())
	}
	waitRouterRunning(t, proc)
	if got := s.runningBuild(); got != "b2" {
		t.Errorf("running build after restart = %q, want b2", got)
	}
	if proc.PID() == firstPID {
		t.Error("restart did not replace the router process")
	}
}

// A benchmark start runs substitute configs from their own preset file.
// The launch snapshot must hold the substitute (so /info reports what
// really runs), the user's preset file must keep the saved config, and
// the "restart needed" marks must stay: the user's edits were not applied.
// An explicit build in the options wins over the saved selection.
func TestSvcStartRouterWithOverrides(t *testing.T) {
	s, proc := evalStopServer(t)
	s.builder = testBuilder(t, testBuild("b1", fakeLlamaServerDir(t)), testBuild("b2", fakeLlamaServerDir(t)))
	s.cfg.ActiveBuild = "b1"
	svcAddModel(t, s, "m1", 8192)
	s.markDirty("m1")

	sub := models.DefaultConfig()
	sub.Enabled = true
	sub.ContextSize = 2048
	if err := s.startRouterWith(routerOptions{buildID: "b2", overrides: map[string]*models.ModelConfig{"m1": &sub}}); err != nil {
		t.Fatalf("start: %v", err)
	}
	waitRouterRunning(t, proc)

	if got := s.runningBuild(); got != "b2" {
		t.Errorf("running build = %q, want the explicit b2", got)
	}
	live, ok := s.runningConfigFor("m1")
	if !ok || live.ContextSize != 2048 {
		t.Errorf("launch config of m1 = %+v, %v; want the substitute (2048)", live, ok)
	}
	sub.ContextSize = 1
	if live.ContextSize != 2048 {
		t.Error("the launch snapshot shares memory with the caller's substitute config")
	}
	if !strings.Contains(readPreset(t, s, models.BenchPresetFileName), "ctx-size = 2048") {
		t.Error("the benchmark preset does not carry the substitute context size")
	}
	if !s.isDirty("m1") {
		t.Error("a benchmark start cleared the user's restart-needed mark")
	}
	if saved, _ := s.registry.GetConfig("m1"); saved.ContextSize != 8192 {
		t.Errorf("saved context size = %d, want 8192", saved.ContextSize)
	}
}

// A benchmark start whose substitute preset cannot be written refuses to
// start: falling back to the saved preset would measure the wrong config.
func TestSvcStartRouterWithOverridesRefusesWhenPresetCannotBeWritten(t *testing.T) {
	s, proc := evalStopServer(t)
	svcAddModel(t, s, "m1", 8192)
	// A directory where the benchmark preset file should go makes the
	// write fail.
	if err := os.MkdirAll(filepath.Join(s.cfg.DataDir, "config", models.BenchPresetFileName, "x"), 0o755); err != nil {
		t.Fatal(err)
	}
	sub := models.DefaultConfig()
	err := s.startRouterWith(routerOptions{overrides: map[string]*models.ModelConfig{"m1": &sub}})
	if err == nil || !strings.Contains(err.Error(), "benchmark preset") {
		t.Fatalf("err = %v, want a benchmark preset error", err)
	}
	if proc.IsRunning() || proc.GetStatus().State != process.StateStopped {
		t.Errorf("the router started anyway: %+v", proc.GetStatus())
	}
}

// The log stream sends the router's buffered lines as server-sent events,
// so a page opened after the router started still sees its start-up log.
func TestSvcServiceLogsStreamsBufferedLines(t *testing.T) {
	s, _, _ := startFakeRouter(t)
	ts := httptest.NewServer(http.HandlerFunc(s.handleServiceLogs))
	t.Cleanup(ts.Close)

	resp, err := http.Get(ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if ct := resp.Header.Get("Content-Type"); ct != "text/event-stream" {
		t.Errorf("Content-Type = %q, want text/event-stream", ct)
	}
	lines := make(chan string)
	go func() {
		sc := bufio.NewScanner(resp.Body)
		for sc.Scan() {
			lines <- sc.Text()
		}
		close(lines)
	}()
	deadline := time.After(30 * time.Second)
	for {
		select {
		case l, ok := <-lines:
			if !ok {
				t.Fatal("the stream ended before the health line was sent")
			}
			if strings.HasPrefix(l, "data: ") && strings.Contains(l, "health check passed") {
				return
			}
		case <-deadline:
			t.Fatal("no buffered log line arrived")
		}
	}
}

// With the router stopped, the JSON answer says so and lists no models as
// an empty array (clients iterate it without a null check); the HTMX
// answer is an empty fragment.
func TestSvcLoadedModelsWhenRouterStopped(t *testing.T) {
	s := newTestServer(t)
	rec := httptest.NewRecorder()
	s.handleLoadedModels(rec, httptest.NewRequest(http.MethodGet, "/api/service/models", nil))
	var out map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("not JSON: %v", err)
	}
	if out["running"] != false || out["schema_version"] != float64(CapabilitiesSchemaVersion) {
		t.Errorf("answer = %v, want running false and the schema version", out)
	}
	if list, ok := out["models"].([]any); !ok || len(list) != 0 {
		t.Errorf("models = %#v, want an empty array", out["models"])
	}

	rec = httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/api/service/models", nil)
	req.Header.Set("HX-Request", "true")
	s.handleLoadedModels(rec, req)
	if rec.Body.Len() != 0 || !strings.HasPrefix(rec.Header().Get("Content-Type"), "text/html") {
		t.Errorf("HTMX answer = %q (%s), want an empty HTML fragment", rec.Body.String(), rec.Header().Get("Content-Type"))
	}
}

// With the router running, each router entry is listed with its status
// and, when a registry model is behind it, the registry ID and public
// name. The registry model is found by the router's ID or by one of the
// entry's aliases; an entry with no registry model is still listed.
func TestSvcLoadedModelsWhenRouterRunning(t *testing.T) {
	s, f, m := startFakeRouter(t)
	f.reset(map[string]string{
		m.PublicName(): "loaded",
		"section-x":    "unloaded",
		"stranger":     "loading",
	}, nil)
	f.mu.Lock()
	f.aliases = map[string][]string{"section-x": {"not-a-model", m.ID}}
	f.mu.Unlock()

	rec := httptest.NewRecorder()
	s.handleLoadedModels(rec, httptest.NewRequest(http.MethodGet, "/api/service/models", nil))
	var out struct {
		Running bool `json:"running"`
		Models  []struct {
			ID           string         `json:"id"`
			Status       string         `json:"status"`
			RegistryID   string         `json:"registry_id"`
			PublicName   string         `json:"public_name"`
			Capabilities map[string]any `json:"capabilities"`
		} `json:"models"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("not JSON: %v\n%s", err, rec.Body.String())
	}
	if !out.Running || len(out.Models) != 3 {
		t.Fatalf("answer = %+v, want running with 3 models", out)
	}
	for _, e := range out.Models {
		switch e.ID {
		case m.PublicName(), "section-x":
			if e.RegistryID != m.ID || e.PublicName != m.PublicName() || e.Capabilities == nil {
				t.Errorf("%s: registry id %q, public name %q, capabilities %v; want the registry model's", e.ID, e.RegistryID, e.PublicName, e.Capabilities)
			}
		case "stranger":
			if e.RegistryID != "" || e.Status != "loading" {
				t.Errorf("stranger = %+v, want no registry model and status loading", e)
			}
		default:
			t.Errorf("unexpected entry %q", e.ID)
		}
	}

	rec = httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/api/service/models", nil)
	req.Header.Set("HX-Request", "true")
	s.handleLoadedModels(rec, req)
	body := rec.Body.String()
	if !strings.Contains(body, "stranger") || !strings.Contains(body, "section-x") || !strings.Contains(body, "loading") {
		t.Errorf("HTMX fragment does not list the router's models:\n%s", body)
	}
}

// resolveRouterModel finds the registry model by the router entry's ID
// first and then by each alias in order; with no match it returns nil, so
// callers list the entry without registry data.
func TestSvcResolveRouterModel(t *testing.T) {
	s := newTestServer(t)
	svcAddModel(t, s, "m1", 4096)
	svcAddModel(t, s, "m2", 4096)

	var direct process.ModelStatus
	direct.ID, direct.Aliases = "m1", []string{"m2"}
	if reg, cfg := s.resolveRouterModel(direct); reg == nil || reg.ID != "m1" || cfg == nil {
		t.Errorf("by ID: got %v, want m1 with its config (the ID wins over aliases)", reg)
	}
	var byAlias process.ModelStatus
	byAlias.ID, byAlias.Aliases = "section", []string{"nobody", "m2"}
	if reg, _ := s.resolveRouterModel(byAlias); reg == nil || reg.ID != "m2" {
		t.Errorf("by alias: got %v, want m2", reg)
	}
	var none process.ModelStatus
	none.ID, none.Aliases = "x", []string{"y"}
	if reg, cfg := s.resolveRouterModel(none); reg != nil || cfg != nil {
		t.Errorf("no match: got %v, %v; want nil", reg, cfg)
	}
}

// A refused write to a read-only registry is a 409: the request was fine
// and the file on disk needs attention. Every other error keeps the
// caller's status, including when the read-only error is wrapped.
func TestSvcRegistryErrorStatus(t *testing.T) {
	dir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dir, "config"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "config", "models.json"), []byte("{broken"), 0o644); err != nil {
		t.Fatal(err)
	}
	reg := models.NewRegistry(dir, filepath.Join(dir, "models"))
	roErr := reg.Remove("anything")
	if !errors.Is(roErr, models.ErrRegistryReadOnly) {
		t.Fatalf("setup: Remove on a broken models.json returned %v, want a read-only error", roErr)
	}

	cases := []struct {
		name string
		err  error
		want int
	}{
		{"read-only", roErr, http.StatusConflict},
		{"wrapped read-only", fmt.Errorf("save: %w", roErr), http.StatusConflict},
		{"other error", errors.New("model not found: x"), http.StatusNotFound},
	}
	for _, c := range cases {
		if got := registryErrorStatus(c.err, http.StatusNotFound); got != c.want {
			t.Errorf("%s: status %d, want %d", c.name, got, c.want)
		}
	}
}

// effectiveFlags shows the flags the preset really gives the model. For
// a chat model they include the context size; for an embedding model the
// embeddings flag, the context capped at the model's own, and a
// micro-batch as large as the context, which the saved config does not
// have. On two GPUs with unequal memory
// the layers are split by size, so the shown flags carry that split even
// though the saved config has none, and the saved config is not changed.
func TestSvcEffectiveFlags(t *testing.T) {
	cfg := &models.ModelConfig{GPULayers: 999, ContextSize: 8192, Threads: 4, GPUAssign: "all"}
	m := &models.Model{ID: "m", ModelID: "org/M-GGUF", NLayers: 32, SizeBytes: 8 << 30}

	one := models.Target{GPUMiB: []int{24576}}
	chat := effectiveFlags(cfg, m, false, one)
	if !strings.Contains(chat, "--ctx-size 8192") || strings.Contains(chat, "--embeddings") {
		t.Errorf("chat flags = %q, want --ctx-size 8192 and no --embeddings", chat)
	}
	enc := &models.Model{ID: "e", ModelID: "org/E-GGUF", NLayers: 12, SizeBytes: 300 << 20,
		ContextLength: 2048, PoolingType: models.PoolingMean, Embedding: true}
	emb := effectiveFlags(cfg, enc, true, one)
	if !strings.Contains(emb, "--embeddings") || !strings.Contains(emb, "--ctx-size 2048") ||
		!strings.Contains(emb, "--ubatch-size 2048") {
		t.Errorf("embedding flags = %q, want --embeddings, --ctx-size 2048 and --ubatch-size 2048", emb)
	}
	if cfg.ContextSize != 8192 || cfg.UBatchSize != 0 {
		t.Errorf("effectiveFlags changed the saved config: ctx %d, ub %d", cfg.ContextSize, cfg.UBatchSize)
	}

	unequal := models.Target{GPUMiB: []int{24576, 8192}}
	split := effectiveFlags(cfg, m, false, unequal)
	if !strings.Contains(split, "--tensor-split") {
		t.Errorf("flags on unequal GPUs = %q, want a size-balanced --tensor-split", split)
	}
	if strings.Contains(cfg.EffectiveFlagsFor(false, unequal), "--tensor-split") {
		t.Error("the saved config itself has a tensor split; the test does not show the balancing")
	}
	if cfg.TensorSplit != "" {
		t.Errorf("effectiveFlags changed the saved config's TensorSplit to %q", cfg.TensorSplit)
	}
}
