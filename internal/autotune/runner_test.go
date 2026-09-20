package autotune

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/benchmark"
	"github.com/tmac1973/llama-toolchest/internal/evaluate"
	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/monitor"
)

// fakeEnv is the job environment the stages run in: it records the config
// each cell applies, and serves a router whose speeds depend on it.
type fakeEnv struct {
	mu      sync.Mutex
	current benchmark.ConfigSnapshot
	saved   benchmark.ConfigSnapshot
	url     string
	// failWhen makes a cell fail, for the case where a setting cannot run
	// on the machine.
	failWhen func(benchmark.ConfigSnapshot) bool
	// delay slows every request, so a test that needs a run to still be
	// going when it does something has time to do it.
	delay   time.Duration
	cleared int
}

func (e *fakeEnv) CheckBuildRunnable(context.Context, string) error { return nil }
func (e *fakeEnv) EnsureBuildActive(context.Context, string, bool) error {
	return nil
}

func (e *fakeEnv) ResolveModel(id string) (benchmark.ModelInfo, error) {
	return benchmark.ModelInfo{
		ID: id, HFRepoID: "org/m-GGUF", Quant: "Q4_K_M", SizeGiB: 5, SizeBytes: 5 << 30,
		DisplayName: "M", RouterName: "m", Config: e.saved,
	}, nil
}

func (e *fakeEnv) ResolveModelPath(id string) (string, error) { return "/models/" + id + ".gguf", nil }

func (e *fakeEnv) ApplyEphemeralConfig(_ context.Context, _ string, cfg benchmark.ConfigSnapshot, _ *models.ModelConfig) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.failWhen != nil && e.failWhen(cfg) {
		return fmt.Errorf("this machine cannot run that setting")
	}
	e.current = cfg
	return nil
}

func (e *fakeEnv) ClearEphemeralConfig(context.Context) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.cleared++
	return nil
}

func (e *fakeEnv) ResolveBuild(id string) benchmark.BuildSnapshot {
	return benchmark.BuildSnapshot{ID: id, Profile: "rocm", GitRef: "b1"}
}
func (e *fakeEnv) CurrentMetrics() monitor.Metrics { return monitor.Metrics{} }
func (e *fakeEnv) RouterURL() string               { return e.url }
func (e *fakeEnv) HFToken() string                 { return "" }
func (e *fakeEnv) HFCacheDir() string              { return "" }

// Capability cells are never part of an autotune, so these are here only
// to satisfy the interface.
func (e *fakeEnv) StopRouterForEval(context.Context) error { return nil }
func (e *fakeEnv) EvalBinary(string) (string, error)       { return "", fmt.Errorf("not used") }
func (e *fakeEnv) EnsureEvalData(context.Context, evaluate.Mode) (string, error) {
	return "", fmt.Errorf("not used")
}
func (e *fakeEnv) ResolveKLReference(string, string) (benchmark.ModelInfo, error) {
	return benchmark.ModelInfo{}, fmt.Errorf("not used")
}
func (e *fakeEnv) EvalFlags(string, benchmark.ConfigSnapshot, string) ([]string, error) {
	return nil, fmt.Errorf("not used")
}
func (e *fakeEnv) EnsureKLBase(context.Context, benchmark.ModelInfo, benchmark.ConfigSnapshot, int, string, func(string)) (string, error) {
	return "", fmt.Errorf("not used")
}
func (e *fakeEnv) RunEval(context.Context, evaluate.Spec) (evaluate.Result, error) {
	return evaluate.Result{}, fmt.Errorf("not used")
}
func (e *fakeEnv) MeasuredMemory(string) (benchmark.MemorySnapshot, bool) {
	return benchmark.MemorySnapshot{}, false
}

// speeds is the machine this fake models: a prompt batch of 1024 reads
// prompts fastest, MTP writes faster than nothing, and an n-gram assist
// on top of MTP is faster again — the combination autotune exists to
// find.
func (e *fakeEnv) speeds() (pp, tg float64) {
	e.mu.Lock()
	cfg := e.current
	e.mu.Unlock()

	pp = 800
	switch cfg.UBatchSize {
	case 1024:
		pp = 1400
	case 2048, 4096:
		pp = 1200
	case 256:
		pp = 600
	}
	tg = 40
	if cfg.SpecType == "draft-mtp" {
		tg = 60
		if cfg.DraftMax >= 9 {
			tg = 64 // a longer draft helps a little
		}
	}
	if cfg.SpecAssist != "" {
		tg += 15
	}
	return pp, tg
}

// newFakeRouter answers model loads and chat completions with timings
// taken from the config the last cell applied.
func newFakeRouter(t *testing.T, env *fakeEnv) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.HasSuffix(r.URL.Path, "/models/load"), strings.HasSuffix(r.URL.Path, "/models/unload"):
			w.WriteHeader(http.StatusOK)
		case strings.HasSuffix(r.URL.Path, "/models"):
			json.NewEncoder(w).Encode(map[string]any{"data": []any{}})
		case strings.HasSuffix(r.URL.Path, "/v1/chat/completions"):
			env.mu.Lock()
			delay := env.delay
			env.mu.Unlock()
			if delay > 0 {
				time.Sleep(delay)
			}
			var body struct {
				MaxTokens int `json:"max_tokens"`
				Messages  []struct {
					Content string `json:"content"`
				} `json:"messages"`
			}
			json.NewDecoder(r.Body).Decode(&body)
			promptTokens := 0
			if len(body.Messages) > 0 {
				promptTokens = len(body.Messages[0].Content) / benchmark.BenchPromptCharsPerToken
			}
			pp, tg := env.speeds()
			json.NewEncoder(w).Encode(map[string]any{
				"choices": []map[string]any{{"message": map[string]any{"content": "ok"}, "finish_reason": "stop"}},
				"timings": map[string]any{
					"prompt_n":             promptTokens,
					"prompt_ms":            float64(promptTokens) / pp * 1000,
					"prompt_per_second":    pp,
					"predicted_n":          body.MaxTokens,
					"predicted_ms":         float64(body.MaxTokens) / tg * 1000,
					"predicted_per_second": tg,
				},
			})
		default:
			w.WriteHeader(http.StatusOK)
		}
	}))
	t.Cleanup(srv.Close)
	env.url = srv.URL
	return srv
}

// tuneFixture is a runner with a model, a starting profile and a fake
// machine to measure on.
type tuneFixture struct {
	runner *Runner
	store  *Store
	reg    *models.Registry
	runs   *benchmark.Store
	env    *fakeEnv
	dir    string
}

const tuneModelID = "org--m-GGUF--m-Q4_K_M"

func newFixture(t *testing.T, mtpLayers int) *tuneFixture {
	t.Helper()
	dir := t.TempDir()
	reg := models.NewRegistry(dir, filepath.Join(dir, "models"))
	m := &models.Model{ID: tuneModelID, ModelID: "org/m-GGUF", Filename: "m-Q4_K_M.gguf", Quant: "Q4_K_M",
		NLayers: 36, ContextLength: 131072, SizeBytes: 5 << 30, NextNLayers: mtpLayers}
	if err := reg.Add(m); err != nil {
		t.Fatal(err)
	}
	if _, err := reg.SaveProfile(tuneModelID, "Autoconfig", models.ProfileSourceAutoconfig, "b1"); err != nil {
		t.Fatal(err)
	}

	env := &fakeEnv{saved: benchmark.ConfigSnapshot{GPULayers: 999, ContextSize: 8192, Threads: 8, FlashAttention: true}}
	newFakeRouter(t, env)
	runs := benchmark.NewStore(dir, nil)
	f := &tuneFixture{
		store: NewStore(dir), reg: reg, runs: runs, env: env, dir: dir,
	}
	f.runner = NewRunner(Deps{
		Store: f.store, Runs: runs, Jobs: benchmark.NewJobQueue(runs, env), Registry: reg,
		ActiveBuild: func() string { return "b1" },
		Hardware:    func() (int, int) { return 1, 16 },
	})
	return f
}

// waitForRun waits for a run to reach a terminal status.
func (f *tuneFixture) waitForRun(t *testing.T, id string) *Autotune {
	t.Helper()
	deadline := time.Now().Add(60 * time.Second)
	for time.Now().Before(deadline) {
		rec, ok := f.store.Get(id)
		if ok {
			switch rec.Status {
			case StatusDone, StatusFailed, StatusCancelled:
				return rec
			}
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatal("the autotune run did not finish")
	return nil
}

// A whole run on a machine where MTP plus an n-gram assist is fastest:
// four stages, and a profile per goal.
func TestRunnerFindsTheFastestSettings(t *testing.T) {
	f := newFixture(t, 1) // the model has its own MTP draft layers

	rec, err := f.runner.Start(tuneModelID, "Autoconfig", UseCode)
	if err != nil {
		t.Fatal(err)
	}
	rec = f.waitForRun(t, rec.ID)
	if rec.Status != StatusDone {
		t.Fatalf("status = %s (%s)", rec.Status, rec.Error)
	}
	if len(rec.Stages) != len(StageOrder) {
		t.Fatalf("stages = %v", stageKeys(rec))
	}
	for _, st := range rec.Stages {
		if st.Status != StatusDone {
			t.Errorf("stage %s = %s", st.Key, st.Status)
		}
	}

	// Generation: MTP with an assist, as the fake machine is built.
	gen := rec.Results[GoalGeneration]
	if !gen.Saved {
		t.Fatalf("generation outcome = %+v", gen)
	}
	if !strings.Contains(gen.Winner.Config.SpecType, "draft-mtp") || gen.Winner.Config.SpecAssist == "" {
		t.Errorf("generation winner = %+v, want MTP with an assist", gen.Winner.Values)
	}
	// Prompt speed: the 1024 prompt batch.
	prompt := rec.Results[GoalPrompt]
	if !prompt.Saved || prompt.Winner.Config.UBatchSize != 1024 {
		t.Errorf("prompt winner = %+v", prompt.Winner.Values)
	}

	// The profiles are saved, with what was measured.
	profiles := f.reg.Profiles(tuneModelID)
	var autotuned []models.ConfigProfile
	for _, p := range profiles {
		if p.Source == models.ProfileSourceAutotune {
			autotuned = append(autotuned, p)
		}
	}
	if len(autotuned) == 0 {
		t.Fatalf("no autotune profile was saved; profiles = %v", profileNames(profiles))
	}
	for _, p := range autotuned {
		if !strings.HasPrefix(p.Name, "Autotune – fastest ") {
			t.Errorf("profile name = %q", p.Name)
		}
		if p.Measured == nil || p.Measured.Workload != string(UseCode) || p.Measured.AutotuneID != rec.ID {
			t.Errorf("profile %q measurements = %+v", p.Name, p.Measured)
		}
		if p.Measured.TGTokPerSec <= p.Measured.BaselineTG && p.Measured.PPTokPerSec <= p.Measured.BaselinePP {
			t.Errorf("profile %q is not faster than the baseline: %+v", p.Name, p.Measured)
		}
		if len(p.Notes) == 0 {
			t.Errorf("profile %q has no explanation", p.Name)
		}
	}

	// The live config is untouched: autotune saves profiles, it does not
	// apply them.
	cfg, _ := f.reg.GetConfig(tuneModelID)
	if cfg.SpecType != "" || cfg.UBatchSize != 0 {
		t.Errorf("the live config was changed: %+v", cfg)
	}
}

// A machine where nothing beats the starting profile saves nothing, and
// says so.
func TestRunnerKeepsQuietWhenNothingIsFaster(t *testing.T) {
	f := newFixture(t, 0) // no MTP: nothing to gain on this fake machine
	f.env.saved.UBatchSize = 1024

	rec, err := f.runner.Start(tuneModelID, "Autoconfig", UseChat)
	if err != nil {
		t.Fatal(err)
	}
	rec = f.waitForRun(t, rec.ID)
	if rec.Status != StatusDone {
		t.Fatalf("status = %s (%s)", rec.Status, rec.Error)
	}
	for _, g := range Goals {
		out, ok := rec.Results[g]
		if !ok {
			continue
		}
		if !out.Saved && !strings.Contains(out.Message, "already the") {
			t.Errorf("%s: message = %q", g, out.Message)
		}
	}
	// Whatever it decided, it never saved a profile that is not faster.
	for _, p := range f.reg.Profiles(tuneModelID) {
		if p.Source != models.ProfileSourceAutotune || p.Measured == nil {
			continue
		}
		m := p.Measured
		if m.TGTokPerSec <= m.BaselineTG && m.PPTokPerSec <= m.BaselinePP && m.ResponseSec >= m.BaselineResponseSec {
			t.Errorf("profile %q was saved without being faster: %+v", p.Name, m)
		}
	}
}

// A setting this machine cannot run is reported, not silently dropped.
func TestRunnerRecordsSettingsThatCannotRun(t *testing.T) {
	f := newFixture(t, 1)
	f.env.failWhen = func(cfg benchmark.ConfigSnapshot) bool { return cfg.UBatchSize == 4096 }

	rec, err := f.runner.Start(tuneModelID, "Autoconfig", UseChat)
	if err != nil {
		t.Fatal(err)
	}
	rec = f.waitForRun(t, rec.ID)
	if rec.Status != StatusDone {
		t.Fatalf("status = %s (%s)", rec.Status, rec.Error)
	}
	batch, _ := rec.Stage(StageBatch)
	if len(batch.Failed) == 0 {
		t.Fatal("a setting that could not run was not recorded")
	}
	found := false
	for _, fc := range batch.Failed {
		if strings.Contains(fc.Label, "4096") {
			found = true
			if fc.Error == "" {
				t.Error("the failure has no reason")
			}
		}
	}
	if !found {
		t.Errorf("failed settings = %+v", batch.Failed)
	}
}

// Start refuses what it cannot measure.
func TestRunnerStartRefusals(t *testing.T) {
	f := newFixture(t, 1)
	if _, err := f.runner.Start(tuneModelID, "No Such Profile", UseChat); err == nil {
		t.Error("a missing profile was accepted")
	}
	if _, err := f.runner.Start(tuneModelID, "Autoconfig", "nonsense"); err == nil {
		t.Error("an unknown use case was accepted")
	}
	f2 := newFixture(t, 1)
	f2.runner.deps.Busy = func() string { return "A benchmark is running." }
	_, err := f2.runner.Start(tuneModelID, "Autoconfig", UseChat)
	if err == nil || !strings.Contains(err.Error(), "benchmark") {
		t.Errorf("err = %v, want the busy reason", err)
	}
}

// A run interrupted by a restart resumes without measuring the stages it
// already finished.
func TestRunnerResumesAfterARestart(t *testing.T) {
	f := newFixture(t, 1)
	rec, err := f.runner.Start(tuneModelID, "Autoconfig", UseChat)
	if err != nil {
		t.Fatal(err)
	}
	rec = f.waitForRun(t, rec.ID)
	if rec.Status != StatusDone {
		t.Fatalf("status = %s (%s)", rec.Status, rec.Error)
	}
	batchJob, _ := rec.Stage(StageBatch)
	cellsBefore := jobCellCount(t, f.runs, batchJob.JobID)

	// Pretend the server stopped during the confirming stage.
	rec.Status = StatusRunning
	confirm, _ := rec.Stage(StageConfirm)
	confirm.Status = StatusRunning
	rec.SetStage(*confirm)
	rec.Results = nil
	if err := f.store.Save(rec); err != nil {
		t.Fatal(err)
	}
	reloaded := NewStore(f.dir)
	again, ok := reloaded.Get(rec.ID)
	if !ok || again.Status != StatusInterrupted {
		t.Fatalf("after a restart: %+v", again)
	}

	f.runner.deps.Store = reloaded
	if _, err := f.runner.Resume(again.ID); err != nil {
		t.Fatal(err)
	}
	f.store = reloaded
	done := f.waitForRun(t, again.ID)
	if done.Status != StatusDone {
		t.Fatalf("resumed status = %s (%s)", done.Status, done.Error)
	}
	if got := jobCellCount(t, f.runs, batchJob.JobID); got != cellsBefore {
		t.Errorf("the first stage was measured again: %d cells, was %d", got, cellsBefore)
	}
	if len(done.Results) == 0 {
		t.Error("the resumed run produced no results")
	}
}

func jobCellCount(t *testing.T, store *benchmark.Store, jobID string) int {
	t.Helper()
	job, err := store.GetJob(jobID)
	if err != nil {
		t.Fatalf("job %s: %v", jobID, err)
	}
	n := 0
	for _, c := range job.Cells {
		n += c.Attempt
	}
	return n
}

func profileNames(ps []models.ConfigProfile) []string {
	var out []string
	for _, p := range ps {
		out = append(out, p.Name)
	}
	return out
}

func TestProfileName(t *testing.T) {
	cases := map[string][]Goal{
		"Autotune – fastest generation":                      {GoalGeneration},
		"Autotune – fastest generation and response":         {GoalGeneration, GoalResponse},
		"Autotune – fastest generation, prompt and response": {GoalGeneration, GoalPrompt, GoalResponse},
	}
	for want, goals := range cases {
		if got := profileName(goals); got != want {
			t.Errorf("profileName(%v) = %q, want %q", goals, got, want)
		}
	}
}

func TestPercentFaster(t *testing.T) {
	if got := percentFaster(40, 60, false); got != "50% faster" {
		t.Errorf("speed 40→60 = %q", got)
	}
	if got := percentFaster(10, 5, true); got != "50% faster" {
		t.Errorf("seconds 10→5 = %q", got)
	}
	if got := percentFaster(60, 40, false); got != "33% slower" {
		t.Errorf("speed 60→40 = %q", got)
	}
}

// Cancelling stops the run and keeps what it had measured, so Resume
// continues rather than starting again.
func TestRunnerCancelKeepsFinishedStages(t *testing.T) {
	f := newFixture(t, 1)
	// Slow enough that the run is still going when the cancel arrives.
	f.env.mu.Lock()
	f.env.delay = 5 * time.Millisecond
	f.env.mu.Unlock()
	rec, err := f.runner.Start(tuneModelID, "Autoconfig", UseChat)
	if err != nil {
		t.Fatal(err)
	}

	// Cancel as soon as the first stage has a job to cancel.
	deadline := time.Now().Add(30 * time.Second)
	for time.Now().Before(deadline) {
		cur, _ := f.store.Get(rec.ID)
		if st, ok := cur.Stage(StageBatch); ok && st.JobID != "" {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	if err := f.runner.Cancel(rec.ID); err != nil {
		t.Fatalf("cancel: %v", err)
	}
	f.env.mu.Lock()
	f.env.delay = 0 // let the resumed run finish quickly
	f.env.mu.Unlock()
	done := f.waitForRun(t, rec.ID)
	if done.Status != StatusCancelled {
		t.Fatalf("status = %s (%s)", done.Status, done.Error)
	}
	if f.runner.Active() != "" {
		t.Error("the runner still holds the run slot after cancelling")
	}
	if len(done.Results) != 0 {
		t.Error("a cancelled run produced results")
	}

	// Resume finishes it.
	if _, err := f.runner.Resume(done.ID); err != nil {
		t.Fatalf("resume: %v", err)
	}
	final := f.waitForRun(t, done.ID)
	if final.Status != StatusDone {
		t.Fatalf("resumed status = %s (%s)", final.Status, final.Error)
	}
}

// Cancelling something that is not going says so.
func TestRunnerCancelUnknown(t *testing.T) {
	f := newFixture(t, 1)
	if err := f.runner.Cancel("at-nope"); err == nil {
		t.Error("cancelling a run that is not going was accepted")
	}
}
