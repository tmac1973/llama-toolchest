package benchmark

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/evaluate"
	"github.com/tmac1973/llama-toolchest/internal/monitor"
)

// fakeEnv records the calls the cell loop makes so a test can assert
// that config overrides were actually applied and then restored, rather
// than merely recorded on the run.
type fakeEnv struct {
	mu sync.Mutex

	routerURL string
	saved     ConfigSnapshot

	applied       []ConfigSnapshot // one per ApplyEphemeralConfig call
	appliedT      []time.Time
	cleared       int
	clearedT      []time.Time
	dirty         bool
	buildSwitches []string
	buildRestarts int

	applyErr error

	// Capability-cell machinery (see the method set below). The fake
	// models a router that starts RUNNING (running bool), a build with
	// or without the eval binary, dataset paths under dataDir, and a
	// KL logits cache under dataDir/logits so EnsureKLBase exercises the
	// real partial/rename discipline of the phase 02 layout.
	running    bool
	evalBinary string
	evalData   map[evaluate.Mode]string // mode → dataset path
	evalResult evaluate.Result
	evalErr    error
	// evalBlock, when non-nil, blocks RunEval until closed (cancel tests).
	evalBlock chan struct{}
	models    map[string]ModelInfo // modelID → registry entry
	klBaseErr error
	// klBaseConfig records the config the last reference generation
	// would have used, so a test can check both sides of a KL
	// comparison were measured the same way.
	klBaseConfig ConfigSnapshot
	// klBaseBlock, when non-nil, holds the generation open AFTER the
	// .kld.partial is written and BEFORE the rename, giving cancel tests
	// a window in which the partial exists on disk.
	klBaseBlock chan struct{}
	stops       int
	evalCalls   []evaluate.Spec

	// dataDir hosts the KL logits cache (EnsureKLBase) and dataset files.
	dataDir string
}

func (f *fakeEnv) CheckBuildRunnable(context.Context, string) error { return nil }
func (f *fakeEnv) EnsureBuildActive(_ context.Context, id string, configFollows bool) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.buildSwitches = append(f.buildSwitches, id)
	// Mirrors the real implementation: when a config apply follows, the
	// restart is deferred to it rather than doubled.
	if !configFollows {
		f.buildRestarts++
	}
	return nil
}

func (f *fakeEnv) ResolveModel(id string) (ModelInfo, error) {
	if m, ok := f.models[id]; ok {
		// Copy: the runner applies overrides onto this.
		return m, nil
	}
	return ModelInfo{
		HFRepoID:    "u/M-GGUF",
		DisplayName: "M",
		RouterName:  "m",
		Config:      f.saved,
	}, nil
}

func (f *fakeEnv) ApplyEphemeralConfig(_ context.Context, _ string, cfg ConfigSnapshot) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	// The real implementation arms the override and marks the router
	// dirty before the restart can fail, so a failed apply still owes a
	// restore.
	f.dirty = true
	if f.applyErr != nil {
		return f.applyErr
	}
	f.applied = append(f.applied, cfg)
	f.appliedT = append(f.appliedT, time.Now())
	return nil
}

// Mirrors the real implementation, which restarts the router and so
// aborts on a dead context. Without this check a test could not tell
// context.WithoutCancel from a plain ctx on the cancel path.
func (f *fakeEnv) ClearEphemeralConfig(ctx context.Context) error {
	f.mu.Lock()
	dirty := f.dirty
	f.mu.Unlock()

	// Mirrors the real implementation: cleanup is always called, but only
	// restarts when a config or build was actually changed. Checking
	// ctx.Err() matters because the real restore restarts the router and
	// so aborts on a dead context — that is what distinguishes
	// context.WithoutCancel from a plain ctx on the cancel path.
	if !dirty {
		return nil
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	f.dirty = false
	f.cleared++
	f.clearedT = append(f.clearedT, time.Now())
	return nil
}

func (f *fakeEnv) ResolveBuild(id string) BuildSnapshot {
	return BuildSnapshot{ID: id, Profile: "rocm", GitRef: "b1"}
}
func (f *fakeEnv) CurrentMetrics() monitor.Metrics { return monitor.Metrics{} }
func (f *fakeEnv) RouterURL() string               { return f.routerURL }
func (f *fakeEnv) HFToken() string                 { return "" }
func (f *fakeEnv) HFCacheDir() string              { return "" }

// --- JobEnv capability methods ---------------------------------------

// StopRouterForEval mirrors the real implementation's recording rules:
// only a stop of a RUNNING router records ownership (the cleanup flag),
// so a user-stopped router is left stopped at job end. Idempotent.
func (f *fakeEnv) StopRouterForEval(context.Context) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.stops++
	if f.running {
		f.running = false
		f.dirty = true // this job stopped a running router: cleanup owes a restart
	}
	return nil
}

func (f *fakeEnv) EvalBinary(string) (string, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.evalBinary == "" {
		return "", fmt.Errorf("build b has no llama-perplexity binary — it predates the binary's installation; rebuild the build to install it")
	}
	return f.evalBinary, nil
}

func (f *fakeEnv) EnsureEvalData(_ context.Context, mode evaluate.Mode) (string, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.evalData == nil {
		return "", fmt.Errorf("dataset for mode %s is unavailable", mode)
	}
	if p, ok := f.evalData[mode]; ok {
		return p, nil
	}
	return "", fmt.Errorf("dataset for mode %s is unavailable", mode)
}

// ResolveKLReference mirrors the real policy: override wins when set;
// otherwise the largest installed quant of the same HF repo (by
// SizeBytes) — the model itself included, so a multi-quant job
// resolves every cell (including the largest quant's own) against the
// same reference, and the runner skips the self cell. Error when the
// repo has no distinct candidate at all.
func (f *fakeEnv) ResolveKLReference(modelID, overrideID string) (ModelInfo, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if overrideID != "" {
		m, ok := f.models[overrideID]
		if !ok {
			return ModelInfo{}, fmt.Errorf("KL reference model %s is not installed", overrideID)
		}
		return m, nil
	}
	base, ok := f.models[modelID]
	if !ok {
		return ModelInfo{}, fmt.Errorf("resolve model %s: not installed", modelID)
	}
	var best *ModelInfo
	others := 0
	for id := range f.models {
		if f.models[id].HFRepoID != base.HFRepoID {
			continue
		}
		if id != modelID {
			others++
		}
		m := f.models[id]
		if best == nil || m.SizeBytes > best.SizeBytes {
			best = &m
		}
	}
	if best == nil {
		return ModelInfo{}, fmt.Errorf("no installed quant found for %s", modelID)
	}
	if best.ID == modelID && others == 0 {
		return ModelInfo{}, fmt.Errorf("no KL reference for %s: it is the only installed quant of its repo", modelID)
	}
	return *best, nil
}

func (f *fakeEnv) EvalFlags(modelID string, snap ConfigSnapshot, _ string) ([]string, error) {
	if snap.UBatchSize > 0 && snap.BatchSize > 0 && snap.UBatchSize > snap.BatchSize {
		// The same named message ValidateBatchSizes produces.
		return nil, fmt.Errorf("%s: micro-batch %d exceeds batch size %d — micro-batch must be less than or equal to batch",
			modelID, snap.UBatchSize, snap.BatchSize)
	}
	return []string{
		"--n-gpu-layers", fmt.Sprintf("%d", snap.GPULayers),
		"--threads", fmt.Sprintf("%d", snap.Threads),
	}, nil
}

// EnsureKLBase implements the real cache contract against a temp dir:
// cache hit on the final file, disk-guard-free generation to the
// .kld.partial path otherwise (the caller owns the file discipline via
// the phase 02 helpers), rename into place on success, delete the
// partial on failure or cancel, and never a cache entry for a failed
// generation.
func (f *fakeEnv) EnsureKLBase(ctx context.Context, ref ModelInfo, underTest ConfigSnapshot, chunks int, _ string, progress func(string)) (string, error) {
	if f.dataDir == "" {
		return "", fmt.Errorf("fake env has no data dir")
	}
	// Record what the reference would actually be generated at, so a
	// test can assert both sides of the comparison match.
	f.mu.Lock()
	f.klBaseConfig = EvalReferenceConfig(ref.Config, underTest)
	f.mu.Unlock()
	key := evaluate.KLBaseKey{
		ModelID: ref.HFRepoID, Quant: ref.Quant,
		Dataset: evaluate.ModeKLDiv.DatasetName(), Chunks: chunks,
		Ctx: evaluate.EvalContextSize,
	}
	final := evaluate.KLBasePath(f.dataDir, key)
	partial := evaluate.KLBasePartialPath(f.dataDir, key)
	if evaluate.HasKLBase(f.dataDir, key) {
		return final, nil
	}
	f.mu.Lock()
	err := f.klBaseErr
	f.mu.Unlock()
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(evaluate.LogitsDir(f.dataDir), 0o755); err != nil {
		return "", err
	}
	if progress != nil {
		progress("generating reference logits")
	}
	if err := os.WriteFile(partial, []byte("logits"), 0o644); err != nil {
		return "", err
	}
	f.mu.Lock()
	block := f.klBaseBlock
	f.mu.Unlock()
	if block != nil {
		select {
		case <-block:
		case <-ctx.Done():
			os.Remove(partial)
			return "", ctx.Err()
		}
	}
	if ctx.Err() != nil {
		os.Remove(partial)
		return "", ctx.Err()
	}
	if err := os.Rename(partial, final); err != nil {
		os.Remove(partial)
		return "", err
	}
	return final, nil
}

func (f *fakeEnv) RunEval(ctx context.Context, spec evaluate.Spec) (evaluate.Result, error) {
	f.mu.Lock()
	f.evalCalls = append(f.evalCalls, spec)
	res, err := f.evalResult, f.evalErr
	block := f.evalBlock
	f.mu.Unlock()
	if block != nil {
		select {
		case <-block:
		case <-ctx.Done():
			return evaluate.Result{}, ctx.Err()
		}
	}
	if err != nil {
		return evaluate.Result{}, err
	}
	out := res
	if out.Mode == "" {
		out.Mode = string(spec.Mode)
	}
	return out, nil
}

func (f *fakeEnv) snapshotCalls() ([]ConfigSnapshot, int) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]ConfigSnapshot(nil), f.applied...), f.cleared
}

// fakeRouter serves the minimum surface Runner needs, and captures every
// chat-completion body so sampling params can be asserted on the wire.
type fakeRouter struct {
	*httptest.Server
	mu     sync.Mutex
	bodies []map[string]any
}

func newFakeRouter(t *testing.T) *fakeRouter {
	t.Helper()
	fr := &fakeRouter{}
	mux := http.NewServeMux()

	// No models loaded, so unloadAllModels returns without its 2s waits.
	mux.HandleFunc("/models", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `[]`)
	})
	mux.HandleFunc("/models/load", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)
		fr.mu.Lock()
		fr.bodies = append(fr.bodies, body)
		fr.mu.Unlock()

		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"timings":{"prompt_n":256,"prompt_ms":100,"prompt_per_second":2560,
			"predicted_n":128,"predicted_ms":1000,"predicted_per_second":128}}`)
	})

	fr.Server = httptest.NewServer(mux)
	t.Cleanup(fr.Close)
	return fr
}

func (fr *fakeRouter) completionBodies() []map[string]any {
	fr.mu.Lock()
	defer fr.mu.Unlock()
	return append([]map[string]any(nil), fr.bodies...)
}

func runJob(t *testing.T, job BenchmarkJob, env *fakeEnv) (*BenchmarkJob, *Store) {
	t.Helper()
	store := NewStore(t.TempDir(), nil)
	q := NewJobQueue(store, env)

	if err := q.Submit(job); err != nil {
		t.Fatalf("submit: %v", err)
	}

	deadline := time.Now().Add(30 * time.Second)
	for time.Now().Before(deadline) {
		got, err := store.GetJob(job.ID)
		if err == nil && got.Status != JobStatusRunning && got.Status != JobStatusPending {
			return got, store
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatal("job did not finish within 30s")
	return nil, nil
}

func oneCellJob(overrides *ConfigOverrides) BenchmarkJob {
	return BenchmarkJob{
		ID:        "job-1",
		Name:      "test",
		Kind:      JobKindBatch,
		ModelIDs:  []string{"m"},
		BuildIDs:  []string{"b"},
		Presets:   []string{"internal-quick"},
		Overrides: overrides,
		Cells:     ExpandCells([]string{"m"}, []string{"b"}, []string{"internal-quick"}),
	}
}

// The regression test for the original bug: a job that declares
// overrides must push them to the router, and the config recorded on the
// run must be the config that was applied.
func TestJobAppliesOverridesBeforeMeasuring(t *testing.T) {
	router := newFakeRouter(t)
	ngl, ctx := 50, 16384
	env := &fakeEnv{
		routerURL: router.URL,
		saved:     ConfigSnapshot{GPULayers: 999, ContextSize: 8192, Threads: 8},
	}

	job, store := runJob(t, oneCellJob(&ConfigOverrides{GPULayers: &ngl, ContextSize: &ctx}), env)
	if job.Status != JobStatusCompleted {
		t.Fatalf("job status = %s, want completed", job.Status)
	}

	applied, cleared := env.snapshotCalls()
	if len(applied) != 1 {
		t.Fatalf("ApplyEphemeralConfig called %d times, want 1", len(applied))
	}
	if applied[0].GPULayers != 50 {
		t.Errorf("applied GPULayers = %d, want 50 (the override)", applied[0].GPULayers)
	}
	if applied[0].ContextSize != 16384 {
		t.Errorf("applied ContextSize = %d, want 16384 (the override)", applied[0].ContextSize)
	}
	if applied[0].Threads != 8 {
		t.Errorf("applied Threads = %d, want 8 (from saved config)", applied[0].Threads)
	}
	if cleared != 1 {
		t.Errorf("ClearEphemeralConfig called %d times, want 1", cleared)
	}

	// The recorded snapshot must match what was pushed to the router —
	// the divergence between these two is the bug being guarded.
	runs := store.RunsForJob(job.ID)
	if len(runs) != 1 {
		t.Fatalf("got %d runs, want 1", len(runs))
	}
	if runs[0].Config != applied[0] {
		t.Errorf("recorded config != applied config:\nrecorded %+v\napplied  %+v", runs[0].Config, applied[0])
	}
}

// With no overrides there is nothing to apply, so the router must not be
// restarted at all — a needless restart per cell would be pure cost.
func TestJobWithoutOverridesDoesNotTouchRouterConfig(t *testing.T) {
	router := newFakeRouter(t)
	env := &fakeEnv{
		routerURL: router.URL,
		saved:     ConfigSnapshot{GPULayers: 999, ContextSize: 8192, Threads: 8},
	}

	job, _ := runJob(t, oneCellJob(nil), env)
	if job.Status != JobStatusCompleted {
		t.Fatalf("job status = %s, want completed", job.Status)
	}

	applied, cleared := env.snapshotCalls()
	if len(applied) != 0 {
		t.Errorf("ApplyEphemeralConfig called %d times for a job with no overrides", len(applied))
	}
	if cleared != 0 {
		t.Errorf("ClearEphemeralConfig called %d times for a job with no overrides", cleared)
	}
}

// If the config can't be applied the cell must fail rather than measure
// the wrong config, and the override must still be cleared afterwards.
func TestJobFailsCellWhenConfigCannotBeApplied(t *testing.T) {
	router := newFakeRouter(t)
	ngl := 50
	env := &fakeEnv{
		routerURL: router.URL,
		saved:     ConfigSnapshot{GPULayers: 999},
		applyErr:  context.DeadlineExceeded,
	}

	job, _ := runJob(t, oneCellJob(&ConfigOverrides{GPULayers: &ngl}), env)

	if job.Status != JobStatusFailed {
		t.Errorf("job status = %s, want failed", job.Status)
	}
	if job.Cells[0].Status != CellStatusFailed {
		t.Errorf("cell status = %s, want failed", job.Cells[0].Status)
	}
	if _, cleared := env.snapshotCalls(); cleared != 1 {
		t.Errorf("ClearEphemeralConfig called %d times, want 1 even on failure", cleared)
	}
}

// Cancelling mid-job must still restore the user's saved config. This is
// the path that relies on context.WithoutCancel in the restore defer: the
// job's ctx is already dead by the time the cleanup runs, so a plain ctx
// would leave the router stuck on benchmark overrides.
func TestCancelledJobStillRestoresConfig(t *testing.T) {
	release := make(chan struct{})
	var once sync.Once
	router := newSlowRouter(t, release)

	ngl := 50
	env := &fakeEnv{
		routerURL: router.URL,
		saved:     ConfigSnapshot{GPULayers: 999, ContextSize: 8192},
	}

	store := NewStore(t.TempDir(), nil)
	q := NewJobQueue(store, env)
	job := oneCellJob(&ConfigOverrides{GPULayers: &ngl})
	if err := q.Submit(job); err != nil {
		t.Fatalf("submit: %v", err)
	}

	// Wait until the cell is actually in flight, then cancel.
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		if applied, _ := env.snapshotCalls(); len(applied) > 0 {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if err := q.Cancel(job.ID); err != nil {
		t.Fatalf("cancel: %v", err)
	}
	once.Do(func() { close(release) })

	deadline = time.Now().Add(30 * time.Second)
	for time.Now().Before(deadline) {
		got, err := store.GetJob(job.ID)
		if err == nil && got.Status != JobStatusRunning && got.Status != JobStatusPending {
			if _, cleared := env.snapshotCalls(); cleared != 1 {
				t.Errorf("ClearEphemeralConfig called %d times after cancel, want 1 — the router would be left on benchmark config", cleared)
			}
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatal("cancelled job did not wind down within 30s")
}

// newSlowRouter blocks completions until release is closed, giving a
// test a window in which the job is reliably mid-cell.
func newSlowRouter(t *testing.T, release <-chan struct{}) *fakeRouter {
	t.Helper()
	fr := &fakeRouter{}
	mux := http.NewServeMux()
	mux.HandleFunc("/models", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `[]`)
	})
	mux.HandleFunc("/models/load", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-release:
		case <-r.Context().Done():
			return
		case <-time.After(20 * time.Second):
		}
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"timings":{"prompt_n":256,"prompt_ms":100,"prompt_per_second":2560,
			"predicted_n":128,"predicted_ms":1000,"predicted_per_second":128}}`)
	})
	fr.Server = httptest.NewServer(mux)
	t.Cleanup(fr.Close)
	return fr
}

// Sampling overrides aren't router config — they have to show up in the
// completion request itself.
func TestSamplingOverridesReachTheRequest(t *testing.T) {
	router := newFakeRouter(t)
	temp, topP := 0.42, 0.87
	topK := 7
	env := &fakeEnv{
		routerURL: router.URL,
		saved:     ConfigSnapshot{GPULayers: 999, ContextSize: 8192},
	}

	job, _ := runJob(t, oneCellJob(&ConfigOverrides{
		Temperature: &temp, TopP: &topP, TopK: &topK,
	}), env)
	if job.Status != JobStatusCompleted {
		t.Fatalf("job status = %s, want completed", job.Status)
	}

	bodies := router.completionBodies()
	if len(bodies) < 2 {
		t.Fatalf("expected a warmup and at least one measured request, got %d", len(bodies))
	}

	// Warmup deliberately runs at server defaults.
	if _, ok := bodies[0]["temperature"]; ok {
		t.Error("warmup request should not carry sampling overrides")
	}

	measured := bodies[len(bodies)-1]
	if measured["temperature"] != 0.42 {
		t.Errorf("temperature = %v, want 0.42", measured["temperature"])
	}
	if measured["top_p"] != 0.87 {
		t.Errorf("top_p = %v, want 0.87", measured["top_p"])
	}
	// JSON numbers decode as float64.
	if measured["top_k"] != float64(7) {
		t.Errorf("top_k = %v, want 7", measured["top_k"])
	}
	if _, ok := measured["min_p"]; ok {
		t.Error("min_p was not overridden and must be omitted")
	}
}

// A sweep must apply each distinct config exactly once, not once per
// cell. Two presets under each of two sweep points is four cells but
// only two configs, so only two reloads.
func TestSweepAppliesEachConfigOnce(t *testing.T) {
	router := newFakeRouter(t)
	env := &fakeEnv{
		routerURL: router.URL,
		saved:     ConfigSnapshot{GPULayers: 999, ContextSize: 8192, Threads: 8},
	}

	job := BenchmarkJob{
		ID: "job-sweep", Name: "sweep", Kind: JobKindBatch,
		ModelIDs: []string{"m"}, BuildIDs: []string{"b"},
		Presets: []string{"internal-quick", "internal-long-ctx"},
		Sweeps:  []SweepAxis{{Field: "gpu_layers", Values: []string{"20", "40"}}},
	}
	job.Cells = ExpandCellsWithSweeps(job.ModelIDs, job.BuildIDs, job.Presets, job.Sweeps)
	if len(job.Cells) != 4 {
		t.Fatalf("expected 4 cells, got %d", len(job.Cells))
	}

	done, store := runJob(t, job, env)
	if done.Status != JobStatusCompleted {
		t.Fatalf("job status = %s, want completed", done.Status)
	}

	applied, cleared := env.snapshotCalls()
	if len(applied) != 2 {
		t.Errorf("ApplyEphemeralConfig called %d times across 4 cells, want 2 (one per distinct config)", len(applied))
	}
	if cleared != 1 {
		t.Errorf("ClearEphemeralConfig called %d times, want 1", cleared)
	}

	got := map[int]bool{}
	for _, a := range applied {
		got[a.GPULayers] = true
	}
	if !got[20] || !got[40] {
		t.Errorf("expected both swept values to be applied, got %v", got)
	}

	if runs := store.RunsForJob(job.ID); len(runs) != 4 {
		t.Errorf("got %d runs, want 4", len(runs))
	}
}

// Sweeping a sampling param changes only the request body, so it must
// not restart the router at all.
func TestSamplingSweepCostsNoRestarts(t *testing.T) {
	router := newFakeRouter(t)
	env := &fakeEnv{routerURL: router.URL, saved: ConfigSnapshot{GPULayers: 999}}

	job := BenchmarkJob{
		ID: "job-temp", Name: "temp sweep", Kind: JobKindBatch,
		ModelIDs: []string{"m"}, BuildIDs: []string{"b"},
		Presets: []string{"internal-quick"},
		Sweeps:  []SweepAxis{{Field: "temperature", Values: []string{"0", "0.7", "1.0"}}},
	}
	job.Cells = ExpandCellsWithSweeps(job.ModelIDs, job.BuildIDs, job.Presets, job.Sweeps)

	done, _ := runJob(t, job, env)
	if done.Status != JobStatusCompleted {
		t.Fatalf("job status = %s, want completed", done.Status)
	}

	// Sampling doesn't change ConfigSnapshot, so all three cells resolve
	// to one identical config and the router is touched exactly once.
	applied, _ := env.snapshotCalls()
	if len(applied) > 1 {
		t.Errorf("sampling sweep caused %d config applications, want at most 1", len(applied))
	}

	// Each swept temperature must reach the wire.
	var temps []any
	for _, b := range router.completionBodies() {
		if v, ok := b["temperature"]; ok {
			temps = append(temps, v)
		}
	}
	if len(temps) != 3 {
		t.Fatalf("got %d requests carrying temperature, want 3: %v", len(temps), temps)
	}
	seen := map[any]bool{}
	for _, v := range temps {
		seen[v] = true
	}
	for _, want := range []any{0.0, 0.7, 1.0} {
		if !seen[want] {
			t.Errorf("temperature %v never reached the router (saw %v)", want, temps)
		}
	}
}

// A restart that fails *after* llama-server came up on the benchmark
// preset must still be restored. Clearing the override in the apply
// error path made the deferred cleanup a no-op, leaving the router
// serving normal traffic under benchmark config.
func TestCleanupRunsWhenApplySucceedsThenRestartFails(t *testing.T) {
	router := newFakeRouter(t)
	ngl := 50
	env := &fakeEnv{
		routerURL: router.URL,
		saved:     ConfigSnapshot{GPULayers: 999},
		applyErr:  context.DeadlineExceeded,
	}

	job, _ := runJob(t, oneCellJob(&ConfigOverrides{GPULayers: &ngl}), env)
	if job.Status != JobStatusFailed {
		t.Errorf("job status = %s, want failed", job.Status)
	}
	if _, cleared := env.snapshotCalls(); cleared != 1 {
		t.Errorf("ClearEphemeralConfig called %d times, want 1 — the router must be restored even when apply failed", cleared)
	}
}

// A zero-valued sweep point must reach the router. This is the runner-
// level guard for the same bug applySnapshotToConfig had: gpu_layers=0
// is a legitimate CPU-only override, not an absent one.
func TestZeroValuedSweepPointIsApplied(t *testing.T) {
	router := newFakeRouter(t)
	env := &fakeEnv{
		routerURL: router.URL,
		saved:     ConfigSnapshot{GPULayers: 999, ContextSize: 8192},
	}

	job := BenchmarkJob{
		ID: "job-zero", Name: "zero", Kind: JobKindBatch,
		ModelIDs: []string{"m"}, BuildIDs: []string{"b"},
		Presets: []string{"internal-quick"},
		Sweeps:  []SweepAxis{{Field: "gpu_layers", Values: []string{"0", "999"}}},
	}
	job.Cells = ExpandCellsWithSweeps(job.ModelIDs, job.BuildIDs, job.Presets, job.Sweeps)

	done, store := runJob(t, job, env)
	if done.Status != JobStatusCompleted {
		t.Fatalf("job status = %s, want completed", done.Status)
	}

	applied, _ := env.snapshotCalls()
	var sawZero bool
	for _, a := range applied {
		if a.GPULayers == 0 {
			sawZero = true
		}
	}
	if !sawZero {
		t.Errorf("gpu_layers=0 never reached the router; applied %+v", applied)
	}

	// And the recorded run must agree with what was applied.
	for _, r := range store.RunsForJob(job.ID) {
		if r.SweepValues["gpu_layers"] == "0" && r.Config.GPULayers != 0 {
			t.Errorf("run recorded gpu_layers=%d for the 0 sweep point", r.Config.GPULayers)
		}
	}
}

// A job with no overrides still asks for cleanup, because it may have
// switched builds and the runner can't see which were already active.
// The implementation decides whether a restart is actually needed.
func TestCleanupIsAlwaysRequested(t *testing.T) {
	router := newFakeRouter(t)
	env := &fakeEnv{
		routerURL: router.URL,
		saved:     ConfigSnapshot{GPULayers: 999, ContextSize: 8192},
	}

	job, _ := runJob(t, oneCellJob(nil), env)
	if job.Status != JobStatusCompleted {
		t.Fatalf("job status = %s, want completed", job.Status)
	}

	applied, cleared := env.snapshotCalls()
	if len(applied) != 0 {
		t.Errorf("no overrides, so no config should be pushed; got %d", len(applied))
	}
	// Nothing was changed, so cleanup finds nothing to restore and the
	// router is never restarted.
	if cleared != 0 {
		t.Errorf("cleanup restarted the router %d times for an unchanged job", cleared)
	}
}

// A build switch followed immediately by a config apply must cost one
// reload, not two. Restarting in EnsureBuildActive and again in
// ApplyEphemeralConfig briefly served the previous cell's config on the
// new build for no purpose, and doubled the reload cost at every build
// boundary — which a two-build comparison hits on every model.
func TestBuildSwitchWithConfigCostsOneReload(t *testing.T) {
	router := newFakeRouter(t)
	env := &fakeEnv{routerURL: router.URL, saved: ConfigSnapshot{GPULayers: 999}}

	job := BenchmarkJob{
		ID: "job-2b", Name: "two builds", Kind: JobKindBatch,
		ModelIDs: []string{"m"}, BuildIDs: []string{"b1", "b2"},
		Presets: []string{"internal-quick"},
		Sweeps:  []SweepAxis{{Field: "ubatch_size", Values: []string{"512"}}},
	}
	job.Cells = ExpandCellsWithSweeps(job.ModelIDs, job.BuildIDs, job.Presets, job.Sweeps)

	done, _ := runJob(t, job, env)
	if done.Status != JobStatusCompleted {
		t.Fatalf("job status = %s, want completed", done.Status)
	}

	env.mu.Lock()
	switches, restarts := append([]string(nil), env.buildSwitches...), env.buildRestarts
	env.mu.Unlock()

	if len(switches) != 2 {
		t.Errorf("build activations = %v, want one per build", switches)
	}
	// Both switches carry a config, so neither restarts on its own; the
	// following ApplyEphemeralConfig does it once each.
	if restarts != 0 {
		t.Errorf("EnsureBuildActive restarted %d times; the config apply should carry it", restarts)
	}
	if applied, _ := env.snapshotCalls(); len(applied) != 2 {
		t.Errorf("config applied %d times, want once per build", len(applied))
	}
}

// Without a config to apply, EnsureBuildActive must still restart — the
// deferral has nothing to defer to.
func TestBuildSwitchWithoutConfigStillRestarts(t *testing.T) {
	router := newFakeRouter(t)
	env := &fakeEnv{routerURL: router.URL, saved: ConfigSnapshot{GPULayers: 999}}

	job := BenchmarkJob{
		ID: "job-nb", Name: "no overrides", Kind: JobKindBatch,
		ModelIDs: []string{"m"}, BuildIDs: []string{"b1", "b2"},
		Presets: []string{"internal-quick"},
	}
	job.Cells = ExpandCells(job.ModelIDs, job.BuildIDs, job.Presets)

	if done, _ := runJob(t, job, env); done.Status != JobStatusCompleted {
		t.Fatalf("job status = %s, want completed", done.Status)
	}

	env.mu.Lock()
	restarts := env.buildRestarts
	env.mu.Unlock()
	if restarts != 2 {
		t.Errorf("EnsureBuildActive restarted %d times, want one per build", restarts)
	}
}
