package benchmark

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/monitor"
)

// fakeEnv records the calls the cell loop makes so a test can assert
// that config overrides were actually applied and then restored, rather
// than merely recorded on the run.
type fakeEnv struct {
	mu sync.Mutex

	routerURL string
	saved     ConfigSnapshot

	applied  []ConfigSnapshot // one per ApplyEphemeralConfig call
	appliedT []time.Time
	cleared  int
	clearedT []time.Time

	applyErr error
}

func (f *fakeEnv) CheckBuildRunnable(context.Context, string) error { return nil }
func (f *fakeEnv) EnsureBuildActive(context.Context, string) error  { return nil }

func (f *fakeEnv) ResolveModel(string) (ModelInfo, error) {
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
	if err := ctx.Err(); err != nil {
		return err
	}
	f.mu.Lock()
	defer f.mu.Unlock()
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
