package benchmark

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/evaluate"
)

// capabilityEnv builds a fakeEnv wired for capability cells: the router
// starts RUNNING (the user had it up), the build carries the eval
// binary, the datasets are "downloaded", and the models map holds the
// registry entries (quant sizes drive automatic KL reference
// resolution).
func capabilityEnv(t *testing.T, running bool) *fakeEnv {
	t.Helper()
	env := &fakeEnv{
		routerURL:  "http://127.0.0.1:0",
		saved:      ConfigSnapshot{GPULayers: 999, ContextSize: 8192, Threads: 8},
		running:    running,
		evalBinary: "/builds/b/llama-perplexity",
		dataDir:    t.TempDir(),
		evalData:   map[evaluate.Mode]string{evaluate.ModePerplexity: "/ds/wikitext"},
		models: map[string]ModelInfo{
			"m4": {ID: "m4", HFRepoID: "u/M-GGUF", Quant: "Q4_K_XL", SizeBytes: 2 << 30,
				DisplayName: "M Q4", RouterName: "m4", FilePath: "/models/m4.gguf",
				Config: ConfigSnapshot{GPULayers: 999, ContextSize: 8192, Threads: 8}},
		},
	}
	return env
}

// twoQuantEnv adds a second (larger) quant of the same repo: m8 is the
// automatic reference for m4, and m4 is a distinct reference for m8.
func twoQuantEnv(t *testing.T) *fakeEnv {
	env := capabilityEnv(t, true)
	env.models["m8"] = ModelInfo{
		ID: "m8", HFRepoID: "u/M-GGUF", Quant: "Q8_0", SizeBytes: 4 << 30,
		DisplayName: "M Q8", RouterName: "m8", FilePath: "/models/m8.gguf",
		Config: ConfigSnapshot{GPULayers: 999, ContextSize: 8192, Threads: 8},
	}
	return env
}

func capIntPtr(v int) *int { return &v }

func capJob(id string, models, presets []string) BenchmarkJob {
	return BenchmarkJob{
		ID: id, Name: "cap", Kind: JobKindBatch,
		ModelIDs: models, BuildIDs: []string{"b"}, Presets: presets,
	}
}

func capJobCells(t *testing.T, models []string, presets []string, sweeps ...string) []JobCell {
	t.Helper()
	var axes []SweepAxis
	for _, s := range sweeps {
		if i := strings.IndexByte(s, '='); i >= 0 {
			axes = append(axes, SweepAxis{Field: s[:i], Values: strings.Split(s[i+1:], ",")})
		}
	}
	return ExpandCellsWithSweeps(models, []string{"b"}, presets, axes)
}

func jobCellStatuses(job *BenchmarkJob) []string {
	out := make([]string, len(job.Cells))
	for i, c := range job.Cells {
		out[i] = c.Status
	}
	return out
}

// The happy path: scores land on the run, the router is stopped before
// the eval and restored at job end, and no router start is attempted for
// the capability cell itself.
func TestCapabilityCellHappyPathLandsScores(t *testing.T) {
	env := capabilityEnv(t, true)
	env.evalData[evaluate.ModeHellaSwag] = "/ds/hellaswag"
	env.evalResult = evaluate.Result{
		Mode: "hellaswag", Dataset: "hellaswag", ContextSize: evaluate.EvalContextSize,
		Accuracy: 79.1, AccuracyCILow: 75.0, AccuracyCIHigh: 83.2, Tasks: 400,
	}

	job := capJob("job-cap", []string{"m4"}, []string{"hellaswag-quick"})
	job.Cells = capJobCells(t, []string{"m4"}, []string{"hellaswag-quick"})
	done, store := runJob(t, job, env)

	if done.Status != JobStatusCompleted {
		t.Fatalf("job status = %s, want completed; cells %+v", done.Status, done.Cells)
	}
	if done.Cells[0].Status != CellStatusCompleted || done.Cells[0].BenchmarkRunID == "" {
		t.Fatalf("cell = %+v, want completed with a run", done.Cells[0])
	}

	runs := store.RunsForJob(done.ID)
	if len(runs) != 1 {
		t.Fatalf("got %d runs, want 1", len(runs))
	}
	run := runs[0]
	if run.Status != StatusCompleted {
		t.Errorf("run status = %s, want completed (error %q)", run.Status, run.Error)
	}
	if run.Eval == nil {
		t.Fatal("run.Eval is nil — scores did not land on the run")
	}
	if run.Eval.Mode != "hellaswag" || run.Eval.Accuracy != 79.1 || run.Eval.Tasks != 400 {
		t.Errorf("run.Eval = %+v", run.Eval)
	}
	if run.Eval.Reference != "" {
		t.Errorf("non-KL run recorded a KL reference: %q", run.Eval.Reference)
	}

	// The router was stopped (for the eval) but never started for the
	// cell: no EnsureBuildActive for a capability cell.
	env.mu.Lock()
	stops, switches, restarts := env.stops, env.buildSwitches, env.buildRestarts
	env.mu.Unlock()
	if stops != 1 {
		t.Errorf("StopRouterForEval called %d times, want 1", stops)
	}
	if len(switches) != 0 || restarts != 0 {
		t.Errorf("EnsureBuildActive was used for a capability cell (switches %v, restarts %d)", switches, restarts)
	}

	// Cleanup restored the router: the fake's dirty flag (set when a
	// running router was stopped for the eval) must be cleared.
	if env.dirty {
		t.Error("cleanup did not restore the router after a capability job (dirty still set)")
	}
}

// A build predating the binary's installation fails the cell with the
// rebuild message, and the run is saved as failed — never left running.
func TestCapabilityCellMissingBinaryFailsWithRebuildMessage(t *testing.T) {
	env := capabilityEnv(t, true)
	env.evalBinary = ""
	env.evalData[evaluate.ModePerplexity] = "/ds/wikitext"

	job := capJob("job-nobin", []string{"m4"}, []string{"perplexity-quick"})
	job.Cells = capJobCells(t, []string{"m4"}, []string{"perplexity-quick"})
	done, store := runJob(t, job, env)

	if done.Status != JobStatusFailed {
		t.Fatalf("job status = %s, want failed", done.Status)
	}
	cell := done.Cells[0]
	if cell.Status != CellStatusFailed {
		t.Fatalf("cell status = %s, want failed", cell.Status)
	}
	if !strings.Contains(cell.Error, "rebuild") {
		t.Errorf("cell error does not name the fix (rebuild): %q", cell.Error)
	}
	if !strings.Contains(cell.Error, "llama-perplexity") {
		t.Errorf("cell error does not name the binary: %q", cell.Error)
	}

	// The run must exist and be failed, not left StatusRunning.
	runs := store.RunsForJob(done.ID)
	if len(runs) != 1 {
		t.Fatalf("got %d runs, want 1", len(runs))
	}
	if runs[0].Status != StatusFailed || runs[0].Error == "" {
		t.Errorf("run = %+v, want failed with an error", runs[0])
	}
	if got := store.List(); len(got) != 1 {
		t.Errorf("store holds %d runs, want exactly 1 (none left running)", len(got))
	}
	if env.dirty {
		t.Error("cleanup did not run after a failed capability cell")
	}
}

// A dataset failure (step b) fails the run with the error; cleanup still
// restores the router.
func TestCapabilityCellDatasetErrorFailsRun(t *testing.T) {
	env := capabilityEnv(t, true)
	env.evalData = nil // EnsureEvalData errors for every mode
	job := capJob("job-nods", []string{"m4"}, []string{"perplexity-quick"})
	job.Cells = capJobCells(t, []string{"m4"}, []string{"perplexity-quick"})
	done, store := runJob(t, job, env)

	if done.Cells[0].Status != CellStatusFailed {
		t.Fatalf("cell status = %s, want failed", done.Cells[0].Status)
	}
	runs := store.RunsForJob(done.ID)
	if len(runs) != 1 || runs[0].Status != StatusFailed {
		t.Fatalf("runs = %+v, want one failed run", runs)
	}
	if !strings.Contains(runs[0].Error, "dataset") {
		t.Errorf("run error does not name the dataset: %q", runs[0].Error)
	}
}

// An eval failure (step f) fails the run with the engine's error — which
// carries the tail of the tool's output.
func TestCapabilityCellEvalErrorFailsRunWithOutputTail(t *testing.T) {
	env := capabilityEnv(t, true)
	env.evalData[evaluate.ModeWinogrande] = "/ds/winogrande"
	env.evalErr = context.DeadlineExceeded
	job := capJob("job-evalfail", []string{"m4"}, []string{"winogrande-quick"})
	job.Cells = capJobCells(t, []string{"m4"}, []string{"winogrande-quick"})
	done, store := runJob(t, job, env)

	if done.Cells[0].Status != CellStatusFailed {
		t.Fatalf("cell status = %s, want failed", done.Cells[0].Status)
	}
	runs := store.RunsForJob(done.ID)
	if len(runs) != 1 || runs[0].Status != StatusFailed {
		t.Fatalf("runs = %+v, want one failed run", runs)
	}
	if !strings.Contains(runs[0].Error, "winogrande") {
		t.Errorf("run error should name the mode: %q", runs[0].Error)
	}
}

// KL automatic resolution: the largest quant (m8) is the reference for
// m4, and the resolved reference identity is recorded on the run. The
// reference model's OWN cell (m8) is skipped with the note and no run,
// while the other cell runs — 2/2 done, not 1/2.
func TestKLCellResolvesReferenceAndSkipsOwnCell(t *testing.T) {
	env := twoQuantEnv(t)
	env.evalData[evaluate.ModeKLDiv] = "/ds/wikitext"
	env.evalResult = evaluate.Result{
		Mode: "kl-divergence", Dataset: "wikitext-2", ContextSize: evaluate.EvalContextSize,
		KLMean: 0.004, KLMeanErr: 0.001, SameTopPct: 98.2,
	}

	job := capJob("job-kl", []string{"m4", "m8"}, []string{"kl-divergence-quick"})
	job.Cells = capJobCells(t, []string{"m4", "m8"}, []string{"kl-divergence-quick"})
	done, store := runJob(t, job, env)

	if done.Status != JobStatusCompleted {
		t.Fatalf("job status = %s, want completed; cells %+v", done.Status, done.Cells)
	}
	// Cell order: models → presets, so cell 0 = m4 (runs), cell 1 = m8
	// (its own reference — skipped).
	c4, c8 := done.Cells[0], done.Cells[1]
	if c4.Status != CellStatusCompleted || c4.BenchmarkRunID == "" {
		t.Errorf("m4 cell = %+v, want completed with a run", c4)
	}
	if c8.Status != CellStatusCompleted {
		t.Errorf("m8 cell status = %s, want completed (skipped-with-reason)", c8.Status)
	}
	if c8.BenchmarkRunID != "" {
		t.Errorf("skipped reference cell kept a run: %s", c8.BenchmarkRunID)
	}
	if c8.SkipReason == "" {
		t.Error("skipped reference cell has no SkipReason note")
	}
	if c4.Error != "" || c8.Error != "" {
		t.Errorf("skip must not reuse Error: m4=%q m8=%q", c4.Error, c8.Error)
	}

	runs := store.RunsForJob(done.ID)
	if len(runs) != 1 {
		t.Fatalf("got %d runs, want 1 (the skipped cell produces none)", len(runs))
	}
	run := runs[0]
	if run.Status != StatusCompleted {
		t.Fatalf("run status = %s, want completed (error %q)", run.Status, run.Error)
	}
	if run.Eval == nil || run.Eval.Mode != "kl-divergence" {
		t.Fatalf("run.Eval = %+v, want kl-divergence scores", run.Eval)
	}
	if run.Eval.Reference != "m8" {
		t.Errorf("run.Eval.Reference = %q, want m8 (the resolved reference)", run.Eval.Reference)
	}
	if run.Eval.KLMean != 0.004 {
		t.Errorf("run.Eval.KLMean = %v, want 0.004", run.Eval.KLMean)
	}
}

// An explicit reference naming the only model resolves to itself — every
// cell would skip, so the runner sees the same skip the validation
// refuses at definition time. Here (bypassing validation, as a resumed
// job would) the single cell completes without a run.
func TestKLCellExplicitSelfReferenceSkips(t *testing.T) {
	env := capabilityEnv(t, true)
	env.evalData[evaluate.ModeKLDiv] = "/ds/wikitext"
	job := capJob("job-kl-self", []string{"m4"}, []string{"kl-divergence-quick"})
	job.KLReference = "m4"
	job.Cells = capJobCells(t, []string{"m4"}, []string{"kl-divergence-quick"})
	done, store := runJob(t, job, env)

	if done.Status != JobStatusCompleted {
		t.Fatalf("job status = %s, want completed", done.Status)
	}
	c := done.Cells[0]
	if c.Status != CellStatusCompleted || c.BenchmarkRunID != "" || c.SkipReason == "" {
		t.Fatalf("self-reference cell = %+v, want completed, no run, skip note", c)
	}
	if runs := store.RunsForJob(done.ID); len(runs) != 0 {
		t.Errorf("self-reference cell produced %d runs, want 0", len(runs))
	}
}

// KL base cache: the first cell generates the base (the fake writes the
// .kld.partial path and renames it into place), the second cell of the
// same (reference, dataset, chunks, ctx) hits the cache — no second
// generation, no partial left behind.
func TestKLBaseGeneratedOnceThenCacheHit(t *testing.T) {
	env := twoQuantEnv(t)
	env.evalData[evaluate.ModeKLDiv] = "/ds/wikitext"
	env.evalResult = evaluate.Result{Mode: "kl-divergence", KLMean: 0.1}

	// Two models of the same repo: both reference m8, so m4's generation
	// must be reused by m8's... no — m8 skips. Use a THIRD model instead:
	// m5 (smaller quant) also references m8.
	env.models["m5"] = ModelInfo{
		ID: "m5", HFRepoID: "u/M-GGUF", Quant: "Q5_K_M", SizeBytes: 3 << 30,
		DisplayName: "M Q5", RouterName: "m5", FilePath: "/models/m5.gguf",
		Config: ConfigSnapshot{GPULayers: 999, ContextSize: 8192, Threads: 8},
	}

	job := capJob("job-klcache", []string{"m4", "m5"}, []string{"kl-divergence-quick"})
	job.Cells = capJobCells(t, []string{"m4", "m5"}, []string{"kl-divergence-quick"})
	done, store := runJob(t, job, env)

	if done.Status != JobStatusCompleted {
		t.Fatalf("job status = %s, want completed; cells %+v", done.Status, done.Cells)
	}
	for _, c := range done.Cells {
		if c.Status != CellStatusCompleted {
			t.Fatalf("cell %s = %s, want completed", c.ModelID, c.Status)
		}
	}
	if runs := store.RunsForJob(done.ID); len(runs) != 2 {
		t.Fatalf("got %d runs, want 2", len(runs))
	}

	// The cache now holds exactly one base file (m8, 100 chunks) and no
	// partial.
	entries, err := os.ReadDir(env.dataDir + "/logits")
	if err != nil {
		t.Fatalf("logits dir: %v", err)
	}
	var final, partial int
	for _, e := range entries {
		switch {
		case strings.HasSuffix(e.Name(), ".kld.partial"):
			partial++
		case strings.HasSuffix(e.Name(), ".kld"):
			final++
		}
	}
	if partial != 0 {
		t.Errorf("%d .kld.partial file(s) left behind; generation must rename or delete", partial)
	}
	if final != 1 {
		t.Errorf("%d cached base file(s), want 1 (shared by both cells)", final)
	}
	// And the second cell hit the cache: EnsureKLBase was called for both
	// cells but the base file was written once (the fake's generation is
	// the only writer, and the file exists exactly once).
}

// Cancel mid-KL-generation: the partial is deleted, no cache entry is
// left, the run is saved as failed (never running), and a subsequent
// EnsureKLBase regenerates — the phase 02 interruption-safety rule.
func TestKLBaseCancelDeletesPartialAndRegenerates(t *testing.T) {
	env := twoQuantEnv(t)
	env.evalData[evaluate.ModeKLDiv] = "/ds/wikitext"
	// Hold the generation open after the partial is written and before
	// the rename, giving the cancel a deterministic window in which the
	// partial exists on disk.
	env.klBaseBlock = make(chan struct{})
	job := capJob("job-klcancel", []string{"m4"}, []string{"kl-divergence-quick"})
	job.Cells = capJobCells(t, []string{"m4"}, []string{"kl-divergence-quick"})

	store := NewStore(t.TempDir(), nil)
	q := NewJobQueue(store, env)
	if err := q.Submit(job); err != nil {
		t.Fatalf("submit: %v", err)
	}

	// Wait until the partial appears (generation in flight), then cancel.
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		if _, err := os.Stat(evaluate.KLBasePartialPath(env.dataDir, evaluate.KLBaseKey{
			ModelID: "u/M-GGUF", Quant: "Q8_0", Dataset: "wikitext-2",
			Chunks: 100, Ctx: evaluate.EvalContextSize,
		})); err == nil {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	if err := q.Cancel(job.ID); err != nil {
		t.Fatalf("cancel: %v", err)
	}

	deadline = time.Now().Add(30 * time.Second)
	var done *BenchmarkJob
	for time.Now().Before(deadline) {
		got, err := store.GetJob(job.ID)
		if err == nil && got.Status != JobStatusRunning && got.Status != JobStatusPending {
			done = got
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if done == nil {
		t.Fatal("cancelled job did not wind down within 30s")
	}
	if done.Status != JobStatusCanceled {
		t.Fatalf("job status = %s, want canceled", done.Status)
	}
	if done.Cells[0].Status != CellStatusFailed {
		t.Errorf("cell status = %s, want failed (cancel mid-generation)", done.Cells[0].Status)
	}

	// No cache entry, no partial, and the run is not left running.
	key := evaluate.KLBaseKey{ModelID: "u/M-GGUF", Quant: "Q8_0", Dataset: "wikitext-2",
		Chunks: 100, Ctx: evaluate.EvalContextSize}
	if evaluate.HasKLBase(env.dataDir, key) {
		t.Error("a failed generation was cached")
	}
	if _, err := os.Stat(evaluate.KLBasePartialPath(env.dataDir, key)); err == nil {
		t.Error(".kld.partial survived the cancel")
	}
	for _, r := range store.List() {
		if r.Status == StatusRunning {
			t.Errorf("run %s left StatusRunning after cancel", r.ID)
		}
	}

	// A subsequent EnsureKLBase regenerates from scratch.
	ref, err := env.ResolveKLReference("m4", "")
	if err != nil {
		t.Fatalf("resolve reference: %v", err)
	}
	env.mu.Lock()
	env.klBaseBlock = nil // release the hold so generation can finish
	env.mu.Unlock()
	path, err := env.EnsureKLBase(context.Background(), ref, ConfigSnapshot{}, 100, "b", nil)
	if err != nil {
		t.Fatalf("regeneration failed: %v", err)
	}
	if !evaluate.HasKLBase(env.dataDir, key) {
		t.Fatal("regeneration did not cache the base")
	}
	if path != evaluate.KLBasePath(env.dataDir, key) {
		t.Errorf("regenerated path = %s", path)
	}
}

// The stop happens BEFORE the eval: StopRouterForEval is called during
// the cell, and cleanup restores the router on success, on failure, and
// on cancel. Success and failure are covered by the individual tests
// above; this one covers cancel with the router running before the job.
func TestCapabilityCancelRestoresRouter(t *testing.T) {
	env := capabilityEnv(t, true)
	env.evalData[evaluate.ModeHellaSwag] = "/ds/hellaswag"
	env.evalBlock = make(chan struct{})
	job := capJob("job-capcancel", []string{"m4"}, []string{"hellaswag-quick"})
	job.Cells = capJobCells(t, []string{"m4"}, []string{"hellaswag-quick"})

	store := NewStore(t.TempDir(), nil)
	q := NewJobQueue(store, env)
	if err := q.Submit(job); err != nil {
		t.Fatalf("submit: %v", err)
	}

	// Wait until the router is stopped (the eval is in flight), then cancel.
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		env.mu.Lock()
		stopped := !env.running
		env.mu.Unlock()
		if stopped {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	if err := q.Cancel(job.ID); err != nil {
		t.Fatalf("cancel: %v", err)
	}
	close(env.evalBlock)

	deadline = time.Now().Add(30 * time.Second)
	for time.Now().Before(deadline) {
		got, err := store.GetJob(job.ID)
		if err == nil && got.Status != JobStatusRunning && got.Status != JobStatusPending {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if env.dirty {
		t.Error("cleanup did not restore the router after a cancelled capability cell (dirty still set)")
	}
}

// The user had the router STOPPED before the job: StopRouterForEval
// records nothing (the router was not running), so cleanup must leave it
// stopped — the job must not start a server the user turned off.
func TestCapabilityRouterAlreadyStoppedBeforeJobStaysStopped(t *testing.T) {
	env := capabilityEnv(t, false) // router already stopped by the user
	env.evalData[evaluate.ModeHellaSwag] = "/ds/hellaswag"
	job := capJob("job-capstopped", []string{"m4"}, []string{"hellaswag-quick"})
	job.Cells = capJobCells(t, []string{"m4"}, []string{"hellaswag-quick"})

	done, _ := runJob(t, job, env)
	if done.Status != JobStatusCompleted {
		t.Fatalf("job status = %s, want completed", done.Status)
	}
	if env.dirty {
		t.Error("cleanup claimed a restore it shouldn't owe: the router was already stopped before the job")
	}
	if !env.running {
		// Still stopped — and it must STAY stopped (no restart happened:
		// the fake only flips running back to true via a restart, which
		// EnsureBuildActive would have logged).
	}
	env.mu.Lock()
	restarts := env.buildRestarts
	env.mu.Unlock()
	if restarts != 0 {
		t.Errorf("router restarted %d times; it was stopped before the job and must stay stopped", restarts)
	}
}

// The mixed-order sequence perf → capability → perf on ONE build with no
// overrides: the capability cell stops the router, and the third cell
// must see EnsureBuildActive re-invoked (the cache invalidation) rather
// than run against a dead router.
func TestCapabilityBetweenPerformanceCellsRevivesRouter(t *testing.T) {
	env := capabilityEnv(t, true)
	env.evalData[evaluate.ModeHellaSwag] = "/ds/hellaswag"
	env.evalResult = evaluate.Result{Mode: "hellaswag", Accuracy: 79.0}

	job := capJob("job-mixed", []string{"m4"}, []string{"internal-quick", "hellaswag-quick", "internal-quick"})
	// Dedupe presets for the matrix; the mixed ORDER is what we want, so
	// build the cells by hand in perf → cap → perf order.
	job.Presets = []string{"internal-quick", "hellaswag-quick"}
	job.Cells = []JobCell{
		{ModelID: "m4", BuildID: "b", Preset: "internal-quick", Status: CellStatusPending},
		{ModelID: "m4", BuildID: "b", Preset: "hellaswag-quick", Status: CellStatusPending},
		{ModelID: "m4", BuildID: "b", Preset: "internal-quick", Status: CellStatusPending},
	}

	done, _ := runJob(t, job, env)
	if done.Status != JobStatusCompleted {
		t.Fatalf("job status = %s, want completed; cells %+v", done.Status, done.Cells)
	}

	env.mu.Lock()
	switches := append([]string(nil), env.buildSwitches...)
	stops := env.stops
	env.mu.Unlock()

	// The performance cells must each re-enter EnsureBuildActive: the
	// first for the job's build, the third AFTER the capability cell
	// invalidated the cache. One capability stop in between.
	if len(switches) != 2 {
		t.Fatalf("EnsureBuildActive calls = %v, want 2 (one per performance cell — the third cell must not see a dead router)", switches)
	}
	if stops != 1 {
		t.Errorf("StopRouterForEval calls = %d, want 1", stops)
	}
	// The router ends up running again (revived by the third cell), so
	// cleanup's restore finds nothing to fix — but the dirty flag was
	// set by the stop and cleared by the restore path either way.
	if env.dirty {
		t.Error("cleanup left the router ownership flag set")
	}
}

// Two consecutive capability cells on the same build: the second must
// NOT re-enter EnsureBuildActive (the invalidation only matters for
// performance cells) — no start/stop churn.
func TestConsecutiveCapabilityCellsDoNotChurnRouter(t *testing.T) {
	env := capabilityEnv(t, true)
	env.evalData[evaluate.ModeHellaSwag] = "/ds/hellaswag"
	env.evalData[evaluate.ModeWinogrande] = "/ds/winogrande"
	env.evalResult = evaluate.Result{Mode: "hellaswag", Accuracy: 79.0}

	job := capJob("job-capcap", []string{"m4"}, []string{"hellaswag-quick", "winogrande-quick"})
	job.Cells = capJobCells(t, []string{"m4"}, []string{"hellaswag-quick", "winogrande-quick"})
	done, _ := runJob(t, job, env)
	if done.Status != JobStatusCompleted {
		t.Fatalf("job status = %s, want completed", done.Status)
	}
	env.mu.Lock()
	switches, stops := append([]string(nil), env.buildSwitches...), env.stops
	env.mu.Unlock()
	if len(switches) != 0 {
		t.Errorf("EnsureBuildActive called %d times for capability-only cells: %v", len(switches), switches)
	}
	if stops != 2 {
		t.Errorf("StopRouterForEval called %d times, want 2 (once per cell; idempotent in the real impl)", stops)
	}
}

// A failed capability cell's run is StatusFailed with the error; the
// reference-skip path deletes its pre-created run and clears
// cell.BenchmarkRunID (asserted here end to end: the pre-created run
// exists mid-flight is not observable from outside, so assert the
// terminal state — exactly zero runs and a cleared ID).
func TestCapabilityRunLifecycleTerminalStates(t *testing.T) {
	// Failure half: dataset error.
	env := capabilityEnv(t, true)
	env.evalData = nil
	job := capJob("job-lifecycle-fail", []string{"m4"}, []string{"perplexity-quick"})
	job.Cells = capJobCells(t, []string{"m4"}, []string{"perplexity-quick"})
	done, store := runJob(t, job, env)
	if done.Cells[0].Status != CellStatusFailed {
		t.Fatalf("cell status = %s, want failed", done.Cells[0].Status)
	}
	runs := store.RunsForJob(done.ID)
	if len(runs) != 1 || runs[0].Status != StatusFailed || runs[0].Error == "" {
		t.Fatalf("failed cell's run = %+v, want StatusFailed with an error", runs)
	}

	// Skip half: single model that is its own only reference — but the
	// fake's ResolveKLReference ERRORS in that case (no distinct
	// candidate), which the cell path surfaces as a failure, not a skip.
	// The skip needs the resolution to SUCCEED to itself: an explicit
	// self-reference.
	env2 := capabilityEnv(t, true)
	env2.evalData[evaluate.ModeKLDiv] = "/ds/wikitext"
	job2 := capJob("job-lifecycle-skip", []string{"m4"}, []string{"kl-divergence-quick"})
	job2.KLReference = "m4"
	job2.Cells = capJobCells(t, []string{"m4"}, []string{"kl-divergence-quick"})
	done2, store2 := runJob(t, job2, env2)
	if done2.Cells[0].Status != CellStatusCompleted {
		t.Fatalf("skip cell status = %s, want completed", done2.Cells[0].Status)
	}
	if done2.Cells[0].BenchmarkRunID != "" {
		t.Errorf("skip cell kept BenchmarkRunID %q; the pre-created run must be cleared", done2.Cells[0].BenchmarkRunID)
	}
	if runs2 := store2.RunsForJob(done2.ID); len(runs2) != 0 {
		t.Errorf("skip cell left %d stored run(s); the pre-created run must be deleted", len(runs2))
	}
}

// A batch/ubatch mismatch on a capability cell is refused with the named
// message (the ValidateBatchSizes call inside EvalFlags), not the
// loader's raw error.
func TestCapabilityCellBatchMismatchRefusedWithNamedMessage(t *testing.T) {
	env := capabilityEnv(t, true)
	env.evalData[evaluate.ModeHellaSwag] = "/ds/hellaswag"
	job := capJob("job-badbatch", []string{"m4"}, []string{"hellaswag-quick"})
	job.Overrides = &ConfigOverrides{BatchSize: capIntPtr(512), UBatchSize: capIntPtr(4096)}
	job.Cells = capJobCells(t, []string{"m4"}, []string{"hellaswag-quick"})
	done, store := runJob(t, job, env)

	if done.Cells[0].Status != CellStatusFailed {
		t.Fatalf("cell status = %s, want failed", done.Cells[0].Status)
	}
	runs := store.RunsForJob(done.ID)
	if len(runs) != 1 {
		t.Fatalf("got %d runs, want 1", len(runs))
	}
	if !strings.Contains(runs[0].Error, "micro-batch 4096 exceeds batch size 512") {
		t.Errorf("run error is not the named ValidateBatchSizes message: %q", runs[0].Error)
	}
}

// A KL reference that cannot be resolved (no distinct quant, no
// override) fails the cell — a real error, not a skip.
func TestKLCellUnresolvableReferenceFails(t *testing.T) {
	env := capabilityEnv(t, true) // m4 is its repo's only quant
	env.evalData[evaluate.ModeKLDiv] = "/ds/wikitext"
	job := capJob("job-kl-unres", []string{"m4"}, []string{"kl-divergence-quick"})
	job.Cells = capJobCells(t, []string{"m4"}, []string{"kl-divergence-quick"})
	done, _ := runJob(t, job, env)

	if done.Cells[0].Status != CellStatusFailed {
		t.Fatalf("cell status = %s, want failed (no distinct reference exists)", done.Cells[0].Status)
	}
	if !strings.Contains(done.Cells[0].Error, "only installed quant") {
		t.Errorf("cell error does not explain the missing reference: %q", done.Cells[0].Error)
	}
}
