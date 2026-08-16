package benchmark

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/evaluate"
	"github.com/tmac1973/llama-toolchest/internal/monitor"
)

// ErrJobAlreadyRunning is returned by Submit when a job is in flight.
var ErrJobAlreadyRunning = errors.New("a benchmark job is already running")

// JobEnv is the integration point the JobRunner needs from outside the
// benchmark package. The api layer implements it against the Server so
// this package stays free of imports for builder / models / process.
type JobEnv interface {
	// CheckBuildRunnable returns nil if the build's binary can satisfy
	// its dynamic-linker dependencies on this host, or an error
	// describing what's missing. Lets the cell loop fail fast on stale
	// builds (e.g. ROCm SONAME bumps that strand older llama-server
	// binaries) instead of restarting the router and watching it crash.
	CheckBuildRunnable(ctx context.Context, buildID string) error

	// EnsureBuildActive switches the router to the build identified by
	// buildID, restarting llama-server when the running build differs.
	// Blocks until the router is reachable.
	//
	// configFollows tells the implementation that the caller will apply
	// a config override immediately afterwards, which restarts the
	// router anyway. It may then record the build and skip its own
	// restart so the pair costs one reload instead of two — the first of
	// which would otherwise serve the previous cell's config on the new
	// build for no purpose.
	EnsureBuildActive(ctx context.Context, buildID string, configFollows bool) error

	// ResolveModel returns everything the cell needs about a model from
	// the registry (HF repo id for tokenizer, router-served name, saved
	// config to apply overrides on top of, display fields).
	ResolveModel(modelID string) (ModelInfo, error)

	// ApplyEphemeralConfig makes modelID run under cfg, restarting the
	// router so it takes effect. Implementations must not persist the
	// change: the user's saved config has to survive a job that is
	// cancelled or crashes. Blocks until the router is reachable.
	ApplyEphemeralConfig(ctx context.Context, modelID string, cfg ConfigSnapshot) error

	// ClearEphemeralConfig drops any active override and restarts the
	// router onto saved config. Must be a no-op when nothing is active.
	//
	// Eval-stop case: when the job stopped the router for a capability
	// cell via StopRouterForEval, "not running" means THIS job stopped
	// it, not the user — ClearEphemeralConfig restarts the router
	// (restoring the pre-job state) instead of leaving it stopped. Only
	// when the router was already stopped before the job started does
	// the user-stop semantics apply and cleanup leave it stopped.
	ClearEphemeralConfig(ctx context.Context) error

	// ResolveBuild returns the snapshot for buildID. Empty struct means
	// the build no longer exists; the cell will fail.
	ResolveBuild(buildID string) BuildSnapshot

	// CurrentMetrics returns the latest GPU metrics for snapshotting.
	CurrentMetrics() monitor.Metrics

	// RouterURL returns the base URL the runner should target.
	RouterURL() string

	// HFToken returns the configured HuggingFace token (empty when
	// unset). Forwarded to llama-benchy as HF_TOKEN so anonymous
	// rate limiting doesn't sink mid-batch tokenizer fetches.
	HFToken() string

	// HFCacheDir returns a persistent directory for the HuggingFace
	// cache. Forwarded as HF_HOME so the tokenizer downloads once
	// and is reused across every benchmark in a batch.
	HFCacheDir() string

	// StopRouterForEval stops llama-server so an evaluation can own the
	// GPU. It records that THIS JOB stopped a RUNNING router — when the
	// router was already stopped (by the user, before the job) nothing
	// is recorded, so cleanup will not start a server the user had
	// turned off. Idempotent.
	StopRouterForEval(ctx context.Context) error

	// EvalBinary returns the llama-perplexity path for a build, or an
	// error naming the fix ("rebuild") when the build predates the
	// binary's installation.
	EvalBinary(buildID string) (string, error)

	// EnsureEvalData downloads/verifies the dataset for a mode.
	EnsureEvalData(ctx context.Context, mode evaluate.Mode) (path string, err error)

	// ResolveKLReference picks the reference for a model: the override
	// when set, else the largest installed quant (by SizeBytes) sharing
	// the model's HF repo. Returns an error when no distinct candidate
	// exists (the model is the only installed quant of its repo).
	ResolveKLReference(modelID, overrideID string) (ModelInfo, error)

	// EvalFlags builds the complete llama-perplexity flag list for a
	// cell: merges the snapshot onto the model's saved config
	// (applySnapshotToConfig), validates the merged config the same
	// way ApplyEphemeralConfig does (ValidateBatchSizes — a -ub > -b
	// sweep fails with the named message, not the loader's raw error),
	// resolves GPU assignment (resolveGPUAssignment), fills
	// evaluate.SnapshotSubset (plain fields + PlacementFlags via
	// models.GPUPlacementFlags with the build's backend), and returns
	// evaluate.MapConfigFlags's output.
	// The single place config becomes CLI flags — phase 01 step 2.
	EvalFlags(modelID string, snap ConfigSnapshot, buildID string) ([]string, error)

	// EnsureKLBase returns the cached logits path for (reference,
	// dataset, chunks, ctx), generating it first when absent. The
	// progress callback receives plain-language status lines; the
	// runner wires it to the run's ProgressDetail through the store —
	// the same transport performance runs already use — which is what
	// makes generation the overview's "visible job step". Progress
	// SOURCE: the generation's own output is newline-less fragments, so
	// the implementation polls the growing .kld.partial file's size on
	// a ticker and reports "generating reference logits: 2.1 of
	// ~4.6 GiB" against the phase 02 estimate — no output parsing. On
	// failure or cancel the partial is deleted, never cached.
	// Generation loads the reference model with the REFERENCE's own
	// saved ModelConfig mapped through EvalFlags (no sweep overrides)
	// — GPU layers and placement apply, so a large reference is not
	// silently evaluated on CPU. Enforces the phase 02 disk guard
	// before generating.
	EnsureKLBase(ctx context.Context, ref ModelInfo, chunks int, buildID string, progress func(string)) (string, error)

	// RunEval executes one evaluation and returns parsed scores.
	RunEval(ctx context.Context, spec evaluate.Spec) (evaluate.Result, error)
}

// ModelInfo bundles registry data for a single model so the JobRunner
// doesn't have to know about the models package.
type ModelInfo struct {
	ID          string // registry model ID
	HFRepoID    string // model.ModelID — passed to llama-benchy --tokenizer
	Quant       string
	SizeGiB     float64
	SizeBytes   int64 // kept alongside the display-oriented SizeGiB
	FilePath    string // path to the GGUF on disk
	DisplayName string // short, human-readable name for the run
	RouterName  string // identifier the router responds to
	Config      ConfigSnapshot // saved baseline; ConfigOverrides overlay on this
}

// JobQueue serializes job execution: only one job runs at a time. Submit
// returns ErrJobAlreadyRunning when the queue is busy.
type JobQueue struct {
	mu      sync.Mutex
	store   *Store
	env     JobEnv
	runner  *Runner
	current *runningJob
}

type runningJob struct {
	id     string
	cancel context.CancelFunc
	done   chan struct{}
}

// NewJobQueue wires up the queue. The returned queue holds no goroutines
// until Submit is called.
func NewJobQueue(store *Store, env JobEnv) *JobQueue {
	return &JobQueue{
		store:  store,
		env:    env,
		runner: NewRunner(store),
	}
}

// Status returns the currently running job (if any).
func (q *JobQueue) Status() (*BenchmarkJob, bool) {
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.current == nil {
		return nil, false
	}
	job, err := q.store.GetJob(q.current.id)
	if err != nil {
		return nil, false
	}
	return job, true
}

// Submit accepts a new job and starts it in a background goroutine.
// Returns ErrJobAlreadyRunning when another job is in flight.
func (q *JobQueue) Submit(job BenchmarkJob) error {
	q.mu.Lock()
	if q.current != nil {
		q.mu.Unlock()
		return ErrJobAlreadyRunning
	}
	if len(job.Cells) == 0 {
		q.mu.Unlock()
		return errors.New("job has no cells")
	}
	job.Status = JobStatusPending
	q.store.SaveJob(job)

	ctx, cancel := context.WithCancel(context.Background())
	rj := &runningJob{id: job.ID, cancel: cancel, done: make(chan struct{})}
	q.current = rj
	q.mu.Unlock()

	go q.run(ctx, job, rj)
	return nil
}

// Cancel signals the running job (if it matches id) to stop. It does
// not wait for the cell loop to wind down.
func (q *JobQueue) Cancel(id string) error {
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.current == nil || q.current.id != id {
		return fmt.Errorf("job %s is not running", id)
	}
	q.current.cancel()
	return nil
}

// RetryFailed re-runs only the cells in CellStatusFailed for the given
// job, treating completed cells as already done. The job must not
// currently be running.
func (q *JobQueue) RetryFailed(id string) error {
	job, err := q.store.GetJob(id)
	if err != nil {
		return err
	}
	any := false
	for i := range job.Cells {
		if job.Cells[i].Status == CellStatusFailed {
			job.Cells[i].Status = CellStatusPending
			job.Cells[i].Error = ""
			any = true
		}
	}
	if !any {
		return errors.New("no failed cells to retry")
	}
	return q.Submit(*job)
}

// run is the per-job orchestration loop. It assumes job.Cells is already
// ordered builds → models → presets so the prevBuildID amortization
// minimizes router restarts.
func (q *JobQueue) run(ctx context.Context, job BenchmarkJob, rj *runningJob) {
	defer func() {
		q.mu.Lock()
		q.current = nil
		q.mu.Unlock()
		close(rj.done)
	}()

	job.Status = JobStatusRunning
	job.StartedAt = time.Now()
	q.store.SaveJob(job)
	slog.Info("benchmark job starting",
		"job", job.ID, "name", job.Name, "cells", len(job.Cells),
		"sweeps", len(job.Sweeps), "has_overrides", job.Overrides != nil)

	var prevBuildID string
	var anyCompleted bool
	var lastApplied appliedConfig

	// Whatever ends this job — completion, failure, cancel — the router
	// has to go back to the user's saved config and build. Unconditional:
	// a job with no overrides can still switch builds, and the runner
	// can't see which builds were already active. The implementation
	// no-ops when there is nothing to restore. WithoutCancel because ctx
	// is already dead on the cancel path, which is precisely when
	// restoring matters most.
	defer func() {
		if err := q.env.ClearEphemeralConfig(context.WithoutCancel(ctx)); err != nil {
			slog.Error("failed to restore saved config/build after benchmark job; the router may still be running under benchmark settings",
				"job", job.ID, "error", err)
		}
	}()

	for i := range job.Cells {
		cell := &job.Cells[i]

		// Already-completed cells (from a prior attempt that's being
		// resumed via RetryFailed) count toward "any completed" but
		// don't re-run.
		if cell.Status == CellStatusCompleted {
			anyCompleted = true
			continue
		}

		if ctx.Err() != nil {
			cell.Status = CellStatusSkipped
			q.store.SaveJob(job)
			continue
		}

		cell.Status = CellStatusRunning
		cell.Attempt++
		cell.Error = ""
		q.store.SaveJob(job)

		cellErr := q.runCell(ctx, &job, cell, &prevBuildID, &lastApplied)

		// A capability cell ran StopRouterForEval (or will on retry), so
		// the router is down and no config is "applied" anymore.
		// Invalidate both caches so the NEXT PERFORMANCE cell
		// unconditionally re-enters EnsureBuildActive/ApplyEphemeralConfig
		// and starts the router back up. Only performance cells ever read
		// those caches — capability cells never call EnsureBuildActive —
		// so consecutive capability cells run back-to-back with the
		// router simply staying stopped: no start/stop churn.
		if GetPreset(cell.Preset).EffectiveSource() == PresetSourceCapability {
			prevBuildID = ""
			lastApplied = appliedConfig{}
		}

		if cellErr != nil {
			cell.Status = CellStatusFailed
			cell.Error = cellErr.Error()
			q.store.SaveJob(job)
			slog.Warn("job cell failed", "job", job.ID, "model", cell.ModelID, "build", cell.BuildID, "preset", cell.Preset, "error", cellErr)
			continue
		}

		cell.Status = CellStatusCompleted
		slog.Info("benchmark cell completed",
			"job", job.ID, "model", cell.ModelID, "sweep", cell.SweepValues)
		anyCompleted = true
		q.store.SaveJob(job)
	}

	job.FinishedAt = time.Now()
	switch {
	case ctx.Err() != nil:
		job.Status = JobStatusCanceled
	case anyCompleted:
		job.Status = JobStatusCompleted
	default:
		job.Status = JobStatusFailed
	}
	q.store.SaveJob(job)
}

// runCell drives one (model, build, preset) cell to completion. The
// caller is responsible for setting the cell to running and recording
// the final status; runCell only returns an error or nil.
//
// prevBuildID is updated when this cell's build differs from the
// previous one so the next iteration knows whether a switch already
// happened.
// runSeq disambiguates run IDs. A millisecond timestamp alone collided
// whenever two cells finished inside the same millisecond, and the store
// silently overwrote the earlier run. Sweeps make that far more likely:
// a sampling sweep needs no router restart, so its cells run back to
// back.
var runSeq atomic.Uint64

func newRunID(attempt int) string {
	return fmt.Sprintf("bench-%d-%d-%d", time.Now().UnixMilli(), runSeq.Add(1), attempt)
}

// appliedConfig is what the router is currently running, so consecutive
// cells needing the same thing don't each pay for a reload. Comparable
// by design — ConfigSnapshot is all scalars.
//
// The build is part of it. Keying on (model, config) alone meant two
// cells on different builds with identical config compared equal, the
// apply was skipped, and — since the build switch now defers its restart
// to that apply — the router never switched at all. The second cell then
// measured the previous build while recording the new one.
type appliedConfig struct {
	modelID string
	buildID string
	cfg     ConfigSnapshot
}

func (q *JobQueue) runCell(ctx context.Context, job *BenchmarkJob, cell *JobCell, prevBuildID *string, lastApplied *appliedConfig) error {
	// Logged before anything can fail, so a failure is always preceded by
	// the cell it belongs to.
	slog.Info("benchmark cell starting",
		"job", job.ID, "model", cell.ModelID, "build", cell.BuildID,
		"preset", cell.Preset, "sweep", cell.SweepValues)

	// Resolved before the build switch so EnsureBuildActive can be told
	// whether a config apply follows it. Fixed overrides form the
	// baseline; this cell's swept values win over them.
	cellOv, err := CellOverrides(job.Overrides, cell.SweepValues)
	if err != nil {
		return fmt.Errorf("resolve cell overrides: %w", err)
	}

	preset := GetPreset(cell.Preset)
	isCapability := preset.EffectiveSource() == PresetSourceCapability

	// The runnable check applies to every cell; starting the router is
	// performance-only. The two used to share one build-change guard,
	// split so a capability cell verifies the build but never calls
	// EnsureBuildActive — starting llama-server just to stop it again
	// would add a full router restart + health wait between every pair
	// of consecutive capability cells.
	if cell.BuildID != *prevBuildID {
		if err := q.env.CheckBuildRunnable(ctx, cell.BuildID); err != nil {
			return fmt.Errorf("build %s not runnable on this host: %w", cell.BuildID, err)
		}
	}
	if !isCapability && cell.BuildID != *prevBuildID {
		if err := q.env.EnsureBuildActive(ctx, cell.BuildID, cellOv != nil); err != nil {
			return fmt.Errorf("activate build %s: %w", cell.BuildID, err)
		}
		*prevBuildID = cell.BuildID
	}

	if isCapability {
		return q.runCapabilityCell(ctx, job, cell, preset, cellOv)
	}

	modelInfo, err := q.env.ResolveModel(cell.ModelID)
	if err != nil {
		return fmt.Errorf("resolve model %s: %w", cell.ModelID, err)
	}
	buildSnap := q.env.ResolveBuild(cell.BuildID)
	if buildSnap.ID == "" {
		return fmt.Errorf("build %s no longer exists", cell.BuildID)
	}

	cfg := applyOverrides(modelInfo.Config, cellOv)

	// Make the merged config real before measuring anything. Without
	// this the cell benchmarks the model's saved config and then records
	// cfg, which is how overrides silently produced mislabeled results.
	//
	// Costs a router restart. At a build boundary EnsureBuildActive
	// deferred its own restart to this one, so the pair costs a single
	// reload rather than two — the first of which would have served the
	// previous cell's config on the new build for no purpose.
	//
	// Sweeping a value that only affects the request (sampling) must not
	// pay for a reload, and consecutive cells that share a config (e.g.
	// several presets under one sweep point) only need one.
	if cellOv != nil {
		want := appliedConfig{modelID: cell.ModelID, buildID: cell.BuildID, cfg: cfg}
		if *lastApplied != want {
			if err := q.env.ApplyEphemeralConfig(ctx, cell.ModelID, cfg); err != nil {
				return fmt.Errorf("apply config overrides for %s: %w", cell.ModelID, err)
			}
			*lastApplied = want
		} else {
			slog.Info("reusing the running config; no reload needed",
				"model", cell.ModelID, "sweep", cell.SweepValues)
		}
	}

	run := BenchmarkRun{
		ID:           newRunID(cell.Attempt),
		JobID:        job.ID,
		CreatedAt:    time.Now(),
		Status:       StatusRunning,
		ModelID:      cell.ModelID,
		ModelName:    modelInfo.DisplayName,
		Quant:        modelInfo.Quant,
		SizeGiB:      modelInfo.SizeGiB,
		Config:       cfg,
		BuildID:      buildSnap.ID,
		BuildRef:     buildSnap.GitRef,
		BuildProfile: buildSnap.Profile,
		Build:        buildSnap,
		GPUs:         GPUSnapshotsFromMetrics(q.env.CurrentMetrics()),
		Preset:       preset.Name,
		SweepValues:  cell.SweepValues,
		PromptTokens: preset.PromptTokens,
		GenTokens:    preset.GenTokens,
	}
	q.store.Save(run)
	cell.BenchmarkRunID = run.ID
	q.store.SaveJob(*job)

	q.runner.Run(ctx, RunConfig{
		Run:        run,
		Preset:     preset,
		RouterURL:  q.env.RouterURL(),
		RouterName: modelInfo.RouterName,
		HFRepoID:   modelInfo.HFRepoID,
		HFToken:    q.env.HFToken(),
		HFHome:     q.env.HFCacheDir(),
		Sampling:   samplingFromOverrides(cellOv),
	}, nil)

	final, err := q.store.Get(run.ID)
	if err != nil {
		return fmt.Errorf("read back run: %w", err)
	}
	if final.Status != StatusCompleted {
		if final.Error != "" {
			return errors.New(final.Error)
		}
		return fmt.Errorf("run ended with status %s", final.Status)
	}
	return nil
}

// runCapabilityCell drives one capability cell (a preset whose
// EffectiveSource is "capability") to completion: llama-perplexity runs
// directly against the model, so the router is stopped for the cell's
// duration.
//
// Deliberately unlike the performance path, which creates its run last,
// the run is created and saved FIRST (status running): the KL base
// generation's progress lands on the run's ProgressDetail, which must
// exist and be stored before generation starts. The performance path
// finalizes its run inside Runner.Run, which this branch bypasses — so
// this branch owns the run's WHOLE lifecycle explicitly, on every exit:
//
//   - any sub-step failure (a: binary, b: dataset, c: reference
//     resolve/generation, e: flags, f: eval) sets StatusFailed + Error
//     and saves;
//   - success (f) sets StatusCompleted and saves;
//   - the reference-model skip (c) DELETES the pre-created run and
//     clears cell.BenchmarkRunID, restoring "skipped cell has no run".
//
// No exit may leave a stored StatusRunning run: the run list renders
// those as live and retry would orphan them.
func (q *JobQueue) runCapabilityCell(ctx context.Context, job *BenchmarkJob, cell *JobCell, preset Preset, cellOv *ConfigOverrides) error {
	modelInfo, err := q.env.ResolveModel(cell.ModelID)
	if err != nil {
		return fmt.Errorf("resolve model %s: %w", cell.ModelID, err)
	}
	buildSnap := q.env.ResolveBuild(cell.BuildID)
	if buildSnap.ID == "" {
		return fmt.Errorf("build %s no longer exists", cell.BuildID)
	}

	// The recorded config and the flags that ran share this one source:
	// the same snapshot goes on the run and into EvalFlags, which is
	// what makes a swept gpu_assign measurably reach llama-perplexity —
	// the config-fidelity criterion.
	cfg := applyOverrides(modelInfo.Config, cellOv)

	run := BenchmarkRun{
		ID:           newRunID(cell.Attempt),
		JobID:        job.ID,
		CreatedAt:    time.Now(),
		Status:       StatusRunning,
		ModelID:      cell.ModelID,
		ModelName:    modelInfo.DisplayName,
		Quant:        modelInfo.Quant,
		SizeGiB:      modelInfo.SizeGiB,
		Config:       cfg,
		BuildID:      buildSnap.ID,
		BuildRef:     buildSnap.GitRef,
		BuildProfile: buildSnap.Profile,
		Build:        buildSnap,
		GPUs:         GPUSnapshotsFromMetrics(q.env.CurrentMetrics()),
		Preset:       preset.Name,
		SweepValues:  cell.SweepValues,
	}
	q.store.Save(run)
	cell.BenchmarkRunID = run.ID
	q.store.SaveJob(*job)

	fail := func(format string, args ...any) error {
		run.Status = StatusFailed
		run.Error = fmt.Sprintf(format, args...)
		run.ProgressDetail = ""
		q.store.Save(run)
		return errors.New(run.Error)
	}

	// a. The build must carry the evaluation binary. Missing means the
	// build predates the binary's installation; the message names the
	// fix.
	binary, err := q.env.EvalBinary(cell.BuildID)
	if err != nil {
		return fail("capability cell cannot run: %v", err)
	}

	// d. Stop the router before any GPU work. Dataset downloads don't
	// need it, but base generation and the eval do; stopping early keeps
	// the ordering simple and the window deterministic.
	if err := q.env.StopRouterForEval(ctx); err != nil {
		return fail("stop the router for the evaluation: %v", err)
	}

	// b. The dataset the mode runs over, downloaded/verified on first
	// use.
	datasetPath, err := q.env.EnsureEvalData(ctx, preset.EvalMode)
	if err != nil {
		return fail("prepare the %s dataset: %v", preset.EvalMode, err)
	}

	// c. KL only: resolve the reference (the job's override when set,
	// else the largest installed quant of the model's own repo) and make
	// sure its cached logits exist.
	var klBasePath, referenceIdentity string
	if preset.EvalMode == evaluate.ModeKLDiv {
		ref, err := q.env.ResolveKLReference(cell.ModelID, job.KLReference)
		if err != nil {
			return fail("resolve the KL reference for %s: %v", modelInfo.DisplayName, err)
		}
		if ref.ID == cell.ModelID {
			// The reference model's own cell (the largest quant's own
			// cell in an all-quants job — the flagship flow): its answer
			// is known (zero), so an expensive eval is waste, and a
			// refusal would break the primary use case. Skip, don't
			// fail: mark completed with a reason, delete the
			// pre-created run, and keep the cell run-free.
			//
			// Deliberately NOT CellStatusSkipped — that status means
			// "job canceled before this cell ran" and such cells are
			// re-run on retry, which is correct for them and wrong for
			// a reference cell. NOT a reuse of Error either (retry and
			// error-styling collisions). Completed-with-reason means the
			// existing loop short-circuit, retry selection, and the
			// Done/Total counting all behave with zero changes.
			cell.SkipReason = "this is the reference model — its difference from itself is zero"
			if err := q.store.Delete(run.ID); err != nil {
				slog.Warn("failed to delete the pre-created run for a skipped reference cell",
					"job", job.ID, "run", run.ID, "error", err)
			}
			cell.BenchmarkRunID = ""
			q.store.SaveJob(*job)
			return nil
		}
		referenceIdentity = ref.ID
		// Generation runs with the reference model loaded (the router is
		// already stopped — the d ordering above). Its progress lands on
		// the stored run's ProgressDetail through the store, the same
		// transport performance runs use.
		klBasePath, err = q.env.EnsureKLBase(ctx, ref, preset.EvalChunks, cell.BuildID, func(line string) {
			run.ProgressDetail = line
			q.store.Save(run)
		})
		if err != nil {
			return fail("KL reference logits for %s: %v", ref.DisplayName, err)
		}
	}

	// e. The complete flag list for this cell's config: the single place
	// config becomes CLI flags.
	flags, err := q.env.EvalFlags(cell.ModelID, cfg, cell.BuildID)
	if err != nil {
		return fail("build the evaluation flags: %v", err)
	}

	// f. Run the evaluation and land the parsed scores on the run.
	// Timings fields stay zero; the detail view renders them as absent.
	spec := evaluate.Spec{
		Binary:      binary,
		ModelPath:   modelInfo.FilePath,
		Mode:        preset.EvalMode,
		DatasetPath: datasetPath,
		Tasks:       preset.EvalTasks,
		Chunks:      preset.EvalChunks,
		KLBasePath:  klBasePath,
		Flags:       flags,
	}
	result, err := q.env.RunEval(ctx, spec)
	if err != nil {
		// The engine's error already carries the tail of the tool's
		// output — keep it in the run's error.
		return fail("%s: %v", preset.EvalMode, err)
	}

	scores := &EvalScores{
		Mode:           result.Mode,
		Dataset:        result.Dataset,
		ContextSize:    result.ContextSize,
		Chunks:         result.Chunks,
		Perplexity:     result.Perplexity,
		PerplexityErr:  result.PerplexityErr,
		Accuracy:       result.Accuracy,
		AccuracyCILow:  result.AccuracyCILow,
		AccuracyCIHigh: result.AccuracyCIHigh,
		Tasks:          result.Tasks,
		KLMean:         result.KLMean,
		KLMeanErr:      result.KLMeanErr,
		KLMax:          result.KLMax,
		KLP999:         result.KLP999,
		SameTopPct:     result.SameTopPct,
		SameTopPctErr:  result.SameTopPctErr,
		Reference:      referenceIdentity,
	}
	run.Eval = scores
	run.Status = StatusCompleted
	run.DurationMs = time.Since(run.CreatedAt).Milliseconds()
	run.ProgressDetail = ""
	q.store.Save(run)
	return nil
}

// samplingFromOverrides extracts the per-request generation settings
// from a job's overrides. These deliberately bypass ConfigSnapshot:
// llama-server takes them per chat-completion request, not from the
// preset INI, so they travel with the benchmark request instead of the
// router config. Previously they were accepted by the form, persisted,
// and then dropped on the floor.
func samplingFromOverrides(o *ConfigOverrides) SamplingParams {
	if o == nil {
		return SamplingParams{}
	}
	return SamplingParams{
		Temperature:   o.Temperature,
		TopP:          o.TopP,
		TopK:          o.TopK,
		MinP:          o.MinP,
		RepeatPenalty: o.RepeatPenalty,
	}
}

// applyOverrides returns base with non-nil ConfigOverrides fields
// applied on top. A nil overrides argument returns base unchanged.
func applyOverrides(base ConfigSnapshot, overrides *ConfigOverrides) ConfigSnapshot {
	if overrides == nil {
		return base
	}
	out := base
	if overrides.GPULayers != nil {
		out.GPULayers = *overrides.GPULayers
	}
	if overrides.ContextSize != nil {
		out.ContextSize = *overrides.ContextSize
	}
	if overrides.Threads != nil {
		out.Threads = *overrides.Threads
	}
	if overrides.BatchSize != nil {
		out.BatchSize = *overrides.BatchSize
	}
	if overrides.UBatchSize != nil {
		out.UBatchSize = *overrides.UBatchSize
	}
	if overrides.FlashAttention != nil {
		out.FlashAttention = *overrides.FlashAttention
	}
	if overrides.KVCacheQuant != nil {
		out.KVCacheQuant = *overrides.KVCacheQuant
	}
	if overrides.DirectIO != nil {
		out.DirectIO = *overrides.DirectIO
	}
	if overrides.GPUAssign != nil {
		out.GPUAssign = *overrides.GPUAssign
	}
	if overrides.TensorSplit != nil {
		out.TensorSplit = *overrides.TensorSplit
	}
	if overrides.SpecType != nil {
		out.SpecType = *overrides.SpecType
	}
	if overrides.DraftModelPath != nil {
		out.DraftModelPath = *overrides.DraftModelPath
	}
	if overrides.DraftMax != nil {
		out.DraftMax = *overrides.DraftMax
	}
	if overrides.DraftMin != nil {
		out.DraftMin = *overrides.DraftMin
	}
	if overrides.DraftPMin != nil {
		out.DraftPMin = *overrides.DraftPMin
	}
	if overrides.NgramSizeN != nil {
		out.NgramSizeN = *overrides.NgramSizeN
	}
	if overrides.NgramSizeM != nil {
		out.NgramSizeM = *overrides.NgramSizeM
	}
	return out
}

// ExpandCells builds the cell matrix in builds → models → presets order.
// Builds is the outermost dimension specifically so EnsureBuildActive
// fires at most once per build per job.
func ExpandCells(modelIDs, buildIDs, presets []string) []JobCell {
	return ExpandCellsWithSweeps(modelIDs, buildIDs, presets, nil)
}

// ExpandCellsWithSweeps builds the cell matrix over the three fixed axes
// plus one axis per sweep, ordered builds → models → sweep combinations
// → presets.
//
// The ordering is about restart cost. Builds stay outermost so
// EnsureBuildActive fires at most once per build. Sweep combinations sit
// above presets because changing a swept config value forces a router
// reload while changing preset does not — so every preset for a given
// config runs before the config changes again. Reversing those two would
// multiply reloads by the preset count.
//
// Capability presets collapse: a capability cell runs only what
// MapConfigFlags lets through, so when the job's sweeps vary parameters
// that never reach the evaluation, each capability preset gets ONE cell
// per distinct eval-reaching configuration, not one per swept value —
// duplicate expensive cells reporting different labels for identical
// runs is the mislabeled-results failure this package guards against
// elsewhere. Performance presets keep the full fan-out from the same
// job.
func ExpandCellsWithSweeps(modelIDs, buildIDs, presets []string, sweeps []SweepAxis) []JobCell {
	combos := sweepCombinations(sweeps)
	cells := make([]JobCell, 0, len(buildIDs)*len(modelIDs)*len(combos)*len(presets))
	for _, b := range buildIDs {
		for _, m := range modelIDs {
			// One emitted eval config per capability preset for this
			// (model, build): the first combo carrying a given
			// eval-reaching configuration wins, later combos collapse
			// onto it.
			emitted := map[string]bool{}
			for _, combo := range combos {
				for _, p := range presets {
					cell := JobCell{
						ModelID: m,
						BuildID: b,
						Preset:  p,
						Status:  CellStatusPending,
					}
					if GetPreset(p).EffectiveSource() == PresetSourceCapability {
						key := capabilityComboKey(p, combo)
						if emitted[key] {
							continue
						}
						emitted[key] = true
						// Record only the eval-reaching values: the
						// cell ran at those and only those, and two
						// cells that differ in the excluded axes would
						// be byte-identical invocations.
						for k, v := range combo {
							if f, ok := LookupSweepField(k); ok && f.AffectsEval {
								if cell.SweepValues == nil {
									cell.SweepValues = map[string]string{}
								}
								cell.SweepValues[k] = v
							}
						}
					} else if len(combo) > 0 {
						// Copy: every cell owns its own map.
						cell.SweepValues = make(map[string]string, len(combo))
						for k, v := range combo {
							cell.SweepValues[k] = v
						}
					}
					cells = append(cells, cell)
				}
			}
		}
	}
	return cells
}

// capabilityComboKey renders a combo's eval-reaching point, namespaced by
// preset, so the collapse dedups per capability preset. Unreachable sweep
// fields are ignored by construction — the key IS the set of values that
// reach the evaluation command line.
func capabilityComboKey(preset string, combo map[string]string) string {
	if len(combo) == 0 {
		return preset
	}
	names := make([]string, 0, len(combo))
	for k := range combo {
		if f, ok := LookupSweepField(k); ok && f.AffectsEval {
			names = append(names, k)
		}
	}
	sort.Strings(names)
	var b strings.Builder
	b.WriteString(preset)
	for _, n := range names {
		b.WriteString(";")
		b.WriteString(n)
		b.WriteString("=")
		b.WriteString(combo[n])
	}
	return b.String()
}

// sweepCombinations returns the Cartesian product of the sweep axes as
// field→value maps. With no sweeps it returns a single empty
// combination, so the caller's loop runs exactly once and unswept jobs
// expand exactly as before.
//
// Axes are sorted by field name so cell order is deterministic across
// runs regardless of how the form serialized them.
func sweepCombinations(sweeps []SweepAxis) []map[string]string {
	ordered := make([]SweepAxis, 0, len(sweeps))
	for _, s := range sweeps {
		if len(s.Values) > 0 {
			ordered = append(ordered, s)
		}
	}
	// Axes processed first vary slowest. Put the reload-affecting ones
	// there so a cheap sampling axis cycles inside them rather than
	// forcing a reload on every cell: sweeping temperature × ubatch_size
	// alphabetically made ubatch alternate every cell, costing one reload
	// per cell instead of one per ubatch value.
	sort.Slice(ordered, func(i, j int) bool {
		ri := SweepRestartsRouter([]SweepAxis{ordered[i]})
		rj := SweepRestartsRouter([]SweepAxis{ordered[j]})
		if ri != rj {
			return ri
		}
		return ordered[i].Field < ordered[j].Field
	})

	combos := []map[string]string{{}}
	for _, axis := range ordered {
		next := make([]map[string]string, 0, len(combos)*len(axis.Values))
		for _, base := range combos {
			for _, v := range axis.Values {
				merged := make(map[string]string, len(base)+1)
				for k, bv := range base {
					merged[k] = bv
				}
				merged[axis.Field] = v
				next = append(next, merged)
			}
		}
		combos = next
	}
	return combos
}
