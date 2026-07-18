package benchmark

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sort"
	"sync"
	"sync/atomic"
	"time"

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
	// buildID, restarting llama-server when the active build differs.
	// Blocks until the router is reachable.
	EnsureBuildActive(ctx context.Context, buildID string) error

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
}

// ModelInfo bundles registry data for a single model so the JobRunner
// doesn't have to know about the models package.
type ModelInfo struct {
	HFRepoID    string // model.ModelID — passed to llama-benchy --tokenizer
	Quant       string
	SizeGB      float64
	DisplayName string         // short, human-readable name for the run
	RouterName  string         // identifier the router responds to
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

		if err := q.runCell(ctx, &job, cell, &prevBuildID, &lastApplied); err != nil {
			cell.Status = CellStatusFailed
			cell.Error = err.Error()
			q.store.SaveJob(job)
			slog.Warn("job cell failed", "job", job.ID, "model", cell.ModelID, "build", cell.BuildID, "preset", cell.Preset, "error", err)
			continue
		}

		cell.Status = CellStatusCompleted
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

// appliedConfig is the (model, config) pair currently pushed to the
// router, so consecutive cells needing the same thing don't each pay for
// a reload. Comparable by design — ConfigSnapshot is all scalars.
type appliedConfig struct {
	modelID string
	cfg     ConfigSnapshot
}

func (q *JobQueue) runCell(ctx context.Context, job *BenchmarkJob, cell *JobCell, prevBuildID *string, lastApplied *appliedConfig) error {
	if cell.BuildID != *prevBuildID {
		if err := q.env.CheckBuildRunnable(ctx, cell.BuildID); err != nil {
			return fmt.Errorf("build %s not runnable on this host: %w", cell.BuildID, err)
		}
		if err := q.env.EnsureBuildActive(ctx, cell.BuildID); err != nil {
			return fmt.Errorf("activate build %s: %w", cell.BuildID, err)
		}
		*prevBuildID = cell.BuildID
	}

	modelInfo, err := q.env.ResolveModel(cell.ModelID)
	if err != nil {
		return fmt.Errorf("resolve model %s: %w", cell.ModelID, err)
	}
	buildSnap := q.env.ResolveBuild(cell.BuildID)
	if buildSnap.ID == "" {
		return fmt.Errorf("build %s no longer exists", cell.BuildID)
	}
	preset := GetPreset(cell.Preset)

	// Fixed overrides form the baseline; this cell's swept values win
	// over them.
	cellOv, err := CellOverrides(job.Overrides, cell.SweepValues)
	if err != nil {
		return fmt.Errorf("resolve cell overrides: %w", err)
	}
	cfg := applyOverrides(modelInfo.Config, cellOv)

	// Make the merged config real before measuring anything. Without
	// this the cell benchmarks the model's saved config and then records
	// cfg, which is how overrides silently produced mislabeled results.
	//
	// Costs a router restart. On a build boundary that's the second one
	// (EnsureBuildActive already restarted), but cells are ordered
	// builds-outermost so that pairing happens once per build, not once
	// per cell.
	// Sweeping a value that only affects the request (sampling) must not
	// pay for a reload, and consecutive cells that share a config (e.g.
	// several presets under one sweep point) only need one.
	if cellOv != nil {
		want := appliedConfig{modelID: cell.ModelID, cfg: cfg}
		if *lastApplied != want {
			if err := q.env.ApplyEphemeralConfig(ctx, cell.ModelID, cfg); err != nil {
				return fmt.Errorf("apply config overrides for %s: %w", cell.ModelID, err)
			}
			*lastApplied = want
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
		SizeGB:       modelInfo.SizeGB,
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
func ExpandCellsWithSweeps(modelIDs, buildIDs, presets []string, sweeps []SweepAxis) []JobCell {
	combos := sweepCombinations(sweeps)
	cells := make([]JobCell, 0, len(buildIDs)*len(modelIDs)*len(combos)*len(presets))
	for _, b := range buildIDs {
		for _, m := range modelIDs {
			for _, combo := range combos {
				for _, p := range presets {
					cell := JobCell{
						ModelID: m,
						BuildID: b,
						Preset:  p,
						Status:  CellStatusPending,
					}
					if len(combo) > 0 {
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
	sort.Slice(ordered, func(i, j int) bool { return ordered[i].Field < ordered[j].Field })

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
