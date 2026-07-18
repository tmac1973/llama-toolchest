package api

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/benchmark"
	"github.com/tmac1973/llama-toolchest/internal/builder"
	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/monitor"
)

// jobEnv adapts *Server to benchmark.JobEnv. Created once at server
// startup and handed to the JobQueue; carries no state of its own.
type jobEnv struct {
	s *Server

	// Transient state for the job that currently owns the router. Held
	// here rather than on Server because it is job-scoped and must never
	// influence an interactive router start; startRouterWith takes it as
	// an explicit parameter. The JobQueue runs one job at a time, and the
	// mutex covers the HTTP goroutine reading ownsRouter.
	mu         sync.Mutex
	jobBuildID string                         // build this job wants; "" = user's
	jobConfigs map[string]*models.ModelConfig // substitute configs; nil = user's
	ownsRouter bool                           // a job has taken over the router
}

// routerOwnedByJob reports whether a benchmark job currently controls the
// router, so interactive start/restart can refuse rather than fight it.
func (e *jobEnv) routerOwnedByJob() bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.ownsRouter
}

// jobRouterOptions renders the job's current intent.
func (e *jobEnv) jobRouterOptions() routerOptions {
	e.mu.Lock()
	defer e.mu.Unlock()
	return routerOptions{buildID: e.jobBuildID, overrides: e.jobConfigs}
}

func newJobEnv(s *Server) *jobEnv { return &jobEnv{s: s} }

// CheckBuildRunnable parses `ldd` output to detect missing shared
// libraries (e.g. a build linked against an older ROCm SONAME than the
// one the host now ships). Linux only; on other OSes returns nil so the
// runner falls through to EnsureBuildActive and any failure surfaces
// from the router instead.
func (e *jobEnv) CheckBuildRunnable(ctx context.Context, buildID string) error {
	if runtime.GOOS != "linux" {
		return nil
	}
	var binary string
	if b, ok := e.s.builder.Find(buildID); ok {
		binary = b.BinaryPath
	}
	if binary == "" {
		return fmt.Errorf("build %s not found", buildID)
	}
	cmd := exec.CommandContext(ctx, "ldd", binary)
	// Mirror what process.Manager does at launch: prepend the binary's
	// directory to LD_LIBRARY_PATH so co-located libs (libllama.so,
	// libggml*.so, etc.) resolve. Without this every build false-flags
	// as broken.
	binDir := filepath.Dir(binary)
	env := os.Environ()
	prepended := false
	for i, kv := range env {
		if strings.HasPrefix(kv, "LD_LIBRARY_PATH=") {
			env[i] = "LD_LIBRARY_PATH=" + binDir + string(os.PathListSeparator) + kv[len("LD_LIBRARY_PATH="):]
			prepended = true
			break
		}
	}
	if !prepended {
		env = append(env, "LD_LIBRARY_PATH="+binDir)
	}
	cmd.Env = env
	// ldd returns non-zero when there are unresolved libs but still
	// prints them, so we deliberately ignore exit code and parse output.
	out, _ := cmd.Output()
	var missing []string
	for _, line := range strings.Split(string(out), "\n") {
		if strings.Contains(line, "=> not found") {
			missing = append(missing, strings.TrimSpace(line))
		}
	}
	if len(missing) > 0 {
		return fmt.Errorf("missing shared libraries: %s — was the build compiled against a different version of the runtime libs (e.g. ROCm SONAME bump)? rebuild llama.cpp on this host", strings.Join(missing, "; "))
	}
	return nil
}

// EnsureBuildActive switches the router to buildID if it isn't already,
// waiting up to 2 minutes for /health to pass.
func (e *jobEnv) EnsureBuildActive(ctx context.Context, buildID string) error {
	if buildID == "" {
		return fmt.Errorf("empty build id")
	}

	// Find the build and verify it's a successful one we can run.
	target, ok := e.s.builder.Find(buildID)
	if !ok {
		return fmt.Errorf("build %s not found", buildID)
	}
	if target.Status != builder.BuildStatusSuccess {
		return fmt.Errorf("build %s is %s, not success", buildID, target.Status)
	}

	e.mu.Lock()
	e.jobBuildID = buildID
	hasConfigs := e.jobConfigs != nil
	e.mu.Unlock()

	// The build is never written to the user's config. It travels as a
	// start parameter, so a crash mid-job leaves the saved selection
	// untouched and there is nothing to restore.
	//
	// Skip the restart when the router is already serving this build on
	// the user's config — which is the common case for a quick benchmark
	// on the build you already have running. Comparing against jobBuildID
	// instead meant the first cell of every job killed a healthy router,
	// unloading models and dropping open sessions for nothing.
	if e.s.process.IsRunning() && e.s.runningBuild() == buildID && !hasConfigs {
		slog.Info("router already on the requested build; not restarting", "build", buildID)
		// Nothing was taken over, so ownership stays clear and cleanup
		// has nothing to restore. Claiming it here reinstated at job end
		// exactly the gratuitous restart this fast path removes from the
		// start — the fix cancelled itself out.
		return nil
	}

	slog.Info("switching router to benchmark build",
		"build", buildID, "was", e.s.runningBuild())

	e.mu.Lock()
	e.ownsRouter = true
	e.mu.Unlock()

	return e.restartRouter(ctx, "build "+buildID)
}

// restartRouter stops the router (if up) and starts it again, blocking
// until it is running or the deadline passes. Factored out of
// EnsureBuildActive so config swaps reuse the same wait semantics.
func (e *jobEnv) restartRouter(ctx context.Context, what string) error {
	if e.s.process.IsRunning() {
		if err := e.s.process.Stop(); err != nil {
			return fmt.Errorf("stop router: %w", err)
		}
	}
	if err := e.s.startRouterWith(e.jobRouterOptions()); err != nil {
		return fmt.Errorf("start router for %s: %w", what, err)
	}
	deadline := time.Now().Add(2 * time.Minute)
	for time.Now().Before(deadline) {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if e.s.process.IsRunning() {
			return nil
		}
		time.Sleep(500 * time.Millisecond)
	}
	return fmt.Errorf("timed out waiting for router after %s", what)
}

// ApplyEphemeralConfig makes modelID run under cfg for the next
// benchmark cell, restarting the router so it re-reads the preset.
//
// The substitute config travels as a start parameter and is written to a
// separate preset file — the user's models.json and preset.ini are never
// modified, and an interactive restart cannot pick it up. Callers must
// pair this with ClearEphemeralConfig.
func (e *jobEnv) ApplyEphemeralConfig(ctx context.Context, modelID string, cfg benchmark.ConfigSnapshot) error {
	base, err := e.s.registry.GetConfig(modelID)
	if err != nil {
		return fmt.Errorf("resolve config for %s: %w", modelID, err)
	}
	merged := applySnapshotToConfig(*base, cfg)
	if err := resolveGPUAssignment(&merged, *base, len(e.s.monitor.Current().GPU)); err != nil {
		return fmt.Errorf("%s: %w", modelID, err)
	}
	if err := merged.ValidateBatchSizes(); err != nil {
		// The model-config form rejects an unusable batch pair; the
		// benchmark path has to as well, or a -ub sweep past the batch
		// size either measures the same clamped value under several
		// labels or fails the cell with a confusing loader error.
		return fmt.Errorf("%s: %w", modelID, err)
	}

	slog.Info("applying benchmark config",
		"model", modelID,
		"changes", configDiff(*base, merged),
	)

	e.mu.Lock()
	e.jobConfigs = map[string]*models.ModelConfig{modelID: &merged}
	e.ownsRouter = true
	e.mu.Unlock()

	// Deliberately left in place when the restart fails. restartRouter
	// can fail after llama-server came up on the benchmark preset (e.g.
	// cancelled during the health poll), and clearing here would make the
	// deferred cleanup a no-op, leaving the router serving normal traffic
	// under benchmark config.
	return e.restartRouter(ctx, "benchmark config override")
}

// ClearEphemeralConfig hands the router back to the user's saved build
// and config. A no-op when no job took it over.
func (e *jobEnv) ClearEphemeralConfig(ctx context.Context) error {
	e.mu.Lock()
	owned := e.ownsRouter
	e.jobBuildID = ""
	e.jobConfigs = nil
	e.mu.Unlock()

	if !owned {
		slog.Debug("benchmark job ended; router was never taken over, nothing to restore")
		return nil
	}
	slog.Info("benchmark job ended; restoring the user's build and config")

	// Nothing was persisted, so there is no saved state to repair — the
	// only thing to undo is which build and preset the process is
	// running. If the user stopped the router mid-job, leave it stopped:
	// their next start already picks up their own settings.
	if !e.s.process.IsRunning() {
		e.releaseRouter()
		slog.Info("router is stopped; benchmark job released it without restarting")
		return nil
	}

	// Ownership is released whatever happens. Holding it after a failed
	// restore was meant to record that work was owed, but nothing ever
	// retried, and the guards read it — so one failed restore locked the
	// user out of starting the router at all.
	//
	// Releasing is also the better recovery: with no job running, the
	// user's next Start uses their own build and config, which is exactly
	// what the restore was trying to achieve.
	defer e.releaseRouter()
	if err := e.restartRouter(ctx, "restore the user's build and config"); err != nil {
		slog.Error("failed to restart the router on the user's saved config after a benchmark job; start it from the Service page to recover",
			"error", err)
		return err
	}
	return nil
}

// releaseRouter marks the router as no longer job-controlled.
func (e *jobEnv) releaseRouter() {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.ownsRouter = false
}

// applySnapshotToConfig overlays a benchmark ConfigSnapshot onto a copy
// of the model's saved config. Zero values mean "not overridden", which
// matches how ConfigSnapshot is built in ResolveModel — a snapshot
// always carries the saved value unless a job override replaced it.
func applySnapshotToConfig(base models.ModelConfig, snap benchmark.ConfigSnapshot) models.ModelConfig {
	out := base

	// Every field is assigned unconditionally. ResolveModel seeds the
	// snapshot from the model's saved config and applyOverrides then
	// replaces only what the job set, so the snapshot is authoritative
	// for every field it models — a zero is the value zero, not "unset".
	//
	// Skipping zeros here silently discarded legitimate overrides
	// (gpu_layers=0 for CPU-only, ubatch/threads/context edge values)
	// while the run still recorded them as applied. That is exactly the
	// mislabeled-result bug this whole mechanism exists to prevent.
	out.GPULayers = snap.GPULayers
	out.ContextSize = snap.ContextSize
	out.Threads = snap.Threads
	out.BatchSize = snap.BatchSize
	out.UBatchSize = snap.UBatchSize
	out.GPUAssign = snap.GPUAssign
	out.TensorSplit = snap.TensorSplit
	out.KVCacheQuant = snap.KVCacheQuant
	out.SpecType = snap.SpecType
	out.DraftModelPath = snap.DraftModelPath
	out.FlashAttention = snap.FlashAttention
	out.DirectIO = snap.DirectIO
	return out
}

// resolveGPUAssignment turns a gpu_assign selection into the fields the
// preset actually emits. writeConfigParams never reads GPUAssign — it
// emits tensor-split, split-mode and main-gpu — so without this a
// gpu_assign sweep generates byte-identical presets for every point and
// reports one measurement under several labels.
//
// Only applied when the snapshot's assignment differs from the base's,
// so a job that doesn't touch gpu_assign leaves the model's saved
// split-mode and main-gpu alone.
func resolveGPUAssignment(out *models.ModelConfig, base models.ModelConfig, numGPUs int) error {
	if out.GPUAssign == base.GPUAssign || out.GPUAssign == "" || out.GPUAssign == "custom" {
		return nil
	}
	// A job setting both is contradictory; it is refused at definition
	// time (validateGPUAssignment). Reaching here means only gpu_assign
	// was set, so deriving the split is unambiguous.
	if numGPUs <= 0 {
		// Telemetry can be absent (tool missing, first poll not yet
		// complete) on a host where inference works fine, so failing the
		// cell here killed jobs that would have run. Jobs that actually
		// sweep gpu_assign are refused at definition time instead, where
		// the message can be acted on.
		return fmt.Errorf("cannot resolve GPU assignment %q: no GPUs detected", out.GPUAssign)
	}
	ts, sm, mg := models.ResolveGPUAssign(out.GPUAssign, numGPUs)
	out.TensorSplit, out.SplitMode, out.MainGPU = ts, sm, mg
	return nil
}

// ResolveModel pulls registry data into the shape the JobRunner expects.
func (e *jobEnv) ResolveModel(modelID string) (benchmark.ModelInfo, error) {
	m, err := e.s.registry.Get(modelID)
	if err != nil {
		return benchmark.ModelInfo{}, err
	}
	cfg, err := e.s.registry.GetConfig(modelID)
	if err != nil {
		return benchmark.ModelInfo{}, err
	}
	return benchmark.ModelInfo{
		HFRepoID:    m.ModelID,
		Quant:       m.Quant,
		SizeGB:      models.BytesToGB(m.SizeBytes),
		DisplayName: shortenModelName(m.ModelID),
		RouterName:  e.s.registry.RouterName(modelID),
		Config: benchmark.ConfigSnapshot{
			GPULayers:      cfg.GPULayers,
			ContextSize:    cfg.ContextSize,
			GPUAssign:      cfg.GPUAssign,
			TensorSplit:    cfg.TensorSplit,
			FlashAttention: cfg.FlashAttention,
			KVCacheQuant:   cfg.KVCacheQuant,
			DirectIO:       cfg.DirectIO,
			Threads:        cfg.Threads,
			BatchSize:      cfg.BatchSize,
			UBatchSize:     cfg.UBatchSize,
			SpecType:       cfg.SpecType,
			DraftModelPath: cfg.DraftModelPath,
		},
	}, nil
}

// ResolveBuild reuses the same builder lookup the migration uses.
func (e *jobEnv) ResolveBuild(buildID string) benchmark.BuildSnapshot {
	return builderResolver(e.s.builder)(buildID)
}

func (e *jobEnv) CurrentMetrics() monitor.Metrics { return e.s.monitor.Current() }

func (e *jobEnv) RouterURL() string {
	return fmt.Sprintf("http://localhost:%d", e.s.cfg.LlamaPort)
}

func (e *jobEnv) HFToken() string { return e.s.cfg.HFToken }

// HFCacheDir is a stable subdir of DataDir so the tokenizer cache
// survives container restarts (DataDir is the mounted volume) and
// doesn't bloat the writable layer.
func (e *jobEnv) HFCacheDir() string {
	return filepath.Join(e.s.cfg.DataDir, "hf-cache")
}

// shortenModelName mirrors the trim done in handleStartBenchmark so cell
// runs label the same way as the existing single-run path.
func shortenModelName(modelID string) string {
	return models.ShortModelName(modelID)
}

// configDiff reports which launch-relevant fields a benchmark override
// changes, so a log line answers "did my override actually reach
// llama-server, and with what value" without diffing structs by eye.
func configDiff(base, merged models.ModelConfig) []string {
	var out []string
	add := func(name string, from, to any) {
		if from != to {
			out = append(out, fmt.Sprintf("%s %v→%v", name, from, to))
		}
	}
	add("ctx-size", base.ContextSize, merged.ContextSize)
	add("gpu-layers", base.GPULayers, merged.GPULayers)
	add("threads", base.Threads, merged.Threads)
	add("batch-size", base.BatchSize, merged.BatchSize)
	add("ubatch-size", base.UBatchSize, merged.UBatchSize)
	add("flash-attn", base.FlashAttention, merged.FlashAttention)
	add("cache-type", base.KVCacheQuant, merged.KVCacheQuant)
	add("direct-io", base.DirectIO, merged.DirectIO)
	add("tensor-split", base.TensorSplit, merged.TensorSplit)
	add("split-mode", base.SplitMode, merged.SplitMode)
	add("main-gpu", base.MainGPU, merged.MainGPU)
	add("spec-type", base.SpecType, merged.SpecType)
	if len(out) == 0 {
		return []string{"none"}
	}
	return out
}
