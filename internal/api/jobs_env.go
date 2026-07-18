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

	// Record the user's selection before the first switch, so the job can
	// restore it. Done before the early return: when the first cell's
	// build is already active, that is still the selection to go back to.
	e.s.captureActiveBuild(e.s.activeBuild())

	if e.s.activeBuild() == buildID && e.s.process.IsRunning() {
		e.s.noteBuildActivated(buildID)
		return nil
	}

	e.s.withConfig(func() {
		e.s.cfg.ActiveBuild = buildID
		e.s.saveConfigLocked()
	})
	e.s.noteBuildActivated(buildID)

	if e.s.process.IsRunning() {
		if err := e.s.process.Stop(); err != nil {
			return fmt.Errorf("stop router: %w", err)
		}
	}
	if err := e.s.startRouter(); err != nil {
		return fmt.Errorf("start router with %s: %w", buildID, err)
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
	return fmt.Errorf("timed out waiting for router to come up on build %s", buildID)
}

// restartRouter stops the router (if up) and starts it again, blocking
// until it is running or the deadline passes. Factored out of
// EnsureBuildActive so config swaps reuse the same wait semantics.
func (e *jobEnv) restartRouter(ctx context.Context, what string) error {
	if e.s.benchOverridesSnapshot() != nil {
		// Record before starting: if the start half-succeeds we still
		// have to restore afterwards.
		e.s.markBenchRouterDirty()
	}
	if e.s.process.IsRunning() {
		if err := e.s.process.Stop(); err != nil {
			return fmt.Errorf("stop router: %w", err)
		}
	}
	if err := e.s.startRouter(); err != nil {
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
// The substitute config is written to a separate preset file and held
// only in memory — the user's models.json and preset.ini are never
// modified, so an interrupted job cannot leave a model misconfigured.
// Callers must pair this with ClearEphemeralConfig.
func (e *jobEnv) ApplyEphemeralConfig(ctx context.Context, modelID string, cfg benchmark.ConfigSnapshot) error {
	base, err := e.s.registry.GetConfig(modelID)
	if err != nil {
		return fmt.Errorf("resolve config for %s: %w", modelID, err)
	}
	merged := applySnapshotToConfig(*base, cfg)
	if err := merged.ValidateBatchSizes(); err != nil {
		// The model-config form rejects an unusable batch pair; the
		// benchmark path has to as well, or a -ub sweep past the batch
		// size either measures the same clamped value under several
		// labels or fails every cell mid-job.
		return fmt.Errorf("%s: %w", modelID, err)
	}

	e.s.setBenchOverrides(map[string]*models.ModelConfig{modelID: &merged})

	// Deliberately left armed when the restart fails. restartRouter can
	// fail *after* llama-server came up on the benchmark preset (e.g.
	// cancelled during the health poll), and clearing here would make the
	// deferred ClearEphemeralConfig a no-op — leaving the router serving
	// normal traffic under benchmark config with nothing to restore it.
	return e.restartRouter(ctx, "benchmark config override")
}

// ClearEphemeralConfig drops any benchmark config override and restarts
// the router back onto the user's saved config. Safe to call when no
// override is active, in which case it does nothing.
func (e *jobEnv) ClearEphemeralConfig(ctx context.Context) error {
	// Config half, keyed off "did we ever start the router for a
	// benchmark" rather than off the override still being armed: a failed
	// apply can leave the router running on the benchmark preset with the
	// override already dropped, and that is precisely the case that must
	// still be restored.
	configDirty := e.s.consumeBenchRouterDirty()

	// Build half. A job that switches builds rewrites the user's saved
	// ActiveBuild, so restore that too — the same leak the ephemeral
	// preset avoids for model config.
	buildDirty := false
	if prev, ok := e.s.consumeActiveBuild(); ok {
		current := e.s.activeBuild()
		switch {
		case prev == current:
			// Never left the user's selection.
		case !e.s.buildStillOurs(current):
			// The user picked a different build while the job ran. Their
			// choice is newer than ours, so leave it alone.
			slog.Info("not restoring pre-benchmark build; the selection changed during the job",
				"pre_job", prev, "current", current)
		default:
			e.s.withConfig(func() {
				e.s.cfg.ActiveBuild = prev
				e.s.saveConfigLocked()
			})
			buildDirty = true
		}
	}

	e.s.setBenchOverrides(nil)
	if !configDirty && !buildDirty {
		return nil
	}

	// Don't resurrect a router the user deliberately stopped mid-job.
	// The saved config and build are restored either way, so the next
	// manual start picks them up.
	if !e.s.process.IsRunning() {
		slog.Info("router is stopped; restored saved config and build without restarting")
		return nil
	}

	// One restart covers both: startRouter re-resolves the active build
	// and rewrites the preset from saved config.
	return e.restartRouter(ctx, "restore saved config and build")
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
