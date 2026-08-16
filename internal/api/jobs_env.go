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
	"github.com/tmac1973/llama-toolchest/internal/evaluate"
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
	// stoppedForEval records that THIS JOB stopped a running router for
	// a capability evaluation (StopRouterForEval). It is the same
	// ownership fact ownsRouter records, but ClearEphemeralConfig needs
	// to tell "the job stopped it" from "the user stopped it mid-job" —
	// both read as IsRunning() == false — and only the first case may
	// restart the router on cleanup.
	stoppedForEval bool
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
func (e *jobEnv) EnsureBuildActive(ctx context.Context, buildID string, configFollows bool) error {
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

	// A config apply is coming immediately, and it restarts the router
	// anyway with this build now recorded. Restarting here as well would
	// cost a second reload whose only effect is to briefly serve the
	// previous cell's config on the new build.
	if configFollows {
		slog.Info("deferring build switch to the config apply that follows",
			"build", buildID, "was", e.s.runningBuild())
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
//
// Eval-stop case: when this job stopped a running router for a
// capability evaluation (stoppedForEval), "not running" does not mean
// the user stopped it — the job did, and cleanup has to restart the
// router, restoring the pre-job state. The user-stop semantics (leave
// it stopped) apply only to the performance-job case.
func (e *jobEnv) ClearEphemeralConfig(ctx context.Context) error {
	e.mu.Lock()
	owned := e.ownsRouter
	stoppedForEval := e.stoppedForEval
	e.jobBuildID = ""
	e.jobConfigs = nil
	e.ownsRouter = false
	e.stoppedForEval = false
	e.mu.Unlock()

	if !owned && !stoppedForEval {
		slog.Debug("benchmark job ended; router was never taken over, nothing to restore")
		return nil
	}
	slog.Info("benchmark job ended; restoring the user's build and config")

	// Nothing was persisted, so there is no saved state to repair — the
	// only thing to undo is which build and preset the process is
	// running. If the user stopped the router mid-job, leave it stopped:
	// their next start already picks up their own settings.
	//
	// The eval-stop case is the exception: this job stopped a running
	// router for an evaluation, so "stopped" is our doing, not the
	// user's, and the router goes back up — including on failure and
	// cancel, which is exactly when this path runs with the job context
	// dead.
	if !e.s.process.IsRunning() && !stoppedForEval {
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

// StopRouterForEval stops llama-server so an evaluation can own the GPU.
// It records that THIS JOB stopped a RUNNING router: ownsRouter (the same
// flag EnsureBuildActive/ApplyEphemeralConfig set) AND stoppedForEval, so
// ClearEphemeralConfig at job end restarts the router instead of treating
// "not running" as a user stop. When the router was already stopped (by
// the user, before the job) nothing is recorded and cleanup leaves it
// stopped. Idempotent: a second call for the same job changes nothing.
func (e *jobEnv) StopRouterForEval(ctx context.Context) error {
	e.mu.Lock()
	if e.stoppedForEval {
		e.mu.Unlock()
		return nil
	}
	wasRunning := e.s.process.IsRunning()
	if wasRunning {
		e.stoppedForEval = true
		e.ownsRouter = true
	}
	e.mu.Unlock()

	if !wasRunning {
		slog.Info("router already stopped before the evaluation; leaving it stopped")
		return nil
	}
	if err := e.s.process.Stop(); err != nil {
		return fmt.Errorf("stop router for evaluation: %w", err)
	}
	// Wait for the process to actually die so the evaluation gets
	// exclusive VRAM; a Stop() that returned early would let
	// llama-perplexity fight llama-server for the model.
	deadline := time.Now().Add(30 * time.Second)
	for e.s.process.IsRunning() && time.Now().Before(deadline) {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(200 * time.Millisecond):
		}
	}
	if e.s.process.IsRunning() {
		return fmt.Errorf("router did not stop in time for the evaluation")
	}
	slog.Info("router stopped for evaluation")
	return nil
}

// EvalBinary returns the path to the build's llama-perplexity, installed
// next to llama-server. A build made before the install step existed
// lacks it: the error names the fix (rebuild) rather than failing deep
// in the cell with a bare "not found".
func (e *jobEnv) EvalBinary(buildID string) (string, error) {
	b, ok := e.s.builder.Find(buildID)
	if !ok {
		return "", fmt.Errorf("build %s not found", buildID)
	}
	bin := filepath.Join(filepath.Dir(b.BinaryPath), "llama-perplexity")
	if _, err := os.Stat(bin); err != nil {
		return "", fmt.Errorf("build %s has no llama-perplexity binary — it predates the binary's installation; rebuild the build to install it", buildID)
	}
	return bin, nil
}

// EnsureEvalData downloads/verifies the mode's pinned dataset under the
// eval-data root (phase 02's EnsureDataset: stat short-circuit, then
// temp download → SHA-256 verify → rename).
func (e *jobEnv) EnsureEvalData(ctx context.Context, mode evaluate.Mode) (string, error) {
	return evaluate.EnsureDataset(ctx, evaluate.EvalDataRoot(e.s.cfg.DataDir), mode.DatasetName())
}

// modelInfoBundle folds a registry model into the JobRunner's ModelInfo
// shape. One builder for both ResolveModel and ResolveKLReference so the
// two cannot drift about which fields exist.
func (e *jobEnv) modelInfoBundle(m *models.Model) (benchmark.ModelInfo, error) {
	cfg, err := e.s.registry.GetConfig(m.ID)
	if err != nil {
		return benchmark.ModelInfo{}, err
	}
	return benchmark.ModelInfo{
		ID:          m.ID,
		HFRepoID:    m.ModelID,
		Quant:       m.Quant,
		SizeGiB:     models.BytesToGiB(m.SizeBytes),
		SizeBytes:   m.SizeBytes,
		FilePath:    m.FilePath,
		DisplayName: shortenModelName(m.ModelID),
		RouterName:  e.s.registry.RouterName(m.ID),
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
			DraftMax:       cfg.DraftMax,
			DraftMin:       cfg.DraftMin,
			DraftPMin:      cfg.DraftPMin,
			NgramSizeN:     cfg.NgramSizeN,
			NgramSizeM:     cfg.NgramSizeM,
		},
	}, nil
}

// resolveKLReference picks the KL reference for modelID: overrideID when
// set, else the largest installed quant (by SizeBytes) sharing the
// model's HF repo. Errors when no distinct candidate exists — the model
// is the only installed quant of its repo, so there is nothing to
// compare it against.
func (e *jobEnv) resolveKLReference(modelID, overrideID string) (benchmark.ModelInfo, error) {
	if overrideID != "" {
		m, err := e.s.registry.Get(overrideID)
		if err != nil {
			return benchmark.ModelInfo{}, fmt.Errorf("KL reference model %s is not installed: %w", overrideID, err)
		}
		return e.modelInfoBundle(m)
	}
	base, err := e.s.registry.Get(modelID)
	if err != nil {
		return benchmark.ModelInfo{}, fmt.Errorf("resolve model %s: %w", modelID, err)
	}
	// The reference is the largest installed quant of the repo — the
	// model itself included, so in a multi-quant job every model (and
	// the largest quant's own cell) resolves against the same
	// reference. A cell whose reference is itself is the skip case the
	// runner handles; an error is reserved for the repo with no
	// distinct candidate at all.
	var best *models.Model
	others := 0
	for _, m := range e.s.registry.List() {
		if m.ModelID != base.ModelID {
			continue
		}
		if m.ID != modelID {
			others++
		}
		if best == nil || m.SizeBytes > best.SizeBytes {
			best = m
		}
	}
	if best == nil {
		return benchmark.ModelInfo{}, fmt.Errorf("no installed quant found for %s", modelID)
	}
	if best.ID == modelID && others == 0 {
		return benchmark.ModelInfo{}, fmt.Errorf("no KL reference for %s: it is the only installed quant of %s — install a second quant of that repo, or pick a reference model", modelID, base.ModelID)
	}
	return e.modelInfoBundle(best)
}

// ResolveKLReference implements the JobEnv method over the registry.
func (e *jobEnv) ResolveKLReference(modelID, overrideID string) (benchmark.ModelInfo, error) {
	return e.resolveKLReference(modelID, overrideID)
}

// buildBackend returns the build's backend profile (the first argument
// to models.GPUPlacementFlags — it decides whether device names exist
// and what they are called).
func (e *jobEnv) buildBackend(buildID string) string {
	if b, ok := e.s.builder.Find(buildID); ok {
		return b.Profile
	}
	return ""
}

// EvalFlags builds the complete llama-perplexity flag list for a cell.
// It is the single place config becomes CLI flags: the same merge and
// validation the performance path applies before restarting the router
// (applySnapshotToConfig → resolveGPUAssignment → ValidateBatchSizes),
// then the merged config through the phase 01 allow-list
// (evaluate.MapConfigFlags) with placement rendered via
// models.GPUPlacementFlags for the build's backend.
func (e *jobEnv) EvalFlags(modelID string, snap benchmark.ConfigSnapshot, buildID string) ([]string, error) {
	base, err := e.s.registry.GetConfig(modelID)
	if err != nil {
		return nil, fmt.Errorf("resolve config for %s: %w", modelID, err)
	}
	merged := applySnapshotToConfig(*base, snap)
	if err := resolveGPUAssignment(&merged, *base, len(e.s.monitor.Current().GPU)); err != nil {
		return nil, fmt.Errorf("%s: %w", modelID, err)
	}
	if err := merged.ValidateBatchSizes(); err != nil {
		// The same named message the performance path refuses with:
		// a -ub > -b sweep fails here, not with the loader's raw error.
		return nil, fmt.Errorf("%s: %w", modelID, err)
	}
	subset := evaluate.SnapshotSubset{
		GPULayers:      merged.GPULayers,
		Threads:        merged.Threads,
		BatchSize:      merged.BatchSize,
		UBatchSize:     merged.UBatchSize,
		FlashAttention: merged.FlashAttention,
		KVCacheQuant:   merged.KVCacheQuant,
		DirectIO:       merged.DirectIO,
		PlacementFlags: models.GPUPlacementFlags(&merged, e.buildBackend(buildID)),
	}
	return evaluate.MapConfigFlags(subset), nil
}

// klFullRunChunksEstimate is the chunk count a FULL run (0 = no cap) is
// estimated at for the disk guard: the wikitext-2 test set yields about
// 650 chunks at the fixed 512-token context. An overestimate is fine —
// the guard may refuse early, never under-reserve.
const klFullRunChunksEstimate = 650

// EnsureKLBase returns the cached logits path for (reference, dataset,
// chunks, ctx), generating it first when absent.
//
// Progress: the generation's own output is newline-less fragments, so it
// polls the growing .kld.partial file (the phase 02 interruption-safe
// temp path — generation never writes the final name directly) on a
// ticker and reports "generating reference logits: X of ~Y GiB" against
// the phase 02 estimate. No output parsing.
//
// Generation loads the reference model with the reference's OWN saved
// ModelConfig mapped through EvalFlags (no sweep overrides), so GPU
// layers and placement apply — a large reference is not silently
// evaluated on CPU. The phase 02 disk guard runs before generating. On
// failure or cancel the partial is deleted, never cached.
func (e *jobEnv) EnsureKLBase(ctx context.Context, ref benchmark.ModelInfo, chunks int, buildID string, progress func(string)) (string, error) {
	root := evaluate.EvalDataRoot(e.s.cfg.DataDir)
	key := evaluate.KLBaseKey{
		ModelID: ref.HFRepoID,
		Quant:   ref.Quant,
		Dataset: evaluate.ModeKLDiv.DatasetName(),
		Chunks:  chunks,
		Ctx:     evaluate.EvalContextSize,
	}
	if evaluate.HasKLBase(root, key) {
		return evaluate.KLBasePath(root, key), nil
	}

	estChunks := chunks
	if estChunks <= 0 {
		estChunks = klFullRunChunksEstimate
	}
	var vocab int
	if m, err := e.s.registry.Get(ref.ID); err == nil {
		vocab = m.VocabSize
	}
	estimate := evaluate.KLBaseSizeEstimate(estChunks, evaluate.EvalContextSize, vocab)
	if err := evaluate.CheckKLBaseSpace(root, estimate); err != nil {
		return "", err
	}

	binary, err := e.EvalBinary(buildID)
	if err != nil {
		return "", err
	}
	datasetPath, err := e.EnsureEvalData(ctx, evaluate.ModeKLDiv)
	if err != nil {
		return "", fmt.Errorf("dataset for KL base generation: %w", err)
	}
	flags, err := e.EvalFlags(ref.ID, ref.Config, buildID)
	if err != nil {
		return "", fmt.Errorf("reference %s: %w", ref.DisplayName, err)
	}

	if err := os.MkdirAll(evaluate.LogitsDir(root), 0o755); err != nil {
		return "", err
	}
	partial := evaluate.KLBasePartialPath(root, key)
	final := evaluate.KLBasePath(root, key)

	if progress != nil {
		progress(fmt.Sprintf("generating reference logits for %s — ~%s", ref.DisplayName, evaluateFormatBytes(estimate)))
	}

	// The ticker reports file growth; progress only stores into the run,
	// so it is safe to call off the job goroutine. The WaitGroup makes
	// the goroutine fully done before EnsureKLBase returns, so no
	// progress write can race the caller's post-generation run writes.
	done := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		ticker := time.NewTicker(2 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-done:
				return
			case <-ticker.C:
				if progress == nil {
					continue
				}
				if info, err := os.Stat(partial); err == nil {
					progress(fmt.Sprintf("generating reference logits: %.1f of ~%.1f GiB",
						float64(info.Size())/1024/1024/1024, float64(estimate)/1024/1024/1024))
				}
			}
		}
	}()

	spec := evaluate.Spec{
		Binary:      binary,
		ModelPath:   ref.FilePath,
		Mode:        evaluate.ModeKLDiv,
		DatasetPath: datasetPath,
		Chunks:      chunks,
		KLBasePath:  partial,
		Flags:       flags,
	}
	genErr := evaluate.GenerateKLBase(ctx, spec)
	close(done)
	wg.Wait()

	if genErr != nil {
		// Failure or cancel: the partial is a corpse, never a cache
		// entry. A later EnsureKLBase regenerates from scratch.
		os.Remove(partial)
		return "", genErr
	}
	if err := os.Rename(partial, final); err != nil {
		os.Remove(partial)
		return "", fmt.Errorf("installing KL base logits: %w", err)
	}
	return final, nil
}

// evaluateFormatBytes is the byte formatter the KL guard errors use.
// (evaluate.formatBytes is unexported; the progress lines only need the
// same units.)
func evaluateFormatBytes(b int64) string {
	const unit = 1024
	if b < unit {
		return fmt.Sprintf("%d B", b)
	}
	div, exp := int64(unit), 0
	for n := b / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %ciB", float64(b)/float64(div), "KMGTPE"[exp])
}

// RunEval implements the JobEnv method over the phase 01 engine.
func (e *jobEnv) RunEval(ctx context.Context, spec evaluate.Spec) (evaluate.Result, error) {
	return evaluate.Run(ctx, spec)
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
	out.DraftMax = snap.DraftMax
	out.DraftMin = snap.DraftMin
	out.DraftPMin = snap.DraftPMin
	out.NgramSizeN = snap.NgramSizeN
	out.NgramSizeM = snap.NgramSizeM
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
	return e.modelInfoBundle(m)
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
	add("draft-max", base.DraftMax, merged.DraftMax)
	add("draft-min", base.DraftMin, merged.DraftMin)
	add("draft-p-min", base.DraftPMin, merged.DraftPMin)
	add("ngram-size-n", base.NgramSizeN, merged.NgramSizeN)
	add("ngram-size-m", base.NgramSizeM, merged.NgramSizeM)
	if len(out) == 0 {
		return []string{"none"}
	}
	return out
}
