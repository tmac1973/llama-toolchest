# Phase 03 — Capability presets, job runner integration, and the job form

**Depends on:** 01 (engine), 02 (datasets/logits cache) · **Enables:**
results display and management UI (04).

## Goal

Capability evaluations become runnable jobs: eight new presets in the
picker, the cell loop branching on preset kind, router exclusivity, KL
reference resolution with on-demand base generation as a visible step, and
scores landing on the run. After this phase a mixed job runs end to end;
scores are only visible via the run JSON until phase 04.

## Files touched

- `internal/benchmark/benchmark.go` — capability presets use the
  EXISTING dispatch key: `Preset.Source` gains the value
  `"capability"` (no separate `Kind` field — `EffectiveSource()` at
  `benchmark.go:248-254` is already the dispatch key and a second one
  would drift). `Preset` gains `EvalMode`, `EvalTasks`, `EvalChunks`;
  the eight capability presets appended in `Presets()`. Every site
  that branches on `EffectiveSource()` handles the new value:
  `runner.go:143`'s switch gains a `case "capability"` returning an
  internal error ("capability presets run through the job runner" —
  the cell loop branches before ever calling it, so reaching this is a
  bug, not a user state); the sweep registry (`sweep.go`) gains the
  per-field `AffectsEval` flag per step 3, while
  `ValidateSamplingSupport` itself is unchanged — the capability
  refusal lives in `bench_jobs.go` (step 7). `bench_about.go:45`
  needs NO change (its
  `!= PresetSourceBenchy → continue` already handles a new source);
  the About-benchmarks modal's preset table is phase 04's concern.
- `internal/benchmark/job.go` — `Job` gains `KLReference string`
  (registry ID; "" = automatic), carried on `jobCreateRequest` in the api.
- `internal/benchmark/job_runner.go` — `JobEnv` additions and the
  capability-cell branch; sampling-axis cell collapsing. `ModelInfo`
  (`job_runner.go:80-87`) gains additive fields the capability path
  needs and today's struct lacks: `ID` (registry model ID),
  `FilePath` (GGUF path), and `SizeBytes int64` (kept alongside the
  display-oriented `SizeGiB`); populated where `ModelInfo` is built
  (`internal/api/jobs_env.go:345-380`, which already has all three
  before converting them away).
- `internal/api/jobs_env.go` — `jobEnv` implementations of the new
  methods.
- `internal/api/bench_jobs.go` — request field, validation, matrix count.
- `internal/api/bench.go` (`:467`) / `web/templates/partials/benchmark_form.html`
  — the single-run quick-benchmark form filters capability presets out
  (step 6).
- `web/templates/partials/job_form.html` — capability presets in the
  preset list; KL reference dropdown revealed when a KL preset is
  checked.
- `web/templates/benchmarks.html` — small JS for the reveal + matrix
  count awareness.
- Tests across `internal/benchmark` and `internal/api`.

## Steps

1. Presets (labels follow the existing duration-hint style; all counts
   are at the fixed evaluation context of 512 — phase 01 — which is
   what makes "all chunks" a defined quantity):
   - `perplexity-quick` (100 chunks, ~2-5 min/model) / `perplexity-full`
     (all chunks — wikitext-2 test at ctx 512 is ~650 chunks)
   - `kl-divergence-quick` (100 chunks) / `kl-divergence-full`
     (all chunks; the base file for a full run is tens of GiB for
     large-vocab models — the phase 02 disk guard is the gate)
   - `hellaswag-quick` (400 tasks) / `hellaswag-full` (all ~10K)
   - `winogrande-quick` (400 tasks) / `winogrande-full`
   Descriptions state what the score means and that the inference server
   is offline while the cell runs.
2. `JobEnv` additions (implemented in `jobs_env.go` with the api server's
   existing machinery; every method documented on the interface):
   ```go
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
   // way ApplyEphemeralConfig does (ValidateBatchSizes,
   // jobs_env.go:204-210 — a -ub > -b sweep fails with the named
   // message, not the loader's raw error), resolves GPU assignment
   // (resolveGPUAssignment), fills evaluate.SnapshotSubset (plain
   // fields + PlacementFlags via models.GPUPlacementFlags with the
   // build's backend), and returns evaluate.MapConfigFlags's output.
   // The single place config becomes CLI flags — phase 01 step 2.
   EvalFlags(modelID string, snap ConfigSnapshot, buildID string) ([]string, error)
   // EnsureKLBase returns the cached logits path for (reference,
   // dataset, chunks, ctx), generating it first when absent. The
   // progress callback receives plain-language status lines; the
   // runner wires it to the run's ProgressDetail through the store —
   // the same transport performance runs already use (runner.go:93-95)
   // — which is what makes generation the overview's "visible job
   // step". Progress SOURCE: the generation's own output is
   // newline-less fragments (perplexity.cpp:633), so the
   // implementation polls the growing .kld.partial file's size (the
   // phase 02 interruption-safe temp path — generation never writes
   // the final name directly) on a ticker and reports "generating
   // reference logits: 2.1 of ~4.6 GiB" against the phase 02
   // estimate — no output parsing (phase 01 step 3). On failure or
   // cancel the partial is deleted, never cached. Generation loads the reference model with the
   // REFERENCE's own saved ModelConfig mapped through EvalFlags (no
   // sweep overrides) — GPU layers and placement apply, so a large
   // reference is not silently evaluated on CPU. Enforces the phase
   // 02 disk guard before generating.
   EnsureKLBase(ctx context.Context, ref ModelInfo, chunks int, buildID string, progress func(string)) (string, error)
   // RunEval executes one evaluation and returns parsed scores.
   RunEval(ctx context.Context, spec evaluate.Spec) (evaluate.Result, error)
   ```
   Router restore is NOT free-riding on today's cleanup — as written,
   `ClearEphemeralConfig` (`internal/api/jobs_env.go:232-269`) would do
   nothing for a capability job: it early-returns when the job never
   took router ownership, and its "user stopped the router mid-job,
   leave it stopped" check reads `IsRunning()`, which is false after
   `StopRouterForEval` for the wrong reason. Concretely:
   `StopRouterForEval` sets the same ownership flag
   `EnsureBuildActive`/`ApplyEphemeralConfig` set AND a new
   `stoppedForEval bool` on `jobEnv`; `ClearEphemeralConfig` is amended
   so that when `stoppedForEval` is set it restarts the router
   (restoring the pre-job state) instead of treating not-running as a
   user stop, then clears the flag. The user-stop semantics for
   performance-only jobs are unchanged. The `JobEnv` doc comment on
   `ClearEphemeralConfig` (`job_runner.go:53-55`) is updated to state
   the eval-stop case. This is what makes the overview's "router is
   back up afterward, including on failure and cancel" criterion true;
   the test in step 5 asserts it.
   `internal/benchmark` importing `internal/evaluate` is cycle-free
   (evaluate imports models, huggingface, and stdlib — none of which
   import benchmark).
3. Cell expansion — the collapse rule: capability cells run only what
   `MapConfigFlags` lets through, so when a job's sweep axes vary
   parameters that never reach the evaluation, capability presets
   generate ONE cell per distinct EVAL-REACHING configuration, not one
   per swept value — duplicate expensive cells reporting different
   labels for identical runs is exactly the mislabeled-results failure
   this codebase guards against elsewhere.
   The discriminator is NOT `RestartsRouter`: three load-time axes
   restart the router yet never reach the eval — `context_size`
   (`sweep.go:340-341`; replaced by the fixed `-c 512`), `spec_type`
   (`sweep.go:366`; speculative decoding excluded), and any future
   axis of that kind — so keying on `RestartsRouter` would fan
   capability cells out into byte-identical invocations under
   different labels. Instead `SweepField` gains an explicit
   `AffectsEval bool` in the registry — true exactly for fields that
   reach the eval command line, whether through the phase 01 direct
   mapping (kv cache quant, batch/ubatch, flash attention, gpu layers,
   threads, direct_io) or through the placement resolution
   (`gpu_assign` AND `tensor_split`, `sweep.go:356` — both become
   `--device`/`--tensor-split` via `PlacementFlags`). That is the
   complete 16-field registry classified: the remaining seven —
   temperature, top_p, min_p, repeat_penalty, top_k (sampling),
   context_size, spec_type — are `AffectsEval == false` and collapse
   for capability presets. The registry stays the single source; a new
   mapped/excluded flag in phase 01 has one place to be reflected. The collapse happens
   at matrix expansion; the form's matrix count reflects it (step 7).
   Relationship to the existing sampling guard: `ValidateSamplingSupport`
   (`sweep.go:748-793`) already REFUSES benchy presets combined with
   sampling settings, for the same mislabeled-results reason. The two
   remedies coexist deliberately: benchy cannot apply sampling at all
   (refusal is the only honest option), while capability cells have a
   well-defined meaning under collapse. `ValidateSamplingSupport`
   itself is UNCHANGED — its signature sees only presets, overrides,
   and sweeps (`sweep.go:756`), while step 7's capability refusal also
   needs the job's model and build counts, so that refusal lives in
   `bench_jobs.go`'s request validation (which has the whole request)
   next to the KL checks. A job mixing benchy, capability, and a
   sampling axis still gets the benchy refusal first, unchanged.
4. Capability cell execution (the branch in the cell loop — entered
   when `preset.EffectiveSource() == "capability"`, before the
   performance path is ever consulted). `CheckBuildRunnable` and
   `EnsureBuildActive` currently share one `prevBuildID`-guarded block
   (`job_runner.go:320-328`); that block is SPLIT so a capability cell
   runs `CheckBuildRunnable` (build must exist and be runnable) but
   NEVER `EnsureBuildActive` — starting llama-server just to stop it
   again would add a full router restart + health wait between every
   pair of consecutive capability cells. The branch creates and saves
   the cell's `BenchmarkRun` (status running) FIRST — deliberately
   unlike the performance path, which creates its run last
   (`job_runner.go:367-389`) — because sub-step c's `EnsureKLBase`
   progress lands on the run's `ProgressDetail`, which must exist and
   be stored before generation starts. The performance path finalizes
   its run inside `Runner.Run` (`runner.go:84-140`), which this branch
   bypasses — so the branch owns the run's WHOLE lifecycle explicitly,
   on every exit: any sub-step failure (a's rebuild message, b's
   dataset error, c's resolve/generation error, f's eval error) sets
   `StatusFailed` + `Error` and saves; success (f) sets
   `StatusCompleted` and saves; the 4c reference-model skip DELETES
   the pre-created run and clears `cell.BenchmarkRunID` (restoring
   "skipped cell has no run"). No exit may leave a stored
   `StatusRunning` run — the run list renders those as live and retry
   would orphan them. Then:
   a. `EvalBinary(buildID)` — absent → cell fails with the rebuild
      message.
   b. `EnsureEvalData` for the mode.
   c. KL only: `ResolveKLReference` (job's `KLReference` as override).
      If the resolved reference IS the cell's own model (the largest
      quant's own cell in an all-quants job — the flagship flow), the
      cell is SKIPPED, not failed and not run: an expensive eval whose
      answer is known is waste, and a refusal would break the primary
      use case. Storage for the skip: the cell is marked
      `CellStatusCompleted` with an additive `SkipReason string` on
      `JobCell` (`job.go:101-113`) and produces no `BenchmarkRun`.
      Deliberately NOT `CellStatusSkipped` — that status already means
      "job canceled before this cell ran" (`job_runner.go:235-237`)
      and such cells are re-run on retry, which is correct for them
      and wrong for a reference cell; and NOT a reuse of `Error`
      (retry and error-styling collisions). Completed-with-reason
      means the existing loop short-circuit (`job_runner.go:229`),
      retry selection, and the Done/Total progress counting
      (`bench_jobs.go:38-46`, `:485`) all behave correctly with ZERO
      changes — a 4-quant KL job shows 4/4 done, not 3/4 forever.
      The note ("this is the reference model — its difference from
      itself is zero") renders where cell status renders (phase 04).
      Otherwise
      `EnsureKLBase` — generation runs with the reference model loaded
      (router already stopped, step d ordering below); the resolved
      reference identity is recorded on the run's `EvalScores`.
   d. `StopRouterForEval` — called before b/c's GPU work: concretely,
      first thing after a; dataset downloads don't need it but base
      generation and the eval do, and stopping early keeps the ordering
      simple and the window deterministic.
   e. Build `evaluate.Spec`: binary, model path (`ResolveModel` →
      `ModelInfo.FilePath`, the field added in this phase),
      mode/dataset/limits from the preset, and `Flags` from
      `EvalFlags(modelID, cellSnapshot, buildID)` — the api-side
      method (step 2) that performs the same merge/resolve the
      performance path uses and runs everything through
      `evaluate.MapConfigFlags`. The cell's `ConfigSnapshot` passed in
      is the same snapshot recorded on the run, so the recorded config
      and the flags that ran share one source; this is what makes a
      swept `gpu_assign` measurably reach `llama-perplexity` — the
      overview's config-fidelity criterion.
   f. `RunEval`; copy `Result` into `run.Eval`; timings fields stay zero
      (phase 04 renders them as absent).
   Failure text includes the engine's tail-of-output.
5. Router lifecycle: a capability cell following a performance cell
   stops the router. The reverse direction does NOT come free: the
   cell loop skips `EnsureBuildActive` when the build is unchanged
   (`job_runner.go:320-328`) and skips `ApplyEphemeralConfig` when the
   wanted overrides match `lastApplied` (`job_runner.go:354-365`), so
   a performance cell after a capability cell would run its warmup
   against a stopped router and fail. Fix in the cell loop: after any
   capability cell runs (equivalently, after `StopRouterForEval`),
   invalidate the loop's `prevBuildID` and `lastApplied` caches so the
   NEXT PERFORMANCE cell unconditionally re-enters
   `EnsureBuildActive`/`ApplyEphemeralConfig`, which start the router
   back up through the existing blocking calls. The invalidation only
   affects performance cells — capability cells never call
   `EnsureBuildActive` (step 4's block split), so consecutive
   capability cells run back-to-back with the router simply staying
   stopped, no start/stop churn. Job-end cleanup
   (`ClearEphemeralConfig` with the `stoppedForEval` amendment of step
   2, run on success, failure, and cancel) restores the user's router
   state — assert both the mid-job revival and the job-end restore in
   tests.
6. Single-run quick-benchmark form: `bench.go:467` feeds
   `benchmark.Presets()` into `benchmark_form.html`'s dropdown. This
   path does NOT bypass the cell loop — `handleStartBenchmark`
   (`bench.go:240-292`) builds a 1-cell job and submits it to the same
   JobQueue, so a capability preset selected there would actually run
   correctly. The form still FILTERS capability presets out as a
   product decision: it has no KL-reference selector, its "router is
   not running — start the server first" gate (`bench.go:260-262`) is
   meaningless for capability cells (they stop the router), and its
   single-run framing is built around timing results. The filter is by
   `EffectiveSource()`, next to the existing benchy handling; the
   JSON API needs no extra guard (a capability preset POSTed directly
   runs fine through the cell loop).
7. Job form: capability presets render in the existing preset checkbox list
   (same fieldset — the Source only matters to the runner and phase
   04's columns). Checking a KL preset reveals a "KL reference model"
   dropdown (options: installed models grouped by repo, default
   "automatic — largest quant of each model's repo"), posted as
   `kl_reference`. Matrix count JS applies the collapse rule to
   capability presets. Capability refusal (in `bench_jobs.go`'s
   request validation, per step 3 — it needs model/build counts):
   refuse ONLY jobs where ALL FOUR hold — capability-only presets, AT
   LEAST ONE sweep axis configured, no configured axis has
   `AffectsEval == true`, and models × builds == 1 — i.e. the
   configured sweep is the job's only variation and none of it reaches
   the evaluation, so the whole job collapses to one cell per preset
   ("the swept parameters do not affect capability evaluations —
   nothing would vary between cells"). Precisely NOT refused:
   sweep-free capability jobs (the overview's flagship four-quant job
   — zero sweeps is the normal case), and multi-model or multi-build
   jobs with inert sweeps (after collapse the cells still vary by
   model/build — meaningful; the matrix count shows the collapse).
   KL-specific
   validation in `bench_jobs.go`: a KL preset whose every cell would be
   skipped under step 4c (each selected model resolves to itself as
   reference — e.g. a single-model job whose model is the only
   installed quant of its repo, or an explicit reference on a
   single-model job naming that model) → refused with a message naming
   what to add (a second quant or a different reference). A reference
   that is merely AMONG the job's models is fine — that is the primary
   all-quants flow; only its own cell skips.

## Build gate

`go build ./... && go vet ./internal/... && go test ./internal/...`

## Test plan

- Runner tests with a fake `JobEnv` (the existing pattern in
  `job_runner_test.go`): capability cell happy path lands scores on the
  run; missing binary fails with the rebuild message; KL resolves
  automatic reference (largest quant fake) and records it; the
  reference model's own cell is skipped with the note, other cells run;
  base generated once then cache hit on the second cell; cancel
  mid-generation deletes the `.kld.partial` and leaves no cache entry,
  and a subsequent `EnsureKLBase` regenerates (the test phase 02's
  interruption-safety rule delegates here); router stop
  called before eval and cleanup restores after success, failure, AND
  cancel (the `stoppedForEval` path — this is the test the old plan
  could not have passed); `stoppedForEval` NOT set when the router was
  already stopped before the job (cleanup leaves it stopped); the
  mixed-order sequence perf → capability → perf on ONE build with no
  overrides — the middle cell stops the router and the third cell must
  see `EnsureBuildActive`/`ApplyEphemeralConfig` re-invoked (cache
  invalidation per step 5), not a dead router.
- Expansion tests: sampling axis + capability preset collapses to one
  cell while the same job's performance preset still fans out;
  context_size axis (RestartsRouter true, `AffectsEval` false) ALSO
  collapses for capability presets — the case a RestartsRouter-keyed
  collapse would miss; eval-reaching axis (kv_cache_quant) fans
  capability cells out normally.
- Validation tests: all-cells-would-skip KL refusal; the four-condition
  capability refusal fires for a single-model, single-build,
  capability-only job whose only sweep is a sampling axis; it does NOT
  fire for the sweep-free four-quant flagship job nor for a
  multi-model job with the same inert sweep; existing benchy refusal
  unchanged; batch/ubatch mismatch on a capability cell refused with
  the named message (the `ValidateBatchSizes` call in `EvalFlags`).
- Run-lifecycle tests: failed capability cell's run is `StatusFailed`
  with the error (never left `StatusRunning`); the reference-skip
  path deletes its pre-created run and clears `cell.BenchmarkRunID`.
- Expansion test addition: `tensor_split` axis fans capability cells
  out (`AffectsEval` true via placement).
- Form render test (existing job-form render pattern): capability
  presets listed, KL dropdown present and hidden by default.
- Manual on the dev box (vulkan build, TWO small models so the
  perf-after-capability ordering actually occurs mid-job): run
  `hellaswag-quick` + `internal-quick` in one job; watch the router
  stop and come back between cells and at job end; inspect the run
  JSON for scores.

## Commit

`feat(benchmarks): capability evaluation presets and job runner integration`

## Rollback

Revert the branch commits; `Eval` on stored runs is additive and ignored
by older code. No persisted format changes beyond additive fields.
