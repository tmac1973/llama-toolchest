# Phase 11 — Autotune records, stage planner and job plumbing

**Depends on:** 01, 02, 04, 05, 06, 10 · **Enables:** 12

## Goal
Everything the autotune runner needs except the running itself:
- a stored autotune record that survives restarts;
- a planner that turns a starting profile, the hardware and earlier stage
  winners into the explicit cells of the next stage;
- two additions to the job queue: running a job against a profile's config
  instead of the live config, and waiting for a job to finish without
  polling.

## Files touched
- `internal/autotune/record.go` (new): the `Autotune` type and its store at
  `<data_dir>/config/autotune.json`, with the same schema gate as Phase 01.
- `internal/autotune/plan.go` (new): the stage planner.
- `internal/autotune/plan_test.go`, `internal/autotune/record_test.go` (new).
- `internal/benchmark/job.go`: `BenchmarkJob` gains
  - `BaseProfile *BaseProfile` (`json:"base_profile,omitempty"`), where
    `type BaseProfile struct{ Name string; Config models.ModelConfig }` holds
    the starting profile's **full** config;
  - `AutotuneID string` (`json:"autotune_id,omitempty"`);
  - `AutotuneStage string` (`json:"autotune_stage,omitempty"`).
- `internal/benchmark/job_runner.go`:
  - when `job.BaseProfile` is set, `runCell` uses
    `SnapshotFromConfig(job.BaseProfile.Config, job.BaseProfile.Name, false)`
    in place of `modelInfo.Config` (line 447; capability cells at line 561 are
    left as they are, because autotune never creates them),
    and it takes the per-request sampling values from the profile's config
    instead of the live config;
  - `JobEnv.ApplyEphemeralConfig` gains a `base *models.ModelConfig`
    argument. `nil` keeps today's behavior (the model's saved config); runCell
    passes `&job.BaseProfile.Config` when it is set;
  - `(*JobQueue).Wait(ctx, jobID) (BenchmarkJob, error)`.
- `internal/benchmark/benchmark.go`: `ConfigSnapshot` gains `MtpPath`, so a
  profile's separate MTP head travels with the base config;
  `SnapshotFromConfig` fills it.
- `internal/benchmark/snapshot.go`: `ApplySnapshotToConfig` (moved there in
  Phase 10) also applies `MtpPath`.
- `internal/api/jobs_env.go`: `ApplyEphemeralConfig` (line 204) uses the
  passed `base` in place of `registry.GetConfig(id)` when it is not nil.
  `EvalFlags` (quality-evaluation cells) is unchanged, because autotune never
  runs capability cells; `BuildStageJob` uses only the `autotune-*` timing
  presets. This covers the base for
  `ApplySnapshotToConfig` and the `GPUAssign` comparison in
  `resolveGPUAssignment`, so fields the snapshot does not carry (sampling,
  `Jinja`, `ReasoningOverride`, `MmprojPath`) come from the profile, not
  the live config.
- Tests: `internal/benchmark/job_runner_test.go` (additions for `BaseProfile`
  and `Wait`).

## Steps
1. **`Wait`.**
   - `runningJob.done` already closes at the end of `run`. Keep a
     `map[jobID]chan struct{}` of done channels on the queue, created in
     `Submit`.
   - `Wait` returns immediately if the stored job is in a terminal state.
     Otherwise it waits on the channel or `ctx.Done()`, then returns
     `store.GetJob(id)`.
2. **`BaseProfile`.**
   - When it is set, a cell's snapshot is
     `ApplyOverrides(SnapshotFromConfig(bp.Config, bp.Name, false), cellOv)`,
     and the launched config is `ApplySnapshotToConfig(copy of bp.Config, snapshot)`.
     What is measured is exactly what Phase 12's `ConfigForValues` later
     saves.
   - The ephemeral preset INI is written from that snapshot, so the live
     config and its profile label are never touched.
   - The run's `ProfileName` comes from `job.BaseProfile.Name`, which
     `BuildStageJob` sets (step 5). `ProfileEdited` is true for every cell
     with sweep values, as Phase 04 defines it, and false for the baseline
     cell.
3. **The record.**
   ```go
   type Autotune struct {
       ID, ModelID, BaseProfile string
       UseCase UseCase
       BuildID string
       Status string            // "planned" | "running" | "interrupted" | "cancelled" | "done" | "failed"
       Stages []StageRecord     // in order
       Skipped []string         // plain-language: "EAGLE3: no EAGLE3 head installed for this model"
       Results map[Goal]Outcome // filled in Phase 12
       SecondsPerCell float64   // measured after stage 1, filled in Phase 12
       CreatedAt, UpdatedAt time.Time
       Error string
   }
   type StageRecord struct {
       Key string        // "batch" | "spec" | "spec-params" | "confirm"
       Title string      // "Batch sizes and attention", …
       JobID string
       Status string
       Failed []FailedCell   // candidates whose runs failed
       Finalists []Candidate // Phase 10 type; carried forward, one per goal, de-duplicated
   }
   type FailedCell struct{ Label, Error string }
   type Outcome struct {
       Goal        Goal
       Saved       bool      // false when the starting profile was already fastest
       ProfileName string    // set when Saved
       Winner      Candidate
       Baseline    Candidate
       Message     string    // plain-language summary line
   }
   ```
   - Store methods: `Get(id)`, `Save(rec)`, `List()`, and
     `LatestForModel(modelID) (*Autotune, bool)`, which returns the most
     recent record by `CreatedAt`.
   - The store uses the Phase 01 pattern: versioned, gated, atomic write,
     saved after every change.
   - On load, any record with status "running" becomes "interrupted".
4. **Planner.**
   `PlanStage(key string, in PlanInput) (cells []map[string]string, axes []benchmark.SweepAxis, skipped []string)`.
   `PlanInput` holds the model, the base config, `Hardware` (Phase 06), the
   installed draft candidates, and the finalists from earlier stages.
   - **Stage "batch":**
     - ubatch ∈ {256, 512, 1024, 2048} with `batch_size=2048`, plus
       (ubatch 4096, batch 4096), leaving out any ubatch larger than the
       context;
     - × flash attention {on, off}, where `ValidateFlashAttention` allows
       both;
     - × `split_mode` {layer, tensor} only when more than one GPU is
       assigned (`DeviceCountForConfig > 1`);
     - × threads {cores/4, cores/2, 3·cores/4} only when `CPUMoE > 0` or
       `GPULayers < NLayers`;
     - plus one cell with no values: the unchanged base profile, which is the
       baseline.
   - **Stage "spec"**, run once for each distinct stage-1 finalist (its
     values are merged into every cell). The draft options are:
     - `draft-mtp` when `NextNLayers > 0` or the base has an `MtpPath`;
     - `draft` with each installed candidate from
       `FindDraftCandidates(id, "draft")`, at most two (smallest first);
     - `draft-eagle3`, `draft-dflash` and `draft-dspark` with each installed
       candidate for that mode.

     Assist options are all five `AssistModes()`. The cells are:
     - `none`;
     - each draft option alone;
     - each assist alone;
     - every draft × assist pair;

     each with default parameters from `SpecDraftParams`/`SpecAssistParams`.
     Each draft mode with no installed file is added to `skipped` with its
     reason.
   - **Stage "spec-params"**, for each distinct stage-2 finalist:
     - if it has a draft method, `draft_max` ∈ {default/2, default,
       default·1.5, default·2} (rounded, min 1; for modes with a blank
       default, use llama.cpp's 3 as the default);
     - if it has an assist, its first parameter (`assist_n_max` for
       ngram-mod, `assist_size_n` for the others) ∈ {default/2, default,
       default·2};
     - ngram-cache has no parameters, so only draft values apply;
     - the product of the two lists, with at most 12 cells for each finalist.
   - **Stage "confirm":**
     - the top 2 stage-1 candidates by the response score, × the distinct
       stage-3 finalists across all goals (at most 3);
     - plus the baseline cell;
     - at most 7 cells.
   - Every cell's values are canonical sweep strings (spec values through
     `benchmark.EncodeSpecValue`), so they display correctly in the existing job detail
     view.
5. **The stage job.** `BuildStageJob(rec, stageKey, cells, axes)` returns a
   `BenchmarkJob` with:
   - `ModelIDs = [rec.ModelID]` and `BuildIDs = [rec.BuildID]`;
   - `Presets` from `Workloads[rec.UseCase]`;
   - `BaseProfile` = `&BaseProfile{Name: rec.BaseProfile, Config: profile.Config}`;
   - `Sweeps = axes`, for display;
   - `Cells`: one `JobCell` per (cell values × preset), with `SweepValues`
     set;
   - `AutotuneID`/`AutotuneStage` set;
   - the name "Autotune: MODEL — Stage N of 4: TITLE".
6. **Time estimate.**
   `EstimateMinutes(cellCount int, sizeGiB float64) int` =
   cells × (load seconds + benchmark seconds). Load seconds is
   `max(20, sizeGiB × 4)`. Benchmark seconds is 45 per preset. The
   constants sit in one place with a comment, and Phase 12 replaces the
   estimate with measured timings once stage 1 has run.

## Build gate
`go build ./... && go vet ./... && go test ./...`

## Test plan
- `Wait`: returns the finished job and returns on context cancel (with
  `fakeEnv`).
- `BaseProfile`: a job whose profile has `ubatch 1024` and `temperature 0.6`
  while the live config has `ubatch 512` and `temperature 1.0` runs its
  cell at 1024, sends temperature 0.6 in the completion request, and writes
  an ephemeral INI built from the profile. The live config is unchanged
  afterwards.
- Planner table tests:
  - single GPU, full offload: 10 batch cells + baseline, no split or threads;
  - two GPUs: split mode added;
  - an MoE model with `CPUMoE`: threads added;
  - a model with built-in MTP and no draft files: spec cells = none + MTP +
    5 assists + 5 MTP pairs = 12, with EAGLE3/DFlash/DSpark/draft listed in
    `skipped`;
  - the spec-params cap is respected;
  - the confirm stage has at most 7 cells.
- Record store:
  - `LatestForModel` returns the newest of two records;
  - a "running" record reloads as "interrupted";
  - a newer schema version sets read-only.
- `BuildStageJob` produces cells × presets for Mixed.

## Commit
`feat(autotune): plan autotune stages and run jobs against a profile`

## Rollback
Revert the commit. `base_profile` and the `autotune_*` job fields are ignored
by older code. `autotune.json` is left on disk, unused.
