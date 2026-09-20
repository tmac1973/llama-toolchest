# Phase 12 — Autotune runner: stages, winners, resume and saved profiles

**Depends on:** 02, 03, 10, 11 · **Enables:** 13

## Goal
The engine that runs an autotune from start to end:
1. submit each stage's job;
2. wait for it and score the results;
3. carry the finalists forward;
4. run the confirmation stage;
5. save up to three profiles with their measurements.

It can be cancelled and resumed from the last finished stage, and it is fully
testable with the fake job environment.

## Files touched
- `internal/autotune/record.go`: fill `SecondsPerCell`, `Failed` and
  `Results` (the fields defined in Phase 11).
- `internal/autotune/runner.go` (new): `Runner` with `Start`, `Resume`,
  `Cancel` and `run`.
- `internal/autotune/runner_test.go` (new), using `benchmark`'s fake
  environment. Export a small `benchmark/benchtest` package with `FakeEnv`
  and `NewFakeRouter`, moved from `job_runner_test.go` so both packages share
  them.
- `internal/api/server.go`: build the `autotune.Runner` with the job queue,
  store, registry and record store. On startup, leave interrupted records as
  they are; the user resumes them.

## Steps
1. **`Start(modelID, baseProfile string, uc UseCase) (*Autotune, error)`** refuses:
   - when the job queue is busy ("A benchmark is running");
   - when another autotune is running;
   - when the base profile does not exist;
   - when the profile fails `models.ValidateProfileConfig` (Phase 03), e.g.
     missing draft files.

   It then creates the record with the active `BuildID`, saves it, and runs
   `run` in a goroutine.
2. **`run(rec)`**, for each stage in order "batch", "spec", "spec-params",
   "confirm", skipping stages whose `Status` is "done":
   1. Plan the cells from the finalists recorded so far (Phase 11).
   2. If the stage has only the baseline cell (e.g. no draft method and a
      spec-params stage with nothing to tune), mark it done with the
      earlier finalists and continue.
   3. Build the job and `jobs.Submit` it. Save `JobID` and set the status to
      "running".
   4. `jobs.Wait`.
   5. Load the runs: `store.RunsForJob(jobID)`, grouped by cell values
      (canonical key) into `map[preset]*BenchmarkRun`.
   6. Score each candidate for each goal with `ScoreRuns`. Candidates whose
      runs failed are left out and listed in the stage record with the
      failure text. (A config that crashes the server, e.g. out of memory,
      is simply not a winner.)
   7. Finalists: `Pick` for each goal. De-duplicate by values, keeping the
      goals each finalist won.
   8. Save the record.
3. **Resume after a partly run stage.**
   - If the stage's job exists but is not complete, call
     `store.UpdateJobDefinition` with the same cells and `jobs.Submit` it
     again. The existing identity logic keeps the cells that completed.
   - A missing job means the stage is planned again from scratch.
4. **Measured timings.**
   - After stage 1, measure the mean wall time per cell from the runs'
     start and end times.
   - Store it on the record as `SecondsPerCell`.
   - Phase 13 uses it for "about N minutes left".
5. **Cancel.**
   - `Cancel(id)` calls the existing job cancel on the running stage job and
     sets the status to "cancelled". Stages already finished stay.
   - `Resume` works for both "cancelled" and "interrupted".
6. **Outcome.** Stage "confirm" re-measures the finalists and the baseline
   side by side. Winners are picked **only from the confirm stage's runs**,
   so every compared number comes from the same period. For each goal:
   - If the winner is the baseline, or does not `Beat` the baseline, the
     outcome is "Your starting profile is already the fastest for GOAL", and
     no profile is saved.
   - Otherwise build the profile config:
     - `benchmark.ConfigForValues(baseProfile.Config, winner.Values, resolve)`,
       where `resolve` is a closure defined in `runner.go`:
       `func(id string) (string, error) { m, err := r.reg.Get(id); if err != nil { return "", err }; return m.FilePath, nil }`
       (Phase 10). This is the same override code the job runner uses, so
       the saved profile launches exactly what was measured;
     - `NormalizeSpec`.
   - The profile name is "Autotune – fastest generation", "Autotune –
     fastest prompt" or "Autotune – fastest response".
   - When one winner covers several goals, it is saved once, with the names
     joined: "Autotune – fastest generation and response".
   - Earlier profiles with those names are replaced.
   - `Source "autotune"`. `Measured` holds the workload, PP/TG/response
     values, the baseline values, the goals and `AutotuneID`.
   - `Notes` hold one plain-language line per changed setting, from the
     candidate `Label` (e.g. "Adds MTP draft layers with an ngram-mod assist:
     generation 38 → 61 tokens/s on the code-editing test.").
7. After saving, set the status to "done". The live config is not touched;
   the user restores a profile from the results screen or the profile bar.
8. **Cleanup.** After the last stage, and on cancel, call
   `ClearEphemeralConfig` through the job queue's existing end-of-job path,
   so the router returns to the live configs.
9. **Logs.** `slog` lines at each stage change, with the record ID.

## Build gate
`go build ./... && go vet ./... && go test ./...`

## Test plan
Runner tests with `benchtest.FakeEnv` and a fake router whose reported
timings depend on the cell's config (e.g. higher TG when `spec_type` contains
`draft-mtp`, higher again with an assist; higher PP at ubatch 1024):
- A full run for "code":
  - four stages run in order;
  - the winners match the scripted speeds;
  - profiles are saved with the expected names and `Measured` values;
  - a goal whose best candidate is within noise of the baseline saves no
    profile.
- One winner shared by two goals produces one profile with a combined name.
- A cell that fails (the fake returns 500) is left out and recorded.
- Cancel during stage 2, then Resume: stage 1 is not re-run and stage 2 keeps
  its completed cells.
- A server restart simulated by reloading the record store: the record is
  "interrupted"; Resume finishes it.
- Start is refused while another job runs, or with a missing base profile.
- A saved profile's config, applied and turned into a preset INI, contains
  the measured `spec-type` and `ubatch-size` values.

## Commit
`feat(autotune): run staged tuning and save the fastest profiles`

## Rollback
Revert the commit. The record store and planner from Phase 11 remain unused.
Profiles already saved by autotune stay as ordinary profiles.

## As implemented

- **Resume carries cells, not definitions.** `UpdateJobDefinition` rebuilds
  a job's cells as a full product of its sweep axes, which is not autotune's
  explicit cell list, so the runner copies the completed cells from the
  previous attempt onto the freshly planned job (`carryCompleted`). The job
  runner already skips completed cells.
- **Failures are read from the job's cells**, not only its runs: a config
  this machine refuses never reaches the router, so it has no run at all.
  That is exactly the case worth reporting.
- **A stage with nothing to measure** carries the previous stage's finalists
  forward, so a model with no speculative decoding still reaches the
  confirming stage.
- **Only the confirming stage decides.** Its numbers were all measured
  minutes apart under the same conditions, rather than an hour apart across
  stages.
- **Tests** use a fake job environment and a fake router whose speeds depend
  on the config each cell applied — a prompt batch of 1024 reads fastest,
  MTP writes faster, an n-gram assist on top faster again — so a whole run
  is exercised end to end in about a second.
- **A bug the tests found:** the settings stage offered a draft length to an
  n-gram-only finalist, which llama.cpp has no setting for; every cell of
  that stage failed. It is now only offered where there is a draft method.
