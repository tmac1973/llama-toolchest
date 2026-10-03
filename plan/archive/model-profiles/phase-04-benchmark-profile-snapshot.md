# Phase 04 — Benchmarks record the profile they ran

**Depends on:** 02 · **Enables:** 12 (autotune compares runs by profile and
settings); completes PR 1 (saved profiles)

## Goal
Every benchmark run records which profile the model's config came from and
whether it had been edited, so runs can be compared by profile. The snapshot
also gains the settings that later phases tune but that are not recorded yet:
split mode, main GPU, parallel slots, and the draft model's resource
overrides.

## Files touched
- `internal/benchmark/benchmark.go`:
  - `ConfigSnapshot` (line 142) gains `ProfileName`, `ProfileEdited`,
    `SplitMode`, `MainGPU`, `Parallel`, `DraftCtxSize`, `DraftGPULayers`,
    `DraftKVCacheQuant`;
  - bump `schemaVersion` to 5 (a new field only, so no migration).
- `internal/api/jobs_env.go`: fill the new fields where the snapshot is built
  from the model config. Merge the two snapshot builders (here and in
  `internal/api/bench.go`) into one exported `SnapshotFromConfig` in
  `internal/benchmark/snapshot.go` (new).
- `internal/benchmark/job_runner.go`: in `runCell`, a cell with sweep values
  or overrides records `ProfileName` with `ProfileEdited=true`. Only a cell
  with no overrides ran the profile exactly.
- `web/templates/partials/benchmark_compare.html` and the CSV export in
  `internal/api/bench.go`: a "Profile" column, shown only when the compared
  runs have different profile names.
- Tests: `internal/benchmark/snapshot_test.go` (new) and an addition to the
  compare render test.

## Steps
1. Add the fields with `omitempty` JSON tags.
2. Write `func SnapshotFromConfig(cfg models.ModelConfig, profile string, edited bool) ConfigSnapshot`
   (it takes a value; callers holding a pointer pass `*cfg`)
   and replace both existing literal builders with it.
3. The API layer passes `registry.ActiveProfileState(id)`.
4. In `runCell`, after `applyOverrides`: if `len(cell.SweepValues) > 0` or
   the job has overrides, set `ProfileEdited = true`.
5. **Compare view:** add the Profile column with the rule "show only if any
   value differs", which the other conditional columns already follow. Label
   it "Profile", with the tooltip "The saved profile the model's settings came
   from when this run started. 'edited' means the settings no longer matched
   the profile exactly."
6. CSV export gains `profile` and `profile_edited` columns.

## Build gate
`go build ./... && go vet ./... && go test ./...`

## Test plan
- Unit test: `SnapshotFromConfig` copies every field of `ConfigSnapshot`
  that has a matching `ModelConfig` field (a reflection test by field name,
  so a future field cannot be forgotten).
- Job runner test with `fakeEnv`: a sweep cell records `ProfileEdited=true`;
  a plain cell of a model whose active profile is unedited records `false`.
- Render test: the compare view shows the Profile column only when names
  differ.
- Manual: restore a profile, run a quick benchmark, and check that the run
  detail shows the profile name.

## Commit
`feat(benchmark): record the config profile and the settings it varies`

## Rollback
Revert the commit. Runs recorded meanwhile carry extra JSON fields that the
older code ignores. `benchmarks.json` will have `version: 5`, which a build
that has Phase 01 but not this phase refuses to write. To go back that far,
set the version to 4 by hand.

## As implemented

- **One builder.** There was only one literal snapshot builder
  (`modelInfoBundle` in `internal/api/jobs_env.go`); the second one the
  plan expected in `bench.go` does not exist. It is replaced by
  `benchmark.SnapshotFromConfig` in `internal/benchmark/snapshot.go`.
- **How "edited" is decided.** `markProfileEdited` compares the cell's final
  snapshot with the model's own, rather than asking whether the job had
  overrides. It therefore also catches the f16 KV cache that capability
  (evaluation) cells default to. `ConfigSnapshot` must stay comparable with
  `==` (all scalar fields); the runner's reuse check already depends on that.
- **Where the profile shows.**
  - The compare table has a "Profile" column that uses the existing
    varies/common mechanism: it is folded into the "same for every run" line
    when all runs share it.
  - Chart labels gain a "profile" dimension.
  - The run detail page shows "Profile: name (edited)".
- **CSV:** both scopes gain `profile` and `profile_edited`.
