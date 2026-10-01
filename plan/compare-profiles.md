# Compare saved profiles in a batch job

## Problem
A model can have several saved profiles ("Fast chat", "Long context 128K",
"MTP + ngram"). Today a user can benchmark only the model's current settings,
so comparing two profiles means restoring one, running a job, restoring the
other and running a second job, then comparing runs from different jobs.

## Goal
In the batch job form, the user can pick one or more starting points for each
model: its current settings, and any of its saved profiles. Each starting
point becomes its own set of cells, and the results are labelled by profile.

Results are **not** written back to the profile. This is only for comparing.

## Decisions
- **Picker:** under each checked model, a list of checkboxes. The first is
  "Current settings", checked by default. It is followed by every saved
  profile for that model. Works for one model or several.
- **Parameters section:** while any saved profile is checked, the parameter
  section (fixed values and sweeps) is disabled. Each profile runs exactly as
  saved, so the results always describe the profile itself. With only
  "Current settings" checked, the form works as it does today.
- **Builds:** profile cells run on the builds picked in the form, as other
  cells do. A profile saved on a build that is not among the picked builds
  shows a warning, the same as when the profile is restored.
- **Profile changes after the job is created:** the job stores a copy of each
  chosen profile's settings when it is created. Runs and retries measure that
  copy. Changing or deleting the profile later does not change the job.
- **Cells are pairs, not every combination:** profiles belong to one model,
  so cells are (model, starting point) pairs × builds × presets.

## How it fits the existing code
Most of the machinery exists. `BenchmarkJob.BaseProfile`
(`internal/benchmark/job.go:67`) already makes a whole job measure from a
copy of one profile. Autotune and Autoconfigure use it. `runCell`
(`internal/benchmark/job_runner.go:516`) already loads those settings instead
of the live config and records the profile name on the run. The compare view
already has a "profile" dimension (`internal/benchmark/compare_labels.go:102`)
and the CSV export already has a `profile` column. The change is moving the
profile choice from the job to each cell.

`BaseProfile` stays as it is, for Autotune and Autoconfigure. The form never
sets it, and a job never has both `BaseProfile` and per-cell profiles.

---

## Phase 1 — Store per-cell profiles and run them
**Files:** `internal/benchmark/job.go`, `internal/benchmark/job_runner.go`,
`internal/benchmark/benchmark.go`, tests.

1. `JobCell` gains `Profile string` (`json:"profile,omitempty"`). Empty means
   "current settings", which is how every existing cell reads.
2. `BenchmarkJob` gains `Profiles []JobProfile` (`json:"profiles,omitempty"`):
   the copies taken when the job was created.
   ```go
   type JobProfile struct {
       ModelID string             `json:"model_id"`
       Name    string             `json:"name"`
       Config  models.ModelConfig `json:"config"`
       SavedAt time.Time          `json:"saved_at"`
       BuildID string             `json:"build_id,omitempty"` // build the profile was saved on
   }
   ```
   Add a lookup `func (j *BenchmarkJob) cellProfile(c JobCell) *BaseProfile`.
   It returns the matching copy as a `BaseProfile`. If the cell has no
   profile, it returns `j.BaseProfile` (nil for normal jobs).
3. Add `ExpandCellsWithStarts(modelIDs, starts, buildIDs, presets, sweeps)`
   with `starts map[string][]string` (model ID → profile names, `""` =
   current settings). `ExpandCellsWithSweeps` calls it with nil. A nil map
   or a missing model means `[""]`, so existing callers and stored jobs
   behave as before. Loop order is build → model →
   starting point → sweep combination → preset. This keeps each profile's
   cells together, so the server reloads once per profile per build. The
   `lastApplied` check in `runCell` already skips a reload when the settings
   did not change.
4. In `runCell`, replace the two uses of `job.BaseProfile` (lines 516 and
   548) and the `samplingForCell(job.BaseProfile, …)` call (line 593) with
   `bp := job.cellProfile(*cell)`. The rest of that code already does the
   right thing: it applies the profile's settings even when nothing is
   overridden, and records `ProfileName` with `ProfileEdited=false`.
   Two more places need the cell's profile:
   - `runCapabilityCell` used the model's current settings even for a job
     with `BaseProfile`. Capability cells must evaluate the profile's
     settings too, or a perplexity score is recorded under the wrong name.
   - `EnsureBuildActive`'s "a config apply follows" flag must be true for
     a profile cell, which always applies its config, so a build switch
     costs one reload rather than two.
5. Cell identity (`benchmark.go:922`): add `Profile` to `cellIdentity`, plus
   a short hash of that profile's copied settings. When a job is edited:
   - completed cells for an unchanged profile keep their results;
   - cells for a profile whose settings changed since the copy run again;
   - without the profile name in the identity, a "Fast chat" result would be
     matched to the "Current settings" cell.
6. `JobDefinition` gains `Starts` and `Profiles`. `UpdateJobDefinition`
   stores them and passes `Starts` to `ExpandCellsWithSweeps`.
7. Bump `schemaVersion` to 6 (new fields only, no migration). An older build
   would ignore the cell profile and measure the live config under a
   profile's name. Opening read-only, which the existing version check
   does, prevents that.

**Tests:** cell expansion with mixed starting points (cell count, order,
grouping by profile). Edit keeps results for an unchanged profile and resets
a changed one. `runCell` applies the profile's copy and not the live config,
and records the profile name. A job with no `Profiles` expands exactly as
before.

## Phase 2 — API: accept, check and copy profiles
**Files:** `internal/api/bench_jobs.go`, tests.

1. `jobCreateRequest` gains `Starts map[string][]string`
   (`json:"starts,omitempty"`).
2. Validation (in `validateJobRequest` and a new
   `(s *Server) resolveJobProfiles`):
   - every model in `Starts` is in `ModelIDs`, and each has at least one
     starting point;
   - each named profile exists for that model (`registry.GetProfile`);
   - if any saved profile is chosen, `Overrides` and `Sweeps` must be empty.
     Error text: "Comparing saved profiles runs each profile exactly as saved,
     so parameters cannot be set or swept in the same job. Clear the
     parameters, or uncheck the profiles."
   - the cell count used by the 500-cell limit counts starting points:
     sum over models of (starting points) × builds × presets.
3. `handleCreateJob` and `handleUpdateJob` copy each chosen profile into
   `job.Profiles` with `registry.GetProfile`. On edit, take a fresh copy.
   Phase 1's identity hash then decides which completed cells are still valid.
   A profile deleted since the job was created cannot be chosen again, so its
   cells drop out on edit, and their runs move to Ad-Hoc as other removed
   cells do.
4. `handleJobForm` passes, for each enabled model, its profiles (name, saved
   time, build ID). It also passes the model's active profile state
   (`registry.ActiveProfileState`) for the "Current settings" label.

**Tests:** create with profiles stores copies, rejects an unknown profile,
rejects profiles together with sweeps, counts cells correctly. Edit after the
profile changed resets only that profile's cells.

## Phase 3 — Form
**Files:** `web/templates/partials/job_form.html`,
`web/templates/benchmarks.html`, `web/jstest/params_test.js`.

1. Under each model checkbox, a nested list that is shown only while the
   model is checked:
   - "Current settings", checked by default. When the live config came from a
     profile, the label says so: "Current settings (Fast chat)" or
     "Current settings (Fast chat, edited)". A tooltip explains that these are
     the settings the model loads today.
   - one checkbox per saved profile, with its saved date in muted text.
   - a model with no saved profiles shows no nested list.
2. When "Current settings" is unedited and its profile is also checked, show
   a note: "Current settings are the same as Fast chat — this measures the
   same settings twice."
3. Build warning: next to a checked profile whose saved build is not among
   the checked builds, show ⚠ with a tooltip: "Saved on build X. Speculative
   decoding methods and options depend on the build, so results on another
   build may differ."
4. While any saved profile is checked, disable the parameter section and show
   a line above it: "Parameters are not available while comparing saved
   profiles. Each profile runs exactly as saved." Keep the user's parameter
   choices in place, so unchecking the profiles brings them back.
5. Submit sends `starts`. `updateMatrixCount` counts starting points instead
   of models. The preview reads, for example,
   "4 starting points (2 models) × 1 build × 2 presets = 8 cells ·
   ~4 llama-server reloads".
6. `prefillJobForm` restores the profile checkboxes from `job.cells` /
   `job.profiles` when a job is edited. A profile that no longer exists is
   listed as "Fast chat (deleted — will be removed on save)", unchecked and
   disabled.

**Tests:** `params_test.js` cases for the cell count with starting points, and
for the parameter section turning off and on.

## Phase 4 — Results
**Files:** `web/templates/partials/job_detail.html`, `internal/api/bench_jobs.go`
(`renderJobDetail`), `internal/api/bench_export.go`,
`web/templates/help.html`.

1. Job detail table: a "Profile" column after "Model", shown only when the job
   has `Profiles`. It shows the profile name, or "Current settings". A tooltip
   gives the time the copy was taken.
2. Job header: for a profile job, one line listing the copied profiles and
   when each was copied, so it is clear the job measures the copies.
3. CSV export: the existing `profile` column already comes from the run. Add
   a `starting_point` column (`current` or `profile`). Without it, an
   unedited live config and the same profile look the same.
4. Compare view: check that the bar labels use the profile name when only the
   profile differs. The dimension exists; add a test case for two profiles of
   one model on one build.
5. Help page, "Jobs" section (`help.html:269`): a short paragraph on comparing
   profiles: what a starting point is, that parameters are off while profiles
   are chosen, and that the job measures a copy taken when it was created.

**Tests:** job detail renders the Profile column only for profile jobs. The
compare label test above.

---

## Not in scope
- Writing results back to a profile (`ProfileMeasurement` stays Autotune's).
- Running each profile on its own saved build.
- Sweeps or fixed parameters on top of profiles. If this is wanted later, the
  runner already supports it: `ApplyOverrides` on a profile base is how
  Autotune works. Only the form and validation would change.
