# Phase 13 — Autotune screens

**Depends on:** 02, 03, 09, 11, 12 · **Enables:** completes PR 3 (autotune) and the
plan

## Goal
The screens a non-expert uses:
- **start**: choose the starting profile and the use case, see how long it
  will take and that the server is unavailable meanwhile;
- **progress**: which stage, what is being tried, time left, cancel or
  resume;
- **results**: what each saved profile gained, in plain language, with
  Restore buttons.

## Files touched
- `internal/api/autotune.go` (new): handlers.
- `internal/api/autotune_test.go`, `internal/api/autotune_render_test.go`
  (new).
- `internal/api/server.go`: routes, and the page route `/autotune/{id}`.
- `web/templates/partials/model_card.html`: an **Autotune** button next to
  Autoconfigure, and a "Last autotune: date, status" link when
  `LatestForModel` (Phase 11) returns a record.
- `web/templates/partials/autotune_start.html` (new).
- `web/templates/autotune.html` (new page): progress and results.
- `web/templates/partials/autotune_progress.html`,
  `web/templates/partials/autotune_results.html` (new).
- `web/templates/benchmarks.html` and `partials/job_detail.html`: a stage job
  shows "Part of Autotune: link" (from `AutotuneID`).
- `web/templates/partials/profile_bar.html`: autotune profiles show their
  measured gain in the option label, e.g. "(autotune, +61% generation)".

## Steps
1. **Routes.**
   - `GET /api/models/{id}/autotune/start`: the start dialog.
   - `POST /api/models/{id}/autotune` (form `profile`, `use_case`): start,
     then redirect to `/autotune/{rid}`.
   - `GET /autotune/{rid}`: the page.
   - `GET /api/autotune/{rid}/progress`: SSE. It polls the record and the
     current stage job every second and sends the rendered progress partial
     when it changes; after "done" it sends the rendered results.
   - `POST /api/autotune/{rid}/cancel` and `POST /api/autotune/{rid}/resume`.
   - `POST /api/autotune/{rid}/restore` (form `profile`): calls the Phase 03
     restore path, then shows a banner.
2. **Start dialog.**
   - A profile `<select>` listing the model's profiles, "Autoconfig"
     first when present, with the note "Autotune keeps this profile's context
     size, memory settings and sampling, and only changes settings that do
     not change the model's answers."
   - When the model has no profiles: "Save your current settings as a
     profile or run Autoconfigure first", with a button that saves the live
     config as "Current settings" and reloads the dialog.
   - A use-case radio group:
     - General chat — "questions, writing, conversation";
     - Coding / editing — "changing code or text you give it; speculative
       decoding helps most here";
     - Mixed — "both; takes about twice as long".
   - An estimate, updated with `hx-trigger="change"`: "About N cells, roughly
     M minutes", from `EstimateMinutes` over all planned stages. Later stages
     are estimated at their maximum size.
   - The list of speculative methods that will be skipped, and why (from the
     planner's `skipped`), with a link to Autoconfigure's draft suggestions.
   - A warning box: "While Autotune runs, the server restarts many times and
     cannot answer other requests."
3. **Progress partial.**
   - A four-step indicator: Batch sizes → Speculative decoding → Fine-tuning
     draft settings → Confirming.
   - The current cell as a plain label (e.g. "Trying MTP with ngram-mod
     assist"), from the Phase 10/11 candidate `Label`.
   - A cell count, and "about N minutes left" from `SecondsPerCell`.
   - Cancel (with `hx-confirm`). On an interrupted or cancelled record,
     Resume.
   - For each finished stage, a small table of its finalists with the three
     scores. Column tooltips explain each metric:
     - "Generation speed: tokens written per second. Higher is better.";
     - "Prompt speed: tokens of input read per second. Higher is better.";
     - "Response time: seconds until a typical answer for this use case is
       finished. Lower is better."
4. **Results partial.**
   - A card for each goal:
     - the saved profile name;
     - the headline, e.g. "Generation: 38 → 61 tokens/s (+61%)";
     - the other two metrics in smaller text;
     - the list of changed settings with their plain-language notes;
     - a **Restore this profile** button.
   - A goal with no gain shows "Your starting profile is already the fastest
     here", with no button.
   - The note "Measured on the Coding / editing test with build BUILD. Other
     workloads may differ."
   - A link to the confirm stage's job, for the full benchmark data.
5. **Model card.** Autotune is disabled with a tooltip reason when a
   benchmark job is running or when no build is active.
6. **Help text.** Add a short "Autoconfigure and Autotune" section to the
   existing Help page (`/help`), explaining:
   - the two features and profiles;
   - which settings each one changes;
   - that speculative decoding does not change answers;
   - the context size and memory choices;

   in plain language.

## Build gate
`go build ./... && go vet ./... && go test ./... && make js-test`

## Test plan
- Render tests:
  - the start dialog with no profiles shows the save-current button;
  - with profiles it lists Autoconfig first;
  - skipped methods are listed;
  - the progress partial for a mid-stage-2 record shows step 2 as active
    with a time estimate;
  - results with one shared winner and one no-gain goal render both cards
    correctly;
  - every metric column has a tooltip.
- Handler tests:
  - start redirects to the page;
  - start while a job runs shows the reason and does not start;
  - cancel and resume change the status;
  - restore applies the profile and marks the model dirty.
- Manual (`make dev`, on a real GPU):
  - Take a model with built-in MTP, run Autoconfigure → Save, then Autotune
    from "Autoconfig" with Coding / editing.
  - Watch the stages, cancel during stage 2, then resume.
  - At the end, check the results show MTP + n-gram combinations were
    tested, restore the "fastest response" profile, restart, and confirm the
    speed in an ordinary quick benchmark is in line with the reported
    numbers.
  - Check the whole run for a single-GPU model finishes in about 1.5 hours or
    less.

## Commit
`feat(autotune): start, follow and use autotune from the model card`

## Rollback
Revert the commit. The runner from Phase 12 remains, but nothing starts it.
Records and saved profiles stay; the profiles remain usable from the profile
bar.
