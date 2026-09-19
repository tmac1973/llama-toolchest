# Phase 09 — Autoconfigure button, review screen and saving

**Depends on:** 02, 03, 05, 06, 07, 08 · **Enables:** 11 and 13 (autotune can start from
the Autoconfig profile); completes PR 2 (autoconfigure)

## Goal
Put the pieces together into the feature the user sees:
- an Autoconfigure button on the model card;
- a dialog that asks one question (context size class);
- progress while it works;
- a review screen that shows every proposed setting with its reason and
  source;
- Save as profile ("Autoconfig") or Save and apply.

After a download, the model card suggests running it.

## Files touched
- `internal/autoconfig/run.go` (new): `Run(ctx, deps, modelID, class) (Result, error)`,
  which combines `PlanFit` and advice into a `ModelConfig` plus notes.
- `internal/api/autoconfig.go` (new): handlers and an in-memory run tracker.
- `internal/api/autoconfig_test.go` (new).
- `internal/api/server.go`: routes.
- `internal/api/hf.go` (`onDownloadComplete`): set `Model.AutoconfigHint = true`.
- `internal/models/registry.go`: `Model.AutoconfigHint bool` (`json:"autoconfig_hint,omitempty"`),
  cleared when the user runs autoconfigure or dismisses the hint.
- `web/templates/partials/model_card.html`: the Autoconfigure button and the
  hint line.
- `web/templates/partials/autoconfig_dialog.html` (new): the size class
  question and warnings.
- `web/templates/partials/autoconfig_review.html` (new).
- `web/templates/partials/profile_bar.html`: the collapsible "Why these
  settings" list of the active profile's `Notes`.
- `internal/api/autoconfig_render_test.go` (new).

## Steps
1. **`Run`:**
   1. Build `Hardware` from `monitor.Current()`.
   2. `base := registry.GetConfig(id)`.
   3. `fit := PlanFit(m, *base, hw, class)`.
   4. `card := FetchCard`, then `adv := llm advice`, then `props := Validate`.
   5. Merge: start from `base`, apply the fit fields, then the validated
      advice proposals. Proposals never touch fit fields, with the single
      exception defined in Phase 08: a lower `context_size` under the
      Maximum class. After that change, recompute only
      `EstimateGiB` with `models.VRAMEstimateForConfigOn`; do not call
      `PlanFit` again, which could change other fields.
   6. `NormalizeSpec`.
   7. Run the same validators the config PUT runs.
   8. Build `[]ProfileNote`, one per changed field plus general notes.
   9. Return the result with `DraftSuggestions`.

   Each step reports progress text through a callback: "Checking what fits
   on your GPU", "Reading the model card", "Asking the helper model (this can
   take a minute)", "Checking the answer".
2. **Speculative decoding in autoconfigure.**
   - If the model has built-in MTP (`m.NextNLayers > 0`, from Phase 05),
     propose `SpecType=draft-mtp` with default draft parameters, even when
     the card does not mention it, with origin "model file" and the reason
     "This model includes its own draft layers (MTP), which usually speeds up
     generation with no change to the answers."
   - Do not propose an n-gram assist; which assist helps depends on the use
     case, and autotune measures it. Add the note "Autotune can test adding
     an n-gram assist, which often speeds up code editing further."
3. **Routes.**
   - `GET /api/models/{id}/autoconfig` returns the dialog.
   - `POST /api/models/{id}/autoconfig` (form `context_class`) starts a run.
   - `GET /api/models/{id}/autoconfig/progress` is an SSE stream using the
     helpers in `sse.go`. It sends the progress text, then the rendered
     review.
   - `POST /api/models/{id}/autoconfig/save` (form `apply=0|1`) saves.
   - `POST /api/models/{id}/autoconfig/dismiss-hint`.

   Only one autoconfigure run at a time for the whole server; a second is
   refused with "Autoconfigure is already running for MODEL".
4. **Dialog.**
   - A radio group, "How long are your conversations or documents?", with
     the options:
     - Short — about 8,000 tokens, a few pages;
     - Medium — about 32,000, a long document (the default);
     - Long — about 128,000, a small codebase or a book chapter;
     - Maximum — the largest this model supports that fits.
   - Warnings shown before starting:
     - no helper model set: a link to Settings, plus the one-click download
       button from Phase 07;
     - a benchmark is running: the Start button is disabled with the reason;
     - other models loaded: "MODEL will be unloaded while the helper model
       runs."
5. **Review screen.** A table with the columns Setting | Current | Proposed |
   Why | Source.
   - Unchanged settings are collapsed under "Settings left as they are (N)".
   - Every setting name has the same tooltip text as its field in the config
     form: move those texts into a shared `fieldHelp` map in
     `internal/api/field_help.go` (new), used by both templates.
   - Above the table, the fit line "Estimated 14.2 GiB of 15.0 GiB of GPU
     memory", with a tooltip explaining the margin.
   - Draft suggestions are listed with size and a **Download** button. It
     uses the existing download endpoint; after the download completes, the
     user runs Autoconfigure again to include the file.
   - "The model card also mentions…" notes appear as a list.
   - The buttons:
     - **Save as profile**: `SaveProfileFrom(id, "Autoconfig", …)` with
       `Source "autoconfig"`, the notes, and `BuildID`;
     - **Save and apply**: the same, then `ApplyProfile` and
       `afterConfigChange`;
     - **Discard**.
   - Saving over an existing "Autoconfig" profile says "Replaced the earlier
     Autoconfig profile".
6. **Clean-up.** After a run, successful or not, unload the helper
   (`llmcall.Client.Unload`) and clear `AutoconfigHint`.
7. **Model card hint.** "Autoconfigure can suggest settings for this model"
   with a Dismiss link. It is shown when `AutoconfigHint` is true.
8. The profile bar (Phase 03) already lists the Autoconfig profile, with
   "(autoconfig)". Restoring it also shows its notes in a collapsible "Why
   these settings" list under the bar.

## Build gate
`go build ./... && go vet ./... && go test ./...`

## Test plan
- `Run` with fake dependencies (fixed hardware, a canned card, canned
  advice):
  - fit and advice merge as described;
  - advice cannot change the context set by fit, except lowering it under
    Maximum;
  - built-in MTP is proposed;
  - the result passes the validators.
- Handler tests:
  - a start while another run is active is refused;
  - a start while a job is running is refused;
  - save with `apply=0` creates the profile and leaves the live config;
  - `apply=1` changes the live config and marks the model dirty.
- Render tests:
  - the review table shows one row per changed field with its reason;
  - unchanged fields are collapsed;
  - the hint shows after `onDownloadComplete`.
- Manual (`make dev`):
  - Download a small model with built-in MTP. Check the hint appears, run
    Autoconfigure with Medium, read the review, and Save and apply.
  - Restart the server and check the model loads and answers a chat request.
  - Repeat with an MoE model larger than VRAM and check that `cpu_moe` is
    proposed with a reason.

## Commit
`feat(autoconfig): suggest a starting profile from the hardware and the model card`

## Rollback
Revert the commit. Profiles already saved as "Autoconfig" stay as ordinary
profiles. The `autoconfig_hint` field is ignored by older code.

## As implemented

- **Polling, not SSE.** Progress is polled once a second
  (`GET /api/models/{id}/autoconfig/status`, which returns the progress
  partial until the run is done and then the review). A run keeps its
  result after the browser leaves, so polling needs no connection state,
  and it reuses the plain partial rendering.
- **Refusals show as messages.** A start refused because a benchmark is
  running or another run is active renders a message instead of an HTTP
  error, because htmx does not swap in a non-2xx response.
- **Help texts.** `fieldHelp` (`internal/api/field_help.go`) holds the
  review table's help texts. The config form's own tooltips were not
  rewritten to read from it; that would be a large template change for no
  change in behavior.
- **Thinking stays a note.** The card's thinking advice is a note only (see
  Phase 08).
- **Named view type.** The model card's view data is now the named
  `modelCardView`; its render test aliases it instead of copying the fields.
- **Hint and container.** New downloads get `AutoconfigHint` (not embedding
  models). Running autoconfigure or dismissing the hint clears it. The card
  holds an `#autoconfig-<id>` container that reuses the config panel's
  styling.
- **Smoke test.** Run against a real server on this machine (16 GB GPU, no
  helper model): dialog, run, review, save and apply all worked, and the
  model's built-in MTP was proposed.
- **Still to check by hand, with a GPU and the helper model installed:**
  - that the helper loads (including the router restart when needed);
  - that it reads a real model card and its advice appears in the review;
  - that it is unloaded afterwards;
  - how the review screen looks in a browser.
