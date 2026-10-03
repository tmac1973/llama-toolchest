# Phase 04 — Settings UI, Models-page ghost rows, and docs

**Depends on:** 01 (export endpoint), 02 (restore endpoint + report
partial), 03 (pending store + discard endpoint) · **Enables:** nothing —
final phase.

## Goal

The user-facing surface: a Backup & Restore card on Settings with export
controls and the selective-restore form (client-side file preview with
per-section counts and a secrets badge), the restore report rendered in
place with one-click download of missing models, pending configs visible as
ghost rows on the Models page with download/discard actions, and help-page
documentation.

## Files touched

- `web/templates/settings.html` — Backup & Restore card + inline preview
  script.
- `web/templates/partials/restore_report.html` — extend Missing rows with
  the download button (partial created in phase 02).
- `internal/api/models.go` — `handleListModels` / `renderModelList` (the
  `GET /api/models` fragment that fills `#model-list`; the models-*page*
  handler only renders the shell) interleaves pending entries into the
  flat card list, and a new `renderPendingCard` renders each ghost card.
  No new routes and no discard-handler changes: phase 03's
  `HX-Trigger: modelsChanged` response makes `#model-list` re-fetch itself
  through the page's existing event listener.
- `internal/api/hf.go` — `handleHFDownload` gains the `inline=1` response
  mode (step 3): always 200 to htmx callers with a minimal text fragment,
  success or error alike.
- `internal/api/model_card_render_test.go` — ghost-card render assertion
  (step 6).
- `internal/api/settings_page_render_test.go` — extend the render smoke
  test.
- `web/templates/help.html` — Backup & Restore section.

## Steps

1. **Backup & Restore card** (after the Runtime environment article), all
   patterns matching the existing Settings page (article card, `title`
   tooltips, htmx status targets):
   - Export row: "Include secrets (HF token, API key)" checkbox with a
     plaintext warning in its tooltip; Export button as a plain link
     `href="/api/backup"` that JS rewrites to `?secrets=1` when checked
     (no htmx — it's a file download).
   - Restore form: `<input type="file" accept=".json">`, four checkboxes
     (`sec_settings`, `sec_env`, `sec_flags`, `sec_models`, all checked),
     Restore button posting to `/api/restore` with explicit
     `hx-target="#restore-report"` on the form (the htmx-inheritance
     lesson) **and `hx-encoding="multipart/form-data"`** — required for
     the file to be sent at all, and worth flagging because no existing
     template in the project uploads a file, so there is no precedent to
     copy.
   - Copy states the caveat: "Backs up configuration, not models, builds,
     or benchmark history."
2. **Client-side preview script** (inline, plain JS like `filterEnvRows`):
   on file selection, FileReader + JSON.parse against the phase 01 schema's
   snake_case keys. Checkbox labels update as:
   - Settings → "Settings" (no count — it's one scalar bundle), with a
     `<mark>` badge appended when `settings.hf_token` or `settings.api_key`
     is present: "contains secrets".
   - Runtime env → "Runtime env (N)" where N = curated entries + non-empty,
     non-`#` lines of `runtime_env.extra`.
   - Flag presets → "Flag presets (N)" = array length.
   - Model configs → "Model configs (N)" = array length.
   Absent sections: checkbox unchecked and disabled. The Restore button is
   enabled only when ALL THREE hold: a file is selected, it parsed as a
   backup (parse failure shows "not a backup file"), and at least one
   section is checked (mirroring the server's 400) — so it renders
   disabled on initial page load, which the smoke test asserts. The server
   remains authoritative — the script only informs.
3. **Report rendering**: `#restore-report` div under the form receives the
   phase-02 partial (explicit `hx-target` on the form — the htmx-inheritance
   lesson). Extend the partial's Missing rows with a Download button:
   per-row htmx `hx-post="/api/hf/download"` with
   `hx-vals='{"model_id": "...", "filename": "..."}'` — form-encoded, which
   the handler's existing form branch accepts (`hf.go` decodes JSON *or*
   `ParseForm`; `size` is optional and omitted). Quant correctness needs no extra plumbing: the download
   registers the file and the registry derives Quant from the filename, so
   a pending entry keyed (ModelID, Quant) claims exactly when the stored
   Filename was fetched. The existing htmx response — the
   `download_progress` partial — is `<td colspan="6">` table markup
   designed for `hf_files.html`'s `<tr id="dl-…">` containers and cannot
   be swapped into a span or card, and the handler reports failures via
   `http.Error` (400/507), which htmx 2.0.4 does not swap by default — so
   errors would be invisible. Both are solved with an inline response
   mode: `handleHFDownload` (add `internal/api/hf.go` to Files touched)
   gains an `inline=1` form value under which it **always returns 200**
   to htmx callers with a minimal text fragment — "queued — progress in
   the Downloads panel" on success, or the error text ("download already
   in progress", "insufficient space…") on failure — the same
   errors-must-be-visible contract the restore handler uses. Each row
   carries a status container
   `<span id="restore-dl-{{cssID .ModelID}}-{{cssID .Quant}}"></span>`;
   the button posts with `hx-vals` including `inline: 1`, `hx-target` on
   that span, `hx-swap="innerHTML"`, `hx-disabled-elt="this"`. The button
   sits outside the swapped span and survives the swap; live progress
   lives in the existing Downloads panel. "Download all
   missing" is a small JS loop that clicks each row's still-enabled
   Download button and then disables itself; a duplicate start for an
   already-queued model surfaces as "download already in progress" text in
   that row's status span (the inline mode's 200-with-error-text
   contract) — benign. Auto-claim
   (phase 03) attaches configs as each lands, visible in the downloads
   panel.
4. **Models-list ghost cards** — grounded in the real structure: the list
   is **flat** (`renderModelList` writes one `renderModelCard` per model
   into `.model-card-list`, sorted by the `OrgAndBase()` key; there are no
   group containers). Changes to `internal/api/models.go`:
   - `renderModelList` takes the pending entries alongside the models and
     merges them into one sorted sequence using the same org/base sort key
     (derivable from the pending entry's ModelID string alone), so a ghost
     card sits next to installed quants of the same base model, or stands
     alone for an absent one.
   - A ghost card renders via a new `renderPendingCard` (greyed, "not
     installed — config waiting (imported <date>)", Download and Discard
     actions) — `PendingConfig` is a different type than `*models.Model`,
     so it gets its own renderer rather than a fake Model.
   - The empty-registry short-circuit ("No models downloaded yet…",
     `models.go:34-37`) must check pending too: on a fresh target server —
     the primary restore-then-provision case — pending entries exist while
     the registry is empty, and the ghost cards must render there.
   - `renderModelList` has two callers, and pending entries route to the
     right one by the **same predicate the real lists use**:
     `Model.IsEmbedding()` is a name regex (`IsEmbeddingModel`,
     `registry.go:956`), not a GGUF inspection, so it's computable from a
     pending entry's ModelID. `handleListModels` gets the
     non-embedding pending entries, `handleListEmbeddingModels` the
     embedding ones (its own empty short-circuit gains the same
     pending-awareness as the chat list's), keeping a pending
     `nomic-embed`-style entry next to its installed siblings.
   Actions: Download uses the same wiring as step 3 (shared row markup:
   status container + `hx-post`/`hx-vals` button); Discard is `hx-post` to
   `/api/backup/pending/discard` with `hx-vals` for model_id/quant and
   `hx-swap="none"` — the 204's `HX-Trigger: modelsChanged` makes
   `#model-list` refresh itself, removing the card.
5. **Help page**: a "Backup & Restore" section under Settings docs covering:
   what's included/excluded, secrets opt-in, selective restore, pending
   configs and auto-claim, topology re-resolution warnings, and the curl
   endpoints for scripting.
6. **Render smoke tests**: extend `settings_page_render_test.go` for the
   card (file input, four checkboxes, preview script marker, report target,
   Restore-disabled default state); extend the existing
   `internal/api/model_card_render_test.go` with a ghost-row assertion
   (pending entry renders greyed with Download and Discard controls).

## Build gate

`go build ./... && go vet ./internal/... && go test ./internal/...`

## Test plan

- Automated: the render smoke tests above; handler test for discard
  asserting 204 with the `HX-Trigger: modelsChanged` header (phase 03's
  final contract) and the entry gone from the registry.
- Manual on the dev box (headless-browser screenshots, as established):
  - Export with and without secrets; diff the two files.
  - Restore the no-secrets file with only flag presets ticked → report shows
    presets applied, others skipped-by-choice.
  - Restore a file referencing a model not installed → ghost row appears on
    Models; Download claims it; config visible on the model afterward.
  - Discard a pending entry → row disappears, models.json updated.
  - Malformed file → preview flags it client-side; force-posting it anyway
    returns the structural rejection.

## Commit

`feat(backup): Backup & Restore settings card, restore preview, pending ghost rows`

## Rollback

Template and handler-glue only — revert the card, script, ghost rows, and
help section; phases 01–03 remain fully functional via curl. No stored
state involved.
