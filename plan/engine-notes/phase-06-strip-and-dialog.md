# Phase 06 — Engine-notes strip and Review dialog

**Depends on:** 01, 05 · **Enables:** the finished feature; closes issue #452

## Goal

Show the data on the Server page: a one-line strip above the log, quiet or amber by the last 24 hours, and a **Review** dialog with a period picker, totals, a per-model table, the crash list and a **Delete old data** control. Add a help section. Verify the whole feature on compute2.

## Files touched

- `web/templates/partials/engine_notes.html` — new: `engine_notes_strip` and `engine_notes_body` templates.
- `web/templates/server.html` — the strip container above `.log-panel` inside `.log-section`, the `<dialog id="engine-notes-dialog">`, CSS, and a small script block for opening the dialog and the strip pulse.
- `internal/api/engine_notes.go` — HTMX branches of the handlers; view types `notesStripView`, `notesBodyView`.
- `internal/api/engine_notes_view_test.go` — new: view and render tests.
- `internal/api/server.go` — register `GET /api/service/notes/strip` in the `/service` route group. The new partial needs no registration: `parseTemplates` loads `templates/partials/*.html`.
- `web/templates/help.html` — an "Engine notes" paragraph in the `#server` section and a line in `#troubleshooting` pointing to it for crashes.

## Steps

1. **Strip.** `GET /api/service/notes/strip` (HTMX only) renders `engine_notes_strip` from `Summary(24h)`. Polled with `hx-trigger="load, every 10s"`. Inside `.log-section`, above `.log-panel`, so it moves with the log and the Phase 01 grip still sits at the log's bottom edge.
   - No crash or failed load in 24 h: muted text "Engine notes — 14 loads in the last 24 h, no crashes" and an outline **Review** button. With no data at all: "Engine notes — nothing recorded yet".
   - One or more: amber background, an inline warning-triangle SVG (`<svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true"><path d="M8 2.2L14.5 13.5h-13z" stroke-linejoin="round"/><path d="M8 6.4v3.2" stroke-linecap="round"/><path d="M8 11.6v.1" stroke-linecap="round"/></svg>`), "3 crashes in the last 24 h (2 in jobs) — last: Qwen3.6-35B-A3B at 13:59", and a filled **Review** button. "(… in jobs)" is left out when it is 0. Failed loads are counted in the same sentence: "1 crash and 1 failed load in the last 24 h".
   - `data-sig` holds an FNV-1a hash of the `Time` and `RouterName` of each `crash` and `load_failed` event in the 24 h window, as in vllm-toolchest. A short script pulses the strip once when the fingerprint changes between polls, never on an unchanged poll.
2. **Dialog body.** `GET /api/service/notes?period=…` with HTMX renders `engine_notes_body` into `#engine-notes-body`. The `<dialog>` lives in `server.html`, outside the polled strip, so a poll never closes it (the reason vllm-toolchest splits them).
   - Period picker: a `role="group"` of four buttons, 24 h / 7 days / 30 days / All, each `hx-get`-ing the body with its period. The active one is marked.
   - Header line: "From <From> to now. Counting since <CountingSince>."
   - Totals row: router starts (user / jobs), router stops, router exits on their own, loads, failed loads, crashes (in jobs), unloads.
   - Per-model table: Model, Loads, Failed, Crashes, Unloads, Time loaded, Avg load, Requests, Failed requests, Prompt tokens, Generated tokens, Last used. Durations: under 1 minute "35 s"; under 1 hour "42 m"; under 1 day "3 h 12 m"; otherwise "2 d 3 h". Average load uses the same rules. Tokens with thousands separators. Sorted by name. Column headers carry short tooltips (how to read each figure).
   - Crash list, newest first: time, model, "crash" or "failed to load", exit status, build, GPUs, the job name when tagged, and the error lines in a `<pre>`. A `<details>` "What llama-server wrote" holds the tail. A note under the heading: "A crash is an exit with an error status that nobody asked for. An unload, an eviction or a stop that had to be forced is not a crash."
   - Empty period: "Nothing recorded in this period."
3. **Delete old data.** At the bottom of the dialog: a `<select>` with "older than 7 days", "older than 30 days", "older than 90 days", "everything", and a **Delete** button. `hx-post="/api/service/notes/delete"` with `hx-confirm` naming the choice ("Delete engine notes older than 30 days? This cannot be undone."). The HTMX branch of the delete handler answers with the re-rendered body for the current period plus a one-line result ("Deleted 412 events and 230 hours of usage.") and triggers `engineNotesChanged`, which the strip also listens to (`hx-trigger="load, every 10s, engineNotesChanged from:body"`).
4. **Opening.** The **Review** button calls `openEngineNotes()`, which `htmx.ajax`-loads the body for 24 h into `#engine-notes-body` and calls `showModal()`.
5. **Times.** Every time shown (the strip's "at 13:59", the From and Counting since header, crash times, Last used) is rendered as `<time datetime="<RFC 3339 UTC>">HH:MM UTC</time>` (with the date when it is not today), and a small script in `server.html` rewrites each `<time>` in the strip and dialog after every htmx swap to the browser's local time with `toLocaleString`. Without the script the UTC text stays and is still correct. Usage stays bucketed by UTC hour; the 24 h, 7 d and 30 d windows are rolling, so the time zone does not change any count.
6. **Text.** All strings follow the tooltip and help conventions: short sentences, about 80% ASD-STE100, no jargon. The help paragraph explains what is counted, that data is kept on disk until deleted, that job runs are tagged, and how to read a crash entry.
7. Use the period keys `24h`, `7d`, `30d`, `all` from Phase 05 unchanged.

## Build gate

- `make build`
- `go vet ./...`
- `go test ./...`
- `make js-test`

## Test plan

- `engine_notes_view_test.go`: strip text for no data, quiet, one crash, crashes with some in jobs, crash plus failed load; fingerprint changes only when the crash set changes; the body renders the per-model table and the crash list from a fixed `Summary`; durations and token counts format as specified; the delete handler's HTMX reply contains the result line and the `HX-Trigger: engineNotesChanged` header.
- Manual, local or on asmodean: the strip is quiet after a clean load; the dialog opens and survives several strip polls; the period buttons change every figure; delete "older than 7 days" asks for confirmation and updates both the dialog and the strip; the log grip from Phase 01 still works with the strip above the log.
- On compute2 with the built binary swapped in: use the server normally for a day and run at least one benchmark job. Check that crashes appear in the strip within 10 s, are tagged with the job when one ran, are still listed after the job's router restarts and after `docker restart llama-toolchest`, and that each one has a WARN line in `docker logs`.

## Commit

`feat(ui): show engine notes above the Server log, with a Review dialog`

## Rollback

Revert the commit. The strip and dialog disappear; recording from Phase 05 continues and the JSON endpoint still answers, so no data is lost.

## As built

- View code is in `internal/api/engine_notes_view.go` (`notesStripView`, `notesBodyView`, `formatSeconds`, `failureSig`); the HTMX branches of the Phase 05 handlers call it.
- `.log-section` is now a flex column, so the strip sits above the log panel and the Phase 01 grip still sizes the whole section.
- Totals show "Failures in jobs" as its own figure (crashes and failed loads), since the count covers both kinds.
- The delete form carries the shown period, so the dialog stays on it after a delete. Its `hx-confirm` text follows the chosen age.
- Pico keeps a dialog's article narrow; `#engine-notes-dialog > article` is set to full width so the model table fits.
- Checked locally in the browser with sample data: strip, dialog, period buttons, delete with its confirm, the strip reload after a delete, and the dialog staying open through polls. The compute2 check is still to do.
