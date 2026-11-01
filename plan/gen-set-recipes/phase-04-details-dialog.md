# Phase 04 — The Details dialog and its JSON routes

**Depends on:** Phase 03 (`SetDetails`) · **Enables:** Phase 06 (the "Copy as custom set" button lives in this dialog)

## Goal

Users can open "How this set runs" from every set on the Models tab and every catalog entry on the Download Models tab. The same routes answer JSON for API clients.

## Files touched

- `internal/generation/details.go` — JSON tags (snake_case) on the Phase 03 types.
- `internal/api/server.go` — two routes in the `/gen/models` group.
- `internal/api/gen_models.go` — the two handlers.
- `web/templates/partials/gen_set_details.html` (new) — the dialog body.
- `web/templates/partials/gen_model_sets.html` — a "Details" button on each row; the dialog element.
- `web/templates/partials/gen_catalog.html` — a "Details" button on each row; the dialog element.
- `internal/api/gen_models_test.go` — handler tests.
- `web/templates/help.html` — one sentence in the generation section about the Details dialog.

## Steps

1. Add `r.Get("/sets/{id}/details", s.handleGenSetDetails)` and `r.Get("/catalog/{id}/details", s.handleGenCatalogDetails)` to the `/gen/models` route group in `server.go`.
2. `handleGenSetDetails` loads the set (404 when missing) and calls `generation.DetailsForSet`. `handleGenCatalogDetails` calls `generation.LookupCatalog` (404 when missing) and `generation.DetailsForCatalog(e, generation.ModelsDir(s.cfg))`. Both read a `tab` query value (`models` or `catalog`; default `models` for the set route and `catalog` for the catalog route) and pass it to the template as `Tab`, so the buttons that later phases add to the dialog know which tab's dialogs to use. As `handleGenModels` does, both render `gen_set_details` for an htmx request (`isHTMX(r)`) and answer JSON (`respondJSON`) for any other request, with snake_case keys (add JSON tags to the Phase 03 types).
3. `gen_set_details.html`, top to bottom:
   - A line with the program, the recipe (`<code>`) and "Custom set" or "Catalog set".
   - **Files**: a table with Slot (label, and the slot name in `<code>` in a tooltip), File (the path, or for a missing catalog file "Not downloaded yet: will be saved to <path>"), and Size.
   - **Settings**: a table with Flag, Value and Why for `Flags`. A flag with no reason shows "—" and the tooltip "No reason recorded for this flag."
   - **Added by toolchest**: the same table for `FixedFlags`, with one line above it: "Toolchest adds these flags for each instance. Values in <angle brackets> are set when the instance starts."
   - **Command**: the command in a `<pre>` with a "Copy" button that calls `navigator.clipboard.writeText` on the `<pre>` text, as `web/templates/server.html` does for its copy buttons.
   - A small note: "An instance can add its own extra arguments on the Server page."
   - Licence and notes at the end, for catalog sets.
4. In `gen_model_sets.html`, add a `<dialog id="gen-set-details-dialog">` with a header (set name, close button) and a body `<div id="gen-set-details-body">`. Add a "Details" button to each row in the column before "New instance". It does `hx-get="/api/gen/models/sets/{{.ID}}/details?tab=models"`, targets the body, and opens the dialog after a successful request, the same way the New instance button does.
5. In `gen_catalog.html`, add the same dialog (different IDs: `gen-catalog-details-dialog`, `gen-catalog-details-body`) and a "Details" button on each row, next to the download button, calling `/api/gen/models/catalog/{{.ID}}/details?tab=catalog`. The catalog tab only lists catalog entries, so a custom set's Details always open with `tab=models`.
6. Use the button size and classes of the existing row buttons. Tooltips: "Shows the files, settings and command this set runs with."

## Build gate

- `go build ./...`
- `go vet ./...`
- `go test ./...`
- `make js-test`

## Test plan

- Handler tests: `GET /api/gen/models/catalog/z-image-turbo/details` without `HX-Request` returns 200 and JSON with `slots`, `flags`, `fixed_flags` and a `command` that contains `--max-vram <vram limit>`. With `HX-Request: true`, it returns HTML that contains the dialog's "Settings" heading. An unknown ID returns 404 on both routes.
- A custom set's details show "Custom set" and its own flags.
- Manual, on asmodean, before merge: open Details from a catalog row that is downloaded and one that is not, and from a custom set row; check the tables, the markers, the Copy button, and that the dialog closes; check it at phone width.

## Commit

`feat(gen): show how each generation set runs in a Details dialog`

## Rollback

Revert the commit. It adds routes and templates only; nothing is stored.
