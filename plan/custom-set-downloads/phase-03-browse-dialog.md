# Phase 03 — Browse dialog and download size

**Depends on:** Phase 01 (`HFLink`, `Recipe.ModelRepo`), Phase 02 (`resolveHFComponents`, `checkDownloadSpace`, the `hfRepoCache`, links in file fields) · **Enables:** Phase 05 (the field test uses Browse)

## Goal

Each file field in the custom set form gets a **Browse** button. It opens a dialog with these parts:

- The recipe's repo, and the exact files that catalog sets use for that slot.
- A box for any other repo.
- The chosen repo's files with their sizes. By default it shows loadable files only. A **Show all files** switch shows the rest, and a note explains Diffusers repos.

**Use** writes the file's link into the field. Under the files, the form shows the download size and the free space, and it updates when a field changes.

## Files touched

- `internal/generation/catalog.go`: `SlotSuggestions(kind, recipe, slot)`.
- `internal/generation/catalog_test.go`: tests for it.
- `internal/api/gen_set_browse.go` (new): `handleGenSetBrowse`, `handleGenSetFormSize`, file filter.
- `internal/api/gen_set_browse_test.go` (new).
- `internal/api/server.go`: routes `GET /api/gen/models/sets/form/browse` and `GET /api/gen/models/sets/form/size`.
- `web/templates/partials/gen_set_browse.html` (new): the dialog body.
- `web/templates/partials/gen_set_form.html`: Browse buttons, the dialog element, the size line.
- `web/static/gen-set-browse.js` (new): `genSetUseFile`.
- `internal/api/gen_set_browse_js_test.go` (new) and `Makefile`: `TestGenSetBrowseJS` in `js-test`.
- `web/templates/layout.html`: load `gen-set-browse.js` with `defer`, next to `errors.js`. The custom set dialog opens from Models and from the Download page, so the script loads on every page.
- `docs/guides/generation.md`: the Browse dialog in `#custom-set-downloads`.
- `web/templates/help.html`: the Help sentence mentions Browse.

## Steps

1. **Catalog suggestions.** Add `generation.SlotSuggestions(kind BackendKind, recipe, slot string) []SlotSuggestion`. `SlotSuggestion` is `{SetName, Repo, File string; Bytes int64}`.
   - It walks `Catalog()`. It keeps entries of the same `Kind`; for sd and video, it also needs the same `Recipe`.
   - It collects the component with the same `Slot`.
   - It removes duplicates by `Repo+File`, and keeps catalog order.
2. **The Browse handler.** `handleGenSetBrowse` takes the query `slot`, `kind`, `recipe`, `repo`, `all`. It renders `gen_set_browse` with:
   - **Repo choices:** the recipe's `ModelRepo()` (sd and video only, when not empty), then the repos of `SlotSuggestions`, without duplicates.
   - **Suggested files:** the `SlotSuggestions`. Each one has a **Use** button with `HFLink(Repo, "main", File)`, its set name and its size.
   - **Current repo:** `repo` when given. Otherwise the first repo choice. With no choices, there is no current repo, and the dialog asks for one.
   - **Files:** when there is a current repo, call `RepoFiles` on it through `hfRepoCache` (keyed by `repo@main`), with a 15-second context. Each file shows its path, its size in GB with one decimal, and **Use** (`HFLink(repo, "main", path)`).
   - **Error:** a failed listing shows the error text (with #657's message for gated repos) in the dialog, not as an HTTP error, so the form stays usable.
3. **The file filter.**
   - Without `all=1`, keep paths that end in `.safetensors`, `.gguf`, `.ckpt`, `.pt` or `.pth`. For the `tokenizer` slot, keep `.json` files only.
   - Sort by path.
   - Show "N other files hidden" with a link that reloads with `all=1`. With `all=1`, show every file and a link back.
   - When the repo has a root `model_index.json`, show the note: "This repo is in the Diffusers layout. sd-server needs one combined file. Look for a single-file or GGUF version of the model."
4. **The dialog markup.**
   - `gen_set_form.html` gets one `<dialog id="gen-set-browse-dialog">` after the form's fields, with an `<article>` body target.
   - Each file field's label gets a small **Browse** button. It has `hx-get="/api/gen/models/sets/form/browse?slot=<slot>"` and `hx-include` of the form's `kind` and `recipe` fields. Its target is the dialog body, and it opens the dialog after the request (the same pattern as the set form dialog).
   - The repo box in the dialog is a text input and a **List files** button. The button runs `hx-get` on the same route with `repo` and keeps `slot`, `kind` and `recipe` in hidden inputs.
   - Upscale and ACE-Step fields get Browse too. Their choices come from `SlotSuggestions` only.
5. **The JS (`gen-set-browse.js`).** `genSetUseFile(button)`:
   - It reads `data-slot` and `data-link` from the button.
   - It finds `input[name="slot_<slot>"]` in the open custom set form, sets its value, and fires `change` with bubbling.
   - It closes `#gen-set-browse-dialog`.
   Add `TestGenSetBrowseJS` with the DOM stub of the other JS tests, and add it to the `js-test` regex in `Makefile`.
6. **The size line.** `handleGenSetFormSize` reads the form's `slot_*` fields with `formComponents` and calls `resolveHFComponents` (20-second context). It returns one `<small>` line:
   - no links: empty,
   - otherwise: "Download: %.1f GB · %.1f GB free",
   - when it does not fit: the same line with the `checkDownloadSpace` message and the error color,
   - when resolving fails: the error text.
   In `gen_set_form.html`, place `<div id="gen-set-download-size" hx-get="/api/gen/models/sets/form/size" hx-include="closest form" hx-trigger="load, change from:closest form delay:300ms">` above the Add set button. This line is a guide only. Add set still checks again (Phase 02).
7. **Docs.**
   - In `#custom-set-downloads`, describe Browse: the suggested repos and files, the repo box, the filter, Show all files, the Diffusers note, and the size line.
   - Update the Help sentence to name Browse.
   - Browse button tooltip: "Choose a file from Hugging Face. Toolchest downloads it when you save the set."

## Build gate

```bash
go build ./...
go vet ./...
go test ./...
make js-test
make docs
```

## Test plan

Automated:

- `SlotSuggestions(KindSD, "flux2-klein-4b", "llm")` returns the klein set's text encoder. A slot that no catalog set uses returns nothing.
- Browse against an `httptest` Hugging Face server:
  - the recipe repo is the default,
  - the filter hides `.md`/`.json` (except for `tokenizer`),
  - `all=1` shows them,
  - `model_index.json` gives the Diffusers note,
  - a 403 shows the gated message inside the dialog with status 200.
- The size line with one link and a too-small free space shows the refusal text.
- `TestGenSetBrowseJS`: Use sets the right field, fires `change`, and closes the dialog.

Manual, on a dev instance:

- Open the form for `flux2-klein-4b`, and press Browse on each slot. Check the suggestions, the repo default, Use, and the size line update.
- Check the dialog at phone width.

## Commit

`feat(generation): browse Hugging Face files in the custom set form`

## Rollback

Revert the commit. Pasted links (Phase 02) keep working. The routes, the template, the JS file and the test go away together. No saved data changes in this phase.
