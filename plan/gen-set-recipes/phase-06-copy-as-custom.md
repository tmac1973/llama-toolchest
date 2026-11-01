# Phase 06 — Copy as custom set

**Depends on:** Phase 02 (`Scale`, `SizeMultiple()`), Phase 04 (the Details dialog and its `Tab` value, where the button lives), Phase 05 (the form dialog it fills) · **Enables:** Phase 08 (the field test copies catalog sets)

## Goal

From the Details dialog of any set or catalog entry, one click opens the custom set form, filled in from that set. Files that are on this machine are filled in; the others show the catalog file name as a hint. This works on the Models tab and on the Download Models tab.

## Files touched

- `internal/api/gen_set_form.go` — `handleGenSetForm` reads `copy` and `copy_catalog`; `setFromForm` reads `video_sizes` and `license`; the POST answers with a short note for `view=catalog`.
- `internal/generation/args.go` (new) — `JoinArgs(args []string) string`, the inverse of `splitWords`.
- `web/templates/partials/gen_set_form.html` — the hidden `video_sizes`, `license` and `view` fields; a line "Copied from <name>" at the top.
- `web/templates/partials/gen_set_details.html` — a "Copy as custom set" button.
- `web/templates/partials/gen_catalog.html` — the form dialog (same markup as on the Models tab, via a shared `gen_set_form_dialog` template), and the form posts with `view=catalog`.
- `web/templates/partials/gen_model_sets.html` — uses the shared `gen_set_form_dialog` template.
- `web/templates/partials/gen_set_saved.html` (new) — the note shown after a save on the Download Models tab.
- `internal/api/gen_set_form_test.go`, `internal/generation/args_test.go` (new) — tests.

## Steps

1. **`JoinArgs`.** Add `JoinArgs(args []string) string` in `internal/generation/args.go`. It joins with spaces and puts a word in double quotes when it has a space, a quote or a shell character, escaping inner double quotes and backslashes. `ValidateExtraArgs(JoinArgs(x))` must give back `x` for every catalog entry's flags.
2. **Prefill.** `handleGenSetForm` accepts `copy=<set id>` or `copy_catalog=<entry id>` (404 when not found; 400 when both are given). From a set, take: kind, recipe, `RunArgs()` (through `JoinArgs`), test size, frames, fps, max fields, `UpscaleScale()`, `HasStemTasks()`, `VideoPresets(set)`, licence, and each component's path as the slot value. From a catalog entry, take: kind, recipe, `JoinArgs(e.LaunchArgs)`, `DefaultWidth`, `DefaultHeight`, `DefaultFrames`, `DefaultFPS`, `MaxFrames`, `MaxWidth`, `MaxHeight`, `Scale`, `StemTasks`, `VideoSizes` and `License`, all from the entry; for each component, use the path from `huggingface.FilePath(modelsDir, c.Repo, c.File)` as the value when that file exists, else leave the value empty and set the hint to `c.File`'s base name. LoRA slots are filled the same way.
3. The suggested name is "<source name> custom". If a set with that ID exists, add " 2", " 3" and so on, as `freeInstanceName` does for instances.
4. The form shows "Copied from <source name>. Change the files or settings, then save." above the fields. Hidden fields: `video_sizes` (the presets as JSON) and `license`.
5. **POST.** `setFromForm` reads `video_sizes` (JSON list of `VideoSize`; video sets only; each size must be within the max fields when they are set) into `ModelSet.VideoSizes`, and `license` into `ModelSet.License`. Both are optional.
6. **Shared dialog.** Move the form dialog markup from Phase 05 into a template `gen_set_form_dialog` in `gen_set_form.html`, with a parameter for the tab (`models` or `catalog`). Include it in `gen_model_sets.html` and `gen_catalog.html`. On the Download Models tab, the form posts to `/api/gen/models/sets?view=catalog` and targets the dialog body.
7. **Saved note.** For `view=catalog`, the POST answers with `gen_set_saved`: "<name> is added. Make an instance for it on the Models tab." with a link to `/models` and a Close button. For other requests it answers as today.
8. **Button.** In `gen_set_details.html`, add "Copy as custom set" at the bottom. It uses the `Tab` value from Phase 04. With `Tab` = `models`, it loads `/api/gen/models/sets/form?copy=<id>&tab=models` into `#gen-set-form-body-models`, closes `#gen-set-details-dialog` and opens `#gen-set-form-dialog-models`. With `Tab` = `catalog`, it loads `/api/gen/models/sets/form?copy_catalog=<id>&tab=catalog` into `#gen-set-form-body-catalog`, closes `#gen-catalog-details-dialog` and opens `#gen-set-form-dialog-catalog`. The `gen_set_form_dialog` template from step 6 uses these IDs (`-models` or `-catalog` after the base ID), and `handleGenSetForm` passes `tab` on so the form posts with `view=catalog` on the catalog tab. Tooltip: "Opens the custom set form with these files and settings filled in. You can change them before you save."

## Build gate

- `go build ./...`
- `go vet ./...`
- `go test ./...`
- `make js-test`

## Test plan

- `JoinArgs` round trip: for every catalog entry, `ValidateExtraArgs(JoinArgs(e.LaunchArgs))` equals `e.LaunchArgs`; a word with a space round-trips.
- Form with `copy_catalog=wan22-i2v-a14b-lightning` and a temp models folder that holds only the VAE: the VAE field has its path, the other fields are empty with the catalog file names as placeholders; launch args, frames, fps, the max fields and `video_sizes` match the entry.
- Form with `copy=<a catalog set on disk>`: every slot field has the set's path.
- Copy then POST with no changes (after the test writes the missing files): the new custom set has the same components, `RunArgs()`, max fields, `VideoSizes`, scale and licence as the source, and `/v1/models` gives it the same `meta.video` as the source set.
- The suggested name avoids an ID that is taken.
- POST with `view=catalog` returns the saved note, not the set list.
- Manual, on asmodean, before merge: copy from a downloaded catalog row on the Download Models tab, save, follow the link, see the set on the Models tab; copy from a custom set on the Models tab.

## Commit

`feat(gen): copy any generation set into a custom set`

## Rollback

Revert the commit. Sets made by a copy are normal custom sets; their `video_sizes` and `license` fields existed before this phase, so they stay readable.
