# Phase 07 — Edit a custom set

**Depends on:** Phase 04 (the Details dialog), Phase 05 (the form dialog and `setFromForm`), Phase 06 (the prefill from a set, and the "Copy as custom set" button the Edit button sits next to) · **Enables:** Phase 08 (the field test edits a set)

`store.UpdateModelSet`, `s.genSetMu`, `s.genMgr.IsActive` and `s.genMgr.IsMeasuring` already exist in the code; this phase only calls them.

## Goal

A custom set can be edited: its files, flags, test sizes, limits, upscale factor and the base DiT checkbox. The name and ID do not change. Editing is refused while an instance of the set runs or measures. After a change to the files or flags, the instance shows its measurement as stale, and the reservation stays.

## Files touched

- `internal/api/server.go` — `PUT /sets/{id}` in the `/gen/models` group.
- `internal/api/gen_set_form.go` — `handleGenSetForm` reads `edit=<id>`; `handleGenUpdateSet`; `setFromForm` gets an edit mode.
- `internal/generation/types.go` — `MeasuredFiles []string` on `Instance`.
- `internal/generation/files.go` (new) — `FilesKey(set ModelSet) []string`.
- `internal/api/gen_vram.go` — Measure stores `MeasuredFiles`.
- `internal/api/gen_instances.go` — `measuredStale` also compares files.
- `web/templates/partials/gen_set_form.html` — the edit mode: the name is read-only, the form sends PUT.
- `web/templates/partials/gen_model_sets.html` — an "Edit" button on custom set rows.
- `web/templates/partials/gen_set_details.html` — an "Edit" button for custom sets.
- `internal/api/gen_set_form_test.go`, `internal/api/gen_instances_test.go` — tests.

## Steps

1. **Files key.** Add `FilesKey(set ModelSet) []string` in `internal/generation/files.go`: for each component, sorted by slot, `slot=path`. Add `MeasuredFiles []string` (JSON `measured_files,omitempty`) to `Instance`.
2. **Measure.** Where Measure stores `MeasuredArgs` (`internal/api/gen_vram.go`), also store `MeasuredFiles = generation.FilesKey(set)`.
3. **Stale.** In `measuredStale` (`internal/api/gen_instances.go`), after the args test, return true when `inst.MeasuredFiles != nil` and it differs from `FilesKey(set)`. Instances measured before this phase have `MeasuredFiles == nil` and are not marked stale by files.
4. **Form.** `handleGenSetForm` accepts `edit=<id>` (404 when not found; 400 "catalog sets cannot be edited. Copy the set to change it." for a catalog set). It fills every field from the set with the Phase 06 prefill code for a set. The name, the backend and the recipe are shown read-only and sent as hidden fields: a different backend or recipe means a different set, so the user copies it instead. The form uses `hx-put="/api/gen/models/sets/<id>"`. The title of the dialog is "Edit <name>".
5. **`setFromForm` edit mode.** Add a parameter `existing *ModelSet`. When it is set, the ID, name, kind, recipe, `Created` and `CatalogID` (empty) come from `existing`, and those form fields are ignored. The files, flags, test sizes, limits, video sizes, scale, licence and base DiT checkbox come from the form and get the same checks as an add, including `CheckFileFormat` on every file.
6. **`handleGenUpdateSet`.** Load the set (404); refuse a catalog set (400, same text as step 4). Under `s.genSetMu`: for each instance of the set, refuse with 409 when `s.genMgr.IsActive(inst.ID)` or `s.genMgr.IsMeasuring(inst.ID)`: "stop <instance name> first". Then parse with `setFromForm`, and save with `store.UpdateModelSet`. Answer with the set list, as the POST does.
7. **Buttons.** On each custom set row in `gen_model_sets.html`, add "Edit" before "Delete". It loads `/api/gen/models/sets/form?edit=<id>` into the form dialog and opens it. Tooltip: "Change this set's files and settings. Stop its instance first." In `gen_set_details.html`, add the same button for custom sets, next to "Copy as custom set". Custom sets only open Details with `Tab` = `models`, so the button loads into `#gen-set-form-body-models`, closes `#gen-set-details-dialog` and opens `#gen-set-form-dialog-models`.

## Build gate

- `go build ./...`
- `go vet ./...`
- `go test ./...`
- `make js-test`

## Test plan

- Edit a custom set's flags with no instance: saved; `RunArgs()` has the new flags; ID, name and `Created` are the same.
- Edit with a running instance (a fake active instance in the manager): 409 with "stop <name> first", and the set is unchanged. Same while measuring.
- Edit of a catalog set: 400 with the copy text, on both the form route and the PUT.
- A PUT that sends another `kind` or `recipe` keeps the set's own kind and recipe.
- An instance measured with `MeasuredFiles` set: after an edit that changes one file path, `measured_stale` is true in the instance list; after an edit that only changes the test size, it is false. Its `reserved_mib` is the same after both edits.
- An instance with `MeasuredFiles == nil` is not marked stale by a file change (old data).
- Edit with a text file in a slot: refused by the format check.
- Manual, on asmodean, before merge: edit a custom set's steps with its instance stopped; see the instance row's existing stale-measurement mark (the one `measured_stale` drives today); try to edit while it runs and read the refusal.

## Commit

`feat(gen): edit a custom set's files and settings`

## Rollback

Revert the commit. `measured_files` stays in the store for instances measured with this phase; an older binary ignores the field. Edits already saved are normal custom sets.
