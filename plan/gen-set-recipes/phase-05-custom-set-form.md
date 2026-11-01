# Phase 05 — The custom set form dialog

**Depends on:** Phase 02 (slot labels, the scale and max fields) · **Enables:** Phase 06 (copy fills this form), Phase 07 (edit reuses this form)

## Goal

Replace the free-text custom set form with a dialog that shows one field per slot of the chosen recipe, offers only the recipes of the chosen backend, and suggests model files from the models folder. Adding a set also checks each file's format, and a missing path gets a message that says where the models folder is.

## Files touched

- `internal/api/server.go` — three GET routes in the `/gen/models` group.
- `internal/api/gen_set_form.go` (new) — the form handlers, the form data type and `setFromForm`, the shared parser for add (and, in Phase 07, edit).
- `internal/api/gen_models.go` — `handleGenCreateSet` uses `setFromForm`.
- `internal/generation/recipes.go` — `CheckFiles` takes the models folder and the container flag; new `CheckFileFormat`.
- `internal/generation/model_files.go` (new) — `ListModelFiles(dir string, max int) (models, tokenizers []string, truncated bool)`.
- `web/templates/partials/gen_set_form.html` (new) — `gen_set_form` (the whole dialog body) and `gen_set_form_fields` (the part that reloads).
- `web/templates/partials/gen_model_sets.html` — the `<details>` form goes; an "Add a custom set" button and the form dialog come in.
- `internal/api/gen_models_test.go`, `internal/api/gen_video_kind_test.go` — update to the new form fields; keep a test for the old `files` field.
- `internal/api/gen_set_form_test.go` (new), `internal/generation/recipes_test.go`, `internal/generation/model_files_test.go` (new) — tests.

## Steps

1. **Routes.** Add to the `/gen/models` group:
   - `GET /sets/form` → `handleGenSetForm`: the whole dialog body. Query: `kind` (default `sd`), `recipe` (default the first of `Recipes()` or `VideoRecipeIDs()` for that kind).
   - `GET /sets/form/fields` → `handleGenSetFormFields`: only `gen_set_form_fields`, for the current `kind` and `recipe`, with the other form values passed through so typed values stay.
   - `GET /files` → `handleGenModelFiles`: the file lists. For htmx, two `<datalist>` elements (`gen-model-file-list`, `gen-tokenizer-file-list`); otherwise JSON `{"models": [...], "tokenizers": [...], "truncated": bool}`.
2. **Form data.** `type genSetForm struct` holds: mode (`add` now; `edit` and copy come later), name, kind, recipe, the recipe list for the kind, a list of slot fields (`Slot`, `Label` from `SlotLabel`, `Required`, `OneOf` note, `Value`, `Hint`, `List` = which datalist), launch args as one string, width, height, frames, fps, max frames/width/height, scale, stem tasks, and the models folder path for the help line.
3. **`gen_set_form_fields`** renders, in order: the Backend select and the Recipe select (only that kind's recipes; hidden for music and upscaler), each with `hx-get="/api/gen/models/sets/form/fields" hx-include="closest form" hx-target="#gen-set-form-fields" hx-swap="innerHTML"` on change; one text input per slot, named `slot_<slot>`, with the label, a "required" mark, `list=` the right datalist, and the hint as `placeholder`; then the fields for the kind (test size for image and video; frames, fps and the three max fields for video; scale for upscaler; base DiT for music). The field help text under the slots says: "Type part of a file name to see matching files in the models folder (<path>). Files elsewhere must be under that folder; in a container, mount them under it."
4. **`gen_set_form`** is the `<form>` with `hx-post="/api/gen/models/sets" hx-target="#gen-model-sets" hx-swap="innerHTML"`, the Name field, `<div id="gen-set-form-fields">` holding `gen_set_form_fields`, the launch args field, a `<div hx-get="/api/gen/models/files" hx-trigger="load">` that loads the datalists once, and Save and Cancel buttons. On success (`hx-on::after-request` with `event.detail.successful`), it closes the dialog.
5. **Models tab.** In `gen_model_sets.html`, remove the `<details>` form. Below the table, add an "Add a custom set" button that loads `/api/gen/models/sets/form` into `#gen-set-form-body-models` and opens `<dialog id="gen-set-form-dialog-models">`, as the New instance dialog does. The `-models` suffix leaves room for the Download Models tab's copy of the dialog in Phase 06. Keep the "or add one from files you already have" text and make "add one" open the dialog.
6. **`setFromForm(r, store) (ModelSet, error)`** in `gen_set_form.go` holds the parsing now in `handleGenCreateSet`: name and ID, kind, recipe, sizes, max fields, scale, stem tasks, launch args (`ValidateExtraArgs`, `CheckLoraArgs`), components, `ValidateComponents`, `CheckFiles`, `CheckFileFormat`, `LoraDir`, capabilities. Components come from the `slot_<slot>` fields; empty ones are skipped. When the old `files` field is sent and no `slot_` field is, parse `files` as today, so scripts that use it keep working. When both are sent, return "send the files as slot_<name> fields or as the files list, not both". `handleGenCreateSet` calls `setFromForm`, then stores the set as today.
7. **Missing paths.** Change `CheckFiles(comps []Component)` to `CheckFiles(comps []Component, modelsDir string, inContainer bool)`. When `os.Stat` gives `os.ErrNotExist`, the error is "<slot>: <path> does not exist. The models folder is <modelsDir>." In a container, add: "Toolchest runs in a container and cannot see host folders. Put the file under <modelsDir>, or mount its folder there." Get `inContainer` from `s.inContainer()` (`internal/api/build.go`). Update every caller.
8. **Format check.** Add `CheckFileFormat(slot, path string) error` in `recipes.go`. Read the first 16 bytes. GGUF: starts with `GGUF`. safetensors: the first 8 bytes are a little-endian header length `n` with `2 <= n < file size`, and byte 8 is `{`. PyTorch: starts with `PK\x03\x04` (zip) or `\x80` (pickle). JSON: the first byte that is not white space is `{`. The `tokenizer` slot must be JSON. Every other slot must be GGUF, safetensors or PyTorch. The error is "<slot>: <path> is not a model file (GGUF, safetensors or PyTorch)" or "<slot>: <path> is not a tokenizer JSON file". It does not run for sets loaded from the store, only in `setFromForm`.
9. **File list.** `ListModelFiles(dir, 5000)` walks `dir` with `filepath.WalkDir`. It skips folders whose name starts with `.`, does not follow symlinked folders, and does not read files. Model files end in `.gguf`, `.safetensors`, `.pth` or `.ckpt`, and not in `.part`. Tokenizer files are `.json` files whose name contains `tokenizer`. Both lists are full paths, sorted, together at most `max`; `truncated` is true when it stopped early. The handler uses `generation.ModelsDir(s.cfg)`.

## Build gate

- `go build ./...`
- `go vet ./...`
- `go test ./...`
- `make js-test`

## Test plan

- `GET /api/gen/models/sets/form/fields?kind=video&recipe=wan22-i2v` renders fields `slot_diffusion_model`, `slot_high_noise_diffusion_model`, `slot_t5xxl`, `slot_vae`, `slot_tae`, `slot_lora`, `slot_high_noise_lora` with their labels, the recipe select lists only video recipes, and the max fields are present. With `kind=upscale`, there is no recipe select, one `slot_upscaler` field and a scale field. With `kind=ace`, the four ace slots and the base DiT checkbox.
- A typed value in `slot_vae` stays after the recipe changes to another recipe that has `vae`.
- POST with `slot_*` fields creates the same set as the old `files` test. POST with only `files` still works. POST with both is refused.
- A missing path gives the "does not exist. The models folder is" message; with the container flag on, the container sentence is added.
- `CheckFileFormat`: small temp files with a GGUF header, a safetensors header, a zip header and a JSON object pass for the right slots; a text file and a JSON file in `diffusion_model` fail; a GGUF file in `tokenizer` fails.
- `ListModelFiles`: a temp tree with model files, a `.part` file, a `config.json`, a `tokenizer.json`, a hidden folder and a symlinked folder returns only the model files and `tokenizer.json`; with `max=2` it sets `truncated`.
- Manual, on asmodean, before merge: add a custom image set through the dialog with the datalist; switch backend and recipe and see the fields change; type a host path and read the message; pick a text file and read the format message.

## Commit

`feat(gen): custom set form with a field per slot, file suggestions and file checks`

## Rollback

Revert the commit. The POST still accepts the old `files` field during this phase, and stored sets do not change, so a revert leaves every saved set working.
