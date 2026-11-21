# Phase 01 — Recipe book, embedded snapshot and overlay

**Depends on:** nothing · **Enables:** 02 (the parser reads build folders), 03 (the registry that 03 swaps), 04 and 05 (recipe names, descriptions, reasons, defaults)

## Goal

Replace the hand-written recipe tables with upstream's own recipe files. Toolchest gets a parser for `examples/server/recipes/*.json`, a snapshot of those files embedded in the binary, a small Go overlay for what upstream does not say, and a package-level recipe book that every current caller reads through the existing function names. After this phase the custom set form offers every snapshot recipe that toolchest can run. Recipes it cannot run are left out of the list for now (phase 04 shows them greyed). Existing catalog and custom sets validate and behave as before.

## Files touched

- `internal/generation/sdrecipes/*.json` — new: upstream's recipe files, copied from stable-diffusion.cpp `master-1047-984eb06` (108 files; `README.md` and `defaults.md` are not copied).
- `internal/generation/sdrecipes/SOURCE` — new: one line, `stable-diffusion.cpp master-1047-984eb06 <full sha> <date>`.
- `internal/generation/recipebook.go` — new: `Recipe`, `RecipeBook`, `ParseRecipes`, `LoadRecipes`, the embedded snapshot, the registry (`ActiveRecipes`, `SetActiveRecipes`).
- `internal/generation/recipe_overlay.go` — new: the toolchest overlay (labels, extra slots, capabilities, edit limits, left-out recipes).
- `internal/generation/recipes.go` — remove `RecipeSlots`, `VideoRecipes`, `RecipeSlotLabels` and the slot helper functions (`wanSlots`, `wanMoESlots`, `ltxSlots`, `hunyuanSlots`, `llmVideoSlots`) and the label vars; `RecipeSlotSet.OneOf` becomes `OneOf [][]string`; `Recipes`, `VideoRecipeIDs`, `RecipeSizeMultiple`, `SlotLabel`, `RecipeCapabilities`, `ValidateComponents` read the active book. `RecipeMaxEditImages` moves to the overlay file with the same name.
- `internal/generation/details.go` — the slot lookup at lines 155–163 reads the active book.
- `internal/generation/testdata/recipes/*.json` — new: a few real upstream files as parser fixtures (`flux1-dev`, `qwen-image-edit-2511`, `wan22-t2v`, `animatediff-sd15`, `flux1-canny`, `pid`) plus two hand-made files: one with `schema_version: 2`, one with an optional slot that has no sd-server flag.
- `internal/generation/recipebook_test.go` — new: parser, overlay, availability, compatibility tests.
- `internal/generation/catalog_test.go` — `TestRecipeCapabilities` and `TestRecipesOrder` updated for the book; a new test that every catalog recipe is in the snapshot and available.
- `scripts/update-sd-recipes/main.go` — new: copies `examples/server/recipes/*.json` from a given checkout into `internal/generation/sdrecipes/` and writes `SOURCE`.
- `Makefile` — new target `update-sd-recipes` (`go run ./scripts/update-sd-recipes $(SD)`), with a comment like the one on `scrape-sampling-presets`.
- `docs/guides/generation.md` — one sentence in the custom set section: the recipe list comes from stable-diffusion.cpp's own recipe files.

## Steps

1. Copy the 108 JSON files from the local checkout (`~/.local/share/llama-toolchest/stable-diffusion.cpp/examples/server/recipes/`, at `master-1047-984eb06`) to `internal/generation/sdrecipes/`. Write `SOURCE`. Write `scripts/update-sd-recipes/main.go` first and use it for this copy, so the script is tested by its first run. It takes the checkout path as its only argument, deletes the old `*.json` files in `sdrecipes/`, copies the new ones, reads `git -C <checkout> describe --tags` and `rev-parse HEAD` for `SOURCE`, and fails if the folder has no `*.json` files.
2. In `recipebook.go`, define the file format as Go structs that match upstream: `schema_version`, `id`, `name`, `description`, `slots[] {id, required, label}`, `required_any [][]string`, `defaults map[string]any`, `context map[string]string`, `features []string`, `size_multiple int`, `video_constraints` (kept only as "present or not"). Unknown top-level keys are ignored.
3. Define `Recipe`:
   - `ID`, `Name`, `Description string`
   - `Kind BackendKind` — `KindVideo` when `video_constraints` is present, else `KindSD`
   - `Slots RecipeSlotSet` — required slots in file order, optional slots in file order, `OneOf` from `required_any`
   - `UpstreamLabels map[string]string`
   - `Features []string`, `SizeMultiple int`
   - `Defaults []DefaultFlag` and `Context []DefaultFlag`, where `DefaultFlag` is `{Key, Flag, Value string}` (filled in phase 05; in this phase the parser keeps the raw maps in unexported fields `rawDefaults`, `rawContext`)
   - `Unavailable string` — empty when toolchest can run it, else the reason
4. `ParseRecipes(fsys fs.FS) (*RecipeBook, error)` reads every `*.json` in the root of `fsys`. A file that does not parse, has a `schema_version` other than 1, or whose `id` is not its file name is skipped and logged with `slog.Warn` (file name and reason). It returns an error only when no recipe could be read. `LoadRecipes(dir string)` is `ParseRecipes(os.DirFS(dir))`. The snapshot is `//go:embed sdrecipes/*.json`, read once by `SnapshotRecipes()`; a snapshot that does not parse is a panic at start (a test guarantees it parses).
5. `RecipeBook` holds `map[string]*Recipe` and a `Source string` ("snapshot master-1047-984eb06", or a build ID in phase 03). Methods: `Get(id) (*Recipe, bool)`, `IDs(kind) []string` (catalog recipes first, in catalog order, then the rest A–Z, only available ones — the current `recipeIDs` order), `Unavailable(kind) []*Recipe` (A–Z).
6. The registry: `var activeRecipes atomic.Pointer[RecipeBook]`, set to `SnapshotRecipes()` in `init`. `ActiveRecipes() *RecipeBook` and `SetActiveRecipes(b *RecipeBook)`. Phase 03 calls the setter; in this phase nothing does outside tests.
7. In `recipe_overlay.go`, apply the overlay after parsing, inside `ParseRecipes`, so every book (snapshot or build) gets it:
   - **Extra slots:** `wan22-i2v` and `wan22-t2v` get optional `lora` and `high_noise_lora` (they are toolchest slots, not upstream ones; the Lightning catalog set uses them).
   - **Labels:** the current `RecipeSlotLabels` content moves here unchanged, as `recipeLabels`.
   - **Capabilities:** `textStart` lists the video recipes that also start from a prompt alone: `wan22-ti2v-5b`, `ltx23-distilled`, `ltx23-dev`, `ltx25-dev`, `lingbot-video-dense`, `lingbot-video-moe`. A video recipe without `init_image` always gets text. `alpha` for `ming-image`. `editNeedsSlot` maps `qwen-image-21`, `qwen-image-21-turbo`, `qwen-image-edit-2511` to `llm_vision`.
   - **Edit limits:** `RecipeMaxEditImages`, unchanged.
   - **Left out on purpose:** `wan21-vace` ("needs control video frames, which llama-toolchest does not send") and `wan22-s2v` ("needs an audio input, which llama-toolchest does not send").
8. Availability, set in `ParseRecipes` after the overlay, first match wins:
   1. a left-out recipe from the overlay → its reason;
   2. a feature in `needsInput` (`control_image`, `control_frames`, `vace`, `ref_audios`, `ref_videos`, `ref_video_audios`) → "needs a <what> input, which llama-toolchest does not send", with `<what>` from a small map (control image, control video frames, audio, reference video);
   3. a **required** slot with no entry in `sdSlotFlags` and not a LoRA slot → "needs a <SlotLabel> file, which llama-toolchest cannot pass to sd-server";
   4. a `context` key other than `vae_format` and `model_args` → "uses a recipe setting (<key>) that llama-toolchest cannot pass to sd-server".
   An **optional** slot with no flag is dropped from `Slots.Optional`, so the form never shows it.
9. Rewrite the callers on top of the book, keeping their names and signatures:
   - `Recipes()` → `ActiveRecipes().IDs(KindSD)`; `VideoRecipeIDs()` → `IDs(KindVideo)`.
   - `RecipeSizeMultiple(recipe)` → the recipe's `SizeMultiple` when it is a video recipe, else 0.
   - `SlotLabel(recipe, slot)` → `recipeLabels`, then the recipe's `UpstreamLabels`, then `SlotLabels`, then the slot ID.
   - `RecipeCapabilities(recipe, comps)` → video: `start_image` from `init_image`, `end_image` from `end_image`, `text` per step 7; image: `edit` from `ref_images` unless `editNeedsSlot` names a slot that `comps` lacks, plus `alpha` for `ming-image`. Order of the result: `text`, `start_image`, `end_image` for video (the order `VideoRecipes` used); `edit` or `alpha` for image.
   - `ValidateComponents` → looks the recipe up in the active book. The current error texts stay. A recipe that exists but is unavailable gives `recipe <id> cannot be used: <reason>`. `OneOf` is checked per group.
   - `details.go` slot lookup → the book's `Slots` for the recipe, empty when unknown.
10. Write the compatibility test before deleting the old tables: copy the current `RecipeSlots`, `VideoRecipes` (slots, caps, size multiple) and `RecipeCapabilities` cases into `recipebook_test.go` as `legacyRecipes`, and assert that the snapshot book gives the same required slots, the same optional slots (as sets), the same `OneOf` group, the same size multiple and the same capabilities for each of the 25 recipes. A difference is allowed only when the snapshot file itself says otherwise (for example a slot that upstream made optional since `master-1000`); any other difference is a bug in the parser or overlay and is fixed. Each allowed difference is listed in the test by recipe and slot, with the upstream file content as the comment, and the pull request text lists them all for the user to approve before merge. Then delete the old tables.
11. Add one sentence to `docs/guides/generation.md` in the "Use your own files" item: the recipe list comes from stable-diffusion.cpp's own recipe files (a copy built into llama-toolchest). Phase 03 extends this sentence when builds bring their own recipes.

## Build gate

```
go build ./...
go vet ./...
go test ./internal/generation/... ./internal/api/...
make js-test
```

## Test plan

- Parser: each fixture parses to the expected kind, slots, `OneOf`, labels, features, size multiple; the `schema_version: 2` file is skipped with a warning; a file whose `id` differs from its name is skipped; an empty folder is an error.
- Snapshot: `SnapshotRecipes()` parses all 108 files without a skip.
- Availability: `animatediff-sd15` (required `motion_module`) and `flux1-canny` (`control_image`) are unavailable with the expected reasons; `wan21-vace` and `wan22-s2v` give the overlay reasons; `sd35-large` (required `clip_g`) is unavailable; a hand-made fixture with an optional unknown slot is available and the slot is gone (no snapshot recipe has one at `master-1047`).
- Overlay: `wan22-t2v` has `lora` and `high_noise_lora`; `qwen-image-21` edits only with `llm_vision`; `ming-image` has `alpha`; `ltx23-dev` has `text`, `start_image`, `end_image`; `wan22-t2v` has only `text`.
- Compatibility: the 25 legacy recipes match (step 10).
- Catalog: every `Catalog()` entry's recipe is in the snapshot, available, and of the entry's kind; `TestCatalogEntriesAreValid` still passes.
- Order: `TestRecipesOrder` still puts catalog recipes first.
- Manual: `make run`, open Models → Add a custom set, check that new recipes (for example `ernie-image`, `pixart-alpha-1024`) are in the image list and `animatediff-sd15` is not in the video list.

## Commit

`feat(generation): read sd-server recipes from upstream recipe files`

## Rollback

Revert the commit. It restores the Go tables and removes the snapshot, the parser and the overlay. No stored data changes: sets keep their recipe ID as before, so nothing needs migrating either way.
