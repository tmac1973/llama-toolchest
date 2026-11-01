# Phase 02 — Flag reasons, slot labels and custom set limits

**Depends on:** nothing · **Enables:** Phase 03 (details use the reasons and labels), Phase 05 (form uses the labels and the new set fields)

## Goal

Put the data this plan needs into `internal/generation`, with no UI yet. Every flag of every catalog entry gets a short reason. Every slot gets a label. Video recipes get the `size_multiple` from upstream. A model set can hold its own upscale factor. Custom sets then give `scale` and `size_multiple` in `/v1/models`, as catalog sets do.

## Files touched

- `internal/generation/flag_reasons.go` (new) — `FlagReasons` shared table and `FlagReason(flag, value string, entry CatalogEntry) string`.
- `internal/generation/catalog.go` — `FlagNotes map[string]string` on `CatalogEntry`; notes for the set-specific flags; `UpscaleScale` and `MadeSize` fall back to the set's own values.
- `internal/generation/recipes.go` — `SlotLabels`, `RecipeSlotLabels`, `SizeMultiple` on `VideoRecipe`, `RecipeSizeMultiple(recipe string) int`.
- `internal/generation/types.go` — `Scale int` on `ModelSet`.
- `internal/generation/catalog_test.go` — the reason coverage test, and tests for the fallbacks.
- `internal/api/gen_models.go` — the custom set POST reads `scale` (upscaler sets) and `max_frames`, `max_width`, `max_height` (video sets) from the form.
- `internal/api/v1models.go` — `meta.video.size_multiple` (around line 198) comes from a new `ModelSet.SizeMultiple()` method, not from the catalog entry only, so custom sets get it. `meta.scale` already uses `UpscaleScale()`.
- `web/templates/partials/gen_model_sets.html` — three optional video fields and an upscale factor field in the existing form, so the API and the form match until Phase 05 replaces the form.

## Steps

1. **Flag reasons.** In `flag_reasons.go`, add `var FlagReasons = map[string]string{...}`. The key is the flag, or `flag value` when the reason depends on the value (`--backend te=cpu`, `--sampling-method euler`). Write one reason for every flag that appears in `Catalog()` launch args, and for toolchest's own fixed flags: `--lora-model-dir`, `--hires-upscalers-dir`, `--max-batch`, `--max-vram`, `--listen-ip`, `--listen-port`, `--host`, `--port`, `--models`. Take the text from the comments in `catalog.go`, `launch.go` and `help.html`. Each reason is one or two short sentences in ASD-STE100 style, for example: `--backend te=cpu` → "Runs the text model on the CPU. This saves 4 to 6 GiB of VRAM and adds a second or two to each job."
2. Add `FlagNotes map[string]string` (JSON `flag_notes,omitempty`) to `CatalogEntry`, with the same key rule. Move set-specific comments into it: LTX-2.3 `--temporal-tiling` (spatial tiling hung on ROCm), Wan2.2 A14B Lightning `--steps`/`--high-noise-steps` (4 + 4 was as sharp as 20 steps; 2 + 2 smeared faces), Ming-Image `--backend te=cpu` (the measured 18.6 → 9.4 GiB), Qwen-Image 2.1 Turbo `--sigmas`. Keep the Go comments short; the reason text now lives in the data.
3. Add `FlagReason(flag, value string, entry CatalogEntry) string`: first `entry.FlagNotes["flag value"]`, then `entry.FlagNotes[flag]`, then `FlagReasons["flag value"]`, then `FlagReasons[flag]`, else "". A custom set calls it with a zero `CatalogEntry`.
4. Add a helper `SplitFlags(args []string) [][2]string` that pairs each `--flag` with its value (the next word when it does not start with `--`), so details and tests read flags the same way.
5. **Slot labels.** In `recipes.go`, add `var SlotLabels = map[string]string` with a label for every slot in `sdSlotFlags`, `LoraSlots`, `AceSlots` and `UpscalerSlot`, for example `diffusion_model` → "Diffusion model", `llm` → "Text model", `llm_vision` → "Picture reader (for edits)", `t5xxl` → "T5 text encoder", `high_noise_diffusion_model` → "High-noise expert", `high_noise_lora` → "High-noise LoRA", `upscaler` → "ESRGAN upscaler". Add `var RecipeSlotLabels = map[string]map[string]string` for the labels upstream gives per recipe (copy them from the recipe JSON files at the commit named in the `RecipeSlots` comment, for example `wan22-i2v`: `diffusion_model` → "Low-noise expert", `t5xxl` → "UMT5-XXL"). Add `SlotLabel(recipe, slot string) string`: the recipe's label, else the shared label, else the slot name.
6. **size_multiple.** Add `SizeMultiple int` to `VideoRecipe`, with the values from the upstream recipe JSON files (sd.cpp `e41fbcf`, read 2026-11-01): 32 for `wan22-ti2v-5b`, `ltx23-distilled`, `ltx23-dev` and `ltx25-dev`; 16 for `wan22-i2v`, `wan22-t2v`, `wan21-t2v`, `wan21-i2v`, `wan21-flf2v`, `hunyuan-video-15-t2v`, `hunyuan-video-15-i2v`, `lingbot-video-dense` and `lingbot-video-moe`. Add `RecipeSizeMultiple(recipe string) int`. Add `ModelSet.SizeMultiple() int` in `catalog.go`, next to `MadeSize`: the catalog entry's value first, then `RecipeSizeMultiple(m.Recipe)`, then 0. `MadeSize` and `internal/api/v1models.go` both use it.
7. **Scale.** Add `Scale int` (JSON `scale,omitempty`) to `ModelSet`. `UpscaleScale()` returns the catalog entry's value for a catalog set, else `m.Scale`, else 0. Old custom upscaler sets have no `Scale`, so they keep 0 as today.
8. **Form and POST.** In `handleGenCreateSet`, read `scale` for upscaler sets (`formInt(r, "scale", 4)`), and `max_frames`, `max_width`, `max_height` for video sets (zero when empty). Refuse a max width or height that is not a multiple of 8, and a test size larger than the max size, with the same wording as `videoMeasureSize`. Add the four fields to the form in `gen_model_sets.html`, with short tooltips.

## Build gate

- `go build ./...`
- `go vet ./...`
- `go test ./...`

## Test plan

- `TestCatalogFlagsHaveReasons`: for every catalog entry and every flag pair from `SplitFlags(entry.LaunchArgs)`, `FlagReason` is not empty. The test names the entry and the flag that fails.
- `TestFixedFlagsHaveReasons`: the toolchest fixed flags from step 1 have reasons.
- Every slot in every `RecipeSlots` and `VideoRecipes` entry, plus `AceSlots` and `UpscalerSlot`, has a `SlotLabel` other than the slot name.
- Every `VideoRecipes` entry has `SizeMultiple > 0`.
- A custom upscaler set with `Scale: 2` gives `UpscaleScale() == 2`; one with no `Scale` gives 0. A catalog upscaler set still gives 4.
- A custom `wan22-i2v` set gives `size_multiple` 16 in `/v1/models`, and `MadeSize(830, 470)` gives 832×480.
- A custom video set POST with `max_frames=81&max_width=1280&max_height=720` stores them, and `/v1/models` shows `max_frames` and `max_size`.

## Commit

`feat(gen): add flag reasons, slot labels and custom set limits`

## Rollback

Revert the commit. Sets saved with `scale` or the max fields keep them in the JSON file. An older binary ignores `scale` and still reads the max fields, which existed before, so the store stays readable.
