# Phase 05 — Prefill the form from upstream defaults

**Depends on:** 01 (the parser keeps `rawDefaults` and `rawContext`), 03 (active book), 04 (the recipe select and its redraw) · **Enables:** 06 (field test)

## Goal

When the user picks a recipe for a new custom set, the form fills in what upstream says the recipe needs: steps, CFG, sampler, scheduler, flow shift and the other defaults as sd-server flags in the launch-argument field, the size (and frames and FPS for video) in the test-size fields, and the recipe's `context` settings (`--vae-format`, `--model-args`). sd-server applies a recipe's defaults and context only when it loads the recipe itself, and toolchest passes files by flag, so without this a new recipe can run with wrong settings, or (for Qwen-Image-Edit 2511, PixArt and PID) give poor images.

## Files touched

- `internal/generation/recipe_defaults.go` — new: the key → flag table, `Recipe.LaunchDefaults() []string`, `Recipe.SizeDefaults() (width, height, frames, fps string)`.
- `internal/generation/recipebook.go` — the parser fills `Recipe.Defaults` and `Recipe.Context` from the raw maps through the table.
- `internal/api/gen_set_form.go` — the prefill rule on a recipe change; a hidden field remembers what was filled.
- `web/templates/partials/gen_set_form.html` — hidden fields `prefill_args`, `prefill_width`, `prefill_height`, `prefill_frames`, `prefill_fps`; a short note under the launch-argument field when it was filled.
- `internal/generation/recipe_defaults_test.go`, `internal/api/gen_set_form_test.go` — tests.
- `docs/guides/generation.md` — one sentence: a new set's settings start from the recipe's own defaults.

## Steps

1. The table in `recipe_defaults.go`, in this order (the order of the flags in the result):
   - launch flags from `defaults`: `steps` → `--steps`; `cfg_scale` → `--cfg-scale`; `img_cfg_scale` → `--img-cfg-scale`; `distilled_guidance` → `--guidance`; `sampling_method` → `--sampling-method`; `scheduler` → `--scheduler`; `flow_shift` → `--flow-shift`; `shifted_timestep` → `--timestep-shift`; `custom_sigmas` → `--sigmas` (a JSON list written comma-separated); `strength` → `--strength`; `high_noise_steps` → `--high-noise-steps`; `high_noise_cfg_scale` → `--high-noise-cfg-scale`; `high_noise_sampling_method` → `--high-noise-sampling-method`; `moe_boundary` → `--moe-boundary`.
   - launch flags from `context`: `vae_format` → `--vae-format`; `model_args` → `--model-args`.
   - size fields from `defaults`: `width`, `height`, `video_frames`, `fps`.
   - skipped on purpose (`skippedDefaults`, with a comment for each): `high_noise_flow_shift` (sd-server has no flag for it; `--flow-shift` covers both experts).
2. Number formatting: integers without a decimal point; floats with Go's `strconv.FormatFloat(v, 'f', -1, 64)`; strings as they are. A value is passed as one word; `ValidateExtraArgs` already splits and checks the field, so a value with a space (none today) is quoted.
3. In the parser, keys in neither the table nor `skippedDefaults` are kept out of the result and logged once per key at `slog.Debug`. The first test in the Test plan makes a new upstream key fail, so it is seen at the next snapshot refresh.
4. `Recipe.LaunchDefaults()` returns the flags as a word list (for example `["--steps", "20", "--cfg-scale", "1", "--guidance", "3.5", "--sampling-method", "euler"]`). `Recipe.SizeDefaults()` returns the four size values as strings, empty when absent; frames and FPS only for video recipes.
5. The prefill rule in `genSetFormFrom`, only for a new set (no `CopiedFrom`, no `EditID`):
   - For each of the five fields (launch args, width, height, frames, FPS): if the field is empty, or its value equals the matching `prefill_*` hidden value (the user has not changed what was filled), set it to the new recipe's value and set the hidden value to it too.
   - Else leave the field as the user typed it, and leave its hidden value empty, so a later recipe change does not touch it either.
   - The launch-arg value is the words joined with single spaces, the same way the form shows a set's flags today.
   - When the launch-arg field was filled, show `<small>From the recipe's defaults. Change or clear them as you need.</small>` under it.
6. The first draw of a new form (the default recipe of the kind) is prefilled in the same way, so the user sees the defaults at once.
7. Docs (see Files touched).

## Build gate

```
go build ./...
go vet ./...
go test ./internal/generation/... ./internal/api/...
make js-test
make docs
```

## Test plan

- Table: every `defaults` and `context` key in the embedded snapshot is either in the table or in `skippedDefaults` (this is the test that catches new upstream keys).
- `flux1-dev`: `--steps 20 --cfg-scale 1 --guidance 3.5 --sampling-method euler`, size 1024×1024.
- `qwen-image-edit-2511`: the flags include `--model-args qwen_image_zero_cond_t=true`.
- `pid`: the flags include `--vae-format flux`.
- `wan22-t2v`: frames 33, FPS 16, `--high-noise-steps -1`, `--moe-boundary 0.875`; no flag for `high_noise_flow_shift`.
- `qwen-image-21-turbo`: `--sigmas` with the comma-separated list.
- Form: new set, first draw → prefilled; recipe A → B with untouched fields → B's values; user edits the steps in the args field, then A → B → the args field is kept, the size fields change; copy and edit never prefill.
- Every `LaunchDefaults()` of every available snapshot recipe passes `ValidateExtraArgs`. This test must pass as written: a failure means a wrong table entry or number format, which is fixed in the table; no recipe or key is skipped to make it pass.
- Manual: in the browser, pick `pixart-alpha-1024`, check the args field holds its `--model-args`; change the steps, pick another recipe, check the args stay.

## Commit

`feat(ui): prefill a new custom set from its recipe's defaults`

## Rollback

Revert the commit. New sets start with empty fields again, as before. Sets saved with prefilled flags keep them; they are normal flags the user could have typed.
