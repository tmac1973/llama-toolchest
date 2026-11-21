# Phase 06 — Field test and closing

**Depends on:** 01, 02, 03, 04, 05 (all merged to `main`) · **Enables:** closing issue #528

## Goal

Check on real hardware that the whole chain works: a new sd build keeps its recipes, the form offers them, a set made from a recipe toolchest did not know before measures and makes an image, and existing sets behave as before. Write the results down, post them on issue #528, and close it.

## Files touched

- `plan/upstream-recipes/field-test.md` — new: the steps below with the results filled in (date, build IDs, pass/fail, notes).
- `docs/guides/generation.md` — fixes found in the test, if any.

## Steps

1. On asmodean (single R9700, live install in `~/llama-toolchest`), deploy a binary built from `main` with phases 01–05, by the usual binary swap. Keep the running binary as `/usr/bin/llama-toolchest.orig`.
2. **Old build, snapshot.** Before building anything new, open Models → Add a custom set. The server log says `recipe list source=snapshot master-1047-984eb06` (the existing sd builds have no `recipes/` folder). The image list has the three groups; `sd35-large` is greyed with the `clip_g` reason.
3. **Existing sets.** Start one catalog image set, one catalog edit set (FLUX.2 klein 4B) and the Wan2.2 Lightning video set on the old build. Each starts, and each makes one image or video as before. Details shows no recipe warning.
4. **New build.** Build the newest stable-diffusion.cpp on the Builds page with the ROCm 10 image (`--rocm-image 10.1.0-full`; 7.x breaks builds on asmodean). Check that the new build's folder (`<data folder>/builds/<new ID>/recipes/`) has as many `.json` files as the checkout's `examples/server/recipes/`. The server log now says `recipe list source=<new build ID>`.
5. **New recipe.** Make a custom set from a recipe toolchest did not know before 01: `sd-turbo` (one `model` file; use the fp16 safetensors from `stabilityai/sd-turbo`, put in `~/models/gen-test`). Check the form prefilled the steps, CFG and sampler from the recipe, and the size. Save, measure, start, and make one 512×512 image through `/v1/images/generations`.
6. **Context flags.** Open Add a custom set (a new set, not a copy), choose recipe `qwen-image-edit-2511`, and check that the launch-argument field holds `--model-args qwen_image_zero_cond_t=true`, from the recipe's `context`. Close the form without saving.
7. **Start check.** Copy the new build's `recipes/` folder into the old build's folder, then delete `sd-turbo.json` from that copy. Restart llama-toolchest (this clears the per-build cache). Set the `sd-turbo` instance to the old build explicitly and start it. The start is refused with "build <old ID> does not know recipe sd-turbo …". Set the instance back to Automatic, delete the copied folder, and restart again.
8. **Delete the new build.** The log goes back to the old build's list (the snapshot). The `sd-turbo` set stays in the list and still starts (it is in the snapshot).
9. Write `field-test.md`, post a short summary on issue #528 (ask the user before posting), and close the issue when the user agrees.

## Build gate

```
go build ./...
go vet ./...
go test ./...
make js-test
make docs
```

## Test plan

The steps above are the test plan. Each step passes when what it describes is seen; `field-test.md` records each result.

## Commit

`docs(plan): field test of recipes from stable-diffusion.cpp`

## Rollback

Nothing to roll back in code. On asmodean, put `/usr/bin/llama-toolchest.orig` back if the test build misbehaves.
