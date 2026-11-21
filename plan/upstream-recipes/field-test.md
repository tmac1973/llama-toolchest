# Field test — recipes from stable-diffusion.cpp (#528)

**Date:** 2026-11-21 · **Machine:** asmodean (1× Radeon AI PRO R9700, gfx1201, container on the ROCm 10.1 image) · **llama-toolchest:** `2.62.0-main-8852b25` (main with phases 01–05), swapped into the container; the 2.62.0 release is kept as `/usr/bin/llama-toolchest.orig`.

Old stable-diffusion.cpp builds on the machine: `sd-master-1006-0886f11-rocm` and `sd-master-1015-e41fbcf-rocm` (neither has a `recipes/` folder). New build made in the test: `sd-master-1106-532d2c1-rocm`.

| Step | Result | Notes |
| --- | --- | --- |
| 1. Deploy | Pass | Binary swap and container restart. |
| 2. Old build, snapshot | Pass | Log: `recipe list source="snapshot master-1047-984eb06" recipes=108`. The image select has the three groups; `sd35-large` is disabled with "needs the CLIP-G text encoder file, which llama-toolchest cannot pass to sd-server". `/api/gen/models`: 79 image, 14 video, 15 unavailable. |
| 3. Existing sets | Pass | Ming Image (catalog image set): a 512×512 image in 18 s. klein (FLUX.2 klein 4B): one edit of that image in 14 s. The plan named the Wan2.2 Lightning set, but asmodean has no instance for it, so the video check used `ti2v` (Wan2.2 TI2V 5B, catalog): a 480×320, 17-frame video in 38 s. Details shows no recipe warning for any of the sets. |
| 4. New build | Pass | Built with the latest commit on ROCm 10.1. `builds/sd-master-1106-532d2c1-rocm/recipes/` has 112 `.json` files, the same as the checkout's `examples/server/recipes/`. Log: `recipe list source=sd-master-1106-532d2c1-rocm recipes=112`. |
| 5. New recipe | Pass | `sd-turbo` with `stabilityai/sd-turbo` `sd_turbo.safetensors` (in `/home/tim/llm/gen-test`, which the container sees as `/data/models/gen-test`). The form prefilled `--steps 1 --cfg-scale 1 --sampling-method euler --scheduler discrete` and 512×512. Measure: peak 4,826 MiB, reservation 5,632 MiB. sd-server ran with the prefilled flags; one 512×512 image in 1.7 s, and the image looks right. |
| 6. Context flags | Pass | A new form with `qwen-image-edit-2511` holds `--steps 20 --cfg-scale 4 --sampling-method euler --flow-shift 3 --model-args qwen_image_zero_cond_t=true`, with the "From the recipe's defaults" note. |
| 7. Start check | Pass | With a copy of the new recipes (less `sd-turbo.json`) in the old build's folder and the instance set to that build, the start is refused: "build sd-master-1015-e41fbcf-rocm does not know recipe sd-turbo, which sd-turbo-test uses. Choose a newer stable-diffusion.cpp build for the instance, or build one on the Builds page". See note 2. Then: build back to Automatic, copy deleted, restart. |
| 8. Delete the new build | Pass | Log: `recipe list source="snapshot master-1047-984eb06" recipes=108`. The `sd-turbo-test` set has no warning; on `sd-master-1015-e41fbcf-rocm` it measured (peak 3,923 MiB, reservation 4,608 MiB), started and made a 512×512 image in 1.9 s. |

## Notes

1. **New upstream recipes work without a toolchest update.** Build 1106 has four recipes the snapshot does not have: `minimax-h3-image`, `p-h3-image` (image) and `hyperflow-h3-flash-fl2va`, `hyperflow-h3-flash-ref2va` (video). All four were offered in the form while the new build was the newest. None of the 112 files has a `defaults` or `context` key that toolchest does not know. `minimax-h3-fl2va` and `minimax-h3-ref2va` also changed upstream since the snapshot.
2. **Order of the start checks.** Changing an instance's build clears its reservation, so the first start after the change was refused with "has no VRAM reservation yet; press Measure". That check comes before the recipe check. To see the recipe refusal, the reservation was set by hand first. A user who measures first would get the recipe refusal from Measure's own start. Nothing to change.
3. **Peak VRAM differs between builds.** sd-turbo peaked at 4,826 MiB on build 1106 and 3,923 MiB on build 1015, with the same flags. This is why a build change clears the measurement.

## Left on asmodean

- llama-toolchest runs `2.62.0-main-8852b25`; the release is `/usr/bin/llama-toolchest.orig`.
- The `sd-turbo-test` custom set and instance (stopped) and `/home/tim/llm/gen-test/sd_turbo.safetensors` (5.2 GB) stay for later tests.
- The new build `sd-master-1106-532d2c1-rocm` was deleted as part of step 8.
- Instances are as before the test: Ming Image running, the others stopped.
