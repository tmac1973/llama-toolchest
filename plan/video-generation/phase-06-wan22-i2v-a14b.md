# Phase 06 — Wan2.2 I2V A14B Set (Two Experts, Start and End Frames)

**Depends on:** Phase 01 (the `high_noise_diffusion_model` slot and the
`wan22-i2v` recipe), Phase 02 (Measure for video), Phase 03 (the job API and
its start/end checks), Phase 04 (the proven pipeline and `verification.md`),
Phase 05 (measure presets and measured limits) · **Enables:** Phase 07, which
adds its set after this one and checks the overview's success criteria,
including this set's results

## Goal

The catalog offers Wan2.2 I2V A14B: image-to-video and start-and-end videos
from two 14B experts, one for high noise and one for low noise. This is the
first set with two diffusion models, and the first that needs a start image
for every video. The set is measured and checked end to end on asmodean,
and tried on a 16 GB A4000, with the results written down.

## Files touched

- `internal/generation/catalog.go`: the Wan2.2 I2V A14B entry; a shared
  `wan21VAE` component.
- `internal/generation/catalog_test.go`: the new set shares the T5 file with
  TI2V 5B; `QuantStack/Wan2.2-I2V-A14B-GGUF` and
  `Comfy-Org/Wan_2.1_ComfyUI_repackaged` join the allowed-repos list.
- `plan/video-generation/verification.md`: an I2V A14B section.
- `web/templates/help.html`: a reservations-table row, and the I2V A14B line
  in the video jobs part, which says it needs a start image.

## Steps

1. **Catalog entry.** Add after Wan2.2 TI2V 5B:
   - ID `wan22-i2v-a14b`, Name "Wan2.2 I2V A14B (Q4)", Kind `KindVideo`,
     Recipe `wan22-i2v`.
   - Components:
     - `diffusion_model`: `QuantStack/Wan2.2-I2V-A14B-GGUF`,
       `LowNoise/Wan2.2-I2V-A14B-LowNoise-Q4_K_M.gguf`, 9651728896 bytes.
     - `high_noise_diffusion_model`: `QuantStack/Wan2.2-I2V-A14B-GGUF`,
       `HighNoise/Wan2.2-I2V-A14B-HighNoise-Q4_K_M.gguf`, 9651728896 bytes.
     - `t5xxl`: the shared `umt5Q8` from Phase 01.
     - `vae`: the shared `wan21VAE` = `Comfy-Org/Wan_2.1_ComfyUI_repackaged`,
       `split_files/vae/wan_2.1_vae.safetensors`, 253815318 bytes.
   - Capabilities `start_image`, `end_image`. License "Apache-2.0".
   - LaunchArgs: `--steps 20 --cfg-scale 3.5 --flow-shift 5
     --high-noise-cfg-scale 3.5 --high-noise-sampling-method euler
     --high-noise-flow-shift 5 --sampling-method euler --diffusion-fa
     --vae-tiling`. These are the `wan22-i2v` recipe's defaults.
   - DefaultWidth 832, DefaultHeight 480, DefaultFrames 33, DefaultFPS 16,
     MaxFrames 81, MaxWidth 1280, MaxHeight 720 (81 frames is 5 s at 16 fps).
   - VideoSizes (Phase 05): Short 33 frames at 832×480; Medium 49 frames at
     1280×720; Full 81 frames at 1280×720.
   - Notes: "Image-to-video and start-and-end videos; needs a start image. Two
     14B experts (high and low noise); 20 steps. Up to 5 seconds at 1280×720.
     Shares its text model with Wan2.2 TI2V 5B." The measured figures are added
     in step 5.
2. **Test.** In `catalog_test.go`, check that `wan22-i2v-a14b` and
   `wan22-ti2v-5b` have the same `t5xxl` component (one download), and that
   `wan22-i2v-a14b` has both diffusion slots.
3. **asmodean.** Deploy, download the set, and measure it alone on GPU 0 at
   the Short preset (Measure uses a drawn start and end picture). If sd-server
   crashes in the VAE decode, as the Wan2.2 VAE did with `--vae-tiling` in
   Phase 01, remove `--vae-tiling` from the entry's LaunchArgs with a comment,
   as for TI2V 5B, and measure again. Through the job API, run:
   1. Image-to-video at the defaults, WebM.
   2. Start-and-end, WebM.
   3. Start-and-end as animated WebP.

   Then measure at the Full preset, which sets the reservation and the
   instance's limit to Full, and run:
   4. 1280×720 at 81 frames, watching that the card stays within the
      reservation.

   Also check that a request without a start image is refused with
   `start_image_required`, and that after the Short measurement a Full-size
   request is refused with `over_measured_size`. Record the times, peaks,
   reservations and file sizes, and set the entry's `FullMeasureTime` to the
   Full measuring time, in the form "about N minutes on a Radeon AI PRO
   R9700".
4. **compute2 (16 GB A4000).** Measure the set at Short and run job 1 on one
   A4000.
   Record what happens: the peak and time if it works, or the error and where
   it failed if it does not. A failure here is written down, not fixed.
5. **Write it down.** Add an I2V A14B section to `verification.md`; add the
   measured figures to the catalog notes and the help table (the Short reservation in the table row, Full in the note under it, as Phase 05 sets out); in the help page's
   video jobs part, list the set as needing a start image.

## Build gate

- `go build ./...`
- `go vet ./...`
- `go test ./internal/...`
- `gofmt -l internal/` prints nothing.

## Test plan

- `TestCatalogEntriesAreValid` and the new sharing test pass.
- The asmodean runs in step 3 all succeed, and the start-image refusal works.
- The compute2 result is recorded either way.

## Commit

`feat(generation): add the Wan2.2 I2V A14B video set`

## Rollback

Revert the commit. Downloaded copies of the set stay on disk as a set whose
catalog entry no longer exists: it keeps working with its own saved launch
settings and can be deleted on the Models page. Nothing else changes.
