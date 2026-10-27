# Phase 08 — Wan2.2 I2V A14B Lightning Set (4-Step LoRAs)

**Depends on:** Phase 06 (the Wan2.2 I2V A14B set, its expert files and the
`--temporal-tiling` choice) · **Enables:** nothing; Phase 07 does not need it

## Goal

The catalog offers a second Wan2.2 I2V A14B set that uses the Lightning
4-step LoRAs, one for each expert. A Short video then takes about 3.5
minutes on the R9700 instead of about 27. The 20-step set stays for those
who want it. To do this, toolchest learns to ship LoRA files with a set:
two new component slots, `--lora-model-dir` at launch, and a `lora` list
in every request to sd-server, Measure's test video included.

## Background (checked 2026-10-26, sd.cpp `master-1015-e41fbcf`)

- sd-server ignores `<lora:…>` tags in prompts. LoRAs are named per
  request: `"lora": [{"path", "multiplier", "is_high_noise"}]`. A LoRA with
  `is_high_noise: true` goes on the `--high-noise-diffusion-model` only;
  with `false`, on the `--diffusion-model` only.
- `path` must be a file sd-server found in `--lora-model-dir`, relative to
  that folder. sd-server skips links that point outside the folder, so the
  files cannot be linked into the instance's own folder the way the ACE
  files are. `--lora-model-dir` must be the folder the files are in.
- The official `lightx2v/Wan2.2-Lightning` I2V files (F32 weights, I64
  `alpha` values) crash sd-server when they load. Kijai's fp16 copies of
  the same weights work.
- Straight to sd-server on asmodean, Short size, `--max-vram 28.5`:

  | Run | Time | Whole-card peak |
  |---|---|---|
  | 20 steps, CFG 3.5, no LoRA (Phase 06) | 1,643 s | 27.9 GB; Measure 29.0 GiB |
  | LoRAs, 4 + 4 steps, CFG 1 | 200 s | 31.3 GB (29.1 GiB) |
  | LoRAs, 2 + 2 steps, CFG 1 | 110 s | 29.5 GB (27.5 GiB) |

  4 + 4 looked about as sharp as the 20-step video. 2 + 2 was softer and
  smeared the face in the last frames. This set uses 4 + 4.

## Files touched

- `internal/generation/types.go`: the `Component.Slot` comment names the
  two LoRA slots.
- `internal/generation/recipes.go`: `lora` and `high_noise_lora` are
  optional slots of `wanMoESlots()`; a `LoraSlots` list; `LoraDir` and
  `Loras` helpers for a set.
- `internal/generation/launch.go`: LoRA slots get no flag of their own; a
  set with LoRAs gets one `--lora-model-dir`.
- `internal/generation/catalog.go`: the Lightning entry; shared
  components for the two experts; a pointer to the new set in the 20-step
  set's notes.
- `internal/api/gen_video.go`: a `Lora` field on `vidGenRequest`.
- `internal/api/gen_video_jobs.go`, `internal/api/gen_vram.go`: video jobs
  and Measure's test video send the set's LoRAs; `lora` is reserved in
  `extra` for a set that has LoRAs.
- Tests: `internal/generation/catalog_test.go`,
  `internal/generation/video_test.go`, `internal/generation/launch_test.go`,
  `internal/api/gen_video_test.go`, `internal/api/gen_video_jobs_test.go`.
- `plan/video-generation/verification.md`: a Phase 08 section.
- `web/templates/help.html`: a reservations-table row and a line in the
  video jobs part.

## Steps

1. **LoRA slots.** In `recipes.go`, add `lora` and `high_noise_lora` to
   the `Optional` slots of `wanMoESlots()`. They are the only recipes with
   two experts, which the Lightning LoRAs need. Other recipes keep refusing
   the slots through `ValidateComponents`, as today. Add:
   - `LoraSlots = map[string]bool{"lora": false, "high_noise_lora": true}`:
     the LoRA slots, and whether each one is for the high-noise expert.
   - `func (m ModelSet) Loras() []Component`: the set's LoRA components.
   - `func LoraDir(set ModelSet) (string, error)`: the one folder all of a
     set's LoRA files are in, or an error when they are in different
     folders: "the LoRA files of a set must be in one folder: … and …".
     A catalog set downloads both files to one folder. A custom set made
     from files in two folders is refused at launch with this message.
2. **Launch.** In `LaunchSpec`, for sd and video sets, skip LoRA slots in
   the flag loop (`sdSlotFlags` stays as it is), and add
   `--lora-model-dir <LoraDir>` once when the set has LoRAs. Add
   `--lora-model-dir` to the flags refused in a set's or instance's extra
   arguments only when the set has LoRAs. Without LoRA components a custom
   set may still give its own `--lora-model-dir` and name LoRAs in
   `extra`, as it can today.
3. **Requests.** In `gen_video.go`, add to `vidGenRequest`:
   ```go
   // Lora is the set's LoRA files, by name within --lora-model-dir.
   Lora []vidLora `json:"lora,omitempty"`
   ```
   with `vidLora{Path string; Multiplier float64; IsHighNoise bool}`
   (`path`, `multiplier`, `is_high_noise`). One helper,
   `setLoras(set generation.ModelSet) []vidLora`, builds the list: each
   LoRA component's file name (`filepath.Base` of its path, since all of
   them are in the `--lora-model-dir` folder), multiplier 1, and
   `is_high_noise` from `LoraSlots`. Use it where video jobs build their
   request (`gen_video_jobs.go`) and in `runMeasureVideo` (`gen_vram.go`),
   so Measure times and measures the set as it runs. For a set with
   LoRAs, a request whose `extra` sets `lora` is refused with `bad_value`
   ("extra may not set "lora"; this set sends its own LoRAs"). For other
   sets `extra` may still set it.
4. **Catalog entry.** Make the two experts shared components (`a14bLow`,
   `a14bHigh`) used by both A14B entries, so the files are downloaded once.
   Add after Wan2.2 I2V A14B:
   - ID `wan22-i2v-a14b-lightning`, Name "Wan2.2 I2V A14B Lightning (Q4,
     4 steps)", Kind `KindVideo`, Recipe `wan22-i2v`.
   - Components: `a14bLow`, `a14bHigh`, `umt5Q8`, `wan21VAE`, and:
     - `high_noise_lora`: `Kijai/WanVideo_comfy`,
       `LoRAs/Wan22-Lightning/old/Wan2.2-Lightning_I2V-A14B-4steps-lora_HIGH_fp16.safetensors`,
       613561776 bytes.
     - `lora`: `Kijai/WanVideo_comfy`,
       `LoRAs/Wan22-Lightning/old/Wan2.2-Lightning_I2V-A14B-4steps-lora_LOW_fp16.safetensors`,
       613561776 bytes.
     A comment says why these files and not lightx2v's own (those crash
     sd-server master-1015 when they load; these are the same weights in
     fp16), and that the weights are lightx2v's, Apache-2.0.
   - LaunchArgs: `--steps 4 --high-noise-steps 4 --cfg-scale 1
     --high-noise-cfg-scale 1 --flow-shift 5 --sampling-method euler
     --high-noise-sampling-method euler --diffusion-fa --temporal-tiling`.
     No `--moe-boundary`: with both step counts given, the high-noise
     expert takes the first 4 steps and the boundary is not used. A comment
     says 2 + 2 is what Lightning's own workflow uses, but it was softer
     here.
   - Capabilities, License, default and maximum sizes, and VideoSizes as
     for Wan2.2 I2V A14B.
   - Notes: "Wan2.2 I2V A14B with the Lightning 4-step LoRAs: 4 steps on
     each expert, about 3.5 minutes for a 2-second video at 832×480 on a
     Radeon AI PRO R9700, against about 27 for the 20-step set. A little
     less detail. Shares its experts, text model and VAE with Wan2.2 I2V
     A14B." The figures are updated in step 7.
   - The 20-step set's Notes end with: "For faster videos, use Wan2.2 I2V
     A14B Lightning."
5. **Tests.**
   - `ValidateComponents` takes `lora` and `high_noise_lora` for
     `wan22-i2v` and refuses them for `wan22-ti2v-5b`.
   - `LaunchSpec` for a set with both LoRAs: one `--lora-model-dir` with
     the folder, no flag for the LoRA files, and an error when the two
     files are in different folders.
   - `--lora-model-dir` in extra arguments is refused for a set with LoRAs
     and allowed for one without.
   - The request body for a set with LoRAs has the `lora` list (file names,
     multiplier 1, the right `is_high_noise`). A set without LoRAs has no
     `lora` key. `extra.lora` is refused for the first and passed through
     for the second.
   - The two A14B entries have the same expert, text and VAE components.
     `Kijai/WanVideo_comfy` is added to the allowed-repos list.
6. **asmodean.** Deploy, download the set (only the LoRAs should download,
   1.2 GB, when the 20-step set is already there), and measure it at
   Short, then Medium, then Full. Record the time and peak of each
   measurement. Larger sizes are refused if they need more than the room
   (Phase 05), and that is recorded, not fixed. Through the job API, run:
   1. Image-to-video at the defaults, WebM: compare three frames with
      Phase 06's 20-step video (the same fox picture, prompt and seed).
   2. Start-and-end, WebM.
   3. A request with `extra: {"lora": []}`: refused with `bad_value`.

   Also check in the sd-server log that both LoRAs were applied (1200 of
   1200 tensors) on every job and on the Measure run.
7. **compute2 (16 GB A4000).** Measure at Short and run job 1 on one
   A4000. On a 16 GB card the decode needs tiles, and that crashed for the
   20-step set in Phase 06. Record the result either way.
8. **Write it down.** Add a Phase 08 section to `verification.md`. Put the
   measured figures in the catalog notes, `FullMeasureTime`, and the help
   page's reservations table (a new row, with a note under it like the
   20-step set's). In the video jobs part, name the set as the fast A14B
   set that also needs a start image.

## Not in this phase

- Kijai's newer lightx2v 4-step I2V LoRAs (`Wan22_Lightx2v/…_260412_rank_64`,
  2026-04-12). They also hold `diff`/`diff_b` tensors and were not tried.
  A later change can swap the files if they load and look better.
- LoRAs for image sets, a LoRA picker, or LoRA strength settings.
- Reporting the crash with lightx2v's own files to stable-diffusion.cpp.

## Build gate

- `go build ./...`
- `go vet ./...`
- `go test ./internal/...`
- `gofmt -l internal/` prints nothing.

## Test plan

- The tests in step 5 and `TestCatalogEntriesAreValid` pass.
- The asmodean measurements and jobs in step 6 succeed, the log shows both
  LoRAs applied, and the `extra.lora` refusal works.
- The compute2 result is recorded either way.

## Commit

`feat(generation): add the Wan2.2 I2V A14B Lightning video set`

## Rollback

Revert the commit. Downloaded copies of the set stay on disk as a set whose
catalog entry no longer exists. That copy has LoRA slots the reverted code
does not know, so its instance fails to start with "sd-server has no flag
for slot "lora"". Delete the set on the Models page. The 20-step set and
its files are not affected.
