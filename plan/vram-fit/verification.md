# VRAM Fit — Verification

## Before Phase 01: the check that shaped the plan (2026-10-28)

**Setup:** asmodean, Radeon AI PRO R9700 (32 GB), Ryzen 9 5950X, 125 GB RAM.
sd-cli from toolchest's `sd-master-1015-e41fbcf-rocm` build, run with
`docker exec` inside the toolchest container, so toolchest's instances and
reservations were not touched. Scripts: `~/scratch/memtest.sh`,
`cputest.sh` and `a14btest.sh` on asmodean; logs and `results.txt` are in
`~/scratch/memtest/`.

**How the numbers were read:**
- **Peak:** the card's used VRAM
  (`/sys/class/drm/card1/device/mem_info_vram_used`, every 0.2 s) less its
  reading just before the run. Idle use was 1.0–1.8 GiB.
- **RAM:** the sd-cli process's largest resident size.
- **Encode:** sd.cpp's `get_learned_condition completed` time.
- **Settings:** prompt "a red fox walking through deep snow, cinematic, soft
  light", seed 42. Video sets used a drawn start picture where they take one
  (TI2V) and start and end pictures for A14B. LTX-2.3 was prompt only.

### One step against all steps

| Run | Time | Peak | RAM |
|---|---|---|---|
| TI2V Medium (81 f, 1280×704), 1 step, `--max-vram 28` | 166 s | 30.0 GiB (idle 1.8) | 2.2 GiB |
| TI2V Medium, 20 steps, `--max-vram 28` | 659 s | 30.7 GiB (idle 1.1) | 2.3 GiB |
| LTX-2.3 Full (121 f, 1280×720), 1 step, `--max-vram 28.4` | 91 s | 29.5 GiB | 5.3 GiB |
| LTX-2.3 Full, 8 steps, `--max-vram 28.4` | 260 s | 29.1 GiB | 5.3 GiB |
| A14B Short (33 f, 832×480), 1+1 steps, `--max-vram 28.5` | 121 s | 27.4 GiB | 2.7 GiB |

- **TI2V:** both runs peaked at about 31.8 GiB on the card. One step took
  27 s and the untiled decode 131 s.
- **LTX-2.3:** the earlier full Measure at Full peaked at 29.7 GiB and took
  about 28 minutes, with start and end pictures (Phase 07 of the video plan).
- **A14B:** the high-noise and low-noise experts each ran one step (46 s
  each). The decode took 15 s. The earlier 20-step Measure at Short took
  1,658 s and the card peaked at 28.2 GiB (Phase 06 of the video plan).

### Inside a smaller budget

| Run | Time | Peak | RAM |
|---|---|---|---|
| TI2V Medium, 1 step, `--max-vram 14` | 324 s | 13.9 GiB | 2.6 GiB |
| TI2V Medium, 20 steps, `--max-vram 14` | 818 s | 14.0 GiB | 2.6 GiB |
| LTX-2.3 Full, 1 step, `--max-vram 14` | 113 s | 15.2 GiB | 29.0 GiB |
| LTX-2.3 Full, 8 steps, `--max-vram 14` | 341 s | 15.3 GiB | 29.0 GiB |
| A14B Short, 1+1 steps, `--max-vram 14` | stopped by hand after 2,886 s | 15.8 GiB | 8.1 GiB |

- **TI2V:** all 12.3 GB of weights still fitted, and sampling ran at the same
  27 s a step. The untiled decode ran out of memory, and sd.cpp retried it
  with temporal tiling ("VAE decode ran out of memory; retrying with temporal
  tiling"). That added about 160 s, and the video finished.
- **LTX-2.3:** sd.cpp kept most of the weights in system RAM (29 GiB) and
  copied them in piece by piece. It went 1.2–1.3 GiB over the budget.
- **A14B:**
  - Sampling worked: 42 s and 58 s for the two experts.
  - The decode did not fit even with temporal tiling. sd.cpp then reduced the
    tiles from 832×480 to 416×240 and retried with spatial and temporal
    tiling. That hung: the GPU was 3% busy and no log line came for 45
    minutes, the same ROCm hang as LTX-2.3's spatially tiled decode in Phase
    07 of the video plan.

### Text encoder on the GPU against the CPU

Image sets at their catalog size and steps, with no `--max-vram` (sd.cpp
uses the free VRAM). Video sets at Short with 1 step.

| Run | Time | Encode | Peak | RAM |
|---|---|---|---|---|
| Qwen-Image 2.1 Turbo, GPU | 35 s | 3.43 s | 9.8 GiB | 2.5 GiB |
| Qwen-Image 2.1 Turbo, `te=cpu` | 33 s | 0.90 s | 5.6 GiB | 6.4 GiB |
| FLUX.2 klein 4B, GPU | 19 s | 5.00 s | 10.4 GiB | 2.8 GiB |
| FLUX.2 klein 4B, `te=cpu` | 13 s | 4.51 s | 7.3 GiB | 5.4 GiB |
| Z-Image Turbo, GPU | 36 s | 7.88 s | 11.2 GiB | 2.0 GiB |
| Z-Image Turbo, `te=cpu` | 20 s | 0.64 s | 7.4 GiB | 5.3 GiB |
| FLUX.3 image klein, GPU | 26 s | 2.98 s | 11.9 GiB | 2.3 GiB |
| FLUX.3 image klein, `te=cpu` | 24 s | 1.14 s | 8.3 GiB | 5.4 GiB |
| Chroma1 Flash, GPU | 53 s | 3.80 s | 15.2 GiB | 1.9 GiB |
| Chroma1 Flash, `te=cpu` | 54 s | 7.47 s | 10.5 GiB | 6.5 GiB |
| Qwen-Image Edit 2511 (plain image), GPU | 129 s | 3.30 s | 17.5 GiB | 2.0 GiB |
| Qwen-Image Edit 2511 (plain image), `te=cpu` | 125 s | 1.52 s | 13.5 GiB | 5.9 GiB |
| Chroma1 HD, GPU | 213 s | 2.28 s | 15.2 GiB | 1.9 GiB |
| Chroma1 HD, `te=cpu` | 224 s | 14.93 s | 10.5 GiB | 6.4 GiB |
| TI2V Short, GPU, `--max-vram 28` | 33 s | 2.40 s | 23.6 GiB | 2.1 GiB |
| TI2V Short, `te=cpu`, `--max-vram 28` | 46 s | 15.74 s | 17.9 GiB | 7.5 GiB |
| LTX-2.3 Short (33 f, 768×512), GPU, `--max-vram 28.4` | 25 s | 6.93 s | 27.2 GiB | 4.2 GiB |
| LTX-2.3 Short, `te=cpu`, `--max-vram 28.4` | 40 s | 22.93 s | 18.2 GiB | 12.8 GiB |
| A14B Short, 1+1, GPU, `--max-vram 28.5` | 121 s | 1.97 s | 27.4 GiB | 2.7 GiB |
| A14B Short, 1+1, `te=cpu`, `--max-vram 28.5` | 126 s | 15.62 s | 27.5 GiB | 8.5 GiB |

- **Image times include loading the model**, since each sd-cli run starts
  from nothing. The encode time is the fair comparison. Some image runs were
  faster overall with `te=cpu` because less had to be copied to the GPU.
- **A14B:** with `te=cpu`, sd.cpp's log shows 18.7 GB of weights in VRAM
  instead of 24.4 GB, but the peak was the same. Without `te=cpu` the encoder
  is dropped from VRAM before sampling, so it is never part of the peak.

The output files were deleted from the container afterwards. Toolchest's
instances were not started, stopped or changed.

## Phase 01 (PR #414, 2026-10-29)

**Setup:** asmodean, with the branch's binary swapped into the toolchest
container. Each run is toolchest's own Measure through the API; the jobs are
real video jobs through `/v1/videos/generations`, with start and end images
where the set takes them. Afterwards the original binary and
`generation.json` were put back, the test instances deleted and the test
videos removed.

| Set and size | Measure | Result | Real job |
|---|---|---|---|
| TI2V 5B Short | 43 s (80 s before) | granted, peak 23.5 GiB | 68 s |
| TI2V 5B Medium | stopped after 43 s (674 s before) | needed more than 29.4 GiB in the decode; room 28.2 GiB | — |
| LTX-2.3 Medium | 93 s | granted, peak 29.0 GiB | 409 s |
| LTX-2.3 Full | stopped after 153 s | 29.3 GiB during sampling; room 28.2 GiB | — |
| A14B Short | 123 s (1,658 s before) | granted, peak 27.3 GiB | 1,650 s |
| A14B Lightning Short | 83 s (about 3 min before) | granted, peak 28.6 GiB | 200 s |

- **LTX-2.3 Full** was granted on 2026-10-27, with a 29.7 GiB peak in a
  28.8 GiB room. It is at the card's edge, and Phase 02's text encoder on the
  CPU should leave it plenty of room.
- **Uneven step times.** The real LTX-2.3 Medium job's step times were 136.6,
  119.2, then 18.0–21.7 s. That is why the time estimate was dropped.
- **A14B with two low-noise steps.** An A14B Measure with two low-noise steps
  and one high-noise step failed in sd-server: "generate_video returned no
  results".
- **Not triggered live:** the stuck-run and ROCm spatial-fallback stops (unit
  tests only).
- **Not checked:** the A4000.

## Phase 02 (PR #415, 2026-10-29)

**Setup:** asmodean, with the branch's binary in the toolchest container.
Every catalog sd-server set was measured through toolchest's API.
Reservations are Measure's suggestion, capped at the room. RAM is the
backend's VmHWM. asmodean was put back afterwards.

| Set | Before | Now | RAM | Measure took |
|---|---|---|---|---|
| Qwen-Image 2.1 Turbo | 11.0 GiB | 6.5 GiB (peak 5.6) | 6.6 GiB | 38 s |
| FLUX.2 klein 4B | 11.5 GiB | 8.5 GiB (peak 7.4) | 5.4 GiB | 43 s |
| Z-Image Turbo | 12.5 GiB | 8.5 GiB (peak 7.4) | 5.3 GiB | 28 s |
| FLUX.3 image klein, with edit | 13.5 GiB | 9.5 GiB (peak 8.6) | 5.7 GiB | 88 s |
| Chroma1 Flash | 17.0 GiB | 12.0 GiB (peak 10.5) | 6.4 GiB | 78 s |
| Chroma1 HD | 17.0 GiB | 12.0 GiB (peak 10.5) | 6.4 GiB | 233 s |
| Qwen-Image Edit 2511, with edit | 22.0 GiB | 16.5 GiB (peak 14.8) | 7.5 GiB | 508 s |
| TI2V 5B Short | 26.0 GiB | 20.0 GiB (peak 17.9) | 7.7 GiB | 53 s |
| TI2V 5B Medium / Full | refused | stopped: decode needs more than 29.4 / 30.2 GiB | | 53 / 88 s |
| LTX-2.3 Full | stopped at 29.3 GiB | 27.7 GiB (peak 26.3), granted | 14.7 GiB | 123 s |
| A14B Short | 28.5 GiB (peak 27.3) | 28.0 GiB (peak 26.0) | 8.5 GiB | 138 s |
| A14B Medium | not measured | stopped by the ROCm small-tile check at 27.9 GiB | | 703 s |
| A14B Lightning Short | 28.6 GiB (peak 28.6) | 28.3 GiB (peak 26.6) | 8.4 GiB | 88 s |

**TI2V with `--temporal-tiling`** as an extra flag, with the encoder on the
CPU:
- Short peaked at 20.5 GiB, more than the 17.9 GiB untiled.
- Medium and Full were stopped by the ROCm small-tile check, at 25.0 and
  25.5 GiB.
- A real Short job took 83 s, the same as untiled with the encoder on the CPU
  (68 s with it on the GPU, Phase 01).

**Qwen-Image Edit 2511,** one 1024² edit through sd-cli:

| Encoder | Time | Encode | Peak |
|---|---|---|---|
| on the GPU | 343 s | 3.7 s | 19.6 GiB |
| on the CPU | 360 s | 21.1 s | 15.1 GiB |

**CPU sharing:** gemma-4-12b Q6 with `-ngl 24` (half its layers on the CPU)
made 7.1 tok/s alone. Z-Image Turbo alone took 23 s with its encoder on the
GPU and 20 s with it on the CPU. Together:

| Image encoder | Image time | LLM speed meanwhile |
|---|---|---|
| on the GPU | 29 s | 6.4 tok/s |
| on the CPU | 32 s | 6.5 tok/s |

**The ROCm small-tile check worked live:** Measure stopped A14B Medium and
the tiled TI2V Medium and Full when sd.cpp began its spatially tiled
fallback.

## Phase 03 (2026-10-29)

**Setup:** asmodean, with the branch's binary in the toolchest container. All
eight sets were downloaded through the catalog. Each downloaded 3 files: the
Qwen2.5-VL file was already there for Qwen-Image Edit 2511. Measure ran
through the API, at Full, then Medium, then Short. Real Short videos (33
frames) went through `/v1/videos/generations`, with a drawn start image for
I2V, and frame 16 of each was looked at. asmodean was put back afterwards
with the downloaded files kept.

| Set | Full | Medium | Short | Real Short video |
|---|---|---|---|---|
| I2V 480p step-distilled (fp8) | 25.6 GiB | 24.8 GiB | 24.3 GiB | 119 s |
| I2V 480p CFG-distilled | 26.0 GiB | 25.3 GiB | 24.8 GiB | 488 s |
| I2V 480p | 26.1 GiB | 25.3 GiB | 24.8 GiB | 944 s |
| T2V 480p CFG-distilled | 25.1 GiB (after the fix) | 24.3 GiB | 23.8 GiB | 489 s |
| T2V 480p | 25.1 GiB | 24.3 GiB | 23.8 GiB | 945 s |
| I2V 720p CFG-distilled | crash (CONCAT failed) | out of memory in the decode | 13.0 GiB, tiled decode | 2,210 s |
| I2V 720p | crash | out of memory | 13.0 GiB, tiled decode | not run |
| T2V 720p | crash | out of memory | 11.6 GiB, tiled decode | 4,194 s |

Measuring took about 4 minutes at Full for the 480p sets (about 6 for the
base ones), and about 5 minutes at Short for the 720p sets.

**The step-distilled file,** through sd-cli: 8 steps, 33 frames at 848×480,
`--max-vram 28`. The three frames looked the same.

| File | Total | Load | Sampling | VRAM |
|---|---|---|---|---|
| fp16 as it is | 141 s | 23.5 s | 94.8 s | 29.2 GiB |
| fp16, converted to Q8_0 as it loads (`--tensor-type-rules '^model\.diffusion_model\.=q8_0'`) | 126 s | 9.5 s | 80.6 s | 24.8 GiB |
| scaled fp8 | 125 s | 5.6 s | 79.2 s | 24.3 GiB |

**The 720p decode,** through sd-cli: I2V CFG-distilled, 33 frames.
- **`--vae-tiling --max-vram 28`:** hung while encoding the start image. The
  GPU sat at 6% and the run was cut off at 15 minutes.
- **`--temporal-tiling --max-vram 20`, and the same with `--max-vram 28`:**
  the untiled decode needed 31,416 MB. sd.cpp reduced the tiles to 640×352
  and retried with spatial and temporal tiling. The decode took 225 s, with a
  13.0 GiB peak.
- **The same with 50 steps:** 2,212 s. Frames 4, 16 and 28 were sharp, with
  no seam at the tile edges.

**The baseline fix:**
- **Before:** T2V 480p CFG-distilled at Full, measured right after another
  HunyuanVideo instance stopped, read 5.9 GiB.
- **After:** the same measurement with the fix read 25.1 GiB.

## Phase 04 (2026-10-29)

**Setup:** asmodean, with the branch's binary in the toolchest container.
Measure ran through the API. The dialog's data was read from the instance
row's `data-video`, and the dialog was looked at in the browser. asmodean
was put back afterwards.

| Step | Result |
|---|---|
| LTX-2.3: Short, then Medium (whole room) | 18.4 GiB and 23.4 GiB. The dialog then predicted 28.8 GiB for Full, "close to the limit" |
| LTX-2.3: Full | 26.3 GiB measured: the prediction was 2.5 GiB high (see the phase file) |
| TI2V 5B: Short, then Medium (whole room) | Short 17.9 GiB. Medium was stopped at 30.3 GiB in the decode. The dialog: Medium "needed more than 30.3 GiB: will not fit", and Full "will not fit" |
| TI2V 5B: Medium inside 14 GiB | Granted in 349 s, peak 10.6 GiB, decoded in smaller pieces. The row says "runs inside a 14.0 GiB budget", and sd-server ran with `--max-vram 14.0` |
| TI2V 5B: a real Medium video inside the budget | 829 s, 81 frames. A frame looked right |
| TI2V 5B: Full inside 14 GiB | Granted in 515 s, peak 11.8 GiB |
| Dialog for TI2V afterwards | For the 14.0 GiB budget: Short "about 8.5 GiB predicted: fits"; Medium "measured 10.6 GiB: fits"; Full "measured 11.8 GiB: fits" |

The reservation inside a budget came out at 12.0 and 13.5 GiB, below the
14 GiB budget. The branch was changed afterwards to reserve at least the
budget; unit tests cover it.
