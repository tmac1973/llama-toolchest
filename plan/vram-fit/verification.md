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
