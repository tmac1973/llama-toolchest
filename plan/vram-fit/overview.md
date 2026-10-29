# VRAM Fit — Project Overview

## Problem

Image and video sets need a VRAM reservation, and Measure is the only way to
get one. For video, Measure is too slow and finds problems too late:

- **Measure runs a whole video job.** Every sampling step runs, then the
  decode. Measuring Wan2.2 I2V A14B at Short took 28 minutes, and LTX-2.3 at
  Full took 28 minutes. Memory peaks in the first step and in the decode; the
  other steps add time but tell Measure nothing new.
- **Running out of memory is seen only at the end.** Measure records the peak
  but does not compare it with the room while the job runs. A size that does
  not fit runs until sd-server crashes or the 2-hour limit ends it.
- **A failed run keeps nothing.** When the test generation fails, the peak it
  reached is thrown away, so the user does not learn how far over it went.
- **More than needs to be stays in VRAM.** Every model part sd.cpp uses sits on
  the GPU when the room allows it. Text encoders (umt5, T5-XXL, Gemma 3 12B,
  the Qwen models) run once per job and could run on the CPU. Only the
  Ming-Image set does this today (`--backend te=cpu`).

sd.cpp (master-1015, the version toolchest builds) already has what is needed:
- **`--max-vram`** sets a GPU budget. Inside it, sd.cpp's auto-fit places
  weights on the GPU, in system RAM or on disk, in that order.
- **`--backend te=cpu`** runs the text encoder on the CPU.
- **Fallback when the decode runs out of memory:** sd.cpp then retries the
  decode with temporal tiling.

Also, the catalog has no HunyuanVideo 1.5 set. HunyuanVideo 1.5 is an 8.3B
video model built to run on consumer cards. sd.cpp supports it, and toolchest
already has its two recipes for custom sets.

## What the asmodean check found (2026-10-28)

All runs used sd-cli (master-1015) on the R9700, with Wan2.2 TI2V 5B at Medium
(81 frames, 1280×704) and a start picture. Peaks are the whole card's used
VRAM above its idle use.

| Run | Time | Peak |
|---|---|---|
| 20 steps, `--max-vram 28` (what Measure does) | 659 s | 30.7 GiB above 1.1 idle |
| 1 step, `--max-vram 28` | 166 s | 30.0 GiB above 1.8 idle |
| 1 step, `--max-vram 14` | 324 s | 13.9 GiB |

- **The peak is reached in one step.** In both runs the card peaked at about
  31.8 GiB in all.
- **The time goes to the steps and the decode.** One step took 27 s and the
  decode 131 s.
- **With 14 GiB, the job still finished.** The untiled decode ran out of
  memory and sd.cpp retried it with temporal tiling, using less than half the
  memory and adding about 160 s. Sampling ran at the same speed, because the
  weights (12.3 GB) still fitted.

LTX-2.3 at Full (121 frames, 1280×720, prompt only) showed the same:

| Run | Time | Peak |
|---|---|---|
| 8 steps, `--max-vram 28.4` | 260 s | 29.1 GiB |
| 1 step, `--max-vram 28.4` | 91 s | 29.5 GiB |
| 8 steps, `--max-vram 14` | 341 s | 15.3 GiB, with 29 GiB of system RAM |

Inside 14 GiB, sd.cpp kept most of the 22B model's weights in system RAM and
copied them in piece by piece, for 31% more time.

Running the text encoder on the CPU saved 3–5 GiB on every image set at no
real cost (the Qwen-family encoders were even faster there), and 5.7 GiB
(TI2V) and 9.0 GiB (LTX-2.3) on video for 13–16 s per job. Phase 02 has the
table, and `verification.md` has every run.

## Goals

- **Measuring a video takes minutes, not half an hour.** Measure runs one
  sampling step (one per expert for A14B) and the full decode, at the size the
  user chose. It reaches the same peak and works out how long a full video
  takes from the measured step and decode times.
- **Measure stops as soon as a size cannot fit.** If memory goes past the room
  plus the 1 GiB allowance, Measure ends the run. It reports the peak it
  reached and which stage was running (sampling or decode). A failed run keeps
  its peak in the message.
- **Every sd-server catalog set uses as little VRAM as it can without becoming
  much slower.** Image and video sets alike:
  - Text encoders run on the CPU (`--backend te=cpu`). This cost at most a few
    seconds per image and 13–16 s per video on the R9700.
  - Weights that do not fit the reservation stay in system RAM, where sd.cpp's
    auto-fit puts them.
  - Untiled video decodes that do not fit fall back to temporal tiling (sd.cpp
    does this itself), and Measure says so.
  - The choices and their measured costs are written in the catalog notes.
  - Measure shows how much system RAM the backend used.
- **An instance measured with older settings says so.** When a set's launch
  settings change after an instance was measured, the instance row says that
  measuring again may free memory.
- **HunyuanVideo 1.5 in the catalog**, in every version that runs on consumer
  cards and that sd.cpp supports (Phase 03 lists them).
- **Later:** before a long run, a prediction of what a larger size needs, from
  two quick measurements (Phase 04).

## Non-goals

- **No change to how LLM memory is worked out.**
- **No change to ace-server**, which takes no memory limit.
- **No VAE on the CPU for video.** Decoding the frames on the CPU is far too
  slow.
- **No HunyuanVideo 1.5 versions sd.cpp cannot run or a consumer card cannot
  use:**
  - the 720p I2V sparse version, which needs NVIDIA H-series cards;
  - the 480p→720p and 720p→1080p upscalers, which sd.cpp does not support;
  - 720p T2V CFG-distilled, which Tencent has not released.
- **No re-encoding of videos.**

## Phases

| Phase | What | Depends on |
|---|---|---|
| 01 | Short Measure for video, and stopping as soon as a size cannot fit | — |
| 02 | Model parts on the CPU, and decode settings, for every sd-server catalog set | 01 (for re-measuring quickly) |
| 03 | HunyuanVideo 1.5 catalog sets | 01, 02 |
| 04 | Predicting what a larger size needs | 01 |

Each phase is one PR, checked on asmodean (R9700, 32 GB) before it merges.
Phase 02 is also checked on compute2 (RTX A4000, 16 GB), where saving memory
matters most.
