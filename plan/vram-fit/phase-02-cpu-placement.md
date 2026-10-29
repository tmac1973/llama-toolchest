# Phase 02 — Model Parts on the CPU, and Decode Settings, for Every sd-server Catalog Set

**Depends on:** Phase 01 (re-measuring every set quickly) · **Enables:**
Phase 03 (the HunyuanVideo sets follow the same rules)

## What was built (PR #415, 2026-10-29)

Steps 1–4 and 6–8 were built as written, except the A4000 check (step 7),
which was not done. What the checks found:
- **Step 2:** sd.cpp places only `diffusion`, `te` and `vae`, so Qwen-Image
  Edit's picture reader goes to the CPU with its text model. An edit takes
  about 17 s longer (360 s against 343 s). `te=cpu` was kept.
- **Step 5:** an LLM with half its layers on the CPU and an image job slowed
  each other no more than before (6.5 against 6.4 tok/s; 32 s against 29 s for
  the image). No `-t` was set.
- **Step 6:** every set measured smaller (see `verification.md`), and LTX-2.3
  Full now fits.
- **Not met: TI2V at Medium** is still not granted. Its untiled decode needs
  more than the card has left. With `--temporal-tiling` too, the decode fell
  back to small tiles and Measure stopped it. It fitted inside
  `--max-vram 14` in sd-cli, so this moves to Phase 04's smaller budget.

## Goal

Every image and video set in the catalog keeps on the GPU only what needs to
be there, so its reservation is as small as it can be without the set
becoming much slower.

After this phase:
- **Every sd-server catalog set runs its text encoder on the CPU**
  (`--backend te=cpu`), as Ming-Image already does. On the R9700 this saved
  3–9 GiB per set. For images it cost nothing, and for video 13–16 s per job
  (see below).
- **The model weights stay where sd.cpp's auto-fit puts them inside
  `--max-vram`.** Toolchest sets no weight placement of its own: inside a
  smaller budget, auto-fit moves weights to system RAM and copies them in
  piece by piece. This worked for LTX-2.3 (see below). Phase 04 lets the user
  choose a smaller budget.
- **Video decodes stay as they are**, untiled for TI2V and temporal tiling
  for A14B and LTX-2.3. When an untiled decode does not fit, sd.cpp retries it
  with temporal tiling by itself, and Phase 01 reports this. The text encoder
  on the CPU leaves more room for the untiled decode.
- **An instance measured with older settings says so.** Its row reads
  "Measured with older settings; measure again, it may need less memory now",
  with a Measure button.
- **System RAM is shown.** Measure records the backend's peak system RAM
  (the CPU encoder and any weights moved there), and the instance row shows
  it.

## What the asmodean check found (2026-10-28)

sd-cli master-1015 on the R9700, Ryzen 9 5950X. Image sets at their default
size and steps; video sets at Short, 1 step. "Encode" is sd.cpp's
`get_learned_condition` time; the peak is the whole card above idle.

| Set | Text encoder | Time GPU → CPU | Encode GPU → CPU | Peak GPU → CPU | Saved |
|---|---|---|---|---|---|
| Qwen-Image 2.1 Turbo | Qwen3-VL 8B Q4_K_M | 35 → 33 s | 3.4 → 0.9 s | 9.8 → 5.6 GiB | 4.2 GiB |
| FLUX.2 klein 4B | Qwen3 4B Q8_0 | 19 → 13 s | 5.0 → 4.5 s | 10.4 → 7.3 GiB | 3.1 GiB |
| Z-Image Turbo | Qwen3 4B 2507 Q8_0 | 36 → 20 s | 7.9 → 0.6 s | 11.2 → 7.4 GiB | 3.8 GiB |
| FLUX.3 image klein | Qwen3-VL 4B Q8_0 | 26 → 24 s | 3.0 → 1.1 s | 11.9 → 8.3 GiB | 3.6 GiB |
| Chroma1 Flash | T5-XXL Q8_0 | 53 → 54 s | 3.8 → 7.5 s | 15.2 → 10.5 GiB | 4.7 GiB |
| Chroma1 HD (with negative prompt) | T5-XXL Q8_0 | 213 → 224 s | 2.3 → 14.9 s | 15.2 → 10.5 GiB | 4.7 GiB |
| Qwen-Image Edit 2511 (plain image) | Qwen2.5-VL 7B Q4_K_M | 129 → 125 s | 3.3 → 1.5 s | 17.5 → 13.5 GiB | 4.0 GiB |
| Wan2.2 TI2V 5B, Short | umt5-XXL Q8_0 | 33 → 46 s | 2.4 → 15.7 s | 23.6 → 17.9 GiB | 5.7 GiB |
| LTX-2.3, Short | Gemma 3 12B Q4_K_M | 25 → 40 s | 6.9 → 22.9 s | 27.2 → 18.2 GiB | 9.0 GiB |

| Wan2.2 I2V A14B, Short, 1+1 steps | umt5-XXL Q8_0 | 121 → 126 s | 2.0 → 15.6 s | 27.4 → 27.5 GiB | none |

**A14B saved nothing at its full room.** sd.cpp's log shows the encoder in RAM
(VRAM weights 18.7 GB instead of 24.4 GB), but the peak stayed the same. With
the encoder on the GPU, sd.cpp drops it from VRAM before sampling (its weight
cache puts the diffusion model first), so it was never part of the peak. The
encoder on the CPU still helps when A14B runs inside a smaller budget, since it
leaves more of that budget to the experts. It costs about 5 s per job, because
sampling was a little faster with the encoder gone. A14B keeps `te=cpu` with
the others, and its notes say it saves memory only inside a smaller budget.

What the table shows:
- **The Qwen-family encoders are as fast or faster on the CPU.** On the GPU,
  the encode time includes copying the encoder into VRAM first.
- **The T5-family encoders are slower on the CPU**, by about 4 s for each
  prompt, since T5 runs twice with a negative prompt. That is 13–16 s on a
  video job of several minutes: 2% of an 11-minute TI2V Medium video, 6% of a
  4-minute LTX Full video. It is 5% of a Chroma1 HD image.
- **Inside a smaller budget, weights move to system RAM at a modest cost.**
  LTX-2.3 Full inside `--max-vram 14`:
  - 341 s against 260 s with the whole room;
  - peak 15.3 GiB against 29.1 GiB;
  - 29 GiB of system RAM held.

  TI2V Medium inside 14 GiB: 818 s against 659 s, peak 14.0 GiB against
  30.7 GiB. Its extra time was the fallback decode, not weights: its 12.3 GB
  of weights still fitted.

## Files touched

- `internal/generation/catalog.go`:
  - every `KindSD` and `KindVideo` entry except SDXL gains `--backend te=cpu`
    in `LaunchArgs`. SDXL's CLIP encoders are small and were not tested; leave
    them as they are;
  - each entry's notes give the new Short reservation and time;
  - each comment says what moving the encoder saved.
- `internal/generation/types.go`: `Instance` gains `MeasuredArgs` (JSON
  `measured_args`, `omitempty`): the run arguments of the measurement. It also
  gains `MeasuredRAMMiB` (JSON `measured_ram_mib`, `omitempty`). Editing an
  instance clears both with the other measured fields.
- `internal/api/gen_vram.go`:
  - the sampler also records the backend's peak resident memory (RSS of the
    process, from `/proc/<pid>/status` `VmHWM` once the run ends, or RSS
    samples);
  - `applyMeasurement` stores `MeasuredArgs` and `MeasuredRAMMiB`.
- `internal/api/gen_instances.go`:
  - `genInstanceRow` gains `MeasuredStale`: true when the instance has a
    measurement and `MeasuredArgs` differs from the set's current run
    arguments plus the instance's extra arguments;
  - the row shows the RAM figure.
- `web/templates/partials/gen_instances.html`: the stale note with its
  Measure button, and the RAM figure with a tooltip.
- `web/templates/help.html`: the reservations section explains what runs on
  the CPU and why.
- Tests:
  - `internal/generation/catalog_test.go`: every sd-server entry but SDXL has
    `--backend te=cpu`;
  - `internal/api/gen_instances_test.go`: `MeasuredStale`;
  - `internal/api/gen_vram_test.go`: the stored args and RAM.
- `plan/vram-fit/verification.md`: a Phase 02 section.

## Steps

1. **The flag.** Add `--backend te=cpu` to each entry's `LaunchArgs`.
   Catalog sets take their run arguments from the current catalog
   (`RunArgs`), so existing instances get it on their next start. A custom set
   gets nothing new. The help page says how to add it there.

2. **Qwen-Image Edit 2511's picture input.** An edit also runs the vision
   part of Qwen2.5-VL (`llm_vision`) on the input picture. Check on asmodean
   whether `te=cpu` puts it on the CPU too, and what that costs per edit. If
   it adds more than about 10 s, keep the vision part on the GPU, if sd.cpp
   allows them apart. Otherwise keep `te=cpu` and write the cost in the notes.

3. **Stale measurements.** Store the run arguments with each measurement.
   When they differ from the current ones, the row shows the note. This also
   covers any later catalog change to a set's arguments.

   Existing instances measured before this phase have no `MeasuredArgs`. Treat
   a catalog instance with a measurement but no `MeasuredArgs` as stale, so
   that this phase's change reaches it.

4. **System RAM.** Record the backend's peak RSS during Measure and show it
   on the row: "uses 6.4 GiB of system RAM", with a tooltip. "The text
   encoder runs on the CPU, and sd.cpp keeps model weights here when they do
   not fit the reservation. Other programs need this RAM too."

   Toolchest's planner does not count it yet. That is left for later and
   written in the plan's open questions.

5. **CPU threads.** sd.cpp uses every physical core for CPU work by default
   (`-t -1`). That competes with a llama-server running layers on the CPU.
   Check on asmodean whether an LLM job and an image job at the same time
   slow each other more than before. If they do, set `-t` to half the
   physical cores in the launch arguments and measure again.

6. **Measure again.** On asmodean, measure every catalog sd-server set with
   the new arguments: Short for video, the test image for image sets. Write
   the reservations and times in the notes. Check that TI2V at Medium is now
   granted.

7. **A4000 check.** On compute2 (RTX A4000, 16 GB), measure TI2V, LTX-2.3
   and three image sets with the new arguments, and make one job each. compute2's
   CUDA crashes (2026-10-28) are a known separate problem: note any crash, and
   retry once before counting it.

8. **Help.** In the reservations section of `help.html`:
   - what runs on the CPU and why;
   - what the system RAM figure means;
   - that measuring again after this change frees memory.

## Done when

- Every catalog sd-server set but SDXL runs its text encoder on the CPU, and
  its notes give the new reservation and time.
- On the R9700, image sets measure 3–5 GiB smaller and no more than 5% slower.
  TI2V and LTX-2.3 measure 5–9 GiB smaller and no more than 20 s per job
  slower. A14B measures the same, as found above.
- TI2V at Medium is granted on the R9700.
- An instance measured before this phase shows the stale note until it is
  measured again.
- `go test ./...` and the JS tests pass.

## Open questions (not in this phase)

- **The planner does not count the system RAM** that sd-server and
  ace-server use. A machine short of RAM can run out when an LLM, an image
  set and a video set all run.
- **SDXL's encoders** were not tested on the CPU.
