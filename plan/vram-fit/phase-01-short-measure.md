# Phase 01 — Short Measure for Video, and Stopping as Soon as a Size Cannot Fit

**Depends on:** nothing · **Enables:** Phase 02 (re-measuring every set
quickly), Phase 03 (measuring seven new sets), Phase 04 (two quick
measurements for a prediction)

## Goal

Measuring a video set takes minutes, and a size that cannot fit is found out
while it runs, not at the end.

After this phase:
- **A video Measure runs one sampling step.** A set with a high-noise expert
  (Wan2.2 I2V A14B and its Lightning version) runs one step on each expert.
  The decode runs in full, at the chosen size, in the heaviest mode the set
  supports, as today.
- **Measure reports how long a full video takes.** It reads the step and
  decode times from sd-server's log and adds them up for the set's real number
  of steps. The instance row and the Measure result show the time, for
  example "a video this size takes about 11 minutes".
- **Measure stops as soon as a size cannot fit.** When the peak goes past the
  room plus the 1 GiB allowance, Measure stops the backend at once. It reports
  the peak, the room and the stage that was running.
- **A failed run keeps its peak** in the error message.
- **A fallback decode is reported.** When sd.cpp's log says the decode ran out
  of memory and was retried with temporal tiling, the result says so: the
  size fits, but the decode is slower than it would be with more room.

Image Measure is unchanged: it runs the set's whole test image (and edit),
which takes about a minute.

## Why one step is enough

On the R9700 (2026-10-28, sd-cli master-1015), TI2V 5B at Medium peaked at
the same 31.8 GiB on the card with 1 step and with 20 steps. One step took 27
s, and the decode 131 s. The full 20-step run took 659 s and the 1-step run
166 s. Memory peaks in the decode, or in the first step when that is larger.
The steps after the first use the same memory again. `verification.md` has
the LTX-2.3 and A14B results.

## Files touched

- `internal/api/gen_video.go`: `vidGenRequest` gains `Steps` and
  `HighNoiseSteps`, sent as `sample_params.sample_steps` and
  `high_noise_sample_params.sample_steps` (sd.cpp master-1015,
  `common.cpp` `from_json_str`). Zero means the set's own.
- `internal/api/gen_vram.go`:
  - `runMeasureVideo` sends `Steps: 1`, and `HighNoiseSteps: 1` for a set
    with a `high_noise_diffusion_model` component;
  - `measureInstance`'s sampler stops the backend when the peak passes the
    room plus `measureOverheadMiB`, and keeps the peak when the run fails;
  - a new `measureLogTimes(lines)` reads the step time, the decode time, the
    encode times and the fallback-decode line from the backend's log;
  - `applyMeasurement` stores the estimate and the fallback note.
- `internal/generation/types.go`: `Instance` gains `MeasuredJobSeconds`
  (JSON `measured_job_seconds`, `omitempty`) and `MeasuredDecodeFallback`
  (JSON `measured_decode_fallback`, `omitempty`). Editing an instance clears
  them with the other measured fields.
- `internal/generation/catalog.go`: a helper `SetSteps(set)` that reads
  `--steps` and `--high-noise-steps` from the set's run arguments (20 when
  not given, sd.cpp's default; for `--sigmas`, the number of sigmas less one).
- `internal/api/gen_instances.go`: the row shows the estimated time next to
  the measured size.
- `web/templates/partials/gen_instances.html`:
  - the Measure dialog's time sentence comes from the new estimate when there
    is one;
  - `FullMeasureTime` sentences are rewritten for the shorter Measure.
- `internal/generation/catalog.go`: every video entry's `FullMeasureTime` is
  measured again.
- `web/templates/help.html`: the Measure section says what a video Measure
  runs and what the time estimate means.
- Tests:
  - `internal/api/gen_vram_test.go`: the request has one step; one on each
    expert for A14B; stopping early over the room; a failed run's message
    holds the peak; log parsing;
  - `internal/generation/catalog_test.go`: `SetSteps`.
- `plan/vram-fit/verification.md`: a Phase 01 section.

## Steps

1. **Step count in the request.** Add `Steps` and `HighNoiseSteps` to
   `vidGenRequest`. In `body()`, set `sample_params.sample_steps` and
   `high_noise_sample_params.sample_steps` when they are above zero. Merge
   them with any `sample_params` already in `Extra`, without replacing it.
   The video job API does not set them: a job runs the set's own steps.

2. **One step in Measure.** In `runMeasureVideo`, set `Steps: 1`. For a set
   with a `high_noise_diffusion_model` component, also set
   `HighNoiseSteps: 1`, so both experts load and run. The start and end
   pictures stay, so that the heaviest mode is measured.

3. **Times from the log.** `measureLogTimes` reads these sd.cpp lines (master-1015):
   - `sampling completed, taking N s`, and for A14B also
     `sampling(high noise) completed, taking N s`;
   - `decode_first_stage completed, taking N ms|s`;
   - `encode_first_stage completed` and `get_learned_condition completed`;
   - `VAE decode ran out of memory; retrying with temporal tiling` (the
     fallback).

   Lines that are missing leave their part out. The full-job estimate is the
   encode times, plus each expert's step time × that expert's steps from
   `SetSteps`, plus the decode time.

   Before any `RecentLogs` call, check that the backend's log buffer holds
   enough of the test run. If it keeps fewer lines than one test run writes,
   read the lines as the run goes.

4. **Stop early.** In the sampler goroutine, compare the current peak with
   `room + measureOverheadMiB` when `haveRoom`. When the peak is past it:
   - record the stage (sampling until a `sampling completed` line, decode
     after it);
   - stop the backend with `s.genMgr.Stop(id)`. Don't cancel the job: sd-server
     checks for a cancel only between steps, and a decode can run for minutes;
   - return an error such as "81 frames at 1280×704 needed more than 29.2 GiB
     during the decode, more than the 28.2 GiB this card can give the
     instance; measure a smaller size". Don't disable the instance (the rule in
     `applyMeasurement` is unchanged: an instance with no reservation is
     disabled).

   A user stop and a crash keep their own messages.

   **Stop a stuck run too.** On ROCm, sd.cpp's spatially tiled decode can
   hang: the GPU sits at a few percent busy and no log line comes for as long
   as Measure waits. On 2026-10-28, A14B inside `--max-vram 14` ran out of
   memory in the untiled decode and retried with spatial and temporal tiling.
   Its log showed "Reducing VAE decode tiles … retrying with spatial+temporal
   tiling", and it then made no progress for 45 minutes. Two ways to catch it:
   - **No progress.** When the backend has written no new log line for 5
     minutes and the GPU has been under 10% busy all that time, stop the
     backend. The error says "the decode stopped making progress; on this
     card sd.cpp's spatially tiled decode can hang. Measure with more room or
     a smaller size".
   - **On ROCm, the spatial fallback line.** When it appears, stop at once
     with the same message.

   The GPU busy reading comes from the monitor, where it has one. Without it,
   only the log check applies.

5. **Keep the peak on failure.** Every error path after the run, in
   `backendStoppedError`'s callers and the timeout, adds "it reached X GiB
   before it stopped" when the peak is above zero.

6. **Store and show.** `applyMeasurement` sets `MeasuredJobSeconds` and
   `MeasuredDecodeFallback` for a video measurement that applies. The row
   shows "about 11 min a video" next to the measured size, and, with the
   fallback, "decoded a few frames at a time to fit". Each has a tooltip:
   - the time: "Worked out from one measured step and the decode, for the
     set's N steps. The first video after a start also loads the model."
   - the fallback: "The decode did not fit the room at once, so sd.cpp
     decoded a few frames at a time. The video is the same, but the decode
     takes longer."

7. **Measure times.** Measure every catalog video set at Short on asmodean
   with the new build. Write the times in `FullMeasureTime` and the catalog
   notes.

8. **Help.** In `help.html`, the Measure step says that a video Measure runs
   one sampling step and the full decode, that it stops as soon as the size
   cannot fit, and how the time estimate is worked out.

## Done when

- Measuring LTX-2.3 at Full on the R9700 takes a few minutes instead of 28,
  and grants the same size with a peak within 0.5 GiB of the 20-step one.
- Measuring A14B (20-step set) at Short takes a few minutes, and both experts
  appear in the log.
- A size too large for the card (TI2V at Full, untiled, on the A4000) is
  stopped within seconds of passing the room. The message names the stage and
  the peak.
- The time estimate for a full TI2V Medium video is within 15% of a real job's
  time.
- `go test ./...` and the JS tests pass.
