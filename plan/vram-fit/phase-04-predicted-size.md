# Phase 04 — Predicting What a Larger Size Needs, and Measuring Inside a Chosen Budget

**Depends on:** Phase 01 (short Measure, times from the log) · **Enables:**
nothing

## What was built (2026-10-29)

Built as written, with these changes:
- **Points carry the decode and stopped runs.** `MeasurePoint` also notes
  whether the decode fitted only in smaller pieces (`Fallback`). With that,
  no line is drawn across two different kinds of decode. It also notes a
  test stopped for going past the room (`Over`): that size, and any size
  above it, "will not fit".
- **A prediction alone never says "will not fit".** For LTX-2.3 the line ran
  2.5 GiB high: 28.8 GiB predicted for Full against 26.3 GiB measured,
  because its tiled decode stops growing with the length. That is outside
  the 1.5 GiB in "Done when". A prediction past the limit says "may not fit,
  measure to check".
- **Inside a budget, the reservation is at least the budget.** TI2V Medium
  inside 14 GiB peaked at only 10.6 GiB, but sd-server may use its whole
  budget.
- **The dialog shows the points of the budget the instance runs in,** and
  says which budget that is.
- **No time estimate.** Phase 01 found step times too uneven to estimate a
  job from Measure. The row's last-video time is what shows how much slower
  a smaller budget is.

## Goal

Before running anything long, the user sees roughly what each video size
needs, and can trade speed for memory on purpose.

After this phase:
- **The Measure dialog predicts each size.** After two short measurements
  of a video instance, the dialog shows, for each preset and for a custom size:
  - the predicted peak;
  - whether it fits the room: "fits", "close to the limit, measure to be
    sure", or "will not fit".

  A prediction never grants a size; only a Measure does.
- **Measure can run inside a smaller budget.** The user can measure with less
  than the whole room, for example to leave space for an LLM. sd.cpp then fits
  the job into that budget: it moves weights to system RAM and decodes a few
  frames at a time. Measure reports the time this costs.

## How the prediction works

A video job's peak has two parts:
- **The weights.** sd.cpp writes these in its log before it starts, for
  example `total params memory size = 12254.72MB (VRAM 12254.72MB, ...)`.
  They do not change with the video's size.
- **Working memory.** This grows with the video's size: frames × width ×
  height, as `PixelFrames` already counts it.

From two measurements at different sizes, the working memory per pixel-frame
is (peak₂ − peak₁) / (size₂ − size₁). The prediction for size S is
peak₁ + that rate × (S − size₁).

The line is per instance, because it depends on the set's decode settings:
- with temporal tiling, the decode holds only a few frames at a time;
- untiled, it holds them all.

Where the prediction can be wrong:
- **sd.cpp's fallback decode.** Near the room, the untiled decode runs out of
  memory and sd.cpp retries it tiled. The peak then stops growing along the
  line. A size predicted to be within 2 GiB of the room is "close to the
  limit".
- **Attention without `--diffusion-fa`** grows faster than the size. Every
  catalog video set uses `--diffusion-fa`, so this concerns custom sets only.
  A custom set without it gets no prediction.

## Files touched

- `internal/generation/types.go`: `Instance` gains `MeasurePoints`, a list of
  up to four `{frames, width, height, peak_mib, weights_mib, budget_mib}`,
  JSON `measure_points`. Editing the instance clears it.
- `internal/generation/video_limit.go`: `PredictPeak(points, size)`, which
  returns the predicted peak and ok. ok is false with fewer than two points
  at different sizes, or two points measured under different budgets.
- `internal/api/gen_vram.go`:
  - every video measurement adds a point (the newest replaces the oldest of
    the same size);
  - the weights come from the log line above;
  - `handleGenMeasure` reads an optional `budget_gib`, which must be at least
    the weights plus 2 GiB and at most the room. It is passed as
    `--max-vram`, and the reservation is capped at it.
- `internal/api/gen_instances.go`: the row's data gains the predictions for
  the presets.
- `web/templates/partials/gen_instances.html`: the Measure dialog shows a
  prediction beside each preset, and a "Measure inside" field with "the
  whole room" as its default.
- `web/templates/help.html`: how predictions are made and when to trust them.
- Tests: `internal/generation/video_limit_test.go` (`PredictPeak`) and
  `internal/api/gen_vram_test.go` (points, budget).
- `plan/vram-fit/verification.md`: a Phase 04 section.

## Steps

1. **Points.** After each video measurement (whether it applied or not, and
   when it was stopped early with a known peak), add a point. Read the weights
   from the `total params memory size` line, VRAM part.

2. **Prediction.** `PredictPeak` uses the two points with the largest
   different sizes measured under the same budget. It returns no prediction
   for a set without `--diffusion-fa` in its run arguments.

3. **The dialog.** Beside each preset and the custom size, show the
   prediction and its word ("fits", "close to the limit", "will not fit").
   Each has a tooltip:
   - "Predicted from your measurements at A and B. Measure to grant the size."
   - for "close to the limit": "Near the limit sd.cpp may decode a few frames
     at a time to fit. That works, but the decode is slower."

   With one point, the dialog says "Measure one more size to see predictions
   for the others".

4. **A suggested second size.** When an instance has one point, the dialog
   suggests the next preset up for the second measurement. With Phase 01's
   Measure, that takes a few minutes.

5. **Measure inside a budget.** Add the "Measure inside" field (GiB, default
   the whole room). `measureInstance` passes it as `MaxVRAMMiB` instead of
   the room, and `applyMeasurement` caps the reservation at it. A budget too
   small for even the temporally tiled decode makes sd.cpp fall back to
   spatial tiling, which hung on ROCm (A14B inside 14 GiB, 2026-10-28).
   Phase 01's stuck-run check stops it, and the dialog's tooltip warns that a
   budget below what the decode needs may not work on Radeon cards. Measure
   cannot say how much slower the smaller budget makes a job (Phase 01 found
   step times too uneven to estimate). The instance row's last-video time
   shows it once a real video has been made.

6. **Help.** Explain the two parts of a peak, what a prediction can and cannot
   say, and why a smaller budget is slower (weights in system RAM, frames
   decoded a few at a time).

## Done when

- On the R9700, with TI2V measured at Short and Medium, the Full prediction is
  within 1.5 GiB of a Full measurement inside the same budget.
- On LTX-2.3 (temporal tiling), the Full prediction from Short and Medium is
  within 1.5 GiB of the measured Full.
- TI2V measured inside 14 GiB is granted Medium, and a real Medium video
  inside that reservation finishes.
- `go test ./...` and the JS tests pass.
