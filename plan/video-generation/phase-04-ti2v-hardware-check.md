# Phase 04 — Wan2.2 TI2V 5B End to End on Real Hardware

**Depends on:** Phase 01 (the TI2V 5B set), Phase 02 (Measure for video),
Phase 03 (the video job API) · **Enables:** Phases 05 and 06, which repeat this
check for their sets against a pipeline proven here

## Goal

The whole video path is proven on real GPUs with the first catalog set before
the bigger sets are added:
- the catalog download;
- Measure;
- a job for each kind of video the set can make, in both formats;
- the waiting route, cancel and eviction.

The measured times and reservations go into the catalog notes, the help page's
reservations table and a verification record. Any bug found is fixed in this
phase's PR.

## Files touched

- `plan/video-generation/verification.md` (new): machines, what was run,
  results, bugs found and fixed, and what was not run, laid out like
  `plan/generation-backends/verification.md`.
- `internal/generation/catalog.go`: the TI2V 5B notes gain the measured time
  per test video and per 5-second 1280×704 video, and the memory used.
- `web/templates/help.html`: the measured reservations table gains a
  "Wan2.2 TI2V 5B" row (R9700 and A4000 columns).
- Any file a bug fix needs, named in `verification.md` with the fix.

## Steps

1. **Build.** On asmodean, build stable-diffusion.cpp with the ROCm profile
   from toolchest's Builds tab, at the newest `master-*` tag on the day, which
   is at or after `master-1000-95ff762`, and record the tag in
   `verification.md`. If that tag's `examples/server/recipes/*.json` differ
   from what Phase 01 copied for the recipes in `VideoRecipes`, update the
   table to match in this phase and note it. Check in the build log that
   libwebm is built: the line `LIBWEBM` appears, and `sd-server` answers
   `output_formats_by_mode.vid_gen` with `webm` on
   `GET /sdcpp/v1/capabilities` once a video model is loaded. Deploy the
   branch build of toolchest with the usual binary swap.
2. **Download and measure on asmodean.** Download Wan2.2 TI2V 5B from the
   Models page and create an instance on GPU 0. Disable the other instances on
   that GPU (their reservations would otherwise block the measurement), then
   press Measure. Record:
   - the peak and the reservation;
   - how long Measure took.
3. **Jobs on asmodean.** Through `/v1/videos/jobs`, with curl:
   1. Text-to-video, defaults (832×480, 33 frames), WebM.
   2. Image-to-video with a photo as `start_image`, sent as a multipart form,
      WebM.
   3. The same request as animated WebP.
   4. 1280×704 at 121 frames (the ceiling), WebM.

   Record for each: the time, the file size, the frame count reported, and
   whether the file plays in Firefox and Chromium. During job 4, check that
   `progress` moves.
4. **Preview cost.** Send job 1's request directly to the instance's
   `/sdcpp/v1/vid_gen` with `"preview": "none"`, bypassing toolchest (which
   always asks for `proj` previews and refuses a `preview` key in `extra`).
   Compare its time with toolchest's run of job 1. Record the difference in
   `verification.md`. If `proj` costs more than 5% of
   the job time, change toolchest's preview setting to send previews every
   5 steps (`preview_interval` 5) and record the new figure.
5. **Refusals.** Check that each of these answers 400 with a message that says
   why, and that `GET /api/gen/queue/history` shows no job for it:
   - 122 frames;
   - an `end_image` (the set has no `end_image`);
   - `output_format: "avi"`.
6. **Waiting route and cancel.** Run one default video through
   `/v1/videos/generations`. Then start one and press Ctrl-C in curl part-way:
   the sd-server log shows the job cancelled, and the GPU queue is free within
   seconds. Cancel one async job with `DELETE` while it generates, with the
   same check.
7. **Eviction and locking.** Turn eviction on in Settings. Check two things:
   1. On GPU 0, keep Ming-Image (10.5 GiB, unlocked) and klein (11.5 GiB)
      running, stop the TI2V instance, and set its reservation by hand to
      12 GiB, so the three cannot fit together (34 GiB against the R9700's
      budget of about 27.6 GiB) but TI2V fits once klein is stopped. A video
      job then stops klein only (the largest), starts TI2V and finishes, and
      Ming-Image keeps running. Set TI2V's reservation back to its measured
      value afterwards.
   2. On GPU 0, lock Ming-Image (10.5 GiB) and keep it running, stop the
      TI2V instance, and set its reservation by hand to 20 GiB, so the two
      cannot fit together in the R9700's budget. A video job is then refused
      with the fit message, and Ming-Image keeps running. Set TI2V's
      reservation back to its measured value afterwards.

   Turn eviction back off and put the instances back as they were.
8. **compute2 (16 GB A4000).** Repeat steps 2 and 3, jobs 1 and 2, with a CUDA
   build on one A4000. Record the peak, the reservation and the times. TI2V 5B
   must succeed here; a failure is a bug to fix in this phase.
9. **Write it down.** Fill in `verification.md`:
   - the machines;
   - each step's result;
   - the bugs found and how they were fixed;
   - anything not run, and why.

   Add the measured figures to the TI2V 5B catalog notes and to the help
   page's reservations table.

## Build gate

- `go build ./...`
- `go vet ./...`
- `go test ./internal/...`
- `gofmt -l internal/` prints nothing.

## Test plan

This phase is the test. It passes when:
- all of steps 2–7 succeed on asmodean;
- steps 2 and 3 (jobs 1 and 2) succeed on compute2;
- every result is in `verification.md`.

Each bug fixed in this phase gets an automated test in the package it touches,
reproducing what failed on hardware.

## Commit

`test(generation): check Wan2.2 TI2V 5B video end to end on R9700 and A4000`

## Rollback

Revert the commit. This removes the verification record, the measured notes
and the help table row; code fixes made here are reverted with it, so revert
them only together with what they fix. No stored data changes.
