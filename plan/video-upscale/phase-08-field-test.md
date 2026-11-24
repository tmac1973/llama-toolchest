# Phase 08 — Field test on compute2 and asmodean

**Depends on:** Phases 01–07 (the whole feature, merged to `main`) · **Enables:** the release, and closing issue #698

## Goal

Run the three catalog sets on both test machines, record what they need and how fast they are, and fix what the test finds. The results go into `plan/video-upscale/field-test.md`, `docs/reference/measured-reservations.md` and the backend timings in `docs/guides/generation.md#video-upscaling`. A set that upstream cannot run on one backend gets a catalog note, not a blocker (see the overview).

## Files touched

- `plan/video-upscale/field-test.md` — new: the steps, the results and the findings.
- `docs/reference/measured-reservations.md` — a new "Video upscalers" table: set, card, Measure size and window, peak, reservation.
- `docs/guides/generation.md` — the timings and VRAM numbers in the Backends part of `#video-upscaling`, and the step 7 numbers in its Limits part.
- `internal/generation/catalog.go` — the catalog notes get the measured VRAM and speed. A set that does not run on a backend gets the sentence "Does not run on <backend> with stable-diffusion.cpp <build> (<short reason>)."
- Any fix the test finds goes in its own PR, with a test, before the field test PR merges.

## Steps

1. **Builds.**
   - Build the newest stable-diffusion.cpp master on both machines (asmodean: `--rocm-image 10.1.0-full`, as always).
   - Deploy the `main` build of toolchest that has Phases 01–07: on compute2 and asmodean, `docker cp` the binary in, as in earlier field tests. Keep the current release binary as `.<version>`.
2. **Test clips.** On asmodean, make three clips with the app's own Wan video set:
   - A: 832×480, 33 frames, webm;
   - B: 832×480, 81 frames, webm;
   - C: 480×272, 49 frames.

   Wan clips have no sound, so on the dev machine make clip D from C with a test tone as its audio: `ffmpeg -i C.webm -f lavfi -i sine=frequency=440 -shortest -c:v mjpeg -q:v 3 -c:a pcm_s16le D.avi`.

   Also keep a short MP4 for the refusal test. Copy the clips to compute2.
3. **Probe check.** Copy the clips to the dev machine and run the Phase 01 `TestProbeField` on them. `Probe` reads clips A and B with the sizes and frame counts that the video jobs reported. This is the real-sd.cpp-output check that Phase 01 deferred to this phase.
4. **For each machine and each set** (RelayVSR ×2, RelayVSR ×4, SeedVR2 3B):
   1. Download the set and add an instance on one GPU (compute2: GPU 2, with the other gen instances, as usual).
   2. Measure:
      - relayvsr at 832×480, 81 frames, window 16;
      - seedvr2 at 832×480, 81 frames, window 1, then again at window 5.

      Record the peak, the reservation and the time.
   3. Upscale clip B with `output_format` webm and again with avi. Upscale clip D with avi and again with webm. Check:
      - size × scale and the same frame count (copy the outputs to the dev machine and run `TestProbeField`);
      - D's tone is kept in the avi output (play it), and `audio_dropped` is true for the webm run of D;
      - the result looks right (no tiling seams, no colour shift, no flicker at window joins). Watch it on the dev machine.
   4. Record the wall time per frame.
5. **Refusals** on asmodean, once:
   - the MP4;
   - a clip over the measured frames (Measure at 33, send B);
   - window 4 on seedvr2;
   - an instance pinned to build 1106 (the Phase 03 message).
6. **Restart and cancel.** Start a long job (B on RelayVSR ×4), and restart the container during the job. The job shows failed with the restart message, and the instance's `input/` folder is empty. Start it again and DELETE it during the run: the instance stops, the job shows cancelled, and the GPU frees within a minute. Send one more job: the instance starts again and the job completes.
7. **VRAM against clip length.** On asmodean RelayVSR ×2, measure at 33 frames and at 81 frames. Record both peaks in `field-test.md`. The ceiling keeps the frame check (Phase 06) either way. The two numbers go in the docs Limits part, so users know how much the frame count costs.
8. **Custom set.** On compute2, download `seedvr2_7b_sharp_int8_convrot.safetensors` and `seedvr2_ema_vae_fp16.safetensors` through the custom set form (Browse). Build a SeedVR2 7B Sharp set with scale 2, then Measure it and upscale clip A.
9. Write `field-test.md`:
   - a results table (machine × set: peak, reservation, s/frame, pass/fail);
   - the refusal results;
   - each finding with its fix PR, or the catalog note it became.
10. Update the docs and catalog notes with the numbers, and open the field test PR. Close #698 when it merges.

## Build gate

- `make build`
- `go vet ./...`
- `go test ./...`
- `make js-test`
- `make docs`

## Test plan

This phase is the test. It passes when every row in step 4 passes, or has a catalog note for an upstream backend failure. Steps 5–8 must also pass.

## Commit

`docs: video upscaling field test results`

## Rollback

Revert the commit. Only notes and docs numbers change, so a revert is safe at any time.
