# Phase 05 — Field test on asmodean and compute2

**Depends on:** Phases 01–04 (all merged to `main`) · **Enables:** closing
issue #682

## Goal

Check the whole feature on real hardware with a current sd-server build (at or
after 6cd0ef1): ROCm on asmodean (one R9700, 32 GB) and CUDA on compute2
(A4000, 16 GB). Fill in the measured numbers that Phase 04 left open. Check
that the 6cd0ef1 rewrite did not break the video sets we already have. Write
the results down. Fix small problems found here in this phase; file a GitHub
issue for anything larger.

## Files touched

- `plan/wan22-s2v/field-test.md` (new) — the results.
- `internal/generation/catalog.go` — the S2V entry:
  - `FullMeasureTime` and the `Notes` timing, from the measured run
  - a dated comment with the measured peaks, as on the A14B entries
  - flag changes only if a run shows they are needed, each with a
    `FlagNotes` reason
- `docs/reference/measured-reservations.md` — the S2V row's measured values.
- Any code file a small fix needs, each with a test.

## Steps

1. **Build.** Build a test binary from `main`. Its version label is the
   version in `.release-please-manifest.json` plus `-s2v682-test` (for
   example `2.63.0-s2v682-test`). Deploy it with the binary swap, as in
   earlier field tests.
   - asmodean: build on ROCm 10 with `--rocm-image 10.1.0-full`; keep the old
     binary as `.orig`.
   - compute2: `docker cp` into the container; keep the old binary.
2. **sd-server.** On each machine, build sd.cpp at the latest master in
   toolchest (it must be at or after 6cd0ef1), and make that build active.
   Check that the custom set form lists `wan22-s2v`.
3. **Old sets on the new build.** Run one Short video with Wan2.2 TI2V 5B and
   one with Wan2.2 I2V A14B Lightning on each machine. Record pass or fail and
   the time. A failure is filed as its own issue; it does not block S2V (as
   the overview says).
4. **Download and Measure.** Download "Wan2.2 S2V 14B (Q4)", start an
   instance, and run Measure for Short, Medium and Full. Record the time and
   peak of each. On compute2, if Full does not fit in 16 GB, record the
   largest size that does; Measure's own limit then applies there.
5. **Jobs.** Use a portrait and a 2.9 s speech clip, made once as WAV, once as
   MP3 and once as FLAC. 2.9 s gives 47 frames, rounded up to 49; the margin
   covers the short padding an MP3 encoder adds (up to 3.06 s still gives
   49). Send each to `POST /v1/videos/jobs` with no length.
   For each job, check:
   - The job shows `frames` 49 and `fps` 16, and the result's `frame_count`
     is 49.
   - The WebM plays, the lips follow the speech, and it has the speech as
     its sound track.
   - The job listing says "a start image and audio".
6. **Refusals**, once on either machine:
   - no audio, and no start image, on the S2V set
   - audio on Wan2.2 TI2V 5B
   - an OGG file, and a broken MP3 (the Phase 01 `broken.mp3`): both get
     `bad_audio` with a message that starts "audio must be WAV, MP3 or FLAC"
   - a 7 s clip with no length (refused)
   - the same clip with `frames: 81` (runs, and the video is cut to about
     5 s)
7. **WebP.** One job with `output_format: webp`. It makes a silent animated
   WebP.
8. **Record and update.**
   - Write `field-test.md`: machines, builds, results, times and peaks.
   - Update the catalog entry and the reservations row with the measured
     values.
   - Run `/v1/models` on both machines and check `fixed_fps` 16 and the
     `audio` capability.
9. Leave both machines on the test build, and update the memory notes for
   compute2 and asmodean.

## Build gate

- `go build ./...`
- `go vet ./...`
- `go test ./...`
- `make`
- JS tests
- `make docs`

These run for the catalog and docs commit that ends this phase.

## Test plan

Steps 3–7 are the test plan. The phase passes when:
- Step 4 completes on asmodean for all three sizes, and on compute2 for at
  least Short (the overview's success criterion).
- Every check in step 5 passes on both machines.
- Every refusal in step 6 gives the expected code: `audio_required`,
  `start_image_required`, `input_not_supported`, `bad_audio`, `over_limit`.

## Commit

`docs(video): record the Wan2.2 S2V field test and measured sizes`

## Rollback

Revert the commit. This brings back the "not measured yet" values. To roll a
test machine back, put its old binary (`.orig`) back.
