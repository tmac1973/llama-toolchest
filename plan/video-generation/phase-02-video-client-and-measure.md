# Phase 02 — sd-server Video Client and Measuring Video Sets

**Depends on:** Phase 01 (the video kind, video set fields and the TI2V 5B
set) · **Enables:** Phase 03 (the video job API uses this client)

## Goal

Toolchest can ask an sd-server instance for a video through sd-server's native
async API and get the finished file back, and Measure works for video sets. A
video set is measured with one test video at its test size and length, in the
heaviest mode it supports (start and end images when it takes them), with a
30-minute limit. After this phase, a Wan2.2 TI2V 5B instance gets its
reservation from Measure like any image set.

## Files touched

- `internal/api/gen_video.go` (new): the sd-server video client
  (`runVidGen`), the request builder, and the drawn test picture.
- `internal/api/gen_video_test.go` (new): client tests against a fake
  sd-server.
- `internal/api/gen_vram.go`: `runMeasureGeneration` handles `KindVideo`;
  `measureInstance` uses a 30-minute limit for video sets.
- `web/templates/help.html`: the Measure step says video sets make one short
  test video, which can take several minutes.

## Steps

1. **Request type.** In `gen_video.go`, define `vidGenRequest`, the subset of
   sd-server's `POST /sdcpp/v1/vid_gen` body toolchest sets:
   - `prompt`, `negative_prompt`, `width`, `height`, `seed`, `video_frames`,
     `fps`;
   - `init_image` and `end_image`, as base64 strings or nil;
   - `output_format` (`webm` or `webp`), `preview` (always `"proj"`, so the job
     reports its step; see step 3) and `preview_interval` (1);
   - `extra map[string]any`, merged into the JSON body last, at the top level.

   Fields the set's launch settings already give (steps, CFG, flow shift,
   sampler) are not sent: sd-server uses its startup defaults for anything a
   request leaves out.
2. **Submit and poll.** `runVidGen(ctx, base string, req vidGenRequest,
   onProgress func(vidProgress)) (vidResult, error)`:
   1. POST the body to `base + "/sdcpp/v1/vid_gen"`. Expect `202`; any other
      status is an error that carries sd-server's `error` text.
   2. Poll `base + "/sdcpp/v1/jobs/{id}"` every 1 s until `status` is
      `completed`, `failed` or `cancelled`.
   3. On each poll while generating, call `onProgress` with `preview.pass`,
      `preview.step` and `preview.total_steps` when present.
   4. On `completed`, decode `result.b64_json` and return
      `vidResult{Data, MimeType, Format, FPS, Frames}` from `result.mime_type`,
      `result.output_format`, `result.fps` and `result.frame_count`.
   5. On `failed`, return sd-server's `error`.
   6. When `ctx` ends, POST `base + "/sdcpp/v1/jobs/{id}/cancel"`, keep polling
      (with a 60-second limit of its own) until the status is `cancelled` or
      terminal, and then return `ctx.Err()`, so the GPU is free when it
      returns.

   All requests use `genHTTPClient`, which has no timeout; the context bounds
   the wait.
3. **Previews stay small.** The job JSON includes `preview.b64_json` on every
   poll. Decode the poll response into a struct that leaves that field out
   (`json:"-"` on a field of the same name), so toolchest never holds preview
   pictures.
4. **Test picture.** `measurePicture(w, h int, end bool) []byte` draws a PNG
   with Go's `image` and `image/png`: a vertical sky-to-ground gradient with a
   filled circle (a sun) at one third of the width, or at two thirds when
   `end` is true, so a start-and-end video has something to move between. No
   picture ships with the program.
5. **Measure for video.** In `runMeasureGeneration`, for `KindVideo`, build a
   `vidGenRequest`:
   - prompt "a sunrise over a calm sea, slow camera pan";
   - `DefaultWidth`×`DefaultHeight`, `DefaultFrames` frames at `DefaultFPS`;
   - format `webm`, seed 42;
   - `init_image` = `measurePicture(w, h, false)` when the set has
     `start_image`, and `end_image` = `measurePicture(w, h, true)` when it has
     `end_image`. This is the heaviest mode the set supports.

   Run it with `runVidGen` and discard the result. A set with neither `text`
   nor `start_image` cannot be measured: return "this set has no supported way
   to make a test video".
6. **30-minute limit.** In `measureInstance`, replace the fixed 10-minute
   context with `measureTimeout(set.Kind)`: 30 minutes for `KindVideo`, 10
   minutes for the others. A named function keeps the difference in one place.
7. **Help.** In `help.html`'s generation setup steps, the Measure item adds:
   "A video set makes one short test video at its test size (using a start and
   an end picture when the set takes them); this can take several minutes."

## Build gate

- `go build ./...`
- `go vet ./...`
- `go test ./internal/api/ ./internal/generation/`
- `go test -race -run 'Video|Measure' ./internal/api/`
- `gofmt -l internal/` prints nothing.

## Test plan

- `TestRunVidGenCompletes` (fake sd-server): POST returns 202 with an ID;
  two polls return `generating` with a preview (step 5 of 20, then 10 of 20);
  the third returns `completed` with base64 data. The client returns the
  decoded bytes, MIME type, frame count and fps, and `onProgress` saw both
  steps.
- `TestRunVidGenFails`: a `failed` job's `error` text comes back as the error.
- `TestRunVidGenCancelsOnContextEnd`: cancelling the context while
  `generating` makes the client POST `/cancel` and wait until the fake reports
  `cancelled`.
- `TestRunVidGenRefusedSubmit`: a 400 from sd-server comes back with its
  error text.
- `TestMeasureVideoUsesHeaviestMode` (fake sd-server): for a set with
  `start_image` and `end_image`, the vid_gen body carries both pictures,
  the test size, `DefaultFrames` and `DefaultFPS`. For a set with only `text`,
  neither picture is sent.
- `TestMeasureTimeout`: 30 minutes for video, 10 for image and music.
- `TestMeasurePictureIsPNG`: decodes as a PNG of the asked size, and the start
  and end pictures differ.
- Manual, on asmodean: Measure the Wan2.2 TI2V 5B instance; it sets a
  reservation, and the measurement note shows its peak.

## Commit

`feat(generation): measure video sets with a test video through sd-server's video API`

## Rollback

Revert the commit. Video sets go back to the Phase 01 message ("set the
reservation by hand"); reservations already set by Measure stay as they are and
keep working. Nothing on disk changes.
