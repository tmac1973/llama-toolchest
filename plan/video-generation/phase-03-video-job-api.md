# Phase 03 — Video Job API

**Depends on:** Phase 01 (video kind, set fields, capabilities), Phase 02 (the
sd-server video client) · **Enables:** Phase 04 (end-to-end checks on real
hardware) and the later catalog sets, which only add sets to this API

## Goal

Programs can make videos through toolchest on port 3000:
- Async jobs: submit, poll (with phase and progress), download, cancel.
- A waiting route that answers with the finished job.

Requests are checked against the set's abilities and ceiling before anything is
queued. Jobs take their turn in the per-GPU queue and start a stopped instance
when eviction is on, like image jobs. Finished videos are saved and kept for 24
hours, at most 50 at a time. `/v1/models` advertises the routes and the help
page documents them. The image and music APIs do not change.

## Files touched

- `internal/api/gen_video_jobs.go` (new): job type, request parsing and
  checks, submit and run, the five route handlers.
- `internal/api/gen_video_jobs_test.go` (new).
- `internal/api/gen_jobs.go`: `jobStore` takes its maximum job count as a
  field, so video jobs can keep 50 while images and music keep 200.
- `internal/api/server.go`: the video routes, the video job store field and
  its prune in `watchGenJobs`.
- `internal/api/gen_images.go`: `pickInstance` needs no change; the video
  handlers call it with `KindVideo`.
- `internal/api/v1models.go`: `meta.api` and video limits for video
  instances.
- `internal/api/v1models_test.go`.
- `web/templates/help.html`: a "Video jobs" part of the generation API
  section.

## Steps

1. **Store limit.** In `gen_jobs.go`, add `max int` to `jobStore[T]`, set at
   creation, and use it in `prune` instead of `jobMax`. Image and music stores
   are created with 200 (behaviour unchanged); the video store with
   `videoJobMax = 50`. `jobKeep` stays 24 h for all three.
2. **Routes.** In `server.go`, inside the `/v1` group next to the image
   routes, behind `apiKeyAuth`:
   - `POST /v1/videos/jobs` → `handleVideoJobCreate`
   - `GET /v1/videos/jobs/{id}` → `handleVideoJobGet`
   - `GET /v1/videos/jobs/{id}/video` → `handleVideoJobVideo`
   - `DELETE /v1/videos/jobs/{id}` → `handleVideoJobDelete` (cancel)
   - `POST /v1/videos/generations` → `handleVideoGenerate` (the waiting route)
3. **Request.** Both POST routes accept either a JSON body or a multipart form
   with the same field names (a body over 128 MiB gives 413, as for images):
   - `model`: an instance or set ID, chosen as `pickInstance` does. Optional,
     as for images: left out, the only eligible video instance is used, and
     with several a 400 `model_required` lists them.
   - `prompt`: required.
   - `negative_prompt`.
   - `size`: `"WxH"`; defaults to the set's test size.
   - `frames` or `seconds`: giving both is a 400. `seconds` becomes
     `round(seconds × fps)`. With neither, the set's `DefaultFrames`.
   - `fps`: defaults to the set's `DefaultFPS`.
   - `seed`: defaults to random (-1).
   - `start_image`, `end_image`: in JSON, base64 or a `data:` URL; in a form,
     file parts with those names.
   - `output_format`: `webm` (default) or `webp`; anything else is a 400.
   - `extra`: in JSON an object; in a form a field holding JSON text. It is
     merged into the sd-server request, and may not hold `prompt`,
     `negative_prompt`, `width`, `height`, `seed`, `video_frames`, `fps`,
     `init_image`, `end_image`, `output_format`, `preview` or
     `preview_interval`. Those have fields of their own, and a 400 names the
     key.
4. **Checks before queueing.** They run in this order, after defaults are
   filled in and `seconds` is turned into frames, and before sd-server's 4n+1
   rounding (which only lowers the count). The first failure answers with a
   400 JSON error (`{"error":{"code":…,"message":…}}`, as image errors do) and
   queues nothing:
   - `frames`, `seconds` or `fps` below 1, `fps` above 240 (sd-server's
     limit), or width or height below 64 → `bad_value`.
   - No start image on a set without `text` → code `start_image_required`.
   - A start image on a set without `start_image`, or an end image on a set
     without `end_image` → `input_not_supported`.
   - An end image without a start image → `start_image_required`.
   - Frames over `MaxFrames`, or width/height over `MaxWidth`/`MaxHeight` (in
     either orientation, so 704×1280 is allowed when the ceiling is 1280×704)
     → `over_limit`. The message names the limit, e.g. "Wan2.2 TI2V 5B makes
     at most 121 frames (5 s at 24 fps) and 1280×704".
   - A width or height that is not a multiple of 8 → `bad_size`. Every size in
     sd.cpp's own Wan and LTX examples (832×480, 1280×720, 640×360) meets
     this; sd-server fits sizes to the model itself beyond that.
   - A ceiling field that is 0 (custom sets) is not checked.

   Start and end images may be any size: sd-server fits them to the requested
   video size.
5. **Job type.** `videoJob` mirrors `imageJob`. Visible JSON fields:
   - `id`, `model`, `status`, `phase`, `queue_position`;
   - `progress`: `{pass, step, total_steps}`, present only while generating;
   - `request`: the parsed fields, without the images; it shows whether a
     start or end image was given (`start_image: true`);
   - `video`: present on success, `{url, mime_type, format, frames, fps,
     width, height, bytes}`. `frames`, `fps`, `mime_type` and `format` come
     from sd-server's result; `width` and `height` are the size sent to
     sd-server; `bytes` is the saved file's size;
   - `error`, `submitted`, `started`, `finished`.

   Phases are the image phases: `queued`, `starting`, `generating`, `saving`.
6. **Submit and run.** `submitVideoJob` follows `submitImageJob`:
   `pickInstance(model, KindVideo)` and `genQueue.Submit(gpu, instance,
   KindVideo)` (a full queue gives 429 with `Retry-After: 30`). Then
   `runVideoJob`:
   1. Wait for the ticket.
   2. `genReady` (phase `starting` while it works).
   3. Phase `generating`, then `runVidGen` with a context the job's cancel
      ends, copying progress into the job.
   4. Phase `saving`: write `<DataDir>/gen-output/videos/<id>.webm` or
      `.webp`, then mark the job succeeded with its `video` block. The URL is
      `/v1/videos/jobs/<id>/video`.
   5. `ticket.Done()` in all cases.
7. **Cancel.** `DELETE` on a queued job takes it out of the queue. On a running
   job it ends the run's context: `runVidGen` cancels the job in sd-server and
   waits for it to stop, so the GPU is free when the job shows `cancelled`.
   Unlike image jobs, nothing is left running in the backend.
8. **Download.** `handleVideoJobVideo` answers 409 `not_ready` unless the job
   succeeded. Otherwise it serves the file with `http.ServeFile`, so range
   requests work in browsers, with the job's MIME type.
9. **Waiting route.** `handleVideoGenerate` submits like `handleVideoJobCreate`
   and waits until the job is finished or the client's request context ends.
   - When the job finishes, it answers with the same JSON as `GET` on the job
     (the URL works while the job is kept: 24 hours, or until 50 newer videos
     exist), with status 200 for a succeeded job, 500 for
     a failed one, and 409 for one cancelled by a `DELETE` from elsewhere.
   - When the client leaves first, it cancels the job, as `DELETE` does.
10. **Retention.** `watchGenJobs` prunes the video store like the others:
    finished jobs older than 24 h, or beyond 50, are dropped with their files.
    The existing `prune` already removes the files a job's `state().files`
    lists, so `videoJob.state()` lists its video file there.
11. **/v1/models.** In `genV1Model`, video sets get:
    - `meta.api` `["video.jobs", "video.sync"]`;
    - `meta.capabilities` as stored (`text`, `start_image`, `end_image`);
    - `meta.video`: `{default_size, default_frames, default_fps, max_frames,
      max_size}`, with the max fields left out when the ceiling is 0.

    Image and music entries stay exactly as they are.
12. **Help.** In the generation API section of `help.html`, after the image
    jobs part, add `<h4 id="generation-video-jobs">Video jobs</h4>`. It covers:
    - the routes;
    - the request fields and their defaults;
    - frames versus seconds, and the 4n+1 rounding;
    - the three kinds of video and which capability each needs;
    - the ceiling and its 400;
    - phases and progress;
    - WebM (default, with sound for sets that make it) versus animated WebP
      (silent);
    - the 24-hour / 50-video retention;
    - that cancelling stops the job in sd-server;
    - that the waiting route cancels when the client leaves.

    Include one curl example for each POST route, the waiting one using
    `-F start_image=@photo.png`.

## Build gate

- `go build ./...`
- `go vet ./...`
- `go test ./internal/...`
- `go test -race ./internal/api/ ./internal/generation/`
- `gofmt -l internal/` prints nothing.

## Test plan

All against a fake sd-server (the Phase 02 fake) and a running fake instance:
- `TestVideoJobSucceeds`: submit by JSON; the job goes `queued`/`generating`
  (with progress) to `succeeded`. `video.url` serves the bytes with the MIME
  type; the file is under `gen-output/videos`.
- `TestVideoJobMultipart`: a form with a `start_image` file reaches sd-server
  as base64 `init_image`.
- `TestVideoJobChecks`, table-driven, each a 400 with nothing queued:
  - end image without start;
  - start image on a set without `start_image`;
  - no image on a set without `text`;
  - 122 frames on a 121-frame ceiling;
  - 1920×1080 on a 1280×704 ceiling;
  - 1004×480 (not a multiple of 8);
  - `frames: 0`, `fps: 300`, width 32;
  - both `frames` and `seconds`;
  - `extra` holding `video_frames`;
  - `output_format: "avi"`.
- `TestVideoJobSeconds`: `seconds: 2` at 24 fps sends `video_frames` 48.
- `TestVideoJobOrientation`: 704×1280 is accepted on a 1280×704 ceiling.
- `TestVideoJobCancelRunning`: `DELETE` while generating sends sd-server's
  cancel, and the job ends `cancelled` with no file.
- `TestVideoGenerateWaits`: the waiting route returns 200 with the same JSON
  as `GET`.
- `TestVideoGenerateCancelledElsewhere`: a `DELETE` on the job while the
  waiting route waits makes it answer 409 with the cancelled job.
- `TestVideoGenerateClientLeaves`: ending the request context cancels the
  job in sd-server.
- `TestVideoJobsPrune`: 51 finished videos keep the newest 50, and the
  oldest's file is removed.
- `TestPickInstanceVideoOnly`: an image instance is never picked for a video
  job, nor a video instance for an image job.
- `TestV1ModelsVideoMeta`: a video instance has `meta.kind` `"video"`, the two
  `api` values and `meta.video` limits. An image instance's `meta` is
  unchanged.
- The existing image, music and Haruspex-contract tests pass unchanged.

## Commit

`feat(generation): add a video job API with a waiting route`

## Rollback

Revert the commit. The routes disappear (clients get 404), and saved videos
stay in `gen-output/videos` until deleted by hand. Image and music stores go
back to the shared 200 limit. Phase 01 and 02 behaviour is untouched.
