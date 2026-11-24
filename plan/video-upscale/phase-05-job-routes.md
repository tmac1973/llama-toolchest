# Phase 05 — Video upscale job routes and /v1/models

**Depends on:** Phase 01 (`videofile.Probe`), Phase 02 (the kind and sets), Phase 03 (launch, work folders), Phase 04 (`RenderVUpscale`, `runDGLJob`) · **Enables:** Phase 06 (Measure adds the ceiling check to these routes), Phase 07 (docs describe the routes)

## Goal

Clients can upscale a video through the API. `POST /v1/videos/upscale/jobs` takes a clip, checks it and starts an async job on a vupscale instance. Clients follow the job, download the result, list jobs and delete them, as with `/v1/videos/jobs`. `/v1/models` lists vupscale instances with `meta.kind = "vupscale"` and their limits in `meta.vupscale`. Jobs persist across restarts like video jobs. After this phase the feature works end to end. The only part left out is the measured ceiling, which Phase 06 adds.

## Files touched

- `internal/api/gen_vupscale_jobs.go` — new: the job type, input reading, checks, submit/run, and the five handlers.
- `internal/api/server.go` — five routes under `/v1` with `apiKeyAuth`, beside `/videos/jobs`.
- `internal/api/gen_jobs.go` — open a third `jobStore` for vupscale jobs in `openGenJobs`, and add it to `watchGenJobs` pruning.
- `internal/api/v1models.go`:
  - `v1ModelKinds` gets `"vupscale"`;
  - `genKindLabel` maps it;
  - `genV1Model` stops skipping `KindVUpscale` (added in Phase 02) and fills `api` and `meta.vupscale` (step 7);
  - the unknown-kind error text lists vupscale.
- `internal/api/server.go:633` — the kind-name list comment and parser accept `vupscale`/`Vupscale`.
- `internal/generation/queue.go` — no change: `Submit` takes any `BackendKind`.
- `docs/api.md` — one row per new route in the route table, and `vupscale` in the kind list (Phase 07 writes the full text and OpenAPI).
- Tests: `internal/api/gen_vupscale_jobs_test.go` (new), `v1models_test.go`, `gen_jobs_test.go`.

## Steps

1. **Routes:**
   - `POST /v1/videos/upscale/jobs` → `handleVUpscaleJobCreate`
   - `GET /v1/videos/upscale/jobs` → list
   - `GET /v1/videos/upscale/jobs/{id}` → get
   - `GET /v1/videos/upscale/jobs/{id}/video` → the file, with `http.ServeFile`
   - `DELETE /v1/videos/upscale/jobs/{id}` → the shared `jobRoutes.delete` (`gen_jobs.go:562`): a finished job's record and files are removed; a queued or running job is cancelled (step 10) and its record stays, with status cancelled, as for video jobs

   Reuse `jobRoutes` (`gen_jobs.go:510`) for list, get and delete, as the image and video jobs do.
2. **Request fields:**
   - `model` (required): a vupscale instance or set ID. The create handler picks the instance at once, with the same code video jobs use (the shared eligibility check at `gen_images.go:64`, then the same choice among several eligible instances). The job is bound to that instance: its folder, its measurement and ceiling (Phase 06), and its runner all use it.
   - `video` (required): the clip.
   - `output_format`: `webm` (default) or `avi`.
   - `seed`: default random, returned in the job.
   - `window_size`: default `DefaultWindowSize(script)`. When the instance is measured and `MeasuredWindow` is smaller, the default is `MeasuredWindow` (Phase 06 adds that field; until then the first rule alone applies).

   Any other field → 400 "unknown field <name>".
3. **Reading the clip.**
   - **Multipart:** read with `r.MultipartReader()` and stream the `video` part straight to `<data>/gen-output/vupscale/incoming/<job id>.part` with a 512 MiB limit (`maxVUpscaleUploadBytes`). The part order does not matter, because this folder does not depend on the model. The other parts are small text fields (max 4 KiB each).
   - **JSON:** `video` is base64 or a data URL. The body limit is `maxImageRequestBytes` (128 MiB), decoded into memory and then written to the same `.part` file.
   - Over a limit → 413 "video is larger than <N> MiB; send it as multipart (up to 512 MiB) or split it".
   - The job ID is made before the body is read, so the file name is known.
4. **Checks before any GPU work**, in this order. Each failure deletes the `.part` file and returns 400 with the given message. Only when every check passes is the file renamed to `<instance dir>/input/<id>.<format>`, after `VUpscaleWorkDirs` (Phase 03) has made sure the folder exists (both folders are under the data folder, so this is a rename, not a copy).
   1. `videofile.Probe` on the file. Its format errors pass through unchanged (they name the supported formats).
   2. `output_format` is webm or avi.
   3. `WindowSizeOK(set.Recipe, window_size)`.
   4. **Input limit:** `info.DecodedBytes() <= 256 MiB` → else "the clip decodes to <X> MiB; sd.cpp reads at most 256 MiB (3 × width × height × frames). This clip fits at most <N> frames at <W>×<H>".
   5. **Output limit:** `3 × (W×scale) × (H×scale) × frames <= 4 GiB` → else the same shape of message with the 4 GiB limit.
   6. **SeedVR2 window limit:** for seedvr2, `4 × 3 × (W×scale) × (H×scale) × padded window <= 256 MiB`, where the padded window is `window_size` rounded up to `4n+1` → else "the window is too large for sd.cpp's 256 MiB tensor limit at <W×scale>×<H×scale>; use window_size <largest that fits>".
   7. **RelayVSR window limit:** for relayvsr, `3 × (W×scale) × (H×scale) × window_size <= 1 GiB`. One byte per value: `relayvsr::decode` returns U8 frames (the example passes them straight to `video::save`), while SeedVR2's decode is F32 (`dtype = "f32"` in its script), hence 4 bytes in check 6. If sd.cpp still reports a tensor-limit error for a clip that passed, Phase 04's error mapping names it → else the same shape of message with 1 GiB.
   8. `info.Frames >= 1`; for relayvsr also `info.Frames >= 2` ("RelayVSR needs at least 2 frames").
5. **Job record** (`vupscaleJob`, embedding `genJob` as `videoJob` does), with these public fields:
   - `progress`: `{stage, detail, completed, total, elapsed_ms}`
   - `cancel_requested`
   - `request`: `{model, output_format, seed, window_size}`
   - `input`: `{format, width, height, frames, fps, has_audio}`. `fps` is `FPSNum / FPSDen` rounded to 3 decimals, or `null` when the file gives no rate.
   - `audio_dropped`: true when `has_audio` and the output is webm
   - `video`: `{url, mime_type, format, frames, width, height, bytes}` once done. `mime_type` is `video/webm` or `video/x-msvideo`.

   The private part keeps `input` and `path`, the paths of the input and output files.
6. **Run** (`runVUpscaleJob`, shaped like `runVideoJob`):
   1. `genQueue.Submit(gpu, inst, KindVUpscale)`, then `ticket.Wait`.
   2. `genReady` starts the bound instance, or evicts others for it, as for video. A start error fails the job with that error as its message.
   3. `RenderVUpscale` with `InputPath = "input/<id>.<ext>"` and `OutputPath = "output/<id>.<format>"`.
   4. `runDGLJob`, with progress written to the job.
   5. On success, move the file from the instance's `output/` to `<data>/gen-output/vupscale/<id>.<format>` (`vupscaleDir`), set `video`, delete the input file, and set the instance's `LastVideo` as video jobs do.
   6. On failure or cancel, delete the input and any partial output.
   7. `Video.Width`, `Height` and `Frames` come from the DGL result. When they differ from `input × scale` (and the input frame count), the job still succeeds, but logs a warning with both sizes.
7. **`/v1/models`** for a vupscale instance:
   - `meta.kind = "vupscale"`, `api = ["video.upscale.jobs"]`.
   - `meta.vupscale = {scale, script, default_window_size, window_sizes, input_formats: ["avi","webm","webp"], output_formats: ["webm","avi"], max_decoded_bytes: 268435456}`.
   - `window_sizes` is `{min, max}` for relayvsr, and the list `[1,5,…,33]` for seedvr2.
8. **Persistence and pruning.**
   - Records go in `<data>/gen-output/jobs/vupscale/`, handled by the generic `jobStore`.
   - On restore, an unfinished job is marked failed with "the server restarted before the job finished", and its input file is deleted.
   - Pruning uses `videoJobMax` (50) and the 24 h rule. Orphan files in `gen-output/vupscale/` (including `incoming/`) and in each vupscale instance's `input/` and `output/` folders are removed by ID prefix.
9. **Errors** use `openAIError`, as the video routes do:
   - unknown model → 404;
   - model is not a vupscale set → 400 "model <id> is a <kind> set; video upscaling needs a vupscale set";
   - no eligible instance → 503 with the existing reasons (for example "has no VRAM reservation yet; measure it, or set a reservation …"). Until Phase 06, a hand-set reservation is the way to use an instance.
   - the bound instance's build fails `BuildHasDGL` (Phase 03) → 503 with Phase 03's build message, checked in the create handler before the 202, so the client sees it at once. Other start errors can only show up later, as the job's failure message (step 6).
10. **Cancel** (DELETE):
   - A queued job (its ticket has not started) is removed from the queue and marked cancelled. Its input file is deleted.
   - A running job, in this order:
     1. Set `cancel_requested` and mark the job cancelled. A cancelled status is final: when `runVUpscaleJob` ends later with any error, it leaves a cancelled job as it is (as `genJob.start` already does).
     2. Cancel the runner's context. `runDGLJob` returns at once (Phase 04).
     3. sd-server cannot cancel a DGL job, so stop the instance with the same stop function the Server page's Stop button uses.
     4. Delete the job's input and any partial output. The next job starts the instance again.
   - Log at info level: "stopping <instance> to cancel video upscale job <id>; sd-server cannot cancel DGL jobs".

## Build gate

- `make build`
- `go vet ./...`
- `go test ./...`
- `make js-test`

## Test plan

- Handler tests with a fake sd-server (the Phase 04 fake) and fixtures from Phase 01:
  - Multipart WebM → 202; the job completes; `GET …/video` returns the fake's output file; `audio_dropped` is false.
  - JSON base64 AVI with PCM audio and `output_format: webm` → `audio_dropped` is true.
  - MP4 → 400 with the Phase 01 message; no file is left in `input/`.
  - Each of the limits in step 4 with a crafted `Info` (call the check function directly with an `Info`, so no huge fixture is needed).
  - `window_size: 4` on seedvr2 → 400.
  - Unknown field → 400. An image set as `model` → 400 with the kind message.
  - DELETE while running → the instance is stopped (the test's stop hook is called); the job ends cancelled; the files are gone.
  - DELETE while queued → the instance is not stopped; the job ends cancelled.
  - Restore: write a running record, reopen the store → the job is failed and its input file is removed.
- `/v1/models?kind=vupscale` lists the instance with the step 7 meta; `?kind=upscale` does not list it.
- Manual on asmodean (RelayVSR ×2 instance):
  - `curl -F model=<id> -F video=@clip.webm http://asmodean:3000/v1/videos/upscale/jobs`
  - Poll the job, download the result, and probe it: 2× size, same frame count.

## Commit

`feat(api): video upscale jobs at /v1/videos/upscale/jobs`

## Rollback

Revert the commit. The routes and the `/v1/models` entries go away. Job records and files under `gen-output/jobs/vupscale/` and `gen-output/vupscale/` stay on disk; delete those two folders by hand. Nothing else reads them.
