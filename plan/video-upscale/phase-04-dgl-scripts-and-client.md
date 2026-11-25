# Phase 04 — The DGL scripts and the sd-server DGL client

**Depends on:** Phase 01 (the probe, for the manual check), Phase 02 (`VUpscaleScripts`, the set's `Recipe` and `Scale`), Phase 03 (a running instance, the `input`/`output` work folders) · **Enables:** Phase 05 (job routes), Phase 06 (Measure)

## Goal

Embed two DGL script templates in the app, one for RelayVSR and one for SeedVR2, based on upstream's examples. Add a client that runs a filled-in script as an sd-server DGL job: submit, poll and read the result. sd-server has no cancel for DGL jobs (its `/sdcpp/v1/jobs/{id}/cancel` route returns 404 for them), so the client only stops waiting; Phase 05 stops the instance to cancel. After this phase, a test (and a manual call from a Go test harness on a test machine) can upscale a clip on a running instance. No public route exists yet.

## Files touched

- `internal/generation/dgl/relayvsr.dgl.tmpl` — new template.
- `internal/generation/dgl/seedvr2.dgl.tmpl` — new template.
- `internal/generation/dgl.go` — new:
  - `//go:embed dgl/*.tmpl`
  - `type VUpscaleJob struct` and `RenderVUpscale(set ModelSet, job VUpscaleJob) (string, error)`
  - `dglString(s string) (string, error)`
  - `WindowSizeOK(script string, n int) error` and `DefaultWindowSize(script string) int`
- `internal/api/gen_dgl.go` — new: `runDGLJob(ctx, inst, source string, progress func(dglProgress)) (dglResult, error)`.
- `internal/generation/dgl_test.go`, `internal/api/gen_dgl_test.go` — new tests.
- `internal/api/gen_dgl_field_test.go` — new, build tag `fieldtest`: a manual check against a real instance.
- `internal/generation/testdata/dgl/*.golden.dgl` — golden scripts.

## Steps

1. **`VUpscaleJob`**:
   ```go
   type VUpscaleJob struct {
       InputPath    string // relative to the instance folder, "input/<id>.<ext>"
       OutputPath   string // "output/<id>.webm" or "output/<id>.avi"
       OutputFormat string // "webm" or "avi"
       Seed         int64
       WindowSize   int
       FPS          int    // only for a clip with no rate (still WebP): 24; else 0
   }
   ```
2. **Window sizes.**
   - relayvsr: default 16, allowed 2 to 32. Each window shares one frame with the next.
   - seedvr2: default 1, allowed 1 or `4n+1` up to 33 (1, 5, 9, … 33).
   - `WindowSizeOK` returns an error that names the allowed values.
3. **`dglString`.** Wrap the value in double quotes. Refuse a value that holds `"`, `\`, a newline or a control character, with "path cannot be used in a DGL script: <reason>". Model paths come from the set (users choose file names), so this check matters.
4. **RelayVSR template.**
   - Start from upstream `docs/dgl/relay_vsr/relayvsr-x2-video-upscale.dgl`. Its text is the same in master-1116 and master-1122 (blob `c678a13`).
   - Change only the `var` block, which takes the template values: `dit_path` and `vae_path` (the set's absolute file paths), `input_path`, `output_path`, `seed`, `window_size`. Drop upstream's `fps = 24`: passing `fps_num` to `video::load` resamples the clip to that rate (sd.cpp `plan_video_sampling`), which would change the frame count. The script calls `video::load(input_path)` and keeps the file's own rate. Only a clip with no rate (a still WebP, which `video::load` cannot read without one) gets `fps_num = 24`, through the job's `FPS` field. The save lines copy `clip.video.fps_num`/`fps_den`. (Changed during implementation.)
   - Change the end of `main` to:
     - for `avi`: `video::save(output_path, {frames = pixels, fps_num = clip.video.fps_num, fps_den = clip.video.fps_den}, audio = clip.audio)`. `video::save` writes MJPEG video with PCM16 audio.
     - for `webm`: `media::save(output_path, video::encode({frames = pixels, fps_num = clip.video.fps_num, fps_den = clip.video.fps_den}, format = "webm"))`
   - Return only scalars, never the frames, so the server does not keep a large tensor in the job result: `return { frames = tensor::shape(pixels)[3], width = tensor::shape(pixels)[1], height = tensor::shape(pixels)[2] }`.
   - The template picks the save line with `{{if eq .OutputFormat "avi"}}`.
   - The first line stays `# version: 0.0.0`, with a comment that names the upstream file and commit it came from.
5. **SeedVR2 template.**
   - Start from upstream `docs/dgl/seedvr2/seedvr2-3b-video-upscale.dgl` (blob `c5ca43c`, the same in master-1116 and master-1122), with the same `var` and ending changes as step 4.
   - It also takes `factor` = the set's `Scale`.
6. **`RenderVUpscale`**:
   - Picks the template by `set.Recipe`; an unknown script is an error.
   - Takes the dit and vae paths from the set's components.
   - Checks every path with `dglString` and the window size with `WindowSizeOK`.
   - Executes the template.
   - Golden tests pin the output for: relayvsr webm, relayvsr avi, seedvr2 webm and seedvr2 avi with factor 3.
7. **Audio note.** The script itself does not report audio. Phase 05 knows from the clip probe whether the input had audio, and sets `audio_dropped` in the job when the output is webm. The webm line never passes audio, so `video::encode` never fails on audio.
8. **Client** (`internal/api/gen_dgl.go`). It uses `genHTTPClient` and `inst.BaseURL()` (`http://127.0.0.1:<port>`) and sends no Origin header. sd-server's local-only check accepts this.
   - **Submit:** `POST /sdcpp/v1/dgl/jobs` with `{"source": <script>}` and `Content-Type: application/json`. A 202 response gives `{id}`. A 429 response ("Job queue is full") becomes the error "the upscaler is busy". A 403 response becomes "sd-server refused the DGL request (local-only check); this is a toolchest bug". Log the request Host so the bug can be found.
   - **Poll:** `GET /sdcpp/v1/dgl/jobs/{id}` every 1 s (reuse `vidGenPoll`).
     - Status `completed` → done.
     - `failed` → the error is `error.message` from the job JSON, prefixed "sd.cpp: ".
     - `cancelled` → a cancelled error.
     - Report `progress.stage`, `progress.detail`, `progress.completed`/`progress.total` when given, and `stage_elapsed_ms` through the `progress` callback.
   - **Stop waiting:** when `ctx` ends, return `ctx.Err()` at once and send nothing to sd-server. The caller decides whether to stop the instance.
   - **Result:** read `result.outputs[]` and return `frames`, `width` and `height` by name. sd-server sends i64 values as strings and i32/f32 values as numbers, so accept both.
9. **Known DGL error texts.** Map these sd.cpp messages to clearer ones and keep the original after a colon:
   - a message that holds `exceeds` and `MiB` → "the clip is too large for sd.cpp's DGL limits".
   - `unknown function relayvsr::` or `unknown function seedvr2::` → "this stable-diffusion.cpp build has no <name> support; use master-1116 or newer".

## Build gate

- `make build`
- `go vet ./...`
- `go test ./...`

## Test plan

- Golden tests for the four scripts (step 6).
- `RenderVUpscale` refuses:
  - a model path with a `"` in it;
  - window 4 for seedvr2;
  - window 1 for relayvsr;
  - script `"wan"`.
- Client tests against an `httptest.Server` that acts like sd-server:
  - 202, then two polls of `running`, then `completed` with outputs `frames="17"`, `width=1280`, `height=720` → result 17/1280/720.
  - `failed` with a message → the error text has the "sd.cpp: " prefix.
  - The context ends while the job runs → `runDGLJob` returns `context.Canceled` within one poll interval, and the fake gets no further request.
  - 429 and 403 → the texts from step 8.
- Manual on asmodean, on a RelayVSR ×2 instance started by Phase 03:
  - Copy a 17-frame Wan WebM from `gen-output/videos/` into the instance's `input/` folder.
  - Run `internal/api/gen_dgl_field_test.go` (build tag `fieldtest`, so CI never runs it) with `go test -tags fieldtest -run TestDGLField ./internal/api/`. It reads `VUPSCALE_BASE_URL`, `VUPSCALE_DIT`, `VUPSCALE_VAE` and `VUPSCALE_INPUT` from the environment, renders the RelayVSR script and runs it with `runDGLJob`.
  - The output file in `output/` has twice the size and 17 frames. Check it on the dev machine with the Phase 01 `TestProbeField`, after copying it there.
  - Run `sd-cli -M dgl --script <rendered file> --check` from the same build. It passes for all four golden scripts.

## Commit

`feat(generation): DGL scripts and client for video upscaling`

## Rollback

Revert the commit. Nothing calls the client yet, so no route or data changes.
