# Video upscaling (issue #698) — Project Overview

## Problem

llama-toolchest can make short videos (Wan, LTX), but it cannot make them larger. Its only upscaler is ESRGAN, which works on one image at a time. A user who wants a sharper clip must split it into frames, upscale each frame, and join the frames again. The result flickers, because ESRGAN does not look at the frames around each one.

stable-diffusion.cpp added two real video upscalers in November 2026: RelayVSR (×2 and ×4, upstream #2285) and SeedVR2 (3B and 7B Sharp, upstream #2300). They look at many frames at once, so the output stays stable over time. Upstream runs them only as DGL scripts (sd.cpp's Diffusion Graph Language), through `sd-cli -M dgl` or sd-server's `/sdcpp/v1/dgl/*` job routes. llama-toolchest does not use DGL anywhere yet. API clients such as Haruspex therefore cannot use these models.

## Goals

- Add a new generation set kind, `vupscale`, that upscales a video with a DGL script on an sd-server instance.
- Ship three catalog sets: RelayVSR ×2, RelayVSR ×4 and SeedVR2 3B.
- Let users make custom `vupscale` sets in the existing custom set form. They choose one of the built-in scripts (RelayVSR or SeedVR2), the DiT file, the VAE file and the scale. SeedVR2 7B Sharp works this way.
- Add async job routes for video upscaling, shaped like the `/v1/videos/jobs` routes, and list `vupscale` instances in `/v1/models` with their limits.
- Accept the input formats that sd.cpp's `video::load` reads: AVI (RGB or MJPEG), animated WebP and VP8 WebM as written by sd.cpp. This covers the app's own video job output.
- Return `webm` (the default, no audio) or `avi` (MJPEG, with the source audio), as the client chooses.
- Check each request against the `video::load` limit (256 MiB of decoded frames) and against a Measure-based ceiling before the job starts. Refuse a clip that does not fit, with a message that gives the limit.
- Support Measure for `vupscale` instances, so the VRAM reservation fits the real peak.
- Document the feature on the docs site and in Help, the API reference and `openapi.yaml`.

## Non-goals

- No ffmpeg. The app does not read or write MP4 or other formats. MP4 input is refused with a clear error.
- No still-image upscaling with these models. ESRGAN upscale sets and `/v1/images/upscale` stay as they are.
- No splitting of long clips inside the app. Clients split clips that are over the limit.
- No page in the app for submitting jobs. Sets and instances are managed on the existing Models and Server pages; jobs come through the API.
- No general DGL support (user-written scripts, a DGL editor, or DGL for other model types). The app uses only its own built-in upscale scripts.
- No SeedVR2 7B Sharp catalog set. It is possible as a custom set.
- No change to the ESRGAN `upscale` kind.

## Users & primary flow

The user is a llama-toolchest admin who also runs an API client (Haruspex or a script).

1. The admin opens **Models → Download Models** and downloads a `vupscale` catalog set, for example RelayVSR ×2 (a DiT file and a VAE file from `Comfy-Org/RelayVSR`).
2. On the **Server** page, the admin adds a generation instance for that set on a GPU. The admin presses **Measure** and enters a test clip size (width, height, frames) and a window size. The app runs a short upscale, records the VRAM peak and sets the reservation.
3. The client calls `GET /v1/models?kind=vupscale` and reads the set's scale and its limits (`meta.vupscale`).
4. The client sends `POST /v1/videos/upscale/jobs` with the video file (multipart or base64 JSON), the `model` and an optional `output_format`.
5. The app checks the format, frame count and size against the limits. It saves the input to the instance's work folder and submits a DGL job to sd-server, with the script filled in for this set and this request.
6. The client polls `GET /v1/videos/upscale/jobs/{id}` for status and progress. When the job is done, it downloads the result from `GET /v1/videos/upscale/jobs/{id}/video`.

## Constraints

- **sd.cpp builds.** DGL video upscaling needs stable-diffusion.cpp master-1116 or newer. That is the first release with the DGL job routes and the RelayVSR and SeedVR2 bindings, and upstream's two example scripts have not changed since then (checked up to master-1122). A build without DGL (no `--dgl-dir` in `sd-server --help`) is refused before start, with a clear message. A later build that lacks a binding fails at its first job, with a message that names the build it needs. The build is chosen per instance (automatic or pinned).
- **sd-server DGL routes are local only.** They accept only requests from `127.0.0.1` or `::1` with a local Host header. The app already calls sd-server at `http://127.0.0.1:<port>` with no Origin header, so it passes this check. Clients never call these routes directly.
- **Model files must be under an sd-server `--model-dir`.** DGL `load_dit` and `load_vae` refuse other paths.
- **Input formats.** `video::load` reads AVI (24-bit RGB or MJPEG, PCM or float audio), WebP (needs `SD_WEBP`), and WebM with independent VP8 keyframes and optional PCM16 audio (needs `SD_WEBM`). It does not read MP4, VP9, interframe VP8 or segmented OpenDML AVI.
- **Size limits.** Decoded input must fit in 256 MiB (3 × width × height × frames bytes). RelayVSR window outputs may use up to 1 GiB, and the joined output up to 4 GiB.
- **Backends.** RelayVSR has a fused CUDA window-attention kernel and a slower tiled fallback on other backends. SeedVR2 was tested upstream on Vulkan only. Both are marked experimental upstream.
- **Test machines.** compute2 (3× A4000, CUDA) and asmodean (one R9700, ROCm; always build on ROCm 10).
- **Repository rules.** Work on branches off `origin/main`, Conventional Commits with no attribution, docs updated in the same pull request, Help kept short with a `docsURL` link.

## Success criteria

- The three catalog sets download, start and pass Measure on both compute2 and asmodean. A set that upstream cannot run on one backend gets a catalog note instead, and this does not block the release.
- A real Wan clip made by the app (WebM) upscales on each catalog set on both machines, in both `webm` and `avi` output. The output has the right size (input size × scale) and the same frame count.
- An AVI input with audio keeps its audio in `avi` output. In `webm` output the job result says that the audio was dropped.
- MP4 input, an unknown format, a clip over the 256 MiB limit and a clip over the measured ceiling are each refused before any GPU work, with a message that names the cause and the limit.
- An instance on an sd.cpp build without DGL upscaling fails to start with a message that names the needed build.
- A custom SeedVR2 7B Sharp set can be built in the form and runs a job.
- A job that is queued or running when the app restarts is marked failed after restart, like video jobs. Cancelling a running job stops its sd-server instance, because sd-server cannot cancel a DGL job, and this frees the GPU. Cancelling a queued job only takes it out of the queue. Both keep the job record, with status cancelled, as video jobs do.
- `/v1/models` lists `vupscale` instances with `meta.vupscale`. The API docs, `openapi.yaml`, the docs site and Help describe the feature.
- `make` build, `go vet`, Go tests and JS tests pass in CI.

## Decisions

- **Models in the first release** → RelayVSR ×2, RelayVSR ×4 and SeedVR2 3B as catalog sets; 7B Sharp only as a custom set.
- **Input formats** → only what `video::load` reads natively (AVI, WebP, VP8 WebM); no ffmpeg; MP4 is refused.
- **Output format** → the client chooses `webm` (default, no audio) or `avi` (keeps audio); `webm` with audio input reports that the audio was dropped.
- **Set kind** → a new kind, `vupscale`, with its own slots, routes and `/v1/models` kind; the ESRGAN `upscale` kind does not change.
- **Still images** → video only; ESRGAN stays the image upscaler.
- **Custom sets** → yes: the form offers the kind, a choice of the built-in scripts, the DiT and VAE slots and the scale.
- **User surface** → API only; sets and instances on the existing Models and Server pages.
- **Long clips** → refuse over the `video::load` limit and over the measured ceiling; clients split clips.
- **Field test** → both machines and all three catalog sets; an upstream backend failure becomes a catalog note, not a blocker.
- **How the app runs the scripts** → through sd-server's DGL job routes, one long-lived sd-server per instance.
- **Where the scripts come from** → two templates embedded in the app, based on upstream's examples.
- **Checking a clip before the job** → small header readers in Go for AVI, WebP and WebM.
- **Client options** → `seed` and `window_size`; the scale is fixed by the set.
- **Route names** → `/v1/videos/upscale/jobs` (create, list, get, video, delete); no synchronous route.
- **Build check** → the `--dgl-dir` flag in `sd-server --help`; a missing binding shows as the job's error.
- **Cancelling a running job** → stop the instance, because sd-server has no cancel for DGL jobs.
