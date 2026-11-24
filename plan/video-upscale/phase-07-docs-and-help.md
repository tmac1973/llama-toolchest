# Phase 07 — Docs, Help and the API reference

**Depends on:** Phase 01 (the format error texts), Phase 02 (sets and the form), Phase 03 (the build rule), Phase 04 (the window rules), Phase 05 (routes and `/v1/models` meta), Phase 06 (Measure and the ceiling) · **Enables:** Phase 08 (the field test follows the docs)

## Goal

Describe video upscaling everywhere a user or a client author looks: the docs site, the in-app Help, the API docs page and `openapi.yaml`. The repository rule says that docs change in the same pull request as the behavior. Phases 02–06 therefore each add one line to the matching docs page, and this phase writes the full sections. After this phase a reader can set up a vupscale instance and call the API from the docs alone.

## Files touched

- `docs/guides/generation.md` — new section `## Video upscaling { #video-upscaling }` with the subsections in step 1.
- `docs/api.md` — the route table and kind list in "Image, video and music API" (line 44 area), plus a curl example.
- `web/apidocs/openapi.yaml`:
  - the five `/v1/videos/upscale/jobs` paths;
  - the `VUpscaleJob` and `VUpscaleMeta` schemas;
  - `vupscale` in the `/v1/models` kind enum.
- `docs/guides/download-models.md` — one sentence: video upscaler sets are in the catalog under the "video" tag.
- `web/templates/help.html` — in `#generation`, one paragraph (step 3) that links to `docsURL "guides/generation#video-upscaling"`. In `#generation-api`, "image, video and music" becomes "image, video, video upscale and music". Keep every anchor.
- `CHANGELOG.md` — no hand edit; release-please writes it from the commits.

## Steps

1. **`docs/guides/generation.md#video-upscaling`.** Write in the docs' natural voice, not STE. It has these parts:
   - **What it does.** RelayVSR (×2 and ×4) and SeedVR2 (3B, and 7B Sharp as a custom set) make a clip larger and keep it stable over time. Name the difference from ESRGAN on single frames (flicker).
   - **What you need { #video-upscaling-build }.** stable-diffusion.cpp master-1116 or newer, because these models run as DGL scripts. Give the exact error text from Phase 03 so a search finds it.
   - **Clip formats { #video-upscaling-formats }.**
     - Inputs: AVI (RGB or MJPEG, with PCM or float audio), WebM with VP8 keyframes only (what toolchest's own video jobs write) and animated WebP. MP4 is not accepted.
     - A one-line ffmpeg command to convert MP4 to MJPEG AVI on the client side: `ffmpeg -i in.mp4 -c:v mjpeg -q:v 3 -c:a pcm_s16le out.avi`. The docs page only shows it; the app never runs ffmpeg.
     - Outputs: webm (no audio) or avi (MJPEG with PCM16 audio, keeps audio).
   - **Cancelling.** Cancelling a running job stops the instance, because sd-server cannot cancel a DGL job. The next job starts it again and reloads the weights.
   - **Limits { #video-upscaling-limits }.** The 256 MiB decoded-input rule with a small table of the most frames that fit at 480p, 720p and 1080p (computed from the formula). Also the 4 GiB output rule, the window rules per script, and the measured ceiling, with each side checked on its own.
   - **Window size { #video-upscaling-window }.** What it trades (smoothness against VRAM) for each script. Add a Mermaid diagram of RelayVSR's windows, which share one frame each: frames → windows of 16 → keyframes → upscaled windows → joined clip.
   - **Custom sets.** The Script select, the two file slots and the scale rule. For 7B Sharp: DiT `diffusion_models/seedvr2_7b_sharp_int8_convrot.safetensors` (8.3 GB) and VAE `vae/seedvr2_ema_vae_fp16.safetensors` from `Comfy-Org/SeedVR2`. The repository also has fp16, fp8, mxfp8 and nvfp4 variants, but upstream's SeedVR2 guide names only the int8_convrot files.
   - **Measure.** The dialog fields and what the ceiling then allows.
   - **Backends.** RelayVSR has a fused CUDA kernel and a slower tiled path on ROCm and Vulkan. SeedVR2 was tested upstream on Vulkan only. (Phase 08 adds the measured timings and VRAM here.)
2. **`docs/api.md`.**
   - Rows for the five routes.
   - `meta.kind` `vupscale`, and `meta.vupscale` with every field from Phase 05 step 7 and Phase 06.
   - The request fields and the error texts that clients will want to match: format, limits and ceiling.
   - Example: `curl -F model=relayvsr-x2 -F video=@clip.webm -F output_format=avi http://localhost:3000/v1/videos/upscale/jobs`, then the poll and the download.
3. **Help paragraph** (STE, two sentences, plus the link): "A video upscaler set (RelayVSR or SeedVR2) makes a clip larger and keeps it steady from frame to frame. It needs stable-diffusion.cpp master-1116 or newer and takes AVI, WebM (VP8) or animated WebP, not MP4."
4. Run `make docs` and check that the new anchors resolve. The app's Help link uses `#video-upscaling`.

## Build gate

- `make build`
- `make docs` (the Zensical build has no broken links or anchors)
- `go test ./...` (help.html renders; the OpenAPI file parses in the existing API docs test)

## Test plan

- Open the built docs locally. Each new heading has its id, and the Help link opens `guides/generation/#video-upscaling`.
- Open `/docs/api` in the app. The five routes show with their schemas.
- Read the Help page at phone width. The new paragraph stays short and wraps cleanly.
- Build this branch, `docker cp` the binary into asmodean's container (keep the running binary as `.<version>`), and run the api.md curl example there exactly as written.

## Commit

`docs: video upscaling with RelayVSR and SeedVR2`

## Rollback

Revert the commit. The feature still works; only the full docs text goes away (the one-line notes from Phases 02–06 stay).
