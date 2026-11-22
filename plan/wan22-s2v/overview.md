# Wan2.2 S2V (speech-to-video) — Project Overview

GitHub issue: #682.

## Problem

Toolchest's video sets turn a prompt, a start image, or a start and an end
image into a video. None of them can make a person speak. Wan2.2 S2V 14B
(Wan-AI, Apache-2.0) does that: it takes a portrait and a speech or singing
track and makes a video of that person, with lips and head moving in time with
the audio. stable-diffusion.cpp supports it (recipe `wan22-s2v`, the
`--audio-encoder` flag, and `ref_audios` in `POST /sdcpp/v1/vid_gen` since
0f76bf5, 2026-10-25). Toolchest builds the latest sd.cpp by default, and each
build brings its own recipe book, so the set works on any build whose recipes
include `wan22-s2v`.

The video plan left S2V out on purpose, because the video job API cannot carry
audio: `recipeLeftOut` in `internal/generation/recipe_overlay.go` marks
`wan22-s2v` "needs an audio input", and the non-goals in
`plan/video-generation/overview.md` say the same. So the work is mostly a new audio input
for video jobs. The catalog set itself is small.

What sd-server does with S2V (read at e41fbcf in `src/pipeline/video.cpp`,
`src/pipeline/request.cpp` and `examples/common/common.cpp`, and at bf31b9c,
2026-11-22, in `src/api/generation_video.cpp`):
- `ref_audios` is an array of base64 strings (a `data:` prefix is allowed).
  Each must be a WAV; any sample rate and channel count, which sd-server
  downmixes and resamples to 16 kHz mono for wav2vec2.
- S2V takes exactly one audio track. The start image (`init_image`) is the
  speaker portrait; an end image is ignored with a warning.
- `video_frames` sets the length. Audio past the end is cut; if the audio is
  shorter, silence fills the rest. The recipe's frame rule is 4n+1 with
  `frame_rounding: "floor"` (in the 984eb06 snapshot), so a frame count that
  is not 4n+1 is rounded down.
- fps is always 16; any other value is overridden with a warning.
- Sound in the result depends on the build. Up to e41fbcf the result is
  silent; the audio only drives the motion. From 6cd0ef1 (2026-11-22, the
  "Diffusion Graph Language" rewrite) sd-server puts the driving audio, cut to
  the video length, into the result as its sound track, as LTX-2.3 does.
  Animated WebP has no sound either way.

## Goals

- **A catalog set, "Wan2.2 S2V 14B (Q4)",** that downloads in one click, with
  a model limit of 81 frames (about 5 s at 16 fps) and 1280×720, a default of
  832×480, and the sizes Short (33 frames, 832×480), Medium (49, 1280×720) and
  Full (81, 1280×720):
  `QuantStack/Wan2.2-S2V-14B-GGUF` → `Wan2.2-S2V-14B-Q4_K_M.gguf` (12.9 GiB),
  the wav2vec2 audio encoder from `Comfy-Org/Wan_2.2_ComfyUI_Repackaged` →
  `split_files/audio_encoders/wav2vec2_large_english_fp16.safetensors`
  (631 MB), and the UMT5 Q8 text model and Wan2.1 VAE that the A14B sets
  already use.
- **The `wan22-s2v` recipe made available** (it comes from sd.cpp's recipe
  files, the built-in snapshot and each build's), with the `audio_encoder`
  slot passed as `--audio-encoder`, so custom sets can use it too.
- **An `audio` input on video jobs** (`POST /v1/videos/jobs` and
  `POST /v1/videos/generations`): a multipart file part, or base64 / a `data:`
  URL in JSON, the same way `start_image` works.
  - Accepted formats: WAV, MP3 and FLAC, found from the file's first bytes.
    Toolchest decodes MP3 and FLAC to WAV in pure Go before it sends them;
    a WAV goes through as it is.
  - A new capability, `audio`. An S2V set has the capabilities
    `start_image` and `audio`, and both inputs are required.
  - Sets without the `audio` capability refuse an `audio` input.
- **Length follows the audio when the client gives none.** With no `frames`
  and no `seconds`, toolchest reads the audio duration and asks for that many
  frames at 16 fps, rounded up to the next 4n+1 so that sd-server's rounding
  down never cuts the end of the audio. A length the client sends wins,
  and the audio is then cut or padded by sd-server. With no length sent, audio
  longer than the set's limit is refused with a clear message.
- **Fixed fps.** For S2V, toolchest uses 16 fps whatever the client sends,
  before it works out frames from `seconds` or from the audio. `/v1/models`
  shows this as `meta.video.fixed_fps: 16`.
- **Everything else works as for other video sets:** reservations, Measure
  (with a generated test audio), the job queue, eviction and locking, the
  Server page, the Models page catalog and custom sets, and the job listing
  (`inputs` says "a start image and audio").
- **Docs:** the API page, the generation guide, measured reservations, the
  OpenAPI file and the Help page describe the new input, capability and set.

## Non-goals

- **No sound added by toolchest.** Toolchest returns sd-server's video as it
  is: with the driving audio on builds from 6cd0ef1, silent on older ones. The
  docs say so and tell users of older builds to add the audio themselves (for
  example with ffmpeg). No ffmpeg in the images, no re-encoding.
- **No sd.cpp version check.** Toolchest does not refuse older builds; a build
  without the `wan22-s2v` recipe simply does not offer it.
- **No other audio inputs:** no `ref_video_audios`, no audio for MiniMax-H3 or
  LTX-2.3, no more than one track.
- **No long videos made of several S2V clips.** sd-server makes one window;
  toolchest does not stitch clips.
- **No Lightning / speed LoRA set** and no Q8 set.
- **No VACE or control videos.**
- **No client work** (Haruspex and others can adopt the input later).

## Users & primary flow

Users are programs that call the video API (for example Haruspex) and people
who run the set from sd-server's own web UI, linked from the Server page.

1. The user downloads "Wan2.2 S2V 14B (Q4)" from the Models page catalog
   (or builds a custom set with the `wan22-s2v` recipe), starts an instance,
   and runs Measure.
2. A client calls `GET /v1/models` and sees the set with
   `capabilities: ["start_image", "audio"]` and `meta.video.fixed_fps: 16`.
3. The client sends `POST /v1/videos/jobs` with a prompt, a `start_image`
   (the portrait) and an `audio` file (WAV, MP3 or FLAC), and usually no
   length.
4. Toolchest checks the inputs, decodes MP3/FLAC to WAV, sets the length from
   the audio, checks it against the limits, and sends the job to sd-server
   with `init_image` and `ref_audios`.
5. The client polls the job and fetches the WebM. On builds from 6cd0ef1 it
   has the speech as its sound; on older builds the client adds it.

## Constraints

- Go, no cgo. New dependencies only `github.com/hajimehoshi/go-mp3` and
  `github.com/mewkiz/flac` (pure Go); they must pass govulncheck in CI.
- sd.cpp is not pinned: the set must work on any build whose recipe book has
  `wan22-s2v`. The field test uses a build at or after 6cd0ef1, and checks
  that the other video sets still work on it, because 6cd0ef1 rewrote sd.cpp's
  generation code.
- Request limit stays 128 MiB (`maxImageRequestBytes`), which covers a
  portrait and several minutes of WAV.
- Video sets keep `--backend te=cpu` (pinned by
  `TestCatalogTextEncodersOnCPU`).
- Test machines: asmodean (one R9700, 32 GB, ROCm, build with
  `--rocm-image 10.1.0-full`) and compute2 (3× A4000, 16 GB each, CUDA).
- Docs and Help change in the same pull requests as the code (repository
  rule); Help anchors stay.
- Work on branches, never on `main`.

## Success criteria

- The catalog set downloads and starts on asmodean and on compute2. Measure
  passes for all three sizes on asmodean and for at least Short on compute2
  (16 GB); where a larger size does not fit, Measure's limit applies.
- On both machines, a job with a portrait and a 2.9-second speech clip (once
  as WAV, once as MP3, once as FLAC) and no length gives a video of about 3 s
  (49 frames) at 16 fps, with the mouth moving in time with the speech, and
  (on the 6cd0ef1+ build) with the speech as its sound track.
- Wan2.2 TI2V 5B and Wan2.2 I2V A14B Lightning are run once each on that
  build. A failure there is recorded and filed as its own issue; it does not
  block this work.
- A job without `audio`, or without `start_image`, on the S2V set is refused
  with a clear error; a job with `audio` on any other set is refused.
- A file that is not WAV, MP3 or FLAC, or a broken one, is refused with an
  error that starts "audio must be WAV, MP3 or FLAC".
- With no length sent, audio longer than the set's limit is refused; with a
  length sent, the job runs and the video has that length.
- `/v1/models` shows the `audio` capability and `fixed_fps`; the catalog card
  shows an "audio" badge.
- `make` build, `go vet`, `go test ./...`, JS tests and govulncheck pass; the
  docs site builds with `make docs`.

## Decisions

- **Sound in the finished video** → Toolchest adds none; it returns
  sd-server's video as it is. Keeps "no re-encode" and needs no ffmpeg.
- **sd.cpp builds** → Any build with the `wan22-s2v` recipe. Builds from
  6cd0ef1 give sound, older ones are silent; the docs say which. Field test
  on a 6cd0ef1+ build and re-check the other video sets on it.
- **Video length** → Match the audio when the client sends no length; a sent
  length wins; with no length, audio over the limit is refused.
- **Required inputs** → Audio and a start image, both required; new `audio`
  capability; caps are `start_image` + `audio`.
- **Audio formats** → WAV, MP3 and FLAC; MP3 and FLAC decoded in pure Go
  (go-mp3, mewkiz/flac), about +0.5–1 MB of binary.
- **Catalog sets** → One set, Q4_K_M.
- **Model limits** → 81 frames and 1280×720, default 832×480, the same sizes
  as the A14B sets.
- **fps other than 16** → Use 16 and show `fixed_fps: 16` in `/v1/models`;
  do not refuse.
