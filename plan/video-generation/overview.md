# Video Generation — Project Overview

## Problem

llama-toolchest runs text, image and music models, but not video. The
generation backends plan (`plan/generation-backends/`) left video out on
purpose. stable-diffusion.cpp's `sd-server`, which toolchest already builds and
runs for images, can also make video: Wan2.1/2.2, LTX-2.3/2.5,
HunyuanVideo 1.5 and others. Today, someone who wants video on the same machine
has to run sd-server by hand or use another tool. That tool does not know what
toolchest has loaded, so they compete for the GPU with no warning, which is the
problem the generation backends work solved for images and music.

The parts are mostly there:
- toolchest already builds sd-server, gives each generation backend a VRAM
  reservation, queues one job per GPU, and can let LLMs and backends take turns
  (eviction, with locking).
- sd-server serves video through its own async API: `POST /sdcpp/v1/vid_gen`,
  then poll `GET /sdcpp/v1/jobs/{id}` (cancel at
  `POST /sdcpp/v1/jobs/{id}/cancel`). A finished job holds one whole video
  file, base64-encoded: WebM (default), animated WebP or AVI.

Video is different from images in ways toolchest does not handle yet:
- A job takes minutes, not seconds.
- The result is a large file, often tens of megabytes.
- Memory grows with the number of frames as well as the size.
- The models need files that no image set has: a second diffusion model for
  the Wan2.2 A14B pairs, plus connectors and an audio VAE for LTX-2.3.

## Goals

- **Three catalog sets that download in one click:**
  - **Wan2.2 TI2V 5B:** text-to-video and image-to-video in one small model.
    Apache-2.0.
  - **Wan2.2 I2V A14B:** image-to-video, plus start-and-end frames, from two
    experts (high-noise and low-noise). The best open quality. Apache-2.0.
  - **LTX-2.3 distilled 1.1:** text-to-video, image-to-video and start-and-end
    frames, at 8 steps, with sound. LTX-2 Community Licence: free unless the
    organisation makes US$10 million a year or more; the licence column says so.
- **A video job API on port 3000**, the same way images and music work:
  - Async jobs: submit, check status (with phases), fetch the video, cancel.
  - A waiting route that takes the same request and answers with the finished
    job.
  - Three kinds of request: text-to-video, start image, and start-and-end
    images. A set says in `/v1/models` which of these it can do.
  - Common request fields (prompt, negative prompt, size, length, frame rate,
    seed, start and end image, format), plus an object of extra sd-server
    settings passed through as they are (like `<sd_cpp_extra_args>` for
    images).
  - Length in frames, or in seconds as a convenience.
  - WebM by default, animated WebP on request.
  - The finished video is saved on disk and fetched from a URL, kept for 24
    hours like image and music results, with at most 50 videos kept at a time
    (the oldest go first).
- **Video sets work with everything generation sets already have:**
  reservations, Measure, the per-GPU job queue, eviction and locking, the
  Server page's instances, and the Models page's catalog and custom sets.
- **Two limits on a video's size, each enforced with a clear message:**
  - **The model's limit** (from the catalog): the most frames and the largest
    size the model can make at all.
  - **The instance's measured size:** Measure lets the user choose the size to
    measure (presets Short, Medium and Full, or a custom size up to the
    model's limit), and the instance then refuses any request with more video
    (frames × width × height) than it was measured for. sd-server therefore
    stays inside its reservation, and users who want longer videos measure at
    a larger size.
- **Measure runs inside what the card can give** (for sd-server, image and
  video), so it never suggests a reservation the card cannot hold.
- **Custom sets can use sd.cpp's other video recipes** (Wan2.2 T2V A14B,
  HunyuanVideo 1.5, LingBot-Video and the rest) from the user's own files,
  without being tested.
- `/v1/models` lists video instances with `meta.kind` `"video"` and `meta.api`
  values for the video routes. The help page documents the API.

## Non-goals

- **No video page in toolchest.** Programs use the API; people use sd-server's
  own web UI, linked from the Server page, as for images.
- **No other catalog sets** in this project: not Wan2.2 T2V A14B,
  HunyuanVideo, LingBot-Video, LTX-2.3 dev or LTX-2.5. MiniMax-H3 is left out
  for its licence, which excludes the US, the EU, the UK and South Korea.
- **No control videos or reference images** (VACE, Wan2.2 S2V, MiniMax-H3's
  reference conditioning). None of the three sets uses them.
- **No AVI output**, and no format sd-server cannot make itself. Toolchest does
  not re-encode videos.
- **No per-request sound switch.** sd-server has none. An LTX-2.3 WebM always
  has sound; asking for animated WebP gives a silent video, since WebP cannot
  hold sound.
- **No client work.** No changes in Haruspex or any other program; they can
  adopt the API later.
- **No change to the image and music APIs.** Haruspex relies on their current
  shapes.

## Users & primary flow

Users are the same as for images and music: someone running toolchest on their
own GPU machine, and the programs they point at it.

1. On the Models page, the user downloads a video set from the catalog, for
   example Wan2.2 TI2V 5B.
2. On the Server page, they create an instance of it on a GPU and press
   **Measure**, choosing a size: Short (the quick test video) or larger,
   for longer videos. Toolchest makes a video of that size and reserves its
   peak memory plus 10%, within what the card can give. The instance will
   make videos up to that size.
3. They start the instance (or, with eviction on, let the first job start it).
4. A program sends a video job to `/v1/videos/jobs` with a prompt, and
   optionally a start image, an end image, a length and a format. Toolchest
   checks the request against the set's abilities, the model's limit and the
   instance's measured size, queues it on the
   instance's GPU, and answers with a job ID.
5. The program polls the job. Its phase moves through `queued`, then
   `starting` (only when eviction has to start the backend), `generating` and
   `saving`.
6. When the job succeeds, the program downloads the video from the URL in the
   job. Toolchest keeps the file for 24 hours, or until 50 newer videos have
   been made.
7. A simple script can instead call the waiting route, which answers with the
   finished job (and its URL) once the video is done.
8. Someone trying things by hand opens sd-server's own web UI from the Server
   page; those jobs skip toolchest's queue, as image jobs from the web UI do.

## Constraints

- **Backend:** stable-diffusion.cpp's sd-server, built by toolchest's existing
  Builds tab. WebM output needs sd-server built with libwebm, which is on by
  default when the submodule is present.
- **Formats and licences:** WebM from sd-server is VP8 video (each frame a VP8
  key frame) with uncompressed PCM sound, in a Matroska-based container written
  by BSD-licensed libwebm. VP8 is BSD-licensed and covered by Google's free,
  irrevocable patent grant. Animated WebP uses the same VP8 coding. Both are
  open and royalty-free; no format with licence costs is offered.
- **Existing APIs:** the image and music routes, `meta.api` values, job phases
  and their JSON shapes must not change; Haruspex relies on them
  (`plan/generation-backends/verification.md`).
- **Long jobs:** toolchest's HTTP server and its client to the backends have no
  timeouts today. Anything time-limited must allow for videos that take many
  minutes: the 10-minute limit on a measurement was one such limit (video
  measurements now have up to 2 hours).
- **Memory:** sd-server is held to its reservation with `--max-vram`; when a
  request needs more, it moves work to system RAM and runs slower. Video memory
  grows with frames × width × height.
- **Frames:** sd-server rounds the frame count down to the nearest 4n+1 (34
  becomes 33) and reports the real count in its result.
- **Test machines:** asmodean (1× Radeon AI PRO R9700, 32 GB, ROCm) is the
  main one. compute2 (3× RTX A4000, 16 GB each, CUDA) is for the 16 GB checks.
- **Platforms:** the same as the other generation backends: Linux, ROCm, CUDA
  and Vulkan.
- **Writing:** UI text, help text and messages follow the project's plain,
  jargon-free English, and every new control has a tooltip.

## Success criteria

- All three catalog sets download from the Models page with no Hugging Face
  token, and their licences show in the licence column.
- On asmodean's R9700, each set:
  - is measured with Measure and gets a reservation;
  - makes a text-to-video (TI2V 5B, LTX-2.3), an image-to-video (all three)
    and a start-and-end video (I2V A14B, LTX-2.3) through the job API as WebM,
    and at least one of them as animated WebP;
  - has its time per video and peak memory written down in the plan's
    verification notes and in the catalog notes.
- An LTX-2.3 WebM plays with sound in a browser.
- On a 16 GB A4000, Wan2.2 TI2V 5B is measured and makes a video through the
  job API. The A14B pair and LTX-2.3 are tried, and what happened is written
  down.
- A request over the model's limit or the instance's measured size, or for a
  kind of video the set cannot make, is refused with a message that says why;
  nothing is queued.
- After measuring at the largest preset, a video of that size stays within the
  instance's reservation on the R9700.
- On a 16 GB A4000, Measure gives a reservation the card can hold, with no
  value set by hand.
- The waiting route returns the same finished job as polling does.
- With eviction on, a video job for a stopped video instance starts it, making
  room the same way image jobs do, and locked instances are left alone.
- Image and music jobs, Haruspex's routes and `meta.api` values behave exactly
  as before (existing tests pass).
- The help page documents the video API, the three sets and their measured
  reservations.

## Decisions

- **How people make videos** → An API for programs plus sd-server's own web
  UI; no video page in toolchest.
- **API shape** → Async jobs plus a waiting route.
- **Kinds of video** → Text-to-video, start image, and start-and-end images.
- **File formats** → WebM by default, animated WebP as an option. Asked whether
  a truly open format could be added: WebM from sd-server already is one (VP8,
  BSD-licensed with a royalty-free patent grant, in an open Matroska-based
  container), and so is WebP. AV1 and Theora are not made by sd-server.
- **How results come back** → Saved file with a download URL, kept 24 hours.
- **How many videos to keep** → 24 hours, at most 50 videos (asked during the
  implementation interview).
- **What the waiting route returns** → The finished job, with its URL.
- **Hardware for "done"** → R9700 32 GB for all three sets; A4000 16 GB best
  effort (TI2V 5B must work, the others tried and written down).
- **LTX-2.3 variant** → Distilled 1.1.
- **LTX-2.3 sound** → Yes, sound in WebM (sd-server has no per-request switch;
  WebP is silent).
- **Request settings** → Common fields plus an object of sd-server extras
  passed through.
- **Video length** → Frames, with seconds as an option; both at once is an
  error.
- **Requests bigger than the test video** → Allowed up to a per-set ceiling of
  frames and size; bigger is refused. Revised after the Phase 04 hardware check
  (a 5-second video went about 4 GiB over its reservation) by the next three
  decisions.
- **Requests above the instance's measured size** → Refused, with a clear
  message that says why and how to allow them (measure at a larger size).
- **Choosing the size to measure** → Presets plus a custom size in the Measure
  dialog. A video measurement always applies its reservation (and so the
  instance's limit), since the user chose the size on purpose; image
  measurements keep "Use suggested" for a second measurement.
- **Measuring on a card the set would fill** → Measure runs inside the card's
  budget and never suggests more than it, for image and video sets; music
  sets keep measuring as before, since ace-server has no memory limit flag.
- **A measured video that needs more than the card can give** (found in the
  Phase 05 hardware check: the untiled decode goes past `--max-vram`) →
  Refuse that size, allowing 1 GiB for the GPU runtime's own memory; keep the
  previous size.
- **Clients** → Built for clients in general and advertised in `meta.api`; no
  client work.
- **Speed-up LoRAs for Wan2.2 I2V A14B** (asked after the Phase 06 hardware
  check: 27 minutes for 2 seconds) → A second catalog set with the Lightning
  4-step LoRAs, 4 steps on each expert; the 20-step set stays (Phase 08).
- **Other video models** → Out of scope for the catalog; custom sets can use
  sd.cpp's other video recipes.
