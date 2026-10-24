# Generation Backends (Images and Music) — Project Overview

## Problem

llama-toolchest can only run text models. It builds and supervises one
llama-server process (in router mode) and sends every `/v1` request to it.
People who also want image or music generation on the same machine have to run
a second tool, such as Lemonade Server or ComfyUI, next to it. That second tool
does not know what llama-toolchest has loaded, and llama-toolchest does not know
what the other tool holds. The two can run out of VRAM, slow each other down
without any warning, and there are two installs, two UIs and two APIs to look
after. Haruspex, which already uses llama-toolchest for text, also uses
Ming-Image for images. Today it has to get those from somewhere else.

Tests on asmodean (1× Radeon AI PRO R9700, 32 GB, Fedora 44, ROCm 7.1.1) on
2026-10-24 showed the pieces are ready:
- stable-diffusion.cpp's `sd-server` and acestep.cpp's `ace-server` both build
  with ROCm using the same compiler settings that llama-toolchest already
  works out for llama.cpp.
- All three servers can run at the same time on one GPU.

The same tests showed the two problems this project has to solve: VRAM sharing
and GPU time sharing.
- At peak, the three servers used 30.8 of 32.6 GB.
- sd-server plans its memory from the VRAM that is free when it starts. It does
  not know that ACE-Step will need 7 GB later.
- While all three were working, each workload ran 2–4x slower. Chat dropped
  from 45 to 11–20 tokens per second.

## Goals

- llama-toolchest can build, run and supervise **stable-diffusion.cpp
  (`sd-server`)** and **acestep.cpp (`ace-server`)** next to llama-server, on
  the same GPU or on a different GPU.
- One install, one UI and one API on port 3000 cover text, image and music
  models.
- **Builds:** the existing Builds tab builds all three projects. It reuses the
  existing GPU profiles and the ROCm/CUDA compiler discovery. The supported
  GPU backends are ROCm, CUDA and Vulkan.
- **VRAM reservations:** each enabled generation backend reserves a fixed
  amount of VRAM on its GPU.
  - The starting value comes from a measured test run, and the user can edit it.
  - The LLM fit planner subtracts the reservations.
  - Nothing is allowed to load past the budget.
- **GPU sharing:** API jobs for images and music are queued, so only one
  generation job runs on each GPU at a time. The UI says when chat will be
  slower because a generation job is running.
- **API:**
  - Images use the standard OpenAI routes `/v1/images/generations` and
    `/v1/images/edits`.
  - Music uses a small llama-toolchest job API: submit a job, check its status,
    fetch the MP3. This wraps ace-server's own job API.
- **Model catalog:** a built-in catalog of known-good model sets that download
  in one click. It includes:
  - **Ming-Image** (required: freely licensed, outputs an alpha channel, used
    by Haruspex)
  - Qwen-Image 2.1 Turbo
  - FLUX.2 klein
  - SDXL
  - ACE-Step turbo and XL

  Users can also define a custom model set from files already on disk.
- **Backend web UIs:** the web UI built into each backend is reachable the same
  way llama.cpp's chat UI is today: its own published port, an "Open … UI →"
  link on the dashboard, and an entry in the Caddy secure setup.
- The design does not assume there are only two generation backends. Speech,
  video and classifier backends can be added later without reworking it.

## Non-goals

- **No other backends in this project:**
  - speech (whisper-server, text to speech)
  - video generation (sd-server's video API)
  - BERT-style classifiers, and reranking work beyond what llama-server
    already does
- **No dynamic VRAM management:**
  - no loading backends on demand when a request arrives
  - no unloading the least-recently-used model when VRAM runs short (the way
    Lemonade does)
- **No chat priority:** a queued generation job does not wait for active chat
  requests, and a running job is never interrupted.
- **No queueing for backend web UIs:** jobs started from a backend's own web UI
  go straight to that backend.
- **No prebuilt binaries:** backends are built from source, as llama.cpp is
  today. llama-toolchest does not download binaries.
- **No new image or music studio:** llama-toolchest does not add its own
  generation UI. The backends' own web UIs and the API are the ways to generate.
- **No autotune, autoconfig or benchmarks for the generation backends.**
- **No other platforms:** no Metal, SYCL, Windows or macOS for the generation
  backends.

## Users & primary flow

**Users:**
- People who self-host llama-toolchest on Linux servers and workstations.
  Today that means NVIDIA A4000s on compute2, AMD R9700s on compute and
  asmodean, and the RX 9070 XT workstation.
- API clients, mainly Haruspex.

**Primary flow:**
1. In Settings, the user turns on **Generation backends**. Until this is on,
   nothing changes for existing installs.
2. In **Builds**, the user picks the project (stable-diffusion.cpp or
   acestep.cpp), the GPU profile (ROCm, CUDA or Vulkan) and a ref, then builds
   it. The build runs and is recorded like a llama.cpp build.
3. In **Models**, the user picks a catalog set (for example Ming-Image, or
   ACE-Step XL) and downloads it, or defines a custom set from files on disk.
4. The user assigns the backend to a GPU and starts a measured test run.
   - llama-toolchest loads the set, runs one generation, and records the VRAM
     it used.
   - It suggests that figure as the backend's reservation.
   - The user accepts it or edits it.
5. llama-toolchest checks that the reservations plus the LLM models currently
   loaded fit on each GPU. LLMs loaded later are checked against the
   reservations at load time, and are refused with a clear message if they do
   not fit. Then it starts the backend process on its own port. The process
   only sees its assigned GPU.
6. **Clients use the API on port 3000:**
   - Image requests go to `/v1/images/generations`.
   - Music jobs go to the music job API.
   - llama-toolchest queues the jobs (one generation job per GPU at a time) and
     forwards each one to the right backend.
7. **The dashboard shows:**
   - each backend's state and its reservation
   - the job queue
   - a warning when chat is sharing a GPU with a running generation job
   - links to each backend's own web UI

## Constraints

- **Codebase:** Go 1.26 with chi v5. The UI is server-rendered `html/template`
  with htmx, and templates are embedded through `web/embed.go`. New code
  follows the existing package layout under `internal/` and the existing UI
  patterns.
- **Existing behaviour must not change** while the feature is off: llama-server
  router mode, the proxy, preset INI generation, fit planning, autotune and
  benchmarks. Existing configs, registries and settings must keep loading
  without changes.
  - The GPU container images gain `nodejs` and `pnpm` (build tools only).
  - Web UI ports are published only when the user says yes in `setup.sh`.
- **Deployment:** both install modes (container and host) are supported, with
  the GPU images (`Dockerfile.{cuda,rocm,rocm-next}`). The CPU image does not
  support generation backends.
  - Backends are compiled inside the container at runtime, the same way
    llama.cpp is.
  - Any new build dependencies must be added to the images. For example,
    sd-server uses `pnpm` to build its web UI.
- **GPU selection:** llama.cpp selects GPUs with `--device` lists. A generation
  backend is a separate process, so it is limited to its GPU by setting
  `CUDA_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES` or the Vulkan equivalent on
  that process only. This is possible because the runtime environment for the
  llama-server router does not apply to it.
- **The APIs differ between backends:**
  - sd-server answers OpenAI image requests directly.
  - ace-server is job-based: `POST /lm` and `POST /synth`, then `GET /job?id=N`,
    then fetch the result as multipart audio.
  - Both upstream projects change often, so their versions are pinned by the
    build ref the user chooses.
- **Measured reference numbers** (asmodean, R9700). VRAM is in MiB, read from
  `rocm-smi`, minus the desktop's ~1,300 MiB:

  | Workload | Time | VRAM |
  |---|---|---|
  | sd-server, Qwen-Image 2.1 Turbo Q4, 1024² | about 36 s per image | 9,240 MiB held while loaded, 10,283 MiB at peak |
  | ace-server, XL SFT, default mode | 17–19 s per 60 s song | 7,239 MiB at peak, about 300 MiB idle |
  | llama-server, Gemma 4 12B Q6, 8K context | 45 tok/s | about 12,150 MiB |

- **16 GB cards:** CUDA cards with 16 GB (A4000) may need smaller quantised sets
  or CPU offload (`--offload-to-cpu`) for some catalog entries.
- **Plain language:** user-facing text uses plain technical English with no
  slang. UI conventions apply: tooltips over static text, and every metric
  explains how to read it.

## Success criteria

- **Feature off:** with generation backends turned off, the existing test suite
  passes. The UI, `/v1/models` and the VRAM figures are unchanged from before.
- **Builds:** stable-diffusion.cpp and acestep.cpp build from the Builds tab
  with the ROCm, CUDA and Vulkan profiles. ROCm is verified on asmodean, CUDA
  on compute2, and Vulkan on asmodean.
- **Image API:** on asmodean, with Ming-Image loaded,
  `POST /v1/images/generations` on port 3000 returns an image with an alpha
  channel.
  - Ming-Image's weights are 17.3 GB, so it needs most of the 32 GB card.
  - Phase 11 records which other workloads the fit check allows beside it.
  - The three-at-once check below uses Qwen-Image 2.1 Turbo.
- **Music API:** the music job API returns a playable 60-second MP3 from the
  ACE-Step catalog set.
- **All three at once:** with llama-server (Qwen3-4B Q8_0), sd-server
  (Qwen-Image 2.1 Turbo) and ace-server (ACE-Step XL SFT) running on one GPU.
  - The two reservations (11,776 + 8,192 MiB) leave 8,746 MiB of the card
    budget for LLMs.
  - Gemma 4 12B Q6, used on 2026-10-24, ran with only 1.8 GB spare, and it is
    refused. The 10% headroom added to each measured peak is enough on its own
    to refuse it, even before the planner's safety margin (8% of the whole
    card).
  - The fit planner refuses an LLM load that would exceed what the
    reservations leave free.
  - None of the three servers fails with an out-of-memory error.
- **One generation job per GPU:** when two image requests and one music request
  are sent at once to backends on the same GPU, the job runs never overlap,
  according to job timestamps. The dashboard shows the queue and the warning
  that chat is slowed.
- **Separate GPUs:** a backend assigned to a different GPU than llama-server
  runs at its stand-alone speed while chat is active (where the hardware
  allows, for example on compute2).
- **Web UIs:** each backend's web UI opens from its dashboard link, in plain
  mode and in Caddy secure mode.
- **Custom sets:** a custom model set built from files on disk can be defined,
  test-run, given a reservation, and served.

## Decisions

- **Where it lives** → Inside llama-toolchest, not a separate sibling project.
  One process knows about all VRAM on the GPU.
- **First release scope** → Images (sd-server) and music (ace-server). The
  design allows more backends later.
- **Opt-in** → Off by default, behind a Settings switch. Existing installs see
  no change until it is turned on.
- **GPU sharing rule** → One generation job per GPU at a time, queued by
  llama-toolchest, with a UI warning that chat will be slower. Chat is never
  held back.
- **VRAM model** → Fixed reservations for each backend, filled in from a
  measured test run and editable. The LLM fit planner subtracts them. There is
  no load-on-demand and no eviction.
- **API shape** → OpenAI routes for images, plus a llama-toolchest music job API
  that wraps ace-server. llama-toolchest owns the queue for both.
- **Models** → A curated catalog of model sets plus custom sets from files on
  disk. The catalog must include Ming-Image (freely licensed, alpha channel,
  used by Haruspex). Its default files come from the project's own GGUF
  quantisation, published for Haruspex at `voltaire321/Ming-Image-0.1-Design-GGUF`.
- **Builds UI** → One Builds tab with a project selector. Profiles, compiler
  discovery and build history are shared.
- **GPU backends** → ROCm, CUDA and Vulkan in the first release.
- **Generation UI** → Expose each backend's own web UI the same way as
  llama.cpp's chat UI (published port, dashboard link, Caddy entry). No
  llama-toolchest generation panel.
- **Web UI jobs vs the queue** → Accepted and documented. Jobs from a backend's
  web UI go straight to that backend and are not queued. The dashboard says so.
