# Generation Backends — Hardware Verification

Results of Phase 11: every success criterion in `overview.md`, checked on real
hardware. Each entry gives the machine, the install mode, the date and the
build refs.

## Machines

| Name | GPUs | Software |
|---|---|---|
| asmodean | 1× Radeon AI PRO R9700, 32 GB | Fedora 44 host (ROCm from Fedora, Vulkan via RADV); Fedora 43 ROCm 7.2.4 container image; Podman 5.8 rootless |
| compute2 | 3× RTX A4000, 16 GB each | CUDA 12.8 container image (Docker) |

## Success criteria

All on 2026-10-25. Build refs are in the sections below.

| Criterion | Result | Where |
|---|---|---|
| Feature off | pass | asmodean host, against v2.42.1 |
| Builds: ROCm | pass | asmodean host and container |
| Builds: CUDA | pass | compute2 container |
| Builds: Vulkan | pass | asmodean host |
| Image API: Ming-Image RGBA on asmodean | pass, with the transparent recipe (a plain request is RGBA but opaque) | asmodean host (ROCm, Vulkan) and container; compute2 |
| Music API: 60 s MP3 | pass | asmodean host and container; compute2 |
| All three at once | pass | asmodean host and container |
| One generation job per GPU | pass | asmodean host and container |
| Separate GPUs | pass, when the LLM is pinned to its own card | compute2 |
| Web UIs: plain mode | pass | asmodean container (8081–8088 and 9081–9088) |
| Web UIs: secure mode | pass | asmodean container |
| Custom sets | pass | asmodean host (`qwen-turbo-local`) |
| Mixed-vendor server (R9700 + RTX 3090) | not run (hardware not installed) | |

## Results

### compute2, container mode (CUDA)

2026-10-25. toolchest built from `main` at `c7a0948` plus the measure fix
(`4ff8cd1`), copied into the existing CUDA container. Builds:
`sd-master-1006-0886f11-cuda`, `ace-master-de858d8-cuda`, llama.cpp
`b11669-cuda-optimized`.

**Builds.** sd.cpp and acestep.cpp both build with the `cuda` profile from the
Builds API.

**Measuring.** Per-process VRAM is not visible to `nvidia-smi` inside the
Docker container, so every reading here is the whole-GPU estimate (marked
"estimated"). Before the measure fix, both instances below were disabled
straight after measuring: the fit check still saw the backend's own memory as
"other programs" (see Bugs).

| Instance | GPU | Peak (MiB) | Reservation (MiB) |
|---|---|---|---|
| FLUX.2 klein 4B (`flux-gpu1`) | 1 | 8,599 | 9,728 |
| ACE-Step 1.5 turbo (`ace-gpu2`) | 2 | 6,945 | 7,680 |
| Ming-Image 0.1 Design (`ming-gpu1`) | 1 | 9,382 | 10,752 |

**Separate GPUs.** Qwen3-4B Q8_0 pinned to GPU 0 (`tensor-split 1,0,0`),
FLUX.2 klein on GPU 1, ACE-Step turbo on GPU 2. Chat ran without a break
during the "beside chat" runs.

| | Alone | Beside active chat |
|---|---|---|
| FLUX.2 klein, 1024×1024 | 5.12 s, 5.14 s | 5.15 s, 5.15 s |
| ACE-Step turbo, 60 s song | 13.3 s | 12.3 s |
| Chat, Qwen3-4B Q8_0 | 77.1–77.5 tok/s | 77.2–77.6 tok/s |

A first run without the pin had the chat model spread over all three cards:
then images took 24% longer and chat fell to 43–58 tok/s. Generation only
runs at full speed on its own card when the LLM is kept off that card.

**Music API.** `/v1/music/jobs` returned a 60-second MP3 (957,696 bytes at
128 kbit/s) from the ACE-Step turbo catalog set.

**Ming-Image on one 16 GB card.** Fits without `--offload-to-cpu`: sd-server's
own auto-fit keeps the 10 GB text model and the VAE in system RAM and puts the
6.2 GB diffusion model on the card. A 2048×2048 image (its default) took
7.6 s of text encoding and 128.5 s of sampling (12 steps); 1024×1024 took
34 s.

**Ming-Image transparency.** A plain request returns an RGBA PNG whose alpha
is fully opaque. Transparency needs both halves of Haruspex's recipe: the
prompt starts with "RGBA, 4-channel, transparent background." and sampling
starts from a fully transparent 1024×1024 canvas at strength 0.9. Through
toolchest that works on `/v1/images/generations`, with the canvas and
strength in sd-server's `<sd_cpp_extra_args>` block in the prompt (toolchest
forwards the prompt unchanged): 83.9% of the pixels clear, 15.7% opaque, a
clean cut-out subject, in 26 s. `/v1/images/edits` is refused for Ming-Image
because its recipe has no edit capability.

### asmodean, host mode (ROCm and Vulkan)

2026-10-25. No release carries the feature yet, so instead of the `.rpm` the
binary built from this branch (`d3964e1`) ran directly on the host, with a
new, empty data folder (`~/scratch/tc-p11-host`) on ports 3000 and 8080.
Builds: llama.cpp `b11869-rocm` / `b11869-vulkan`,
`sd-master-1006-0886f11-rocm` / `-vulkan`, `ace-master-de858d8-rocm` /
`-vulkan`. ROCm from Fedora 44's packages.

**Feature off.** `go test ./...` passes. Against the last release before the
feature (v2.42.1), on the same data folder with Qwen3-4B Q8_0 loaded:
- `/api/gpu-map`: identical.
- `/v1/models`: no `kind` field. The only difference is three fields from
  #304 (shared context), which is not part of this feature.
- Server and Models pages: only the version string differs.
- Builds page: no project choice, nothing about sd.cpp or acestep.cpp. The
  "pull tags" tooltip now says "the project's remote", and the release link
  uses llama.cpp's current address (`ggml-org`).
- Settings: the new Generation backends switch, the one intended change.

**Builds.** All three projects build with both the `rocm` and `vulkan`
profiles.

**Measuring** (all on GPU 0, R9700, nothing else running):

| Instance | Build | Peak (MiB) | Reading | Reservation (MiB) |
|---|---|---|---|---|
| `qwen-turbo-local` (custom set, Qwen-Image 2.1 Turbo) | ROCm | 9,777 | per process | 11,264 |
| `ace-xl` (ACE-Step 1.5 XL SFT) | ROCm | 7,236 | per process | 8,192 |
| `ming-image` (Ming-Image 0.1 Design) | ROCm | 18,947 | per process | 20,992 |
| `ming-vk` | Vulkan | 18,817 | estimated (whole-GPU) | 20,992 |
| `ace-vk` | Vulkan | 6,793 | estimated (whole-GPU) | 7,680 |

The Qwen-Image figures are lower than the plan's (10,283 / 11,776, from
2026-10-24 with sd.cpp `master-1000`). The Vulkan instances show the
"estimated (whole-GPU)" label with its explanation in the instance table.

**Ming-Image.** Plain 1024×1024 request: 24 s, RGBA PNG, fully opaque. With the
transparent recipe (RGBA phrase, transparent start image at strength 0.9 in
`<sd_cpp_extra_args>`): 21 s, 84.9% clear pixels. Vulkan, same recipe: 24.5 s,
89.1% clear.

**What fits beside Ming-Image** (reservation 20.5 GiB on the 31.9 GiB card):
- ACE-Step XL SFT: refused. "ace-xl needs 8.0 GiB, but only 7.5 GiB is left
  after other programs 1.4 GiB, ming-image 20.5 GiB and the safety margin."
- Qwen3-4B Q8_0: loads.
- Gemma 4 12B Q6_K: refused with 507.

**All three at once.** `qwen-turbo-local` and `ace-xl` running, Qwen3-4B Q8_0
loaded (102 tok/s): the GPU map shows Qwen3-4B 17.8%, `qwen-turbo-local`
34.5% and `ace-xl` 25.1% of the card. Gemma 4 12B Q6_K and Qwen3.6 27B Q4_K_M
are both refused with 507: "Not enough VRAM: 19.0 GiB reserved for
generation backends on GPU 0 (qwen-turbo-local 11.0 GiB, ace-xl 8.0 GiB)".
No out-of-memory error in toolchest's log or any instance log.

**APIs.**
- `/v1/images/generations`, `model: qwen-turbo-local`: 1024×1024 PNG in 26.6 s.
- `/v1/music/jobs`: 60 s MP3 (951,168 bytes, 128 kbit/s, 48 kHz stereo) in 21 s.
- `/release_task` → `/query_result` → `/v1/audio`, no `model` field: a 30 s
  MP3 (477,696 bytes).
- Vulkan: 60 s song in 18 s on `ace-vk`.

**One job per GPU.** Two image requests and one song sent at once: the queue
history shows them one after another (image 21.4 s, image 21.4 s, song
10.8 s), no overlap. While they ran the instance table showed 1 running and 2
waiting on GPU 0 and the slowed-chat banner was up; it cleared when the queue
emptied.

**Enabling Ming-Image beside the other two** is refused: "ming-image needs
20.5 GiB, but only 9.0 GiB is left after other programs 1.4 GiB,
qwen-turbo-local 11.0 GiB, ace-xl 8.0 GiB and the safety margin."

### asmodean, container mode (ROCm image)

2026-10-25. `setup.sh install --container --from-source` from this branch
(rootless Podman), Fedora 43 ROCm 7.2.4 image. The install reused the
`llama-toolchest-data` volume from the Phase 10 test. Builds made inside the
container: llama.cpp `b11869-rocm`, `sd-master-1006-0886f11-rocm` (web UI on,
pnpm found), `ace-master-de858d8-rocm`.

**Measuring.** In the container every reading is "estimated (whole-GPU)": the
per-process figures come from the kernel's per-PID GPU entries, which use the
host's PIDs, not the ones toolchest sees inside the container. The
reservations came out the same as in host mode:

| Instance | Peak (MiB) | Reservation (MiB) |
|---|---|---|
| `ming-image` | 19,032 | 20,992 |
| `qwen-turbo-local` | 9,863 | 11,264 |
| `ace-xl` | 7,362 | 8,192 |

**Steps 1.3–1.7 repeated.** Ming-Image transparent recipe: 84.7% clear pixels
in 33 s. With `qwen-turbo-local` and `ace-xl` running and Qwen3-4B Q8_0
loaded: image 38 s (first request after the start), 60 s MP3 in 27 s,
`/release_task` → `/query_result` → `/v1/audio` work. Two images and a song at
once ran one after another (22.5 s, 22.5 s, 10.0 s), no overlap; 1 running and
2 waiting showed while they ran. No out-of-memory errors.

**Web UIs.**
- Plain mode, ports 8081–8088: the dashboard links (`http://asmodean:8082`,
  `:8083`) open sd-server's page (158 KB) and acestep.cpp's "AI Music
  Generator" page (191 KB). ace-server only sends its page gzip-compressed,
  which every browser accepts.
- Secure mode (`--secure --tls self-signed`): both pages answer 401 without
  the login and load with it through Caddy on 8082 and 8083. The loopback
  debug port moved from 8081 to 18080, and the installer said so.
- `--gen-port-start 9081`: the container publishes 9081–9088 and the
  dashboard links point at 9082 and 9083 (the instances listen on 8082 and
  8083 inside the container); both open.

## Bugs found and fixed

Each has its own `fix(...)` commit with a regression test where one is
possible.

- **Measuring disabled the instance on CUDA** (`fix(generation)`, compute2).
  Right after the test run, the card's reading still included the stopped
  backend, so the fit check counted it as another program and refused the
  backend's own suggested reservation. The measure step now waits, up to 20
  seconds, for the reading to fall back near its level before the run.
- **Ming-Image catalog note** (`fix(generation)`). It said Ming-Image outputs
  transparent backgrounds and needs more than 17 GB of VRAM. It runs on a
  16 GB card, and transparency needs the RGBA phrase plus a transparent start
  image; the note now says how.
- **Secure → plain switch moved the chat port** (`fix(setup)`, asmodean).
  `load_env_ports` read a secure `.env`'s loopback debug port (8081 or 18080)
  back as the public chat port.

Found but not fixed here, because they are older than this feature:
- Saving any setting writes environment-variable overrides into the config
  file. The secure install's `LLAMA_TOOLCHEST_EXTERNAL_URL` ended up there, so
  after switching back to plain mode the links kept `https://localhost:8443`.
- The secure install takes the external host name from `hostname -f`, which
  is `localhost` on asmodean, so its links point at `localhost`.
- A router started before a model's settings change, or before a model was
  downloaded, keeps the old preset until it restarts (the 4B model's tensor
  split on compute2; the new models on asmodean answered "restart the router
  to pick it up").
- The three `setup.sh` problems found in Phase 10 (SELinux and
  `--from-source`, custom Caddy ports, an old `llamactl.yaml` in the volume)
  are fixed separately, in #318. The asmodean checks here ran before it was
  merged, with a one-line SELinux relabel added to the test copy only.

## Not run

- **Mixed-vendor server (step 4):** the RTX 3090 is not installed in asmodean
  yet. Mixed-vendor servers are not verified.
- **Haruspex end to end (step 5):** Haruspex has no image provider for an
  OpenAI-style image server like toolchest (its providers are ComfyUI, its own
  bundled sd-server, and Lemonade's `/api/v1/…` routes). The request
  Haruspex's Lemonade adapter builds — the RGBA phrase and the transparent
  start image in `<sd_cpp_extra_args>` — was sent to toolchest directly and
  gave transparent images on ROCm, Vulkan and CUDA. Pointing Haruspex at
  toolchest needs a provider for it in Haruspex.
- **Host install from the `.rpm`:** no release has the feature yet; the host
  checks ran the same binary directly.

## Release notes

> **Image and music generation (optional).** llama-toolchest can now run
> stable-diffusion.cpp's `sd-server` and acestep.cpp's `ace-server` next to
> llama-server, on the same GPU or other ones. It is off until you turn it on
> in Settings. Build both projects from the Builds page with the ROCm, CUDA or
> Vulkan profile (Vulkan in host installs only), download a model set from the
> catalog (Ming-Image, Qwen-Image 2.1 Turbo, FLUX.2 klein, SDXL, ACE-Step) or
> add your own, and create an instance on a GPU. Each instance measures its
> peak GPU memory once and keeps a reservation that the LLM fit planner leaves
> free, so loading a model that would crowd it out is refused with HTTP 507
> instead of running out of memory. One image or music job runs per GPU at a
> time. Images use the OpenAI routes `/v1/images/generations` and
> `/v1/images/edits`; music has its own `/v1/music/jobs` API and the
> ACE-Step 1.5 server's `/release_task`, `/query_result` and `/v1/audio`.
> Each backend's own web page opens from the Server page; jobs started there
> are not queued. Container installs publish those pages' ports only if you
> say yes in `setup.sh` (`--gen-ports`), behind the admin login in secure
> mode. Verified on a Radeon AI PRO R9700 (ROCm and Vulkan, host and
> container) and three RTX A4000s (CUDA, container); servers with GPUs from
> two vendors have not been verified.
