# Llama Toolchest

Click this screenshot to watch the explainer video:

[![Watch the video](https://img.youtube.com/vi/N4Xe6OL1Og4/maxresdefault.jpg)](https://youtu.be/N4Xe6OL1Og4)

A web-based management interface for [llama.cpp](https://github.com/ggerganov/llama.cpp) inference servers. Build llama.cpp from source, download models from HuggingFace, configure and run inference, and expose an OpenAI-compatible API — all from a single containerized application.

**Linux only.** Supports NVIDIA CUDA, AMD ROCm, Vulkan (host install only), and CPU backends. Works with Docker and Podman on all major Linux distributions.

## Features

- **Build management** — Compile llama.cpp with CUDA / ROCm / Vulkan / CPU backends. Toggleable build options, real-time SSE log streaming.
- **Model management** — Download GGUF models from HuggingFace, scan existing files, configure per-model parameters.
- **Multi-model loading** — Run multiple models simultaneously via llama.cpp's router. Per-model isolated subprocess, LRU eviction at the VRAM limit.
- **Per-model config** — Context size, KV cache quant, GPU layers, tensor split, flash attention, sampling, aliases, and speculative-decoding draft model.
- **Sampling presets** — Fetched automatically at download time from the GGUF file's embedded defaults (`general.sampling.*`), the Unsloth docs (thinking / non-thinking / coding variants), and the base model's `generation_config.json`, and offered as one-click presets in the config UI.
- **Vision / multimodal** — Auto-detect and pair `mmproj` files. Send images via the OpenAI chat API (requires OpenSSL build).
- **Embedding models** — Curated one-click downloads (nomic-embed, bge, mxbai-embed, snowflake-arctic-embed) with automatic `--embeddings` injection.
- **Speculative decoding** — Pair a small draft model with a large model; draft picker auto-filters by architecture.
- **Capability detection** — Tool calling, vision, and reasoning-mode detection from GGUF metadata, surfaced as badges and via the `capabilities` block on `/api/service/loaded-models` (single-round-trip auto-discovery).
- **VRAM estimation** — Architecture-aware estimates from GGUF metadata, accounting for KV cache size and quantization.
- **Measured memory** — What llama.cpp really allocated on the last load — weights, KV cache and working buffers, per GPU — read from its own buffer report and shown beside the estimate in the Available Models tooltip. Needs *Model loading detail* 4 in Settings.
- **Benchmarks** — Batch jobs sweeping models × builds × presets × any model parameter, with results compared across runs and exported to CSV/JSON. Each cell also records the memory its load actually used, so a sweep shows what a setting costs in VRAM as well as what it buys in speed — as a column, a visualization metric, and export columns.
- **OpenAI-compatible API** — Chat completions (streaming, tool calling, JSON schema), completions, embeddings, model listing. Optional Bearer auth.
- **Coding agent configs** — Ready-made config files for opencode, Kilo Code, pi, Goose and Crush, listing the models the server is serving with each one's context and capabilities (Server tab → API Endpoint → *Agent Configs*). See [Coding agents](#coding-agents).
- **Built-in chat UI** — llama.cpp's native chat interface with a model-selector dropdown.
- **Agent CLI** — Lightweight terminal chat client (`cmd/agent`) with optional filesystem tool use.

## Installation

### Quick start

```bash
git clone https://github.com/tmac1973/llama-toolchest.git
cd llama-toolchest
./setup.sh install              # default: container install
# or, for a containerless install on the host:
./setup.sh install --host
```

The setup script detects your GPU and container runtime, installs missing prerequisites, shows a summary, then builds and starts. The management UI is at `http://localhost:3000`.

### Install modes

| Mode | When to use | What it does |
|------|-------------|--------------|
| `--container` (default) | Most users — keeps GPU SDKs and the build toolchain isolated from the host. | Builds a Docker/Podman image and runs llama-toolchest inside it. The image installs the released `.deb`/`.rpm`, so the binary inside is byte-identical to a host install. |
| `--host` (default: `--from-package`) | You already have a working GPU driver and want a leaner setup, faster startup, or Vulkan support. | Downloads the latest released `.deb`/`.rpm` for your distro from the GitHub release, verifies its checksum, and installs via `dnf`/`apt`. Writes a config, registers a systemd user unit, and (optionally) enables the service. |
| `--host --from-source` | You're testing uncommitted changes from the source tree. | Builds the binary via `go build` and drops it in `~/.local/bin/llama-toolchest`. Otherwise the same flow. |

Host mode is managed via `systemctl --user start|stop|status llama-toolchest` (user install) or `sudo systemctl ...` (system install). Container `up`/`down`/`logs`/`enable`/`disable` are container-mode-only.

### Backend SDK selection (host mode)

By default `--host` auto-detects your primary GPU and asks whether to also install the Vulkan SDK as a portable fallback. To pick explicitly — including stacking multiple SDKs in one install — pass any combination of `--cuda`, `--rocm`, `--vulkan`. Each implies `--host`.

The similarly named `--rocm-next` and `--rocm-image` are different: they choose which ROCm *container* to build and do **not** imply `--host`. See [GPU Backend Notes → ROCm](#choosing-a-rocm-version-container-mode).

```bash
./setup.sh install --rocm --vulkan    # AMD GPU + Vulkan as a fallback
./setup.sh install --vulkan           # Vulkan-only (cross-vendor)
./setup.sh install --cuda             # NVIDIA only, skip the Vulkan prompt
```

For multi-GPU SDK installs, prefer the additive flags over `GPU=`.

### Switching modes

`./setup.sh migrate` moves your model registry, per-model configs, benchmarks, and main settings between sides:

```bash
./setup.sh migrate --to-host         # container → host
./setup.sh migrate --to-container    # host → container
```

Migration refuses to run if the destination side is already populated — uninstall the unwanted side first (`./setup.sh uninstall` or `./setup.sh uninstall --host`). The source's snapshot is kept at `~/llt-migrate-<timestamp>` as a safety net.

`builds.json` is wiped during migration: container-built `llama-server` binaries don't run on the host (different glibc/CUDA/ROCm runtime) and vice versa. Open the **Builds** page after migration and rebuild.

### Setup script reference

```
./setup.sh <command>

Lifecycle:
  install     Detect environment, install prerequisites, build & start
  uninstall   Stop, disable auto-start, remove container + image (or host package)
  migrate     Move state between container and host installs
              (--to-host or --to-container required)
  quick       Container only: pull the latest released package and reinstall
              it inside the existing image (reuses cached GPU SDK / base layers)
  rebuild     Container only: full rebuild with no cache, then start

Runtime (container only):
  up / down / logs           Start, stop, follow logs
  enable / disable           Auto-start on boot

Info:
  status      Show detected environment and planned actions
  deps        Verify prerequisites and print install commands for anything missing
  detect      Print detected GPU backend (cuda/rocm/vulkan/cpu); for AMD also
              prints which ROCm container variant would be used
  help        Show full help
```

Override detection: `GPU=cpu ./setup.sh install`, `RUNTIME=podman ./setup.sh install`.

### Manual install

If you'd rather skip `setup.sh` and install the released `.deb`/`.rpm` packages by hand, see [docs/manual-install.md](docs/manual-install.md).

### Supported GPUs

| GPU | Backend | Build profiles | Notes |
|-----|---------|----------------|-------|
| NVIDIA (Maxwell+) | CUDA 12.8 | cuda, cpu, vulkan† | GTX 900 series and newer. Driver >= 570. |
| AMD | ROCm 7.2 (default), ROCm 10 ‡ | rocm, cpu, vulkan† | RDNA and newer. |
| Other (Intel Arc, etc.) | Vulkan† | vulkan, cpu | Cross-vendor; install with `./setup.sh install --vulkan`. |
| None | CPU-only | cpu | No GPU required. |

† Vulkan is host-install only — see [GPU Backend Notes → Vulkan](#vulkan).

‡ ROCm 10 is available in both modes: container installs build on AMD's ROCm 10 image, host installs use AMD's ROCm 10.1 packages — see [GPU Backend Notes → ROCm](#rocm).

CUDA and ROCm provide native GPU compute for best performance; Vulkan is portable but typically slower than the vendor-specific backend on the same hardware.

### Supported distros

| Distro family | Package manager | Tested |
|---------------|-----------------|--------|
| Debian / Ubuntu | apt | Yes |
| Fedora / RHEL | dnf | Yes |
| Arch / CachyOS | pacman | Yes |
| openSUSE | zypper | Planned |

Both Docker and Podman (including rootless) are supported.

### First run

1. Open `http://localhost:3000`
2. Go to **Builds** and compile llama.cpp for your backend
3. Go to **Browse** to download a GGUF model from HuggingFace
4. Go to **Models**, click **Configure** to set GPU layers / context / sampling, then enable the model and restart
5. The OpenAI API is at `http://localhost:3000/v1`; the proxy routes by the `model` field

## Configuration

The YAML config lives at `/data/config/llama-toolchest.yaml` (container) or `~/.config/llama-toolchest/llama-toolchest.yaml` (host user install). Most settings are also editable from the Settings page.

```yaml
listen_addr: ":3000"        # Management UI listen address
data_dir: "/data"           # Base directory for builds, models, config
models_dir: ""              # Optional override; empty → <data_dir>/models
llama_port: 8080            # Port for llama-server inference
external_url: ""            # Public URL for link generation (e.g. http://myserver:3000)
hf_token: ""                # HuggingFace token for gated downloads
api_key: ""                 # Bearer token for /v1 proxy (empty = no auth)
log_level: "info"
active_build: ""            # Active llama.cpp build ID
models_max: 1               # Max simultaneously loaded models (0 = unlimited)
auto_start: false           # Start the inference router on container startup
```

To persist models on the host filesystem (so they survive `docker volume rm`), set `LLAMA_TOOLCHEST_MODELS_DIR=/path/to/models` in `.env` before running `setup.sh`. Existing models in the volume won't be visible after switching — move them first.

`external_url` and `llama_port` can also be set from the environment (`LLAMA_TOOLCHEST_EXTERNAL_URL`, `LLAMA_TOOLCHEST_LLAMA_PORT`), which overrides the YAML — handy for container deploys that inject values via compose.

### Securing with HTTPS + a login

By default the UI, management API, and inference ports are served over plain HTTP with no authentication. For deployments reachable beyond a trusted LAN, `./setup.sh install --secure` puts a [Caddy](https://caddyserver.com) reverse proxy in front for HTTPS (self-signed or Let's Encrypt) and a single admin login. It's interactive, or fully scriptable with flags. See **[docs/secure.md](docs/secure.md)**.

> ⚠️ The bundled Caddy config is a best-effort starting point, provided as-is — you are responsible for auditing it for your own threat model.

## Ports

| Port | Service |
|------|---------|
| 3000 | Management UI + OpenAI proxy (`/v1`) |
| 8080 | llama.cpp router + built-in chat UI |

In a [secure install](docs/secure.md), Caddy fronts these on `443` (UI/API/`/v1`) and `8080` (chat UI) with `80` redirecting to HTTPS, and the app itself is bound to loopback.

## GPU Backend Notes

### ROCm

`setup.sh` auto-detects the AMD GPU architecture and sets `HSA_OVERRIDE_GFX_VERSION` in `.env` when needed. Only required for older GPUs not natively supported by ROCm 7.2 (RDNA 1 → `10.1.0`, Vega → `9.0.0`).

#### Choosing a ROCm version (container mode)

Container installs offer two ROCm versions. An AMD install asks which, defaulting to whichever is already installed:

```bash
./setup.sh install                              # asks; stable on a fresh machine
./setup.sh install --rocm-next                  # experimental, latest known release
./setup.sh install --rocm-image 10.1.0-full     # experimental, pinned to a tag
```

`ROCM_VARIANT=stable|next` and `ROCM_BASE_IMAGE=<tag>` are the environment equivalents. The choice is stored in `.env`, so `rebuild`, `up` and `down` keep it without re-passing anything, and `./setup.sh detect` prints which one is in use.

| | Stable | Experimental |
|---|---|---|
| ROCm | 7.2.4 | 10.1.0 (any tag you name) |
| Base image | Fedora 43 + RPMs from `repo.radeon.com` | `rocm/dev-ubuntu-24.04` |
| Image size | 14.1 GB | 21.1 GB |
| Dockerfile | `Dockerfile.rocm` | `Dockerfile.rocm-next` |

**Why the experimental one exists.** `repo.radeon.com`'s `el9`, `el10`, `rhel9` and `rhel10` paths all stop at 7.2.4, as does the `amdgpu-install` route, so the Fedora image cannot reach anything newer. AMD now packages ROCm 10 in a separate repository (`stable.repo.amd.com`, used by [host installs](#rocm-on-a-host-install)); the container variant builds on AMD's own ROCm image instead, which also offers the 7.14.x line.

Any tag from [rocm/dev-ubuntu-24.04](https://hub.docker.com/r/rocm/dev-ubuntu-24.04/tags) works, and a full image reference is accepted too. `10.1.0-full` and `7.14.1-full` are both known to build here. The tag is checked before anything is downloaded, so a typo fails in about a second rather than part-way through a 20 GB pull.

**What it requires.** ROCm 10 supports RDNA 1 and newer plus the CDNA cards — it does **not** need RDNA 4. The host kernel it needs depends on your card, not on ROCm, because the container carries no kernel components: RDNA 4 wants 6.12, RDNA 3 6.0, RDNA 2 5.9, RDNA 1 5.3. `setup.sh` checks your card against both lists and warns without blocking.

**Is it faster?** On an RX 9070 XT, with the same llama.cpp and the same settings:

| workload | ROCm 7.2.4 | ROCm 10.0.0 |
|---|---|---|
| generation, no speculative decoding | 116.3 tok/s | 116.3 tok/s |
| MTP + n-gram assist, warm | 206.9 tok/s | **258.4 tok/s** |

So: nothing measurable for ordinary generation, and about 25% for speculative decoding once the n-gram assist has warmed up. Two caveats. The images use different compilers (gcc 15.3.1 on Fedora, 13.3.0 on Ubuntu), so this compares images rather than purely ROCm versions. And an n-gram assist gets faster the more it has seen of the text it is generating — the same measurement reads 87 tok/s cold and 207 warm — so any figure needs to say which it is.

**Switching means rebuilding llama.cpp.** A build is linked against the libraries of the image it was made in, so a build from one ROCm version may fail to load under the other. Nothing is deleted: the Builds page marks builds that cannot run in the current image and explains why, the Server tab refuses to select them, and switching back makes them work again. This is the same rule as the one under [Switching modes](#switching-modes) above, for the same reason: a `llama-server` is built for the place it will run.

#### ROCm on a host install

A host install on an AMD GPU asks which ROCm to set up, when AMD publishes ROCm 10.1 for your distro and GPU:

```
  1) Distro        Your distro's ROCm (7.1), or the one already installed (default)
  2) ROCm 10.1     AMD's ROCm 10.1 packages
  3) Both          Your distro's ROCm 7.1 and ROCm 10.1, side by side
```

`HOST_ROCM=distro|10|both` answers it for a scripted install. ROCm 10.1 comes from AMD's `stable.repo.amd.com`, installs into `/opt/rocm/core-10.1`, and only the package for your GPU is installed (`amdrocm-core-dev10.1-gfx1100`, say). It is offered on Ubuntu 22.04/24.04/26.04, Debian 12/13, RHEL-family 8–10 and Fedora 44+ (Fedora through AMD's RHEL 10 packages, which AMD does not publish for Fedora but which install and work there). **Both** is only offered when your distro's own ROCm is new enough to build llama.cpp with — Ubuntu 24.04 and Debian 13 ship 5.7, which isn't.

What can sit side by side:

| Combination | |
|---|---|
| Your distro's ROCm (in `/usr`) + ROCm 10.x | Supported. |
| Several ROCm 10.x releases | Supported. |
| AMD's own 7.2.4-or-older packages + ROCm 10.x | **Not supported by AMD.** ROCm 10 installs itself inside the older release and takes over parts of it. `setup.sh` refuses to add 10.1 here and explains how to fix it; a machine already in this state gets a warning. Builds still work (see below). |

#### Several ROCm installs

When a machine has more than one ROCm, the Builds page shows a **ROCm Install** picker on the build form, defaulting to the newest, and each build uses only the install it was given: its compiler, headers, libraries and CMake packages. The preview of effective CMake flags shows the extra flags this adds. Each build records which install it was built against (`rocm 10.1.0 @ /opt/rocm/core-10.1`), and the Builds page flags a build whose install has since been removed or upgraded. The same ref built against two installs gets two builds rather than a replace prompt.

A ROCm 10.x install is always built against this way, even on its own: its packages point `/usr/bin/hipconfig` at themselves, which otherwise sends the build to the wrong place. A single distro or 7.x install builds as it always has.

### CUDA

CUDA 12.8 requires NVIDIA driver >= 570. The CUDA build auto-detects GPU architecture at compile time — no manual target configuration needed.

### Vulkan

Vulkan is **host-install only** — container mode would need GPU driver / ICD passthrough that this project doesn't manage. Use it as a portable fallback alongside CUDA or ROCm, or as the sole backend on hardware where the vendor SDK isn't a fit.

`./setup.sh install --vulkan` (or `--rocm --vulkan`) installs:

| Distro | Packages |
|--------|----------|
| Debian / Ubuntu | `glslc libvulkan-dev spirv-headers vulkan-tools` |
| Fedora / RHEL | `glslc vulkan-headers vulkan-loader-devel spirv-headers-devel vulkan-tools` |

`vulkan-tools` provides `vulkaninfo`, which the backend probe uses to enumerate hardware Vulkan devices. The runtime loader (`libvulkan1` / `vulkan-loader`) is typically already installed by your GPU driver.

### Multi-GPU

Two modes:

- **Layer parallelism (default)** — Layers are split sequentially across GPUs. Best for most cases.
- **Tensor parallelism (experimental)** — Tensors are split across all GPUs simultaneously. More memory-efficient but needs fast interconnect (NVLink / PCIe 4.0+).

GPU selection: pick "All GPUs" (tensor mode), specific GPUs ("GPU 0", "GPUs 0–1"), or a custom tensor-split string like `1,1,0,0`.

**Tensor parallelism on CUDA — NCCL.** llama.cpp's `-sm tensor` path uses NCCL for the AllReduce that fires after attention and the FFN in every transformer block. NCCL is the optimized GPU-to-GPU collective communications library; without it the build silently falls back to a slower generic path. The CUDA container image installs `libnccl-dev` automatically, and `setup.sh install --cuda` does the same on host installs. On ROCm the counterpart is RCCL, off by default upstream — enable the **RCCL Collectives** build toggle (the ROCm image ships `rccl-devel`; host installs offer it as an optional extra, since it only matters for multi-GPU); without it the log shows `falling back to meta-backend butterfly` and every token pays for the slow path. At runtime, pair tensor parallelism with **Flash Attention on** and **KV cache at `f16`** (no quant) — those are hard constraints in the current implementation. Note that the per-tensor split requires the model's attention head count to divide cleanly by the number of GPUs, so 3-way splits crash on many architectures (most pick power-of-2 head counts). 2 or 4 GPUs split far more reliably.

## Architecture

A single Go binary serves the web UI and manages llama-server subprocesses. Server-rendered HTML with [htmx](https://htmx.org/) and [Pico CSS](https://picocss.com/) — no JS build step.

```
cmd/
  llama-toolchest/   Server entry point
  agent/             Terminal chat client with tool use
internal/
  api/               HTTP handlers, SSE streaming, /v1 proxy
  benchmark/         benchmark jobs, parameter sweeps, comparison
  builder/           llama.cpp build pipeline (git, cmake, ninja)
  config/            YAML configuration
  huggingface/       HF API client and model downloader
  models/            Registry, GGUF parser, VRAM estimation, preset INI
  monitor/           GPU/CPU/memory metrics (ROCm + NVIDIA)
  process/           llama-server router lifecycle
web/                 Templates + static assets (htmx, Pico CSS)
scripts/             API smoke tests (test-api, test-embeddings, test-tools, etc.)
```

Container files: `Dockerfile.{cuda,rocm,cpu}`, `docker-compose.{cuda,rocm,cpu}.yml`, plus `setup.sh`.

## Development

```bash
make dev          # go run with hot reload
make build        # compile bin/llama-toolchest + bin/agent
make run          # build and run
make agent        # compile just the agent CLI
```

### Agent CLI

```bash
agent                                 # connect to localhost:3000
agent -host 192.168.1.50              # remote server
agent -port 8080                      # custom port
agent -url http://gpu-box:3000/v1     # full endpoint URL
agent -model qwen3-32b                # target a specific model
agent -system "You are..."            # set a system prompt
agent -no-tools                       # plain chat (no filesystem tools)
agent -work-dir /path/to/project      # working dir for tools
LLAMA_API_KEY=sk-xxx agent            # authenticate
```

Pass the API key through `LLAMA_API_KEY` rather than `-api-key`: a value on the command line shows in the process list (`ps`) and your shell history.

### Test scripts

All scripts include an interactive model picker; pass a model name to skip selection.

```bash
./scripts/test-api.sh           # API smoke test
./scripts/test-embeddings.sh    # embedding dimensions + cosine similarity
./scripts/test-info.sh          # /api/models/{id}/info + /api/ps
./scripts/test-structured.sh    # JSON schema + json_object output
./scripts/test-tools.sh         # tool / function calling
./scripts/test-vision.sh        # vision via URL or local image
```

## API

### OpenAI-compatible (`/v1/*`)

All requests are forwarded to llama.cpp's router. Per-model sampling defaults are injected for requests that don't specify them; user-defined model aliases work in the `model` field. Sampling precedence is: request parameters > per-model config > GGUF-embedded defaults (`general.sampling.*`, applied by llama-server itself) > llama.cpp built-ins.

- `GET /v1/models` — list available models
- `GET /v1/models/{model}` — single model info
- `POST /v1/chat/completions` — streaming, tool calling, JSON schema / `json_object`
- `POST /v1/completions` — text completion
- `POST /v1/embeddings` — vector embeddings (auto-configured for `--embeddings` builds)

```bash
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen35", "messages": [{"role": "user", "content": "Hello!"}]}'
```

When `api_key` is set, all `/v1/*` requests require `Authorization: Bearer <key>`. The management UI is unaffected.

### Coding agents

Terminal coding agents can use these models, but most only find a local server through a hand-edited config file that has to name every model, its context size and what it can do. **Server tab → API Endpoint → Agent Configs** generates those files from what the server is serving right now:

| Agent | File | Where it goes | How it's picked up |
|---|---|---|---|
| [opencode](https://opencode.ai) | `llama-toolchest.opencode.json` | `~/.config/opencode/` | `export OPENCODE_CONFIG=…` layers it over your own config |
| [Kilo Code](https://kilo.ai) | `kilo.json` | `~/.config/kilo/` | read on start (merge the `provider` block if you have a kilo.json) |
| [pi](https://github.com/earendil-works/pi) | `models.json` | `~/.pi/agent/` | read on start (merge the provider if you have a models.json) |
| [Goose](https://github.com/aaif-goose/goose) | `llama_toolchest.json` | `~/.config/goose/custom_providers/` | `goose session --provider llama_toolchest` |
| [Crush](https://github.com/charmbracelet/crush) | `llama-toolchest.crushrc` | `~/.config/crush/` | one `source` line in your `crushrc` |

Each file lists the models in the Available Models card, the loaded one first as the default, with:
- the context **one conversation** gets (a 32K context split across 2 parallel slots is 16K per conversation, which is what an agent has to compact against);
- a max output of a quarter of that, capped at 32K;
- tool, image and reasoning support.

The dialog shows the paths for Linux, macOS and Windows and the exact command for each agent. Its *Server URL* field sets the address the files use: change it if the agent runs on another machine. When an API key is set, the files read it from `LLAMA_TOOLCHEST_API_KEY` instead of containing it, so set that variable wherever the agent runs.

A file is a snapshot: **download it again after changing which models are served or their context size.** Each agent was tested making tool calls against llama-server through these files, with and without an API key (opencode 1.18, Kilo Code 7.8, pi 1.1, Goose 1.54, Crush 0.98).

### Management API

Routes under `/api/` cover builds, models, HuggingFace search & download, the inference service, settings, monitoring, and benchmarks. SSE endpoints stream build / download / log progress. Browse the routes in [`internal/api/server.go`](internal/api/server.go) — that's the source of truth.

A few useful ones:

- `GET /api/ps` — loaded models with status (Ollama-style)
- `GET /api/models/{id}/info` — enriched metadata with capabilities (tools, vision)
- `GET /api/models/{id}/vram-estimate` — VRAM estimate for a given config
- `GET /api/service/loaded-models` — models available to the router, each with a `capabilities` block (see below)
- `GET /api/monitor/stream` — GPU/CPU/memory metrics over SSE

#### Capability discovery (`capabilities`)

`GET /api/service/loaded-models` returns a top-level `schema_version` and folds a
`capabilities` object into every model entry, so a client can auto-configure
itself from a single round-trip — no per-model `/info` fan-out and no model-name
heuristics. The object is a stable contract: keys use explicit `null`/`false`
rather than being omitted, so "known absent" is distinguishable from "server too
old to report". `schema_version` is bumped only on breaking changes.

```jsonc
{
  "schema_version": 1,
  "running": true,
  "models": [
    {
      "id": "Qwen3-32B",
      "status": "loaded",
      "public_name": "unsloth-Qwen3-32B.Q8_K_XL",
      "registry_id": "unsloth--Qwen3-32B-GGUF--...",
      "capabilities": {
        "schema_version": 1,

        // context — compact requests on context_per_request, never context_length
        "context_size": 32768,          // served runtime n_ctx (0 → trained max)
        "context_length": 131072,       // model's trained max (informational)
        "parallel": 4,                  // conversations served at once
        "context_shared": false,        // true: the conversations share one pool of context_size
        "context_per_request": 8192,    // the most one conversation can use

        // modalities / tools
        "vision": false,
        "tools": true,
        "embedding": false,

        // reasoning / thinking — detected from the chat template
        "reasoning": {
          "supported": true,
          "default_enabled": true,
          "toggle": "chat_template_kwargs", // or "reasoning_effort" | "none"
          "kwarg": "enable_thinking"        // key when toggle == chat_template_kwargs, else ""
        },

        // recommended sampling — per-model override wins, else the card
        // preset. source is where the card preset came from: "gguf"
        // (defaults embedded in the GGUF header — the common case),
        // "unsloth-docs", "generation_config.json", or "params".
        "sampling": {
          "source": "gguf",
          "source_url": null,
          "default": { "temperature": 0.6, "top_p": 0.95, "top_k": 20,
                       "min_p": null, "presence_penalty": null, "repeat_penalty": null },
          "presets": [ /* SamplingPreset list, verbatim */ ]
        },

        "max_output_tokens": null         // null = no server-imposed cap
      }
    }
  ]
}
```

`context_per_request` depends on how the model is set to serve conversations:

| Parallel Conversations | `parallel` | `context_shared` | `context_per_request` |
|---|---|---|---|
| blank (llama.cpp's default) | 4 | true | `context_size` |
| 1 | 1 | false | `context_size` |
| N ≥ 2 | N | false | `context_size` ÷ N |
| N ≥ 2, Shared Context on | N | true | the Max Context per Conversation, or `context_size` without one |

With a shared pool, `context_per_request` is what one conversation may
reach, not what is guaranteed: when several long conversations run at
once the pool can fill first.

The same `parallel`, `context_shared` and `context_per_request` values are
also added to the `config` map in `GET /api/models/{id}/info` (the
detailed per-model view), together with `context_pool`.

## License

[GNU Affero General Public License v3.0](LICENSE)
