# LlamaCtl — Implementation Plan
> Custom llama.cpp inference server manager for headless Linux deployments

---

## Overview

**LlamaCtl** is a self-hosted web application for managing llama.cpp inference on bare-metal or containerized Linux servers. It provides a browser-based UI for downloading models from HuggingFace, compiling llama.cpp against ROCm or Vulkan backends, configuring multi-GPU tensor splits, and controlling a `llama-server` systemd service — all without touching a terminal.

**Target environment:** Debian 13, dual AMD Radeon AI PRO 9700 (64GB total VRAM), ROCm + Vulkan backends.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Container / Host                  │
│                                                     │
│  ┌──────────────────────────────────────────────┐   │
│  │           llamactl (Go binary)               │   │
│  │  ┌─────────────┐   ┌──────────────────────┐  │   │
│  │  │  SvelteKit  │   │     REST API         │  │   │
│  │  │  (embedded) │◄──│  /api/models         │  │   │
│  │  └─────────────┘   │  /api/build          │  │   │
│  │                    │  /api/service        │  │   │
│  │                    │  /api/hf             │  │   │
│  │                    │  /v1/* (proxy)       │  │   │
│  │                    └──────────┬───────────┘  │   │
│  └───────────────────────────────┼──────────────┘   │
│                                  │                  │
│          D-Bus / systemctl        │                  │
│                                  ▼                  │
│  ┌───────────────────────────────────────────────┐  │
│  │         llama-server.service (systemd)        │  │
│  │   llama-server --model <path>                 │  │
│  │     --n-gpu-layers 999                        │  │
│  │     --tensor-split 0.5,0.5                    │  │
│  │     --ctx-size 8192                           │  │
│  │     --port 8080                               │  │
│  └───────────────────────────────────────────────┘  │
│                                                     │
│  /opt/llamactl/                                     │
│    builds/     ← compiled llama.cpp binaries        │
│    models/     ← downloaded GGUF files              │
│    config/     ← model configs + service state      │
└─────────────────────────────────────────────────────┘
```

---

## Technology Stack

| Layer | Choice | Rationale |
|-------|--------|-----------|
| Backend | Go 1.22+ | Single static binary, great `os/exec` + D-Bus support, easy embed |
| Frontend | SvelteKit (static build) | Already in your stack, fast, embeds cleanly into Go binary |
| HTTP router | `chi` | Lightweight, idiomatic Go |
| Systemd control | `go-systemd/v22/dbus` | Proper D-Bus integration, avoids shelling out to `systemctl` |
| Frontend embedded | `//go:embed` | Serves SvelteKit dist from Go binary, zero separate file serving |
| HF downloads | Go `net/http` + SSE | Native streaming, progress events to UI |
| Build process | `os/exec` → cmake + ninja | Subprocess with log streaming over SSE |

---

## Project Structure

```
llamactl/
├── cmd/
│   └── llamactl/
│       └── main.go              # Entry point, flag parsing, startup
├── internal/
│   ├── api/
│   │   ├── routes.go            # chi router setup
│   │   ├── models.go            # Model CRUD handlers
│   │   ├── build.go             # llama.cpp build handlers
│   │   ├── service.go           # Systemd service control handlers
│   │   ├── hf.go                # HuggingFace proxy/download handlers
│   │   └── proxy.go             # OpenAI-compat passthrough (/v1/*)
│   ├── builder/
│   │   ├── builder.go           # CMake build orchestration
│   │   ├── backends.go          # ROCm / Vulkan / CPU flag sets
│   │   └── detect.go            # GPU backend auto-detection
│   ├── huggingface/
│   │   ├── client.go            # HF API client (search, model info)
│   │   └── downloader.go        # Resumable GGUF download with progress
│   ├── models/
│   │   ├── registry.go          # Local model registry (JSON manifest)
│   │   └── vram.go              # VRAM estimation by quant + param count
│   └── systemd/
│       ├── manager.go           # D-Bus service start/stop/status/reload
│       └── unit.go              # Unit file template rendering
├── web/                         # SvelteKit project (separate build step)
│   └── dist/                    # Built output, embedded into Go binary
├── systemd/
│   └── llamactl.service         # Systemd unit for the Go daemon itself
├── config/
│   └── config.go                # App config (YAML, env vars)
├── Dockerfile
├── Makefile
└── README.md
```

---

## Phase 1 — Foundation (Week 1–2)

### 1.1 Go Project Scaffold

- Initialize Go module (`github.com/yourname/llamactl`)
- Set up `chi` router with middleware (logging, CORS, recovery)
- Embed SvelteKit `dist/` via `//go:embed web/dist`
- YAML config file for: data dir, llama-server port, HF token, listen addr
- Graceful shutdown with `context` cancellation

### 1.2 Data Directory Layout

```
/opt/llamactl/
├── builds/
│   ├── rocm-<git-sha>/
│   │   └── llama-server
│   └── vulkan-<git-sha>/
│       └── llama-server
├── models/
│   └── <model-id>/
│       ├── model.gguf
│       └── meta.json
├── config/
│   ├── models.json              # Registry of downloaded models + their configs
│   └── active.json              # Currently active model + build
└── llamactl.yaml
```

### 1.3 SvelteKit Shell

- Basic layout: sidebar nav (Models, Builds, Service, Settings)
- API client module (`src/lib/api.ts`) with typed fetch wrappers
- SSE utility for streaming log/progress events
- Tailwind CSS + shadcn-svelte for components

---

## Phase 2 — llama.cpp Build Manager (Week 2–3)

### 2.1 Backend Detection

```go
// internal/builder/detect.go
func DetectBackends() []Backend {
    // Check `rocminfo` exit code → ROCm available
    // Check `vulkaninfo` exit code → Vulkan available
    // Always include CPU fallback
}
```

Auto-detect on startup and expose via `GET /api/build/backends`.

### 2.2 Build Configuration

**ROCm flags for 9700 AI Pro (gfx1201, RDNA4):**
```cmake
-DGGML_ROCM=ON
-DAMDGPU_TARGETS=gfx1201
-DCMAKE_BUILD_TYPE=Release
```

**Vulkan flags:**
```cmake
-DGGML_VULKAN=ON
-DCMAKE_BUILD_TYPE=Release
```

Store named build profiles in config. Allow user to create custom profiles with arbitrary cmake flags for experimentation.

### 2.3 Build Process

- `GET /api/build/list` — list available compiled builds with sha + backend + timestamp
- `POST /api/build/trigger` — start a build (body: `{backend, ref}` where ref is tag/commit/`latest`)
  - Clones or fetches `https://github.com/ggml-org/llama.cpp`
  - Runs cmake + ninja in a temp dir
  - Streams stdout/stderr to frontend via SSE (`GET /api/build/logs/:id`)
  - On success, moves binary to `/opt/llamactl/builds/<backend>-<sha>/llama-server`
  - Records build metadata in `builds.json`
- `DELETE /api/build/:id` — remove a build

### 2.4 Build UI

- Build list with backend badge (ROCm / Vulkan / CPU), git sha, date
- "New Build" modal: backend selector, git ref input (default: `latest release`)
- Live log stream panel during build (SSE → scrolling terminal-style div)
- Active build indicator (which binary is currently in use)

---

## Phase 3 — HuggingFace Model Manager (Week 3–4)

### 3.1 HF API Client

```go
// internal/huggingface/client.go

// Search models with GGUF filter
GET https://huggingface.co/api/models?search=<q>&filter=gguf&limit=20

// Get model file tree
GET https://huggingface.co/api/models/<owner>/<model>

// Resolve sibling files → filter for .gguf extensions
// Group by quantization (parse filenames: Q4_K_M, Q8_0, IQ4_XS, etc.)
```

Expose via:
- `GET /api/hf/search?q=<query>` — proxied search with GGUF filter
- `GET /api/hf/model?id=<owner/model>` — file list + metadata

### 3.2 Downloader

- `POST /api/hf/download` — body: `{model_id, filename, hf_token?}`
- Resumable: check `Content-Range` support, store partial files as `.part`
- Progress streamed via SSE: `{bytes_downloaded, total_bytes, speed_bps}`
- On completion: validate file, write `meta.json`, register in model registry
- `DELETE /api/models/:id` — removes model files + registry entry

### 3.3 VRAM Estimator

```go
// internal/models/vram.go
// Rough formula: param_count_billions * bytes_per_weight * 1.1 (overhead)
// Quant multipliers: Q4_K_M ≈ 4.5 bits, Q8_0 ≈ 8 bits, F16 ≈ 16 bits
// Surface as: estimated_vram_gb per GPU given a tensor split
```

### 3.4 Model Browser UI

- Search bar → results list with model name, downloads, likes, license
- Model detail drawer: file list grouped by quant, with VRAM estimate badges
  - Green = fits in single GPU (32GB), Blue = fits across both (64GB), Red = too large
- Download progress panel with speed + ETA
- Local models list: name, size, quant, last used, quick-select button

---

## Phase 4 — Service Control (Week 4–5)

### 4.1 Systemd Unit Template

```ini
# /etc/systemd/system/llama-server.service (managed by llamactl)
[Unit]
Description=llama-server inference daemon
After=network.target

[Service]
Type=simple
User=llamactl
ExecStart=/opt/llamactl/builds/{{.BuildPath}}/llama-server \
    --model /opt/llamactl/models/{{.ModelPath}}/model.gguf \
    --n-gpu-layers {{.GPULayers}} \
    --tensor-split {{.TensorSplit}} \
    --ctx-size {{.ContextSize}} \
    --threads {{.Threads}} \
    --host 0.0.0.0 \
    --port 8080 \
    {{.ExtraFlags}}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

LlamaCtl renders this template from the active model config, writes it to disk, then calls `daemon-reload` + `restart` via D-Bus.

### 4.2 Service Manager

```go
// internal/systemd/manager.go
type Manager struct { conn *dbus.Conn }

func (m *Manager) Status() (ServiceStatus, error)
func (m *Manager) Start() error
func (m *Manager) Stop() error
func (m *Manager) Restart() error
func (m *Manager) ApplyConfig(cfg ModelConfig) error
// ApplyConfig: renders unit template, writes file, daemon-reload, restart
```

### 4.3 Model Configuration Schema

```json
{
  "model_id": "bartowski/Qwen2.5-72B-Instruct-GGUF",
  "filename": "Qwen2.5-72B-Instruct-Q4_K_M.gguf",
  "build_id": "rocm-abc1234",
  "gpu_layers": 999,
  "tensor_split": "0.5,0.5",
  "context_size": 8192,
  "threads": 8,
  "extra_flags": "--flash-attn"
}
```

Persisted per model in registry, editable from UI.

### 4.4 Service Control UI

- Status card: running / stopped / failed + uptime
- Start / Stop / Restart buttons
- Active model display with quick-swap dropdown (stops service, swaps config, restarts)
- Live log tail: SSE stream from `journalctl -u llama-server -f`
- Config panel per model:
  - GPU layers slider (0 to 999)
  - Tensor split: dual slider showing GPU 0 / GPU 1 allocation with VRAM usage bars
  - Context size selector
  - Raw extra flags text input (escape hatch)

---

## Phase 5 — OpenAI-Compatible Proxy (Week 5)

### 5.1 Passthrough Proxy

All requests to `GET|POST /v1/*` on the LlamaCtl port are reverse-proxied to `http://localhost:8080/v1/*` (llama-server's port).

```go
// internal/api/proxy.go
func ProxyHandler(target *url.URL) http.Handler {
    proxy := httputil.NewSingleHostReverseProxy(target)
    // Add error handling for when llama-server is down
    // Return 503 with JSON error body instead of default proxy error
}
```

### 5.2 API Key Middleware (optional)

Simple Bearer token auth on the proxy path if you want to expose it on your network without open access. Configured via `llamactl.yaml`.

### 5.3 Proxy UI

- Settings page: upstream llama-server URL (default `localhost:8080`)
- Connection test button (hits `/v1/models` and shows response)
- Displays the LlamaCtl OpenAI-compat endpoint URL for pasting into other tools

---

## Phase 6 — Container + Deployment (Week 6)

### 6.1 Dockerfile

```dockerfile
# Stage 1: Build SvelteKit
FROM node:22-slim AS web-builder
WORKDIR /app/web
COPY web/package*.json .
RUN npm ci
COPY web/ .
RUN npm run build

# Stage 2: Build Go binary
FROM golang:1.22-bookworm AS go-builder
WORKDIR /app
COPY go.mod go.sum .
RUN go mod download
COPY . .
COPY --from=web-builder /app/web/dist ./web/dist
RUN CGO_ENABLED=0 go build -o llamactl ./cmd/llamactl

# Stage 3: Runtime (Debian 13 + ROCm runtime libs)
FROM debian:trixie-slim
# Install ROCm runtime (not full SDK — just the libs llama-server needs at runtime)
# Install Vulkan loader
# Don't include build tools — builds happen on host or in a build sidecar
COPY --from=go-builder /app/llamactl /usr/local/bin/llamactl
VOLUME ["/opt/llamactl"]
EXPOSE 3000
ENTRYPOINT ["llamactl"]
```

### 6.2 ROCm in Container

Two viable strategies:

**Option A — Build on host, mount binary into container**
- Host has full ROCm SDK, build happens there
- Container only needs ROCm runtime libs
- Mount `/opt/llamactl/builds` as a volume
- Simpler, driver updates don't require container rebuild

**Option B — Build sidecar container**
- Separate `llamactl-builder` image with full ROCm SDK
- Triggered by llamactl API, shares the builds volume
- Cleaner separation but more moving parts

Recommend **Option A for v1**.

### 6.3 Systemd in Container

Running a container that controls the *host's* systemd requires:
- Mount `/run/systemd/private` and `/run/dbus/system_bus_socket` into container
- Run container with `--pid=host` and appropriate capabilities
- **Or:** Run llamactl directly on the host as a systemd service (simpler)

For v1, recommend running llamactl itself as a systemd service on the host:

```ini
# /etc/systemd/system/llamactl.service
[Unit]
Description=LlamaCtl inference manager
After=network.target

[Service]
Type=simple
User=root
ExecStart=/usr/local/bin/llamactl --config /opt/llamactl/llamactl.yaml
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Containerization becomes cleaner in v2 once the systemd integration approach is validated.

---

## Data Flow: Model Swap

```
User clicks "Activate" on a downloaded model
  → PUT /api/models/:id/activate
    → Load model's saved ModelConfig
    → Render llama-server.service unit template
    → Write to /etc/systemd/system/llama-server.service
    → D-Bus: daemon-reload
    → D-Bus: restart llama-server.service
    → Poll service status until active (or failed)
    → SSE event to frontend: {status: "active", model: "..."}
  ← UI updates service status card + active model display
```

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/builds` | List compiled builds |
| POST | `/api/builds` | Trigger new build `{backend, ref}` |
| GET | `/api/builds/:id/logs` | SSE stream of build output |
| DELETE | `/api/builds/:id` | Remove a build |
| GET | `/api/models` | List local models |
| PUT | `/api/models/:id/activate` | Activate model (rewrites + restarts service) |
| PUT | `/api/models/:id/config` | Update model's launch config |
| DELETE | `/api/models/:id` | Delete model files |
| GET | `/api/hf/search` | Search HuggingFace `?q=` |
| GET | `/api/hf/model` | Model file list `?id=owner/name` |
| POST | `/api/hf/download` | Start download `{model_id, filename}` |
| GET | `/api/hf/download/:id/progress` | SSE download progress |
| GET | `/api/service/status` | llama-server service status |
| POST | `/api/service/start` | Start llama-server |
| POST | `/api/service/stop` | Stop llama-server |
| POST | `/api/service/restart` | Restart llama-server |
| GET | `/api/service/logs` | SSE stream of journalctl output |
| GET | `/v1/*` | Proxy to llama-server OpenAI API |

---

## Non-Goals (v1)

- Multi-instance (multiple llama-server processes simultaneously) — complex, low value for solo use
- Model conversion (GGUF generation from base weights) — too heavy, separate tooling exists
- Chat UI — use Open WebUI or SillyTavern pointed at the `/v1` proxy endpoint
- Remote HF dataset browser, fine-tuning, quantization in UI
- Windows / macOS support

---

## Open Questions

1. **D-Bus permissions**: LlamaCtl needs to manage systemd units. Decide between running as root vs. a dedicated `llamactl` user with a polkit policy allowing service management.

2. **ROCm runtime in container vs host**: If running containerized, the ROCm userspace libs in the container must match the kernel driver version on the host. Pin the ROCm version in the Dockerfile.

3. **gfx1201 support maturity**: RDNA4 / 9700 AI Pro (gfx1201) ROCm support was added in ROCm 6.x. Verify `rocminfo` correctly identifies both cards before building. May need `HSA_OVERRIDE_GFX_VERSION=11.0.0` workarounds if gfx1201 isn't fully recognized by all llama.cpp ROCm paths.

4. **Tensor split on dual 9700**: With identical cards, `--tensor-split 0.5,0.5` should be optimal. UI should still expose the control for experimentation, and display per-GPU VRAM usage live via `rocm-smi` polling.

5. **llama-server port conflict**: LlamaCtl and llama-server both need ports. Default plan: LlamaCtl on `:3000`, llama-server on `:8080`, proxied through LlamaCtl at `/v1/*`.

---

## Phased Milestones

| Phase | Deliverable | Est. Time |
|-------|-------------|-----------|
| 1 | Go scaffold + SvelteKit shell + config + embedded serving | 1–2 weeks |
| 2 | llama.cpp build manager (ROCm + Vulkan) with log streaming | 1–2 weeks |
| 3 | HuggingFace browser + resumable downloader + VRAM estimator | 1–2 weeks |
| 4 | Systemd service control + unit template + model configs + swap | 1–2 weeks |
| 5 | OpenAI proxy + auth middleware + connection tester | 3–5 days |
| 6 | Dockerfile + deployment docs + README | 3–5 days |

**Total estimated:** 6–9 weeks solo, working evenings/weekends

---

## Future (v2 Ideas)

- `rocm-smi` / `radeontop` live GPU stats dashboard (VRAM usage, GPU %, temp)
- Multiple named configurations per model (e.g. "fast" vs "quality" context settings)
- Scheduled model preloading (warm the service before peak use)
- Prometheus metrics endpoint (inference tokens/sec, queue depth)
- Container-native mode with proper D-Bus socket mounting
- Support for `whisper.cpp` and `stable-diffusion.cpp` alongside llama.cpp
