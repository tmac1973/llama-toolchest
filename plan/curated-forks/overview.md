# Curated llama.cpp Forks — Project Overview

## Problem

Some llama.cpp forks run one model much faster than upstream. Avalon (github.com/freriadoc/Avalon, MIT) is tuned for Qwen3.8-Flash-Next (125B MoE, ~89 GiB GGUF) on one RDNA3 GPU. It keeps the dense weights, the KV cache and a cache of the hottest experts in VRAM and the rest of the experts in system RAM, and it reports 91 t/s on an RX 7900 XT. That makes it a good match for a friend's RX 7900 XTX cards. Today the toolchest cannot build or run it: the llama.cpp repo URL is hard-coded (`internal/builder/builder.go`, `llamaCppRepo`), the router runs one global build, and the launch options written for that build are chosen by upstream build number (`internal/models/launch.go`), which a fork's history does not follow.

Tracked in issue #494.

## Goals

- Add a short list of curated forks to the toolchest source. Each entry gives the repo, a pinned tested commit, its build profiles, a launch recipe, its paired model files and its hardware needs. Avalon is the first and only entry in this work.
- Build a curated fork from the Builds page, at its pinned commit or, as an advanced choice marked untested, at the fork's latest `main`.
- Download a fork's paired model set with one click from a fork catalog, with resume, including extra files (Avalon's MTP head in `MTP/` and `moe-cache-ranking.bin`).
- Run a fork as a **fork instance**: a separate llama-server process (not router mode) on its own port and one whole GPU, next to the main router and the generation instances.
- Launch each fork instance from its curated recipe, plus two free-text fields per instance: extra arguments and extra environment variables (for example `GGML_CUDA_MOE_PIN=0`).
- Send OpenAI-compatible requests whose `model` is a fork instance's name to that instance, and list running fork instances in `/v1/models`.
- Hide all of it behind a new opt-in setting, "llama.cpp forks (experimental)", separate from the generation setting.

## Non-goals

- Fork builds as the router's active build, or as the build for benchmark, Autotune or Autoconfigure jobs. Fork instances are not tuned or benchmarked by the toolchest.
- A second router for forks that serve the whole model collection (ik_llama.cpp style). A separate issue tracks it.
- Any repo URL the user types. Only curated forks in the source.
- Any git ref other than the pinned commit and the fork's latest `main`.
- More than one GPU per fork instance (Avalon's multi-GPU mode is experimental: Avalon#1, Avalon#9).
- A fixed VRAM budget or sharing a GPU with anything else.
- Load on first request, idle stop or auto-evict for fork instances.
- Engine notes (crash and usage tracking) for fork instances.
- ik_llama.cpp or any fork other than Avalon.
- Using a model folder the user downloaded by hand.

## Users & primary flow

The user runs the toolchest on a Linux host with a supported GPU (here: a friend's machine with RX 7900 XTX cards) and serves models to their own clients.

1. In Settings the user turns on "llama.cpp forks (experimental)". Router memory turns off, as it does for generation. The page reloads and the fork sections appear.
2. On the Builds page the user picks the project "Avalon", the profile ROCm and the ref "Pinned (tested)", and builds. The build appears in the builds table with an `avalon-` ID.
3. On the Models page, under "Fork model sets", the user downloads "Qwen3.8-Flash-Next Avalon IQ3_KS" (~89 GiB). The set is marked complete when every file is in place.
4. On the Server page, under "Fork instances", the user adds an instance: the fork, the model set, the build (newest Avalon build by default), one GPU, a port, extra arguments, extra environment variables and "start with toolchest". If the host falls short of the recipe's needs (32 GB RAM, AVX2, an RDNA3 or supported CUDA GPU), the form shows a warning. The user can still save and start.
5. The user starts the instance. It claims the whole GPU; the router and generation instances cannot place work on that GPU while it runs. Its log shows on the Server page.
6. A client sends `/v1/chat/completions` with `"model": "<instance name>"` to the toolchest port. The toolchest sends it to the fork instance. Other model names still go to the main router.
7. The user stops the instance when done, or leaves it running. Turning the setting off stops every fork instance; the builds, model sets and instance config stay on disk, so turning it on again restores them.

## Constraints

- Go backend in `internal/`, HTML templates with inline scripts in `web/templates/`, htmx. Same patterns as the generation feature (`internal/generation/`, `internal/api/gen_*.go`).
- Container and host installs must both work. Host mode runs under systemd with `ProtectHome`, and needs a test before release.
- Avalon is tested upstream with ROCm 7.2 on gfx1100. The toolchest's ROCm containers may use other ROCm versions (asmodean must build on ROCm 10).
- Avalon has no release tags and changes daily; nine issues opened in its first weeks. Pin a commit that has been run, and update it on purpose.
- Avalon needs at least 32 GB RAM, AVX2 and ~98 GiB disk for the model. With less RAM than the model it may hang at load on ROCm (Avalon#3; workaround `GGML_CUDA_MOE_PIN=0`). Its expert cache hangs on gfx1201 (Avalon#2; workaround `GGML_CUDA_MOE_CACHE_MODE=off`).
- One fork instance uses one whole GPU. With `--fit`, Avalon fills free VRAM with expert cache.
- Avalon loads in 30 s to several minutes. Requests sent while it loads must get a clear "loading" error, not hang.
- All UI text and tooltips follow the project's writing rules (short, ASD-STE100 style, no slang).

## Success criteria

- With the setting off, no fork UI shows, fork API routes return 404, and no fork instance runs.
- With it on, the Builds page builds Avalon at the pinned commit for ROCm and CUDA, and at latest `main` with an "untested" mark.
- The fork catalog downloads the full Avalon set, resumes after an interruption, and marks it complete only when every file is present.
- A fork instance starts, reaches healthy, serves `/v1/chat/completions` through the toolchest port, and appears in `/v1/models`; the main router keeps serving its own models.
- While a fork instance runs, its GPU is unavailable to the router's placement and to generation instances.
- Extra environment variables reach the fork process (checked with `GGML_CUDA_MOE_PIN=0`).
- Turning the setting off stops running fork instances; turning it on again with "start with toolchest" ticked starts them again.
- Build, download, start/stop and routing pass on asmodean. Avalon serves Qwen3.8-Flash-Next at the expected speed on the friend's RX 7900 XTX.
- `go build`, `go vet`, `go test ./...` and the JS tests pass in CI.

## Decisions

- **Scope of the first version** → a generic curated-fork mechanism, with Avalon as the only entry.
- **Who decides which forks exist** → a curated list in the toolchest source; no user repo URLs.
- **Which refs can be built** → the pinned tested commit, plus the fork's latest `main` as an advanced, untested choice.
- **What turning the opt-in off does** → hide the UI and stop fork instances; keep builds and config on disk.
- **How a fork serves models** → first considered as the router's active build; changed to contained fork instances, because the router writes options by upstream build number and a fork's number does not match upstream. A fork instance takes its options from a curated recipe instead. Fork instances do not get Autotune, Autoconfigure or benchmarks. A second router for whole-collection forks is a separate issue.
- **Opt-in setting** → a separate "llama.cpp forks (experimental)" setting; it turns router memory off as generation does.
- **GPU memory** → a fork instance reserves one whole GPU, exclusively.
- **Multi-GPU** → one GPU per instance.
- **Model files** → one-click download from a fork catalog.
- **Launch settings** → fixed recipe plus per-instance extra arguments and extra environment variables.
- **Lifecycle** → manual start and stop, plus a "start with toolchest" option; no eviction or load on request.
- **Hardware checks** → warn when the host falls short of the recipe's needs, but allow start.
- **Test hosts** → plumbing on asmodean; final speed and stability check on the friend's RX 7900 XTX machine.
