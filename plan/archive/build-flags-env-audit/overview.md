# Build Flag Toggles & Runtime Environment Audit — Project Overview

## Problem

The Build tab's preset toggles have drifted out of sync with upstream llama.cpp.
Several toggles are silent no-ops on every ref a user would realistically build
today: `GGML_CUDA_F16` and `LLAMA_HIP_UMA` were removed upstream,
`GGML_CUDA_ENABLE_UNIFIED_MEMORY` is a runtime environment variable that was
never a CMake flag on current refs, and `GGML_CUDA_FORCE_CUBLAS_COMPUTE_16F` and
HIP fast-math are already annotated as dead. Meanwhile, genuinely useful flags —
`GGML_HIP_RCCL` (the ROCm collective library, directly relevant to
tensor-parallel all-reduce), `GGML_CUDA_FA_ALL_QUANTS` (FlashAttention kernels
for quantized KV cache), `GGML_CUDA_NO_PEER_COPY` (fixes corrupt output on
non-P2P PCIe topologies) — have no toggle at all, so less-expert users never
find them. The `GGML_HIP_ROCWMMA_FATTN` toggle is a special case: upstream
removed the rocWMMA path in PR #26046 (cleanup landed in b10332), yet it remains
the fastest prompt-processing option on RDNA4 (open regression #26220 shows the
native replacement kernel is 25–49% slower at depth on gfx1201) — so the toggle
must stay, with a description that explains it only works on refs older than
b10332 and why an RDNA4 user would pin one.

The Settings tab's runtime environment section has the opposite problem: four
stacked label+dropdown+help blocks consume a lot of vertical space for four
variables, there is no way to set variables outside the curated list, and
newer high-value variables (`GGML_CUDA_P2P`, `CUDA_SCALE_LAUNCH_QUEUES`,
`GGML_VK_DISABLE_COOPMAT`, `GGML_VK_FORCE_MAX_ALLOCATION_SIZE`) are missing.

## Goals

- Curate the Build tab toggle lists for ROCm, CUDA, and Vulkan around one bar: a
  toggle earns its place by offering a real performance or compatibility payoff
  a non-expert might want (~4–6 per backend beyond the common set).
- Remove toggles that are no-ops on any realistically-built ref; keep
  `GGML_HIP_ROCWMMA_FATTN` with a rewritten description covering the b10332
  removal, the RDNA4 prompt-processing regression (#26220), and the
  pin-an-older-ref workflow.
- Add the researched new toggles (ROCm: `GGML_HIP_RCCL`,
  `GGML_CUDA_FA_ALL_QUANTS`, graphs toggle; CUDA: `GGML_CUDA_FORCE_CUBLAS`,
  `GGML_CUDA_FA_ALL_QUANTS`, `GGML_CUDA_NO_PEER_COPY`, graphs toggle; common:
  `GGML_LTO` — exact list finalized in the phase plan).
- Rename the ROCm profile's `AMDGPU_TARGETS` base flag to the current
  `GPU_TARGETS` name (upstream honors both; `GPU_TARGETS` is the documented
  form).
- Rebuild the runtime environment section as a compact table: one slim row per
  curated variable (label, code name, value dropdown, help collapsed behind a
  disclosure), filtered to the active build's backend so a ROCm box never shows
  CUDA-only variables.
- Extend the curated variable set with `GGML_CUDA_P2P`,
  `CUDA_SCALE_LAUNCH_QUEUES` (CUDA), `GGML_VK_DISABLE_COOPMAT`,
  `GGML_VK_FORCE_MAX_ALLOCATION_SIZE` (Vulkan), keeping the existing four.
- Add a free-form "Extra environment" entry for arbitrary `KEY=VALUE` pairs and
  a read-only "Effective environment" preview, following the Extra
  flags/Effective Flags pattern used in model config and the Build tab. Any
  variable is accepted; a small named list of risky variables (`CUDA_DEVICE_ORDER`,
  `HIP_VISIBLE_DEVICES`/`CUDA_VISIBLE_DEVICES`/`ROCR_VISIBLE_DEVICES`,
  `HSA_OVERRIDE_GFX_VERSION`) gets an inline warning explaining the specific
  hazard but saves anyway.
- Build the environment feature as a self-contained, reusable component
  (curated options + free-form entries + validation + risky-variable warnings +
  effective-env rendering) used at global scope in Settings now, shaped so a
  per-model copy can slot into model config later without rework.
- Update the help page sections for the Build tab and Settings to match the new
  toggle lists and environment UX.

## Non-goals

- No per-model environment UI. Children inherit the router's environment
  identically and preset INI carries only CLI args, so a per-model editor would
  be a control that does nothing. The reusable component is the groundwork; the
  UI waits for upstream support.
- No migration of saved build-option overrides for removed toggles. Stored
  overrides for unknown flags are already ignored harmlessly, and the removed
  toggles were no-ops anyway — they vanish silently.
- No changes to the Metal or CPU profiles beyond receiving any new common
  toggles.
- No changes to benchmark build-option sweep mechanics.
- No blocking validation on free-form environment entries — warn-and-save only.
- No deep rewrite of help documentation beyond the sections the changed UI
  touches.

## Users & primary flow

The toolchest's users are self-hosters running llama.cpp on ROCm, CUDA, or
Vulkan GPUs who are comfortable with a web UI but not necessarily with
llama.cpp's CMake surface or ggml's environment variables.

Build flow: the user opens the Build tab, picks a profile (rocm/cuda/vulkan),
and sees a short list of toggles where every switch does something real on a
current ref — each with a description saying when to enable it. An RDNA4 user
reading the rocWMMA toggle learns they must pin a ref older than b10332 to get
it, and why. Experts bypass toggles entirely via Extra CMake flags, and the
effective flags preview shows exactly what cmake will receive either way.

Settings flow: the user opens Settings and sees a compact table of environment
variables relevant to their active backend, sets values from dropdowns, adds
anything exotic in the free-form Extra environment box (getting a warning — not
a refusal — if the variable is a known-risky variable), and confirms the final
`KEY=VALUE` list in the Effective environment preview. The change applies on
the next router restart, process-wide.

## Constraints

- Go backend (`internal/builder/profiles.go` for toggles,
  `internal/config/runtime_env.go` for env), Pico.css + htmx server-rendered
  templates (`web/templates/`), no JS framework.
- `ApplyOptionOverrides` is the single source of truth shared by real builds
  and the flag preview; toggle changes must preserve that property. The same
  single-source rule applies to the env component: validation, warnings, and
  the effective preview must come from one place.
- The runtime environment applies to the router process and is inherited by
  every spawned model instance identically; the UI copy must keep saying so.
- Backend filtering for env rows keys off the active build's backend
  (`Server.activeBackend()` exists as of the `--device` work).
- `applyExtraEnv` in `internal/process/manager.go` gives inherited service
  environment precedence over UI-set values, and `pinCUDADeviceOrder` pins
  `CUDA_DEVICE_ORDER=PCI_BUS_ID`; the effective preview and risky-variable warnings
  must be consistent with both behaviors.
- Existing `config.yaml` `runtime_env` maps must keep loading unchanged; the
  free-form additions extend the schema rather than replace it.
- Toggle flag knowledge is ref-dependent (e.g. rocWMMA < b10332); descriptions
  carry the ref boundaries since the toolchest builds arbitrary tags.

## Success criteria

- Every toggle shown for rocm, cuda, and vulkan either affects current-ref
  builds or (rocWMMA only) documents exactly which refs it affects.
- `GGML_CUDA_F16`, `LLAMA_HIP_UMA`, `GGML_CUDA_ENABLE_UNIFIED_MEMORY`,
  `GGML_CUDA_FORCE_CUBLAS_COMPUTE_16F`, and the HIP fast-math toggle no longer
  appear; existing saved configs with those overrides still load and build.
- The new toggles appear with accurate descriptions, and enabling any of them
  changes the effective CMake flags preview and the actual cmake invocation
  identically.
- The ROCm profile emits `GPU_TARGETS` instead of `AMDGPU_TARGETS`.
- The runtime env section renders at roughly a third of its current height,
  shows only backend-relevant curated rows, and offers all eight curated
  variables across backends.
- A free-form entry like `FOO=bar` reaches the router process's environment on
  restart; setting `HSA_OVERRIDE_GFX_VERSION` shows a warning and still saves.
- The Effective environment preview matches what `RuntimeEnvPairs` +
  `applyExtraEnv` actually produce, including showing when an inherited service
  variable wins.
- The env component's Go types and template are structured so adding a
  per-model instance would not require modifying the component itself
  (demonstrated by a unit test exercising the component at a non-global scope).
- `go build ./... && go test ./internal/...` passes; help page reflects the new
  reality.

## Decisions

- **Stale toggles** → Drop dead toggles; keep rocWMMA with a rewritten
  ref-pinning description (removed upstream at b10332, RDNA4 PP regression
  #26220, pin an older ref to use it).
- **Bar for new toggles** → Perf + fixes, curated: real performance or
  compatibility payoff for non-experts, ~4–6 per backend; debug-only flags
  excluded.
- **Vulkan toggle set** → Drop both debug toggles (Check Results, Validation
  Layers); Vulkan shows common toggles only, and Vulkan tuning moves to runtime
  env variables.
- **Runtime env UX** → Compact table with one slim row per variable, help
  behind a disclosure, filtered by the active build's backend; free-form Extra
  environment + read-only Effective environment preview follow the Extra
  flags/Effective Flags pattern.
- **Free-form validation policy** → Allow any `KEY=VALUE`; warn inline on known
  known-risky variables (`CUDA_DEVICE_ORDER`, visible-devices vars,
  `HSA_OVERRIDE_GFX_VERSION`) but save anyway.
- **Migration of removed-toggle overrides** → Drop silently; stored overrides
  for unknown flags are already ignored and the flags were no-ops.
- **Scope** → Toggles + env + help-text updates + `AMDGPU_TARGETS` →
  `GPU_TARGETS` rename; per-model env groundwork included.
- **Per-model env groundwork (after upstream reality check)** → Env stays
  global in Settings — the only thing that can work today, since the router's
  children inherit its environment identically and preset INI carries only CLI
  args. Build the env feature as a reusable component so a per-model copy can
  slot in later; no dead per-model UI.
