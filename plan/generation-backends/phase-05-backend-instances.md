# Phase 05 — Backend Instances: Launch and Supervise

**Depends on:** Phase 01 (supervisor), Phase 02 (store, ports), Phase 03
(sd/ace builds), Phase 04 (model sets) · **Enables:** Phase 06 (measuring and
reserving VRAM), Phase 07 (image API), Phase 08 (music API)

## Goal

Run each enabled instance as its own supervised process: one model set, on one
build, on one GPU, on one port.

Each process only sees its assigned GPU, through a per-process device
environment variable. The Server page gets a "Generation backends" section
where users can:
- create an instance
- start, stop and restart it
- watch its logs

Instances with auto-start begin with toolchest.

## Files touched

- `internal/generation/launch.go` (new): `LaunchSpec(inst, set, build, gpu,
  host, opts LaunchOptions)`, which returns binary, args, env and health URL
  for each backend kind.
  - `build` is a `builder.BuildResult`, so `generation` imports `builder` and
    not the other way round (see Phase 03).
  - `LaunchOptions{OmitMaxVRAM bool}` is used by the measured test run in
    Phase 06.
- `internal/generation/launch_test.go` (new).
- `internal/generation/manager.go` (new): `Manager` holds one
  `*process.Supervisor` per instance ID. It provides `Start(id)`, `Stop(id)`,
  `Restart(id)`, `Status(id)`, `StopAll()`, `Subscribe(id)` and `PID(id)`.
- `internal/api/gen_instances.go` (new): the instance CRUD and lifecycle
  handlers and the log SSE.
- `internal/api/server.go`:
  - Create `generation.Manager`.
  - Register the `/api/gen/instances/*` routes, which return 404 while the
    switch is off.
  - `Shutdown()` (L220) also calls `gen.StopAll()`.
- `internal/api/service.go`: start instances at toolchest startup (see step 8).
- `internal/models/gpu_assign.go`: factor the monitor-index → backend-index
  mapping out of `DeviceList` into the exported
  `BackendDeviceIndex(backend, gpu)`. `DeviceList` keeps its behaviour.
- `internal/api/settings.go`: when `GenerationEnabled` is switched off, call
  `gen.StopAll()` (see step 9).
- `web/templates/server.html` and new
  `web/templates/partials/gen_instances.html`:
  - the instance table
  - a create/edit form: model set, build, GPU, auto-start, extra launch
    arguments
  - start/stop/restart and enable/disable buttons
  - a log viewer that reuses the existing log tab component

## Steps

1. **Map the device environment.** `LaunchSpec` sets one environment variable
   to pick the GPU; there is no separate device flag.
   - **The build's profile decides which variable is set.**
   - **The GPU must suit the profile.** A ROCm build only accepts an AMD GPU
     and a CUDA build only an NVIDIA GPU, checked when the instance is created
     or edited. A Vulkan build accepts any GPU.
   
   The variable and value for each profile:

   | Profile | Variable | Value |
   |---|---|---|
   | ROCm | `HIP_VISIBLE_DEVICES` | the GPU's ROCm index |
   | CUDA | `CUDA_VISIBLE_DEVICES` | the PCI-bus-ordered CUDA index; `CUDA_DEVICE_ORDER=PCI_BUS_ID` is already set by `BuildEnv` |
   | Vulkan | `GGML_VK_VISIBLE_DEVICES` | the Vulkan index |

   - The build's profile decides which variable is used.
   - Use the same monitor-index → backend-index mapping as
     `gpu_assign.go` `DeviceList`.
   - Factor that mapping out into `models.BackendDeviceIndex(backend, gpu)` so
     both the LLM code and the generation code use it.
   - The variables are set only on that process's environment, built with
     `process.BuildEnv(buildDir, nil)`. The router's `RuntimeEnv` is not applied
     to generation backends.

2. **Build the `sd` arguments.**

   ```
   sd-server --listen-ip <host> --listen-port <port>
   ```

   - Then one flag per component, mapped from its slot:

     | Slot | Flag |
     |---|---|
     | `diffusion_model` | `--diffusion-model` |
     | `model` | `--model` |
     | `llm` | `--llm` |
     | `llm_vision` | `--llm_vision` |
     | `clip_l` | `--clip_l` |
     | `t5xxl` | `--t5xxl` |
     | `tokenizer` | `--tokenizer` |
     | `vae` | `--vae` |
     | `tae` | `--tae` |

   - Then the set's `LaunchArgs`, then the instance's `ExtraArgs`.
   - Then `--max-vram <GiB>` when `ReservedMiB > 0` and `opts.OmitMaxVRAM` is
     false. The value is `ReservedMiB/1024`, rounded down to one decimal place.
   - Health URL: `GET /v1/models`.

3. **Build the `ace` arguments.**

   ```
   ace-server --host <host> --port <port> --models <instance dir> --max-batch 1
   ```

   - `<instance dir>` is `<data>/gen-instances/<id>/models`.
   - It is created at start and holds symlinks to the set's four component
     files, because ace-server scans a directory.
   - There is no `--keep-loaded`. Default mode measured 7.2 GB at peak and
     0.3 GB idle, against about 20 GB with keep-loaded.
   - The instance's `ExtraArgs` are added last.
   - Health URL: `GET /health`.

4. **Use the router's host value.** `<host>` is the same value the router uses
   (`RouterConfig.Host`, default `0.0.0.0`), so each web UI is reachable on its
   published port in the same way the chat UI is on 8080.

5. **Create and edit instances.**
   - The form takes a name. It defaults to the model-set ID, with `-2`, `-3`
     and so on added when that name is taken. The ID is the slug of the name
     (Phase 02).
   - Creating an instance from a set with `Complete=false` is refused.
   - The build is resolved with `resolveBackendBuild(kind, inst.BuildID)`, and
     creation fails with a clear message when no successful build exists.
   - `Port` is assigned with `Store.AssignPort` when the instance is created.
   - `Enabled` is set to true.
   - `ExtraArgs` are checked with `generation.ValidateExtraArgs` (Phase 04).
   - An instance can be edited only while it is stopped. Editing extra
     arguments or the model set clears `ReservedMiB` and the measured fields.
   - **Disable** stops the instance and sets `Enabled=false`.
   - **Start** is refused on a disabled instance.

6. **Use the Phase 01 log format.** Logs go through the supervisor
   broadcaster, named `sd-server:<instance id>` or `ace-server:<instance id>`.
   The Server page log tabs list each running instance.

7. **Leave VRAM checks for Phase 06.** No VRAM checks are made in this phase.
   Starting an instance whose `ReservedMiB` is 0 is allowed here.

8. **Start instances at toolchest startup.**
   - When `GenerationEnabled` is on, every instance with `Enabled && AutoStart`
     starts.
   - This does not depend on the router's `Config.AutoStart`.
   - If the router is also auto-starting, instances start after the router
     reaches `running` or `error`. Otherwise they start straight away.

9. **Turning the switch off.**
   - Switching `GenerationEnabled` off stops every running instance.
   - Instance records, reservations and model sets stay in `generation.json`,
     so turning it back on restores them.
   - While the switch is off, every generation route and page is hidden or
     returns 404, as in the earlier phases.

## Build gate

- `go build ./...`
- `go vet ./...`
- `go test ./internal/generation/... ./internal/api/... ./internal/process/...`

## Test plan

- **`launch_test.go`, table-driven:**
  - The Ming-Image set produces `--diffusion-model`, `--llm`, `--tokenizer`
    and `--vae` in that order, followed by its launch args.
  - A ROCm build on GPU 1 sets `HIP_VISIBLE_DEVICES=1` and no CUDA variable.
  - A CUDA build sets `CUDA_VISIBLE_DEVICES`.
  - A Vulkan build sets `GGML_VK_VISIBLE_DEVICES`.
  - `ReservedMiB=10240` adds `--max-vram 10.0`, and `OmitMaxVRAM` leaves it
    out.
  - Instance `ExtraArgs` come after the set's `LaunchArgs`.
  - The ace set creates four symlinks.
- **Manager test with fake `sd-server` / `ace-server` scripts:**
  - start → running → stop
  - `StopAll` on shutdown
  - auto-start with router auto-start both on and off
  - a disabled instance cannot start
- **Manual on asmodean:**
  - Create an instance of the `qwen-turbo-dev` custom set (Phase 04) and an
    ACE-Step XL SFT instance on GPU 0, and start both.
    - Qwen-Image's peak of 10,283 MiB plus ACE-Step's peak of 7,239 MiB fits
      on the 32 GB card even with no fit checks yet.
    - Do not pair Ming-Image with anything in this phase, because the fit
      checks only arrive in Phase 06.
  - Start a Ming-Image instance on its own afterwards, with the other two
    stopped.
  - Generate directly against their ports with `curl`, as on 2026-10-24.
  - Open both web UIs on ports 8081 and 8082.
  - Restart toolchest and confirm the auto-start instances come back.

## Commit

`feat(generation): launch and supervise sd-server and ace-server instances`

## Rollback

Revert the commit. Running instances are stopped by the old binary's shutdown
only if it knows about them, and the reverted binary does not. So stop all
instances from the Server page before rolling back. `<data>/gen-instances/` can
be deleted.
