# Phase 05 — Fork instances: run, reserve the GPU, stop

**Depends on:** 01 (the setting), 02 (recipes, store, needs), 03 (`resolveForkBuild`), 04 (complete model sets, `forkModelsDir`) · **Enables:** Phase 06 (the Server page card drives this API), Phase 07 (the proxy sends requests to running instances)

## Goal

Add the backend for fork instances: create, edit and delete them, start and stop them under a supervisor, auto-start them with the toolchest, and stop them all when the setting is turned off. A running or starting fork instance reserves its whole GPU: router model loads and generation instances cannot place memory on it, and auto-evict never stops it. A start is refused, with a message that names what is there, when the GPU already holds a router model, a generation instance or another fork instance. Hardware needs that are not met give warnings but do not block. This phase is API only; Phase 06 adds the UI.

## Files touched

- `internal/forks/manager.go` — new: `Manager` with one `process.Supervisor` per instance, modelled on `internal/generation/manager.go` (`Start`, `Stop`, `Status`, `IsRunning`, `IsActive`, `PID`, `WaitRunning`, `Subscribe`, `Unsubscribe`, `RecentLogs`, `Forget`, `StopAll`).
- `internal/api/fork_instances.go` — new: CRUD, start, stop, logs and auto-start.
- `internal/api/fork_hostinfo.go` — new: `forks.HostInfo` for this host (RAM from `/proc/meminfo`, AVX2 from `/proc/cpuinfo`, GPU arch and backend from the detected hardware, free disk of the models dir).
- `internal/api/gen_vram.go` — fork instances added to the per-GPU reservation sum.
- `internal/api/settings.go` — fill `forksSwitched` from Phase 01.
- `internal/api/server.go` — `forkMgr` field, routes, auto-start after the router at startup, `StopAll` on shutdown.
- `internal/api/gen_instances.go` — port assignment and checks also look at fork instances.
- `internal/api/fork_instances_test.go`, `internal/api/fork_vram_test.go`, `internal/forks/manager_test.go` — new tests.

## Steps

1. `forks.Manager`: same shape as `generation.Manager`, without the generation kinds and measuring. The supervisor health URL is `http://127.0.0.1:<port>/health`. The start timeout is 20 minutes, because Avalon reads an 89 GiB model and fills its expert cache before `/health` answers 200. Log lines go to the supervisor's ring buffer, as for generation instances.
2. Ports: fork instances take ports from the generation port range (`cfg.GenPorts(inContainer)`, called the same way `gen_instances.go:627` gets `start, count`). Change the port helpers so that assigning or checking a port, for either kind of instance, treats ports used by both stores as taken. A new fork instance with no port gets the first free one.
3. Routes, all 404 while forks are off:
   - `GET /api/forks/instances`, `POST /api/forks/instances`, `PUT /api/forks/instances/{id}`, `DELETE /api/forks/instances/{id}`.
   - `POST /api/forks/instances/{id}/start`, `POST /api/forks/instances/{id}/stop`.
   - `GET /api/forks/instances/{id}/logs` (SSE, same as the generation logs endpoint).
   The body's `extra_args` is text; split it with `generation.ValidateExtraArgs` and refuse it on error. Saving validates: the fork exists, the GPU index exists, the port is in range and free, `ExtraEnv` lines are valid (Phase 02), and the ID does not match a registry model ID or a generation instance ID (the ID is the API model name). Edits and deletes are refused while the instance runs.
4. `startForkInstance(id)`:
   1. Forks on, the instance exists, its fork's model set is complete, and `resolveForkBuild` returns a build whose backend matches the GPU's vendor. Each failure has its own message.
   2. **Busy GPU check.** Collect what is on the GPU: router models whose planned memory (`loadedLLMMiB`) uses that GPU, generation instances on that GPU that are running or reserved, and other fork instances on it that are running or starting. If the list is not empty, refuse with 409 and "GPU 1 holds Qwen3.6-35B-A3B (router) and flux-dev (generation). Unload or stop them first." Nothing is stopped for the user.
   3. **Needs.** Call `forks.CheckNeeds` with `HostInfo`. Return the warnings in the response, but go on.
   4. Launch: binary from the build directory, `forks.Args`, and `process.BuildEnv` with each `forks.Env` line set over it by `builder.SetEnv` (as generation launches set the GPU selector), so an inherited `HIP_VISIBLE_DEVICES` cannot win; `device` comes from `models.BackendDeviceIndex`. The working directory is the model set's folder, so `moe-cache-ranking.bin` is found next to the model.
5. Reservations in `gen_vram.go`: keep the name `genVRAM` (its callers and tests stay as they are) and change its doc comment to say it covers fork instances too. Change its guard from "generation on" to "generation on or forks on", and skip the generation loop when generation is off. For each fork instance that is starting or running, add an entry for its GPU with `Reserved` = the GPU's total MiB, `Using` = the process's measured VRAM (the same `s.genProcVRAM.read(pid)` call, a `procVRAMCache`), `Evictable` = 0, and a name "<instance> (fork, whole GPU)". `genReservationsApply` uses the same guard. The effect: `applyGenVRAM` marks the GPU as full for the router's plan, `reservationFitError` refuses an LLM load that needs that GPU and names the fork instance, and `genFitCheck` refuses a generation instance on it.
6. Auto-evict (`gen_evict.go`) only walks generation instances; add a test that proves a fork instance is never chosen.
7. `forksSwitched(was, now)`: off → `forkMgr.StopAll()` in a goroutine under a new `forkSwitchMu`; on → `autoStartForkInstances()`. Both store files stay as they are.
8. `autoStartForkInstances()`: at startup after the router, and when the setting is turned on, start every instance with `AutoStart`, one after another. Each one goes through `startForkInstance`, so a busy GPU or a missing build is logged as a WARN line and skipped.
9. Server shutdown calls `forkMgr.StopAll()` next to `genMgr.StopAll()`.
10. Engine notes stay router-only: do not attach the engine notes tracker to fork supervisors (overview non-goal).

## Build gate

- `make build`
- `go vet ./...`
- `go test ./...`
- `make js-test`

## Test plan

- `manager_test.go`: start, health wait and stop with a fake server binary (the pattern the generation manager tests use).
- `fork_instances_test.go`: CRUD validation (bad ID, ID equal to a registry model, bad env line, port out of range, port used by a generation instance); edit refused while running; start refused with each reason (set not complete, no build, wrong vendor); busy GPU refused with a message that names a router model and a generation instance; needs warnings returned while the start goes on; with forks off every route is 404; turning the setting off calls `StopAll`, turning it on starts `AutoStart` instances.
- `fork_vram_test.go`: a running fork instance on GPU 1 makes `reservationFitError` refuse an LLM planned on GPU 1 and accept one on GPU 0; `genFitCheck` refuses a generation instance on GPU 1; the reservation is gone after stop; auto-evict never picks it.
- Manual on asmodean: create an Avalon instance on the R9700, start it with `ExtraEnv` `GGML_CUDA_MOE_CACHE_MODE=off` (Avalon#2), check with `/proc/<pid>/environ` that the variable is set, wait for healthy, `curl` its port directly, stop it.

## Commit

`feat(forks): run fork instances on a reserved GPU`

## Rollback

Revert the commit. `forks.json` keeps the instance entries, which nothing reads. The `genVRAM` change is reverted with it, which brings back the generation-only sum.

## As built (2026-10-31)

- `reservationFitError` keeps its wording ("reserved for generation backends on GPU N"): `reservedOnGPURE` in `autoconfig.go` parses it. A fork reservation still shows in the message by name, as "<instance> (fork, whole GPU)".
- `genReservationsApply` is now true whenever any reservation exists (generation or fork); `reservationNote` shows while either feature is on.
- The instance list (`GET /api/forks/instances`) answers JSON only; Phase 06 adds the HTML card. Start and stop are `POST /api/forks/instances/{id}/{start|stop}` and answer `{"warnings": [...], "state": ...}`.
- Generation instances now also avoid ports held by fork instances (`instancePorts`).
