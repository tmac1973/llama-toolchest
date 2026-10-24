# Phase 01 — Generic Process Supervisor

**Depends on:** nothing · **Enables:** Phase 05 (backend instances run as
supervised processes)

## Goal

Split the process lifecycle code out of `internal/process/manager.go` into a
generic supervisor. That code covers start, stop, restart, health polling, log
broadcasting and environment assembly. After the split, the llama-server router
becomes one user of the supervisor, and later phases can run `sd-server` and
`ace-server` with the same, already tested code.

This is a refactor only. Router behaviour, log text, SSE log streams and the
public `Manager` API do not change.

## Files touched

- `internal/process/supervisor.go` (new): the `Supervisor` type and its
  `SupervisorConfig`.
- `internal/process/manager.go`: `Manager` keeps its exported API but delegates
  the lifecycle work to an embedded `*Supervisor`. Router-specific argument
  building (`--models-preset`, `--models-max`) and the routerclient methods stay
  in `Manager`.
- `internal/process/manager_unix.go`, `internal/process/manager_windows.go`:
  move the process-group and signal helpers so the supervisor uses them.
- `internal/process/supervisor_test.go` (new): tests that run a fake binary.
- `internal/process/manager_test.go`: unchanged. It must still pass.

## Steps

1. **Create `SupervisorConfig`** with these fields:

   | Field | Type | Purpose |
   |---|---|---|
   | `Name` | `string` | Used in log lines, e.g. `"llama-server"` or `"sd-server:ming-image"` |
   | `BinaryPath` | `string` | Program to run |
   | `Args` | `[]string` | Command-line arguments |
   | `Env` | `[]string` | Full environment, already assembled |
   | `WorkDir` | `string` | Working directory |
   | `HealthURL` | `string` | Polled to decide when the process is ready |
   | `HealthTimeout` | `time.Duration` | Defaults to 5 min, as `pollHealth` does today |
   | `StopTimeout` | `time.Duration` | Defaults to 30 s, as `Stop` does today |
   | `LogHistory` | `int` | Defaults to 500 lines |

2. **Move the lifecycle code into `Supervisor`.**
   - Move the body of `Manager.Start` (process start, `monitorProcess`,
     `pollHealth`), `Stop` (SIGTERM, then kill after `StopTimeout`), `Restart`,
     `GetStatus`, `IsRunning`, `WaitRunning`, `CheckHealth`, and the
     log broadcaster methods (`Subscribe`, `Unsubscribe`, `RecentLogs`,
     `ClearLogs`) into it.
   - Keep the existing `Status` values (`stopped`, `starting`, `running`,
     `error`).
   - Add `PID() int`, which returns 0 when the process is not running. Phase 06
     needs it to attribute VRAM to a process.

3. **Add `process.BuildEnv(binDir string, extra []string) []string`.** It
   returns `applyExtraEnv(pinCUDADeviceOrder(libpath.Prepend(os.Environ(),
   binDir)), extra)`, which is the expression at `manager.go` L118 today.
   `Manager` calls it with `cfg.ExtraEnv`.

4. **Rewrite `Manager.Start(cfg RouterConfig)`.**
   - Build the router `Args`, exactly as at L98-111 today.
   - Build `Env` with `BuildEnv`.
   - Set `HealthURL` to `http://<host>:<port>/health`.
   - Call `supervisor.Start`.
   - Keep the "Router" wording in the log lines by passing
     `Name: "llama-server"` and formatting the messages as they are today.

5. **Keep `Manager`'s exported API unchanged,** so `internal/api` (`server.go`
   L103, `service.go` `startRouterWith`) needs no changes.

## Build gate

- `go build ./...`
- `go vet ./...`
- `go test ./internal/process/... ./internal/api/...`

## Test plan

Add these tests to `supervisor_test.go`, each using a shell-script fake binary
in `t.TempDir()`, following the pattern of `memory_measured_test.go` L221:

- **Start and stop:**
  - The fake serves `/health` through a small Go `httptest` listener whose
    address is passed in as an argument.
  - Status moves `starting` → `running`, and `PID()` is non-zero.
  - `Stop` brings the status to `stopped`.
- **Ignores SIGTERM:** a fake that traps SIGTERM is killed after `StopTimeout`.
- **Exits early:** a fake that exits before it becomes healthy leaves the
  status at `error`, with its last log lines available from `RecentLogs`.
- **Logs:** lines printed by the fake reach a subscriber.

Run the existing `proxy_test.go` fake-router tests and
`job_capability_test.go` (`fakeLlamaServerDir`) unchanged.

Manual check: start the real router from the Server page on compute2 and
confirm that the logs, status and restart work as before.

## Commit

`refactor(process): extract a generic supervisor from the router manager`

## Rollback

Revert the single commit. Nothing outside `internal/process` changes, and there
is no data or config migration, so a partial revert is not possible and is not
needed.
