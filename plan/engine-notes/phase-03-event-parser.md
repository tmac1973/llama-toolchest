# Phase 03 — Engine-notes event parser

**Depends on:** 02 · **Enables:** Phase 04 stores the outcomes it produces; Phase 05 feeds it the live router log

## Goal

A tested parser that turns the router's log, line by line, into lifecycle outcomes: an instance loaded, failed to load, crashed or was unloaded, and the router stopped or exited on its own. For a crash or failed load it keeps the exit status, the error lines and the instance's last 30 raw lines. It runs on recorded logs in tests, with no server, no clock and no I/O.

## Files touched

- `internal/enginenotes/doc.go` — new: package comment (what the package reads, why it exists, that it imports only `internal/serverlog`).
- `internal/enginenotes/tracker.go` — new: `Tracker`, `Outcome`, `OutcomeKind`.
- `internal/enginenotes/tracker_test.go` — new: tests on recorded router output.
- `internal/enginenotes/testdata/*.log` — new: recorded router logs (see Test plan).

## Steps

1. Types:
   - `OutcomeKind` is a string type with the values `"load"`, `"load_failed"`, `"crash"`, `"unload"`, `"router_stop"`, `"router_exit"`. A string, because it is written to JSON and read in templates.
   - `Outcome` has `Kind`, `Time time.Time`, `RouterName string` (the router's `name=`), `Port string`, `ExitStatus int`, `LoadSeconds float64` (spawn to ready; set on `load`), `LoadedSeconds float64` (ready to exit; set on `crash` and `unload` after a ready), `ErrorLines []string`, `Tail []string`, `RouterError string` (set on `router_exit`).
2. `Tracker` keeps, per port, an `instance{name, spawnedAt, readyAt, tail ring of 30 lines}`, and a set of names with a stop requested. `NewTracker()` returns an empty one. It is not safe for use from more than one goroutine; Phase 05 calls it from one reader.
3. `Add(line string, now time.Time) []Outcome`. The caller passes `now`, so tests control time. In this order:
   - Strip a leading `data: ` (as `ErrorLines` does).
   - `serverlog.ParseSpawn` → open the instance for that port; clear any stop flag for that name (a new instance of a name starts fresh).
   - `serverlog.InstancePort` → append the line to that instance's tail ring.
   - `serverlog.LoadedPort` (the instance's own `srv server_run_i: model loaded`, printed at the default verbosity) or `serverlog.ReadyPort` (the router's `cmd_child_to_router` ready line, printed only at verbosity 5) → set `readyAt`, emit `load` with `LoadSeconds`. The first one seen counts. Checked before the `[port]` test, since the router's ready line starts with a timestamp. `LoadedPort` is added to `internal/serverlog` in this phase: the compute2 log at verbosity 4 has no router ready line at all.
   - `stopping model instance name=(\S+)` → mark the name as stopping.
   - `instance name=(\S+) exited with status (-?\d+)` → find the open instance with that name and close it:
     - stop was requested → `unload`, whatever the status (a stop that timed out is killed and reports 1);
     - status 0 → `unload`;
     - status ≠ 0 and the instance had reached ready → `crash`;
     - status ≠ 0 before ready → `load_failed`.
     For `crash` and `load_failed`, `ErrorLines = serverlog.ErrorLines(tail)` and `Tail` is a copy of the ring.
   - `==> Router stopped` → emit `router_stop`; close every open instance as `unload`.
   - `==> Router exited with error: (.*)` → emit `router_exit` with `RouterError`; close every open instance as `unload` (they ended with their parent, not on their own). `==> Router exited normally` without a preceding stop line is also `router_exit`, with an empty `RouterError`.
   - Any other line → nothing.
4. An exit for a name with no open instance (the toolchest started after the spawn) still emits an outcome, with no tail and no `ErrorLines`: a stop request or status 0 → `unload`; any other status → `crash`. Without the spawn there is no way to know whether it was ready, and an instance that outlived a toolchest restart had almost certainly loaded.
5. `Open() []OpenInstance` returns `{RouterName, Port, ReadySince}` for every instance that is ready now, so Phase 04 can count time loaded for models still running. It returns a new slice the caller may keep.
6. The tracker clears itself: `router_stop` and `router_exit` close every open instance and clear the stop flags. The supervisor always prints one of those lines when a router ends (`Supervisor.Stop` and `monitorProcess`), and always before the next router starts, so ports reused by the next run never meet stale state. No outside reset is needed.
7. Why exit status 1 is enough: the router's `subprocess_join` reports `WEXITSTATUS` for a normal exit and `EXIT_FAILURE` (1) for a child killed by a signal, so an abort after a CUDA error is 1. Write this in the code comment on the crash rule.

## Build gate

- `make build`
- `go vet ./...`
- `go test ./...`

## Test plan

Recorded fixtures in `testdata/`, built from real v0.8.0 router output (the compute2 log captured during this planning, extended with exit lines in the router's exact format):

- `load_unload.log`: spawn, ready, stopping, exited 0 → `load` then `unload`, `LoadSeconds` and `LoadedSeconds` from the passed times.
- `crash_cuda.log`: spawn, ready, a CUDA error and a backtrace, exited 1 → `crash`; `ErrorLines[0]` contains `CUDA error`; no backtrace frame in `ErrorLines`; `Tail` holds at most 30 lines and ends with the instance's last line.
- `stop_timeout.log`: stopping, then exited 1 → `unload`, not `crash`.
- `load_failed.log`: spawn, an allocation error, exited 1 before ready → `load_failed` with the allocation line first.
- `interleaved.log`: two instances on two ports with lines mixed; one crashes → its tail holds only its own port's lines.
- `router_exit.log`: one ready instance, then `==> Router exited with error: signal: aborted` → `router_exit` and an `unload` for the instance.
- A respawn of the same name after a stop: the stop flag does not carry over, so a later non-zero exit is a crash.
- An exit with no matching spawn: status 1 → `crash`; status 1 after a stopping line → `unload`; status 0 → `unload`.
- After `router_stop`, a new spawn on the same port starts a fresh instance with an empty tail.

## Commit

`feat(enginenotes): read model loads, crashes and unloads from the router log`

## Rollback

Revert the commit. Nothing uses the package yet, so removal is safe until Phase 04 lands.
