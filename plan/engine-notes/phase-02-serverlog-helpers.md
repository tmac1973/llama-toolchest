# Phase 02 — Shared llama-server log helpers

**Depends on:** nothing · **Enables:** Phase 03 reads spawn, ready and error lines with the same code that benchmarks and the memory report use

## Goal

Move the llama-server log-reading helpers that two packages already need, and that the engine-notes parser will be the third user of, into one leaf package `internal/serverlog`. It is a pure refactor: no behaviour changes, and the existing tests move with the code and still pass.

## Files touched

- `internal/serverlog/serverlog.go` — new. Holds the line shape, the instance prefix, the error-line picker and the spawn and ready matchers.
- `internal/serverlog/serverlog_test.go` — new. The tests for `ErrorLines` moved from `internal/benchmark/failure_log_test.go`, plus spawn/ready tests moved from `internal/memreport/collector_test.go`.
- `internal/benchmark/failure_log.go` — delete `serverLogRE`, `instancePrefixRE`, `rawFailureWords`, `hasFailureWord`, `stackFrameRE` and `serverErrorLines`; call `serverlog.ErrorLines`. `linesAfter`, the memory words and `failureReason` stay here: they are about a benchmark run.
- `internal/benchmark/runner.go` — `serverErrorLines(...)` becomes `serverlog.ErrorLines(...)`.
- `internal/benchmark/failure_log_test.go` — keep the tests for what stays; the moved ones go to `serverlog`.
- `internal/memreport/collector.go` — replace `spawnRE` and `readyRE` with `serverlog.ParseSpawn` and `serverlog.ReadyPort`.

## Steps

1. Create `internal/serverlog` with a package comment: it reads llama-server and router log lines, imports nothing else from this project, and is the one place that knows their shape.
2. Move, with their comments, and export:
   - `LineRE` (was `serverLogRE`) and `InstancePrefixRE` (was `instancePrefixRE`).
   - `ErrorLines(log []string) []string` (was `serverErrorLines`), with `rawFailureWords`, `hasFailureWord` and `stackFrameRE` as unexported helpers. `MaxErrorLines = 12` replaces `maxFailureLogLines` here; `benchmark` uses the exported constant where it referred to the old one.
3. Add, moving the expressions from `memreport/collector.go` unchanged:
   - `ParseSpawn(line string) (name, port string, ok bool)` for `spawning server instance with name=… on port …`.
   - `ReadyPort(line string) (port string, ok bool)` for the child's `cmd_child_to_router:state` ready line.
   - `InstancePort(line string) (port string, ok bool)`: the port in a `[ port] ` prefix, spaces trimmed. Phase 03 uses it to give each instance its own tail of lines.
4. Update `benchmark` and `memreport` to call the new functions. Remove the now-unused imports.
5. Move the tests as listed. Keep their recorded log fixtures byte for byte.

## Build gate

- `make build`
- `go vet ./...`
- `go test ./...`

## Test plan

- The moved `ErrorLines` tests (OOM log, repeated warm-up, CUDA crash log with backtrace frames) pass unchanged in `serverlog`.
- New table tests for `ParseSpawn`, `ReadyPort` and `InstancePort`, including a port below 10000 with the router's leading-space padding (`[ 8081]`).
- `internal/memreport` and `internal/benchmark` tests pass with no edits other than the moved cases.
- `grep -rn "serverErrorLines\|spawnRE\|readyRE" internal` finds nothing.

## Commit

`refactor(serverlog): share the llama-server log helpers between benchmark and memreport`

## Rollback

Revert the commit; both callers go back to their private copies. Nothing is stored or shown differently, so a revert is safe at any time before Phase 03 lands. After Phase 03, revert that first.
