# Phase 01 — Clear start errors and safe catalog deletes

**Depends on:** nothing · **Enables:** Phase 08 (field test of the error message); fixes two gaps from the 2026-11-01 test on its own

## Goal

Two small fixes from the custom set test that need nothing else from this plan. First: when a generation instance fails to start or measure, the error shows the backend's last error line, for example `get sd version from file failed`, not only `exit status 1`. This helps catalog and custom sets alike. Second: deleting a catalog set keeps every file that another set uses, also when that set reaches the file by a different path (a symlink, `..`, or a doubled `/`).

## Files touched

- `internal/generation/manager.go` — `WaitRunning` adds the last error line from the instance's log to a start failure.
- `internal/generation/logline.go` (new) — `LastErrorLine(lines []string) string` and `withLastLogLine(err error, lines []string) error`.
- `internal/generation/logline_test.go` (new) — tests for both functions.
- `internal/api/gen_models.go` — `handleGenDeleteSet` compares files by identity, not by path text.
- `internal/api/gen_models_test.go` — a test for the delete through a symlink.

## Steps

1. Add `LastErrorLine(lines []string) string` in `internal/generation/logline.go`. It walks `lines` from the end, over at most the last 50 lines. It skips empty lines and toolchest's own lines (they start with `==>`). It returns the first line, from the end, that contains `[E]` or the word `error` (case-insensitive). If none matches, it returns the last line it did not skip. It trims the line, removes a leading `data: `, removes sd.cpp's ` --- file.cpp:NNN` suffix, and cuts the result to 200 characters.
2. Add `withLastLogLine(err error, lines []string) error` in the same file. It returns `err` when `err` is nil or `LastErrorLine(lines)` is empty; otherwise `fmt.Errorf("%w. Last log line: %s", err, line)`. Keeping `%w` lets callers that test the error still work.
3. In `Manager.WaitRunning` (`internal/generation/manager.go`), when `s.WaitRunning(ctx)` returns an error and the supervisor state is `StateFailed`, return `withLastLogLine(err, s.RecentLogs())`.
4. No change is needed in the callers: Measure (`internal/api/gen_vram.go`: "the instance did not start: %w") and start (`internal/api/gen_evict.go`: "backend %s did not start: %w") already wrap the error with its full text.
5. In `handleGenDeleteSet` (`internal/api/gen_models.go`), replace the `shared` test. Before the loop, `os.Stat` every component path of every other set and keep the `os.FileInfo` values (skip paths that fail). A file is shared when `os.Stat(path)` succeeds and `os.SameFile` matches one of them, or when its cleaned path (`filepath.Clean`) equals a cleaned path of another set. Keep the existing `huggingface.FilePath(root, o.Repo, o.File)` comparison for catalog sets whose files are not yet downloaded.

## Build gate

- `go build ./...`
- `go vet ./...`
- `go test ./internal/generation/... ./internal/api/...`

## Test plan

- `LastErrorLine`: a log with `[E] get sd version from file failed: '' --- diffusion_engine.cpp:972` near the end returns `get sd version from file failed: ''` with the `[E]` kept. A log with only info lines returns the last one. A log whose last line is `==> sd-server x exited with error: exit status 1` skips that line. Empty input returns "".
- `withLastLogLine(errors.New("sd-server x failed to start: exit status 1"), []string{"[E] boom"})` gives an error that contains `exit status 1` and `Last log line: [E] boom`, and `errors.Is` finds the original error. With no log lines, the error is unchanged.
- The manual start test in Phase 08 (step 4.8) checks the full path with a real sd-server.
- Delete: a catalog set and a custom set; the custom set's path is a symlink to the catalog set's file, in another folder. Deleting the catalog set keeps the file. A second case with a `dir/../dir/file` path does the same. The existing test that a catalog set's own files are deleted still passes.

## Commit

`fix(gen): show the backend's last error line on a failed start, and keep files another set uses by any path`

## Rollback

Revert the commit. The two changes do not depend on each other or on stored data, so a partial revert of either one is safe.
