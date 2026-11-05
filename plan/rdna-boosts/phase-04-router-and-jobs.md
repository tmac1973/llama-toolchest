# Phase 04 — Router, jobs and build pickers

**Depends on:** 02 (patched builds exist), 03 (for the manual checks) · **Enables:** Phase 06 (field test)

## Goal

A patched build works everywhere an upstream llama.cpp build works: as the router's build, and as the build for benchmark, Autotune and Autoconfigure jobs. Its launch options are written for its base version. "Auto" does not choose it on its own. Every build picker shows that it is patched.

Most of this already works, because a patched build is recorded under the llama.cpp project. This phase adds the "Auto" rule and the picker labels, and adds tests that hold the rest in place.

## Files touched

- `internal/api/build.go` — `resolveBuild`'s "Auto" fallback.
- `internal/api/build.go` and the router build picker partial — labels.
- `web/templates/benchmarks.html` and the job build pickers — labels.
- `internal/api/build_test.go`, `internal/api/jobs_env_test.go`, `internal/models/launch_test.go` — tests.

## Steps

1. **Auto.** In `resolveBuild`, when no build is saved, walk `SuccessfulBuildsRanked()` twice: first skipping builds with `PatchSet` set, then (only if nothing was found) including them. Keep the existing "runs in this image" filter in both passes. Update the function's comment.
2. **Explicit choice.** `resolveBuild(id)` already accepts any successful llama.cpp build. Add a test that a saved patched build ID is returned as it is.
3. **Launch options.** `buildTarget` uses `Rank()`, which is `CommitCount`, which Phase 02 sets to the base count. Add a test with a patched `BuildResult` at `CommitCount: 11887`: `RouterMemoryOption()` is false, and every option gated at or below 11887 is on.
4. **Jobs.** `EnsureBuildActive` and `CheckBuildRunnable` accept any llama.cpp build; `refuseForkJobBuild` checks for fork projects only. Add a test that a patched build passes both and is not refused.
5. **Labels.** Wherever a build is offered as a choice (the router build picker on the Server page, the benchmark and Autotune build pickers, the Autoconfigure build picker), add " · <Name> r<N>" to a patched build's label. Use one helper, `buildLabel(b)`, so the text is the same everywhere. Results and history rows that name a build use the same helper.
6. **Engine notes.** Crash and usage notes are keyed by build ID, so patched and upstream builds keep separate notes. No change; confirm with a test that a patched build's notes do not appear on the upstream build of the same base.

## Build gate

- `make build`
- `go vet ./...`
- `go test ./...`
- `make js-test`

## Test plan

- `build_test.go`:
  - With an upstream b11880 build and a patched b11887 build and nothing saved, "Auto" returns the upstream build.
  - With only the patched build, "Auto" returns it.
  - A saved patched build ID is returned.
- `launch_test.go`: a target built from a patched build at 11887 writes the same options as an upstream 11887 build.
- `jobs_env_test.go`: a patched build passes `CheckBuildRunnable` and `refuseForkJobBuild`.
- Render tests: the router picker and the benchmark picker show "RDNA boosts r206" on the patched build only.
- Manual on asmodean:
  - Select `rdna-b11887-r206-rocm` as the router build. The router starts and serves a model. Its launch command has no option newer than b11887.
  - Run one benchmark job and one Autoconfigure job on that build. Both finish and name the build with its label.

## Commit

`feat(router): run patched llama.cpp builds as the router and in jobs`

## Rollback

Revert the commit. "Auto" can then choose a patched build again when it ranks highest, and picker labels lose the patch set suffix. Saved choices are not changed.
