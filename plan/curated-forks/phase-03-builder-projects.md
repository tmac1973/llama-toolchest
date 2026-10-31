# Phase 03 — Fork builds on the Builds page

**Depends on:** 01 (the setting), 02 (the curated list) · **Enables:** Phase 05 (instances need a fork build to run)

## Goal

Each curated fork becomes a builder project. With the setting on, the Builds page offers "Avalon" in its project picker, with two refs: "Pinned (tested)" and "Latest main (untested)". Fork builds get the `avalon-` prefix, are stored like other builds, and show an "untested" mark when their commit is not the pinned one. The router, benchmark jobs and generation instances never choose a fork build.

## Files touched

- `internal/builder/project.go` — append one `Project` per `forks.Curated()` entry in `projects()`; a new `ForkKind = "fork"` constant; `buildIDRef` shortens a full SHA ref.
- `internal/builder/builder.go` — no logic change expected; check that `SuccessfulBuildsRanked()` and `buildRank` stay llama.cpp-only (they filter by `ProjectID()`).
- `internal/api/server.go` — the Builds page project list: generation projects when generation is on, fork projects when forks are on, llama.cpp always.
- `internal/api/build.go` — the build form accepts a fork project only while forks are on; `resolveBuild` and `resolveBackendBuild` skip `Kind == "fork"`; a new `resolveForkBuild(forkID, buildID string) (builder.BuildResult, error)`.
- `internal/api/jobs_env.go` — `EnsureBuildActive` and `CheckBuildRunnable` refuse a fork build ID, so no benchmark, Autotune or Autoconfigure job can run one.
- `internal/builder/archs.go` — `BackfillArchs` only reads llama.cpp builds.
- `web/templates/builds.html` — ref picker labels for fork projects.
- `web/templates/partials/build_card.html` — "untested" badge.
- `internal/builder/project_test.go`, `internal/api/build_test.go` — tests.

## Steps

1. In `projects()`, after `ace`, loop over `forks.Curated()` and append:
   - `ID: f.ID`, `Kind: ForkKind`, `Repo: f.Repo`, `CloneDir: f.ID`, `IDPrefix: f.ID + "-"`.
   - `BaseCMakeArgs` and `Required`/`Optional` copied from the llama.cpp project (build `llama-server`; `llama-bench` optional).
   - `Backends: f.Backends`.
   - `ReleaseNotesURL: f.Repo + "/commit/{ref}"`, `LatestNotesURL: f.Repo + "/commits"`.
   - `latestRef: defaultBranchRef` (origin/main), and `listRefs` returning `[]string{f.PinnedSHA}`.
   `internal/builder` imports `internal/forks`; `internal/forks` does not import the builder.
2. `buildIDRef`: when `ref` is a 40-character hex SHA, use its first 7 characters. So the pinned Avalon build on ROCm is `avalon-ad2c6e7-rocm`, and a latest-main build is `avalon-main-<sha7>-rocm`.
3. In the Builds page handler (`server.go` around line 1027), replace the `if cfg.GenerationEnabled { for … builder.Projects() }` loop with one loop over `builder.Projects()` that adds a project when its `Kind` is "" (llama.cpp), or is `sd`/`ace` and generation is on, or is `fork` and forks are on. Show the picker when there is more than one project.
4. In `builds.html`, when the chosen project is a fork, the ref picker shows two choices: the pinned SHA labelled "Pinned (tested)" and selected by default, and `latest` labelled "Latest main (untested)". The free-text ref field is hidden for fork projects. The project's tooltip uses `Fork.Notes`.
5. In `build.go`, the build POST refuses a fork project with 404 while `ForksEnabled` is off, and refuses any ref other than the pinned SHA or `latest`.
6. `resolveBuild` (router) and `resolveBackendBuild` (generation) already filter by project or kind. Add a test that proves a fork build is never returned by either, even when it is the newest build.
7. `resolveForkBuild(forkID, buildID)`: when `buildID` is set, return it if it succeeded and its project is `forkID`; else return the newest successful build of that project. Prefer a build whose `GitSHA` equals the pinned SHA when two builds have the same rank.
8. Job builds: in `jobs_env.go`, `EnsureBuildActive` and `CheckBuildRunnable` return "Fork builds run as fork instances and cannot run benchmark jobs." for a build whose project kind is `fork`. The job build pickers list builds from `SuccessfulBuildsRanked()`, which is llama.cpp-only; a test confirms a fork build is not in it.
9. `build_card.html`: for a fork build whose `GitSHA` differs from its fork's `PinnedSHA`, show an "untested" badge with tooltip "This build is not the tested commit. It can fail to start or give wrong output."
10. Architecture lists: the build step at `builder.go:862` only reads `llama-arch.cpp` for `ProjectLlamaCpp`, so fork builds get none. `BackfillArchs` (`archs.go:162`) does not filter by project and would run `git show` in the llama.cpp clone for every fork build at each start; add `br.ProjectID() == ProjectLlamaCpp` to its filter.

## Build gate

- `make build`
- `go vet ./...`
- `go test ./...`
- `make js-test`

## Test plan

- `project_test.go`: `Projects()` includes `avalon` with `Kind "fork"`, prefix `avalon-` and backends rocm and cuda; `FindProject("avalon")` finds it; `ProjectForKind("sd")` still returns sd; `buildIDRef` shortens a 40-character SHA and leaves `b9000` and `origin/main` as before.
- `build_test.go`: with forks off, a build POST for `avalon` returns 404; with forks on, a ref other than pinned or `latest` is refused; `resolveBuild` and `resolveBackendBuild` skip a newer fork build; `resolveForkBuild` honours a set build ID and otherwise returns the newest; a job with a fork build ID is refused.
- Manual on asmodean (container, `--rocm-image 10.1.0-full`): with forks on, the picker shows Avalon; a pinned ROCm build succeeds and lands as `avalon-<sha7>-rocm` without an "untested" badge; the router keeps its active llama.cpp build after a restart.

## Commit

`feat(forks): build curated forks from the Builds page`

## Rollback

Revert the commit. Fork builds already on disk stay in `builds.json` with `Project: "avalon"`. The builds list (`handleListBuilds`) does not look projects up, so they still show and can be removed with the existing delete action; nothing selects them, because every resolver filters by project.
