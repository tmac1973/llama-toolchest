# Phase 03 — Multi-Project Builder

**Depends on:** Phase 02 (the feature switch decides whether the project
selector appears) · **Enables:** Phase 05 (backend instances need
`sd-server` / `ace-server` builds), Phase 10 (container images add `pnpm`)

## Goal

Make the builder build more than llama.cpp. A **project** describes:
- the repo
- the CMake flags it always uses
- how a GPU profile maps to that project's own CMake options
- which binaries a build must produce
- how its refs are ranked

llama.cpp is the first project, and it behaves exactly as today.
stable-diffusion.cpp and acestep.cpp are added as two more. The Builds tab gets
a project selector, shown only when generation backends are on.

The existing pieces are shared by all projects:
- GPU profiles and their option toggles
- ROCm and CUDA compiler discovery and ROCm install pinning
- the build log stream
- flag presets
- the build history in `builds.json`

## Files touched

- `internal/builder/project.go` (new): the `Project` type, the
  `DefaultProjects()` registry and `FindProject`.
- `internal/builder/builder.go`:
  - `BuildResult` gains `Project string` and `HasWebUI bool` fields.
  - `Build`, `runBuild`, `ensureRepo`, `checkoutRef`, `FetchRefs`, `CachedRefs`
    and `ReleaseAnchors` take a project.
  - Binary collection uses the project's binary lists.
  - Ranking goes through `Project.Rank`.
- `internal/builder/profiles.go`: add `ProjectOptions(project, profile)`,
  which returns the project-specific toggles (sd.cpp's frontend option).
- `internal/builder/project_test.go` (new). The existing builder tests are
  updated so they pass the llama.cpp project.
- `internal/api/build.go`:
  - `handleTriggerBuild` reads the form field `project`, defaulting to
    `llama.cpp`.
  - `handleListBuilds` filters on `?project=`.
  - `resolveActiveBuild` and `resolveBuild` only consider llama.cpp builds.
  - New: `resolveBackendBuild(kind, id)`.
- `internal/api/server.go`: `handleBuildsPage` passes the project list and
  `GenerationEnabled`.
- `web/templates/builds.html`, `web/templates/partials/build_card.html`,
  `build_options.html`, `git_ref_options.html`:
  - A project selector.
  - A project column in the build table.
  - Refs and options reload when the project changes.

## Steps

1. **Define `Project`:**

   | Field | Notes |
   |---|---|
   | `ID string` | `"llama.cpp"`, `"stable-diffusion.cpp"`, `"acestep.cpp"` |
   | `Repo string` | |
   | `CloneDir string` | Relative to the data dir: `llama.cpp`, `stable-diffusion.cpp`, `acestep.cpp` |
   | `Submodules bool` | |
   | `BaseCMakeArgs []string` | |
   | `MapProfileFlags func(map[string]string) map[string]string` | |
   | `RequiredBinaries []BinarySpec` | |
   | `OptionalBinaries []BinarySpec` | |
   | `Rank func(BuildResult) int` | |
   | `Kind string` | `""` for llama.cpp, `"sd"`, `"ace"` |

   `Kind` is a plain string that holds the same values as
   `generation.BackendKind`. That way `builder` never imports `generation`, and
   `generation` (Phase 05) can import `builder` without an import cycle.

   `BinarySpec` is `{Name string; Candidates []string}`, with candidate paths
   relative to the build dir.

2. **Register the three projects in `DefaultProjects()`.**

   **llama.cpp:**
   - Repo `https://github.com/ggml-org/llama.cpp`; this replaces the
     `llamaCppRepo` variable.
   - Base args are today's
     `-DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON`.
   - The flag map is the identity.
   - Required: `llama-server` (`bin/llama-server`, `bin/server`).
   - Optional: `llama-bench`, plus `llama-perplexity` through
     `llamaPerplexityCandidates`.
   - Rank: today's `buildRank` (CommitCount, then `bN`).

   **stable-diffusion.cpp:**
   - Repo `https://github.com/leejet/stable-diffusion.cpp`, with submodules.
   - Base args: `-DSD_BUILD_EXAMPLES=ON`.
   - Flag map:

     | Profile flag | sd.cpp flag |
     |---|---|
     | `GGML_HIP=ON` | `SD_HIPBLAS=ON` |
     | `GGML_CUDA=ON` | `SD_CUDA=ON` |
     | `GGML_VULKAN=ON` | `SD_VULKAN=ON` |
     | `GPU_TARGETS` / `AMDGPU_TARGETS` | passed through |
     | all other `GGML_*` toggles | passed through |

   - Also add `-DCMAKE_POSITION_INDEPENDENT_CODE=ON` and
     `-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON`, which are the flags that worked on
     asmodean.
   - Required: `sd-server` (`bin/sd-server`). Optional: `sd-cli`.
   - Rank: parse `master-<N>-<sha>` from `git describe --tags`. If that fails,
     use CommitCount.

   **acestep.cpp:**
   - Repo `https://github.com/ServeurpersoCom/acestep.cpp`, with submodules.
   - No base args.
   - The flag map is the identity: it uses `GGML_*` flags directly.
   - Required: `ace-server` (`ace-server`, `bin/ace-server`). Optional:
     `ace-lm`, `ace-synth`.
   - Rank: CommitCount.

3. **Add the sd.cpp web UI option.**
   - `ProjectOptions` adds a `SD_SERVER_BUILD_FRONTEND` toggle for
     stable-diffusion.cpp, labelled "Build the built-in web UI (needs pnpm)".
   - It defaults to ON when `exec.LookPath("pnpm")` finds pnpm, and OFF
     otherwise.
   - After the build, record `BuildResult.HasWebUI` as true when the option was
     on and the build succeeded.
   - acestep.cpp always has a web UI, because its `index.html.gz` is committed
     upstream, so its builds record `HasWebUI = true`.

4. **Thread the project through `Build`.**
   - The build ID becomes `<project-prefix>-<ref>-<profile>[-tag|-hash6]`. The
     prefix is empty for llama.cpp, so existing IDs keep their form. It is `sd`
     for stable-diffusion.cpp and `ace` for acestep.cpp.
   - Output stays at `<data>/builds/<ID>/`.
   - The temp dir is `<CloneDir>/build-<profile>`.

5. **Keep the existing container rule.** Vulkan stays refused in containers
   (`builder.go` L315) for every project. Use the same error message, with the
   project name added.

   **Profiles for the new projects.** stable-diffusion.cpp and acestep.cpp
   offer only the `rocm`, `cuda` and `vulkan` profiles. The project selector
   hides `cpu` and `metal` for them, and `Build` refuses those combinations.
   Generation backends need a GPU, which matches the overview's GPU backend
   decision.

6. **Handle old build records.** `loadBuilds` treats records with no `Project`
   as `llama.cpp`, and `saveBuilds` writes the field from then on.

7. **Keep the Server page and autotune on llama.cpp builds.**
   `resolveActiveBuild`, `serverBuildChoices` and the router start only look at
   llama.cpp builds. `resolveBackendBuild(kind, id)` returns the named build,
   or the best-ranked successful build of that project.

8. **Show the selector only when the feature is on.** With
   `GenerationEnabled` false the template does not render the project selector
   and the page is identical to today. With it on, the selector sits above the
   profile picker. Changing it reloads refs (`/api/builds/refs?project=…`) and
   options.

## Build gate

- `go build ./...`
- `go vet ./...`
- `go test ./internal/builder/... ./internal/api/...`

## Test plan

- **`project_test.go`:**
  - sd.cpp flag mapping turns `GGML_HIP=ON` into `SD_HIPBLAS=ON` and keeps
    `GPU_TARGETS`.
  - Binary candidate lookup finds `ace-server` at the build root.
  - Rank parses `master-1000-95ff762` as 1000.
  - An old `builds.json` without a `project` field loads as llama.cpp.
- **Existing tests:** builder and `build_render_test.go` pass with the
  llama.cpp project.
- **Manual on asmodean (ROCm, host mode):**
  - Build sd.cpp `master` and acestep.cpp `master` from the Builds tab with the
    `rocm` profile.
  - Expected build times are about 3 min and 2 min, as measured on 2026-10-24.
  - Confirm `sd-server` and `ace-server` are recorded.
  - Confirm the llama.cpp Server page still picks a llama.cpp build.
- **Manual on compute2 (CUDA):** build both projects with the `cuda` profile.
- **Manual on asmodean, Vulkan:** build both projects with the `vulkan` profile
  in host mode.

## Commit

`feat(builder): build stable-diffusion.cpp and acestep.cpp alongside llama.cpp`

## Rollback

Revert the commit. If `builds.json` already contains sd/ace records, a reverted
binary loads them as llama.cpp builds whose `llama-server` binary is missing,
and they show as failed. Delete them from the Builds tab, or delete their
entries from `builds.json`, before rolling back.
