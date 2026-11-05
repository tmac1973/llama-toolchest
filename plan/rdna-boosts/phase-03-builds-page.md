# Phase 03 — Builds page and build API

**Depends on:** 02 · **Enables:** Phase 04 (users can make patched builds), Phase 06 (field test)

## Goal

The Builds page offers "llama.cpp + RDNA boosts" in its project picker, with ROCm profiles only and a release picker. The build POST turns that choice into a llama.cpp build with a patch spec. Build cards show which release a patched build carries.

## Files touched

- `internal/api/server.go` — the project picker gets one choice per curated patch set.
- `internal/api/build.go` — the ref list and the build POST for a patch set choice.
- `web/templates/builds.html` — labels and the tooltip for the patch set choice.
- `web/templates/partials/build_card.html` — the "RDNA boosts rN" badge and the "untested" badge.
- `web/templates/partials/build_info.html` — patch set, release and base in the build details.
- `internal/api/build_test.go`, `internal/api/server_test.go` (or the existing Builds page render test) — tests.

## Steps

1. **Picker choice.** In the Builds page handler (`server.go`, the `builder.Projects()` loop), add one `projectChoice` per `patchsets.Curated()` entry after llama.cpp:
   - `ID: "patchset:" + ps.ID`;
   - `Label: "llama.cpp + " + ps.Name`;
   - `Backends: ps.Backends`;
   - `Notes: ps.Notes`;
   - `NotesURL: ps.Repo + "/releases/tag/{ref}"`.

   It shows when the open question in the overview is settled: always, or while forks are on. With the patch set choice, the picker always has at least two entries, so it always shows.
2. **Release list.** In the ref handler (`build.go`, the `refresh` handler around line 180), a `patchset:` project returns the `git_ref_options` partial with:
   - the pinned tag, labelled "r206 (tested)" and selected;
   - `latest`, labelled "Newest (untested)";
   - the ten most recent releases other than the pinned one, labelled "r220 · b11887", in a "Recent releases (untested)" group.

   The recent releases are read from the patch set's clone (`Builder.PatchSetReleases`), newest base first, then highest number. The refresh button fetches the set's tags first. Before the first patched build there is no clone, so the picker shows only the first two choices and a note. The builder resolves `latest` at build time (Phase 02).
3. **Build POST.** When `req.Project` starts with `patchset:`, look up the patch set (404 if unknown). Accept a release tag (`v<blocks>-<base>-r<N>`) or `latest` as the ref; the builder checks that the release exists. Then call the builder with project llama.cpp and a `PatchSpec`. Keep the existing ROCm profile and option handling. The ROCm install picker and the option checkboxes apply as for llama.cpp.
4. **Build card.** For a build with `PatchSet` set, show a badge "<Name> r<N>". Its tooltip is the release tag and "Base: <short base> (b<CommitCount>)". When `PatchRelease` is not the patch set's pinned tag, also show the "untested" badge that fork builds use, with the same tooltip text.
5. **Build details.** In `build_info.html`, show "Patch set", "Release" (linked to the release page) and "Upstream base" rows for patched builds.
6. **Picker layout.** Keep the radio buttons for now. With five choices (llama.cpp, RDNA boosts, two generation projects, Avalon) the row is full. A later change must replace the radios before more build sources are added (e.g. a grouped drop-down: llama.cpp variants, generation backends, forks). That change is not part of this plan.
7. **Text.** All new labels and tooltips follow the writing rules. The picker tooltip: "Upstream llama.cpp with the RDNA boosts patches. Faster on AMD RDNA GPUs. ROCm only."

## Build gate

- `make build`
- `go vet ./...`
- `go test ./...`
- `make js-test`

## Test plan

- Builds page render test: the picker has "llama.cpp + RDNA boosts" with backends `rocm`.
- `build_test.go`:
  - The ref handler for `patchset:rdna-boosts` lists the pinned tag first, then `latest`.
  - The build POST with `patchset:rdna-boosts`, the ROCm profile and the pinned tag calls the builder with project llama.cpp and the right `PatchSpec` (use the builder test double the other build tests use).
  - A CUDA profile is refused. Another ref is refused. An unknown patch set is 404.
- Build card render test: a patched `BuildResult` shows the badge, and one with a non-pinned release also shows "untested".
- Manual on asmodean: build r206 from the page; the card shows "RDNA boosts r206"; the details show base `6050b467c` and `b11887`.

## Commit

`feat(builds): build llama.cpp with RDNA boosts from the Builds page`

## Rollback

Revert the commit. Patched builds stay usable as llama.cpp builds. They lose their badge until the commit is back.
