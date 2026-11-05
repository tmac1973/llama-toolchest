# Phase 02 — Apply and check a patch set in the builder

**Depends on:** 01 · **Enables:** Phase 03 (the API calls the new builder entry point), Phase 04 (patched builds exist)

## Goal

`BuildProject` can build llama.cpp with a patch set release applied. It checks out the release's base, applies the patches with strict `git am`, checks the tree, and records the base's commit count as the build's version. If any step fails, the build fails before cmake runs.

## Files touched

- `internal/builder/builder.go` — a patch set argument on the build path; the version and ID rules for patched builds.
- `internal/builder/patchset.go` — new: fetch the patch repo, read a release, apply it, check the tree.
- `internal/builder/patchset_test.go` — new, with a local fixture repo pair.
- `internal/builder/archs.go` — no change expected; confirm `BackfillArchs` can read a patched build's `GitSHA` (step 6).

## Steps

1. **Entry point.** Add a `PatchSpec` (patch set ID and release tag, where an empty tag means the pinned release and `"latest"` means the newest) to the build options. Change `BuildProject` so it takes it with the other build options, without adding a tenth positional argument. Refuse a patch spec for any project but llama.cpp, and for a profile backend the patch set does not list.
2. **Patch repo clone.** Keep a clone per patch set at `<data>/src/patchsets/<id>` (next to the existing clones), managed like the project clones in `ensureRepo`: clone when missing, `git fetch --tags` when present, discard an incomplete one. The repo is small, but its `archive/` and `benchmarks/` folders hold history, so clone with `--filter=blob:none`.
3. **Read the release.** For `"latest"`, take the newest tag that matches `^v\d+-[0-9a-f]+-r\d+$`, ordered by its release number. Read `release.json` at that tag with `git show <tag>:release.json` and parse `base`, `tree` and `n_blocks`. For the pinned release, the values from the registry must equal the ones in its `release.json`. If they differ, fail with "The pinned release does not match its release.json." List the patch files with `git ls-tree --name-only <tag> patches/`, keep `NNNN-*.patch`, sort them, and check that the count equals `n_blocks`.
4. **Check out the base.** `release.json.base` may be abbreviated. Resolve it in the llama.cpp clone (`git rev-parse --verify <base>^{commit}`) after the usual fetch. Then check out that commit with the existing forced checkout, on a branch `patchset/<id>/<tag>` created with `-B`. The branch keeps the patched commits reachable after the clone moves to another ref, so later `git show <GitSHA>:…` calls (arch backfill) still work.
5. **Apply.** Write the patch files from the patch repo into a temporary directory (`git show <tag>:patches/<file>`). Then run `git am --committer-date-is-author-date <files…>` in the llama.cpp clone with this environment:
   - `GIT_COMMITTER_NAME` and `GIT_COMMITTER_EMAIL` taken from the first patch's `From:` header, as the patch repo's `apply-all.sh` does;
   - `GIT_CONFIG_NOSYSTEM=1` and `HOME` set to the temporary directory, so the host's git config cannot change the result.

   With these values the patched commit IDs match the release's `tip`. On failure, run `git am --abort` and fail with the name of the patch that did not apply. Do not retry with `-3`.
6. **Check the tree.** Compare `git rev-parse HEAD^{tree}` with `release.json.tree`. If they differ, fail with "The patched source does not match release <tag>." Log the tree hash on success.
7. **Version.** Set `CommitCount` from `git rev-list --count <base>`, not from `HEAD`. Set `GitRef` to the release tag, `GitSHA` to the patched HEAD, `PatchSet`, `PatchRelease` and `PatchBase` from the release. `buildRank` then ranks the build at the base, and `buildTarget` writes options for the base.
8. **Build ID.** For a patched build, the ID is `<short name>-b<base count>-r<N>-<profile>` (e.g. `rdna-b11887-r206-rocm`); a tag still adds `-<tag>`. The existing same-ID check and the flag-hash suffix work unchanged. A rebuild of the same release and profile gives `DuplicateBuildError`.
9. **Arch lists.** The `ProjectLlamaCpp` case at the end of `runBuild` reads `llama-arch.cpp` from the checkout, so patched builds get their own arch lists. Check that `BackfillArchs` reads `GitSHA` (the patched HEAD), which step 4's branch keeps reachable.
10. **Logs.** Log each phase with the existing `==>` lines: "Patch set: RDNA boosts v16-6050b467c-r206", "Base: 6050b467c (b11887)", one line per applied patch, "Tree matches release …".

## Build gate

- `make build`
- `go vet ./...`
- `go test ./...`

## Test plan

- `patchset_test.go` with a fixture: a local "upstream" repo with a few commits, and a local "patch repo" with two release tags. Each release has `release.json` and two `git format-patch` files made from the fixture.
  - The pinned release applies; the tree matches; `CommitCount` is the base's count, not base + 2; `GitRef` is the tag; the ID is `<short>-r<N>-<profile>`.
  - `"latest"` picks the higher release number, not the newer tag date.
  - A `release.json` with a wrong tree fails at the tree check, and no binary is installed.
  - A patch that does not apply fails with its file name. The clone is clean afterwards: the next plain llama.cpp build in the same clone succeeds.
  - A patch count that differs from `n_blocks` fails before `git am`.
  - Applying twice in a row with different committer config in `HOME` gives the same HEAD SHA.
  - After building an unpatched ref in the same clone, `git show <patched GitSHA>:src/llama-arch.cpp` still works.
  - A patch spec with project `sd` or a CUDA profile is refused.
- Manual on asmodean (container, ROCm 10.1): a pinned build's log shows the tree match, and `git rev-parse HEAD` in the clone equals the release's `tip` (`efaa60db0…` for r206).

## Commit

`feat(builder): build llama.cpp with a curated patch set release applied`

## Rollback

Revert the commit. Patched builds already in `builds.json` keep working as llama.cpp builds: they have an ID, a binary and a version. The `patchset/…` branches and the patch repo clone stay on disk and can be removed by hand.
