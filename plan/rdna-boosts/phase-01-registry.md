# Phase 01 — Patch set registry and build fields

**Depends on:** none · **Enables:** Phase 02 (the builder applies a registry entry), Phase 03 (the UI lists the entries)

## Goal

A new package `internal/patchsets` holds the curated patch sets. rdna-boosts is its only entry. `BuildResult` gets the fields that say a build is patched and with which release. Nothing builds or shows patch sets yet.

## Files touched

- `internal/patchsets/patchsets.go` — new: `PatchSet`, `Release`, `Curated()`, `Find(id)`.
- `internal/patchsets/patchsets_test.go` — new.
- `internal/builder/builder.go` — new `BuildResult` fields.
- `internal/builder/builder_test.go` — JSON round trip of the new fields.

## Steps

1. In `internal/patchsets`, define:

   ```go
   // PatchSet is one curated set of patches on upstream llama.cpp.
   type PatchSet struct {
       ID       string   // "rdna-boosts"; also the build ID prefix's source ("rdna-")
       Name     string   // "RDNA boosts"
       Repo     string   // https://github.com/stew675/llama-cpp-rdna-boosts
       Pinned   Release  // the tested release
       Backends []string // ["rocm"]
       ROCmArchs []string // gfx1030 gfx1100 gfx1150 gfx1151 gfx1200 gfx1201
       License  string
       Notes    string   // one or two sentences, shown as a tooltip
   }

   // Release is one release of a patch set, as its release.json records it.
   type Release struct {
       Tag     string // "v16-6050b467c-r206"
       Base    string // upstream commit the patches apply to, full SHA
       Tree    string // git tree of the patched source
       NBlocks int    // number of patch files
       Date    string // YYYY-MM-DD the release was pinned (Pinned only)
   }
   ```

   The package imports nothing from the toolchest, so the builder and the API can both import it.
2. Add the rdna-boosts entry with the pinned release `v16-6050b467c-r206`:
   - base `6050b467c07a95c7b87debd4ddac2f8ba8d9b807` (upstream `b11887`);
   - tree `1bac1001f1780a5907fbce256ceeeaac8b8561ec`;
   - 16 blocks.

   Copy the values from that tag's `release.json`, not from the branch. Short build ID name: `rdna`.
3. Add a `ReleaseNumber(tag string) (int, bool)` helper that reads N from `v<blocks>-<base>-r<N>`. Build IDs and badges use it ("r206").
4. In `BuildResult`, add:
   - `PatchSet string` (`json:"patch_set,omitempty"`): the patch set ID, or empty for unpatched source.
   - `PatchRelease string` (`json:"patch_release,omitempty"`): the release tag.
   - `PatchBase string` (`json:"patch_base,omitempty"`): the upstream base SHA. `GitSHA` stays the SHA of the patched HEAD.

   Comment each field as the existing fields are commented. Builds from before these fields load with them empty, which means "not patched".

## Build gate

- `make build`
- `go vet ./...`
- `go test ./...`

## Test plan

- `patchsets_test.go`: `Find("rdna-boosts")` returns the entry; every curated entry has a 40-character `Base`, a 40-character `Tree`, `NBlocks > 0`, and a tag that `ReleaseNumber` parses; IDs are unique.
- `builder_test.go`: a `BuildResult` with the three fields set survives a save and load of `builds.json`; an old `builds.json` without them loads with them empty.

## Commit

`feat(patchsets): add the curated patch set registry with rdna-boosts`

## Rollback

Revert the commit. Nothing reads the new fields yet.
