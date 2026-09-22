# Phase 04 — Build environment stamp

**Depends on:** nothing · **Enables:** Phase 05 (the mismatch flag can be
verified against two real ROCm lines)

## Goal

Record, on every new build, the backend and version it was compiled against, and
flag builds on the Builds page whose stamp does not match the container that is
running now. A `llama-server` linked against ROCm 7.2.4's `libhipblas` will not
resolve under a ROCm 10 userspace, and today that surfaces as an opaque load
failure. This phase makes it legible before the user tries.

This phase is independent of the container work and can be implemented at any
point, including before Phase 02 — it is placed here because it is only
*observable* once two ROCm lines exist to switch between. Builds made before this
change carry no stamp and are shown as unknown, never as mismatched: guessing
would risk flagging a build that works perfectly.

## Files touched

- `internal/builder/builder.go` — one new `BuildResult` field, populated where
  the result is constructed.
- `internal/builder/buildenv.go` — new; detection of the backend version, for
  both stamping a build and reporting what is running now.
- `internal/builder/buildenv_test.go` — new; parsing and comparison tests.
- `internal/api/build.go` — a view type carrying the stamp and the mismatch
  verdict; the builds table gains a column; the info modal gains a row.
- `web/templates/partials/build_card.html` — render the stamp and the flag.
- `internal/api/build_stamp_test.go` — new; render tests for the three states.

## Steps

1. **The field.** Add to `BuildResult` (`internal/builder/builder.go:37`):
   ```go
   // BuiltAgainst is the GPU toolchain this build was compiled against,
   // as "<backend> <version>" — e.g. "rocm 10.0.0", read from the ROCm
   // release version file rather than from hipconfig, whose number is
   // HIP's own and reads as 7.x even on ROCm 10. A llama-server
   // linked against one ROCm line does not run under another, and the
   // build directory outlives the container image, so this is what lets
   // the Builds page say which builds need rebuilding. Empty on builds
   // from before the field existed, and on backends with no version to
   // read; empty means unknown and is never reported as a mismatch.
   BuiltAgainst string `json:"built_against,omitempty"`
   ```
   `omitempty` plus "empty means unknown" mirrors the `CommitCount` precedent
   directly above it, so `builds.json` written by an older version still loads.
2. **Detection.** Create `internal/builder/buildenv.go` with:
   - `func backendVersion(backend string, rocmInfoPaths []string) string` — the
     testable core. For `rocm`, read each path in order and use the first that
     yields a non-empty value, trimming whitespace and taking the first
     whitespace-separated field so a longer line cannot become the version. The
     production order, both confirmed by Phase 01, is:
     1. `/opt/rocm/.info/version` — the Fedora/RPM layout the stable image uses,
        which reports `7.2.4` there.
     2. `/opt/rocm/core/.info/version` — the TheRock layout the AMD Ubuntu images
        use, which reports `10.0.0`. `/opt/rocm/core` is a symlink through
        `/etc/alternatives/core`, so this path names no version and keeps working
        across releases.
     Do **not** fall back to `hipconfig --version`. Phase 01 found it reports
     `7.15.26333-0000000` in the ROCm 10.0.0 image — that is HIP's own component
     version, not the ROCm release, and stamping it would label a ROCm 10 build
     as 7.x and make the mismatch comparison wrong in the most confusing way
     available. If neither file is readable the answer is unknown, which the rest
     of the design already handles.
     For `cuda`, run `nvcc --version` and extract the `release X.Y` number. For
     anything else, return `""`. Any error returns `""` — an unreadable toolchain
     must not fail a build.
   - `func BuildEnvStamp(backend string) string` — calls `backendVersion` with
     the real path list and returns `""` or `backend + " " + version`.
   - `func CurrentBuildEnv(backend string) string` — the same value for the
     running process, memoised per backend (a `sync.Mutex`-guarded map, since
     `sync.Once` cannot be keyed), because neither version file can change while
     the process lives.
   - `func StampMismatch(buildStamp, currentStamp string) bool` — true only when
     both are non-empty, their backend words are equal, and their version words
     differ. Different backends return false: a CUDA build listed on a ROCm host
     is a separate situation, out of scope here, and flagging it would be noise.
3. **Populate.** In `Builder.Build` (`internal/builder/builder.go:252`), at the
   `result := &BuildResult{…}` literal, add
   `BuiltAgainst: BuildEnvStamp(prof.Backend)`. `prof` is already in scope from
   `FindProfile`. Nothing else in the build path changes, and a failed build keeps
   its stamp — knowing what a failure was compiled against is useful.
4. **The view type.** In `internal/api/build.go`, add:
   ```go
   // buildRow is a build plus what the page needs to say about it. The
   // BuildResult is embedded so the template's existing field references
   // keep working.
   type buildRow struct {
       *builder.BuildResult
       BuiltAgainstText  string // the stamp, or "—" when unstamped
       Mismatch          bool
       BuiltAgainstTitle string // the cell's tooltip; "" only when stamped and matching
   }
   ```
   One text field and one title field cover all three states, so the template has
   no conditional wording of its own: `BuiltAgainstText` is the stamp or an
   em-dash, and `BuiltAgainstTitle` is the mismatch explanation when flagged, the
   not-recorded sentence when unstamped, and empty when stamped and matching.
   Add `func (s *Server) buildRowFor(b *builder.BuildResult) buildRow` that fills
   it using `builder.StampMismatch` and `buildBackend`, which already exists at
   `internal/api/build.go:336` and resolves a build's backend through its profile.
   It reads the running environment through a function field on `Server` —
   `currentBuildEnv func(string) string`, left nil in production and treated as
   `builder.CurrentBuildEnv` when nil. That seam is what makes the render test in
   Step 9 possible: the test sets it to return a known stamp, so "matching" and
   "differing" are decided by the test rather than by whatever ROCm happens to be
   installed on the machine running `go test`. Without it the memoised real value
   would make the test pass or fail depending on the host.
5. **The tooltips.** Follow the project's convention that a flag explains how to
   read it. Define both strings once, in `buildRowFor`, and use them verbatim in
   the table and the info modal so the two cannot drift. The backend is named
   from the stamp rather than hard-coded, because
   `BuildEnvStamp` also stamps CUDA builds and `StampMismatch` fires for any
   backend whose versions differ — ROCm-specific wording on a CUDA row would be
   wrong. When flagged:
   `"Built against <stamp>; this container runs <current>. A llama-server built
   against one <backend> version does not run under another, so this build needs
   rebuilding before it will load. Nothing has been deleted — it is still here if
   you switch back."` with `<backend>` taken from the stamp's first word. When the
   stamp is empty, exactly this string, capital N included, everywhere it appears:
   `"Not recorded — this build predates build-environment tracking."`
6. **The table.** In `handleListBuilds` (`internal/api/build.go:190`), add a
   `<th>Built against</th>` to the inline header string between `Status` and
   `Date`, and render `s.buildRowFor(b)` instead of `b`. Keep the non-HTMX JSON
   branch returning the raw builds — the new field appears there through the
   struct tag, with no handler change.
7. **The card.** In `web/templates/partials/build_card.html`, add one `<td>`
   in the matching position rendering `.BuiltAgainstText`, and when `.Mismatch` a warning mark
   (`&#9888;`) carrying `title="{{.BuiltAgainstTitle}}"` and `cursor:help`, styled
   with `var(--pico-del-color)` as the job table's error text is. For an
   unstamped build `.BuiltAgainstText` is already the em-dash and
   `.BuiltAgainstTitle` already holds the not-recorded sentence, so the `title`
   attribute is the same expression in every state and the template needs no
   conditional of its own — matching how absent measurements read elsewhere in
   the app. Field references such as `.ID` and
   `.GitSHA` keep working through the embedded struct, so nothing else in the
   template changes.
8. **The info modal.** In `handleBuildInfo` (`internal/api/build.go:365`), add a
   `Built against` row to the `<dl>` after `Status`, rendering the same
   `BuiltAgainstText` and `BuiltAgainstTitle` the table uses. For an empty stamp
   that is the exact string from Step 5 — the same wording, capitalisation
   included, as the table tooltip. It reads as a near-twin of the existing
   cmake-flags fallback further down the handler ("cmake flags not recorded —
   this build predates flag tracking"), which is the phrasing being echoed.
9. **Tests.** In `internal/builder/buildenv_test.go`:
   - `backendVersion("rocm", paths)` against a temp file containing `10.0.0`, a
     file with trailing whitespace and a trailing newline, a file with extra
     fields on the line, a missing path, and an empty file.
   - Path ordering: first path present wins; first path missing falls through to
     the second; first path present but empty also falls through; both missing
     returns `""`. The fall-through cases are the ones that matter — the stable
     image has only the first path and the ROCm 10 image only the second, so a
     bug here would stamp one of the two images as unknown.
   - `backendVersion("vulkan", …)` and an unknown backend return `""`.
   - A table for `StampMismatch`: equal stamps false; `rocm 7.2.4` vs
     `rocm 10.0.0` true; either side empty false; `cuda 12.4` vs `rocm 10.0.0`
     false.
   In `internal/api/build_stamp_test.go`, set the server's `currentBuildEnv` to
   return a fixed `"rocm 10.0.0"` and render the builds list for three builds —
   one stamped `rocm 10.0.0`, one stamped `rocm 7.2.4`, one unstamped — and assert: exactly one row carries the warning mark; the
   flagged row's title names both versions; and the unstamped row renders an
   em-dash and the Step 5 string `Not recorded — this build predates
   build-environment tracking.` byte for byte, with no warning mark. Assert the
   exact string, not a lowercase fragment, so a reworded constant fails the test
   rather than passing on a partial match. Then render `handleBuildInfo` for the
   same unstamped build and assert its `Built against` row carries that identical
   string — the two handlers read one shared constant from `buildRowFor`, and
   this is what proves the second handler is actually wired to it.

## Build gate

```
go build ./...
go vet ./...
go test ./internal/builder/ ./internal/api/
gofmt -l internal | grep -v -e sampling_presets -e monitor.go   # no new findings
```

The three pre-existing `gofmt` findings (`internal/models/sampling_presets.go`,
`sampling_presets_data.go`, `internal/monitor/monitor.go`) are not ours; the gate
is that no new file appears. The two `internal/builder` tests that fail locally
on `tag.gpgsign` are a known local-environment issue, not a regression.

## Test plan

1. **A new build is stamped.** Trigger a build on the current stable container
   and confirm its `builds.json` entry has `built_against: "rocm 7.2.4"`.
2. **An old build is not guessed at.** Existing entries in `builds.json` (for
   example `b10453-rocm-optimized`) have no `built_against` key; confirm the
   Builds page shows an em-dash with the not-recorded tooltip and no warning
   mark.
3. **The flag appears only on a real mismatch.** Hand-edit a copy of
   `builds.json` to set one entry's `built_against` to `rocm 10.0.0` while the
   container runs 7.2.4; the page flags exactly that row, and the tooltip names
   `rocm 10.0.0` and `rocm 7.2.4`. Restore the file.
4. **A flagged build is still usable.** Select the flagged build as the active
   build and confirm the app accepts it — the flag informs, it does not block.
5. **The info modal agrees with the row.** Open Info on each of the three builds
   and confirm the `Built against` row matches what the table showed.
6. **Old state still loads.** Start the app against a `builds.json` with no
   `built_against` keys at all and confirm no error is logged and the list
   renders.
7. **JSON API carries it.** `curl -s localhost:3000/api/builds | jq '.[0]'`
   includes `built_against` for a stamped build and omits it for an unstamped
   one.

## Commit

```
feat(builds): record what each build was compiled against

A llama-server linked against one ROCm version does not run under
another, and the build directory outlives the container image — so
switching ROCm lines leaves builds that fail to load for a reason
nothing on screen explains. New builds now record their backend and
version, and the Builds page flags any build that does not match the
running container, with a tooltip saying what to do about it. Older
builds have no stamp and are shown as unknown rather than guessed at,
and a flagged build can still be activated.
```

## Rollback

Revert the six files. `built_against` keys left in `builds.json` are ignored by
the reverted code — Go's JSON decoding skips unknown fields — so no data
migration is needed in either direction. Safe to leave partially applied: the
field alone, without the UI, simply records information nothing displays, and the
build path treats detection failure as unknown, so a half-finished
`buildenv.go` cannot fail a build.
