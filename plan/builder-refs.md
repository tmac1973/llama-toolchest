# Building from Branches, PRs and Commits

Let the builder compile any ref from `ggml-org/llama.cpp` — not just the
`v*` / `b*` tags the picker lists today. The ref picker gains a
**Custom…** entry that reveals a text field taking a branch, a PR, or a
bare commit SHA.

Motivating case: model support routinely lands in a PR weeks before it
reaches a nightly tag. Qwen3.8-Flash-Next MTP needs
[#27836](https://github.com/ggml-org/llama.cpp/pull/27836) +
[#28097](https://github.com/ggml-org/llama.cpp/pull/28097); GLM-5.3-Flash
(`glm5next`) needs [#27754](https://github.com/ggml-org/llama.cpp/pull/27754).
None is buildable today, so the only way to run either model is to leave
the toolchest behind and build llama.cpp by hand — which also means the
build stops being tracked, benchmarkable, or selectable as active.

**The reason this needs no multi-remote support:** GitHub publishes every
PR head on the *upstream* repo as `refs/pull/N/head`, including PRs
authored from forks. All three PRs above are fork-authored and all three
are fetchable from `ggml-org`. Building other people's repositories is a
separate, larger feature (see Non-goals).

---

## Goals

1. **Build any ggml-org ref** — tag, branch, `refs/pull/N/head`, or a
   bare SHA.
2. **One control** — the existing picker gains a `Custom…` entry that
   reveals a text input; tags stay a dropdown for the common case.
3. **Identify what was built** — a build from a moving ref carries its
   commit in the build ID, so a build can never claim to be "the PR"
   after the PR has moved on.
4. **Never auto-activate an experiment** — a PR build must not become
   the build `LatestSuccessfulBuild` hands out.
5. **No migration** — existing tag builds keep their IDs, their ranking
   and their directories.

## Non-goals

- **Other remotes.** No forks, no arbitrary URLs, no local paths.
  Revisit if tracking a fork *continuously* becomes the norm; the
  blocking work there is build-ID identity and the fact that
  `CommitCount` stops ordering anything once two histories are in play.
- **Listing open PRs in the picker.** git alone cannot enumerate them,
  ggml-org has thousands, and it would put a rate-limited GitHub API
  call in the build form.
- **Tracking a PR.** A build is a snapshot. Nothing re-builds when the
  PR moves; you rebuild deliberately.
- **Combining PRs.** The working recipe for qwen4exp MTP in the wild is
  #27836 *plus* a loader fix from a third party. Composing that is a
  local branch you prepare in the managed clone by hand — which the
  Custom field can then build by name. We do not automate the compose.
- **Pruning.** SHA-suffixed builds accumulate; deleting them stays
  manual.

## Constraints we design around

- The clone at `<dataDir>/llama.cpp` is a build cache we own outright.
  `checkoutRef` already force-checks-out on the stated grounds that
  nothing in it is worth preserving — so a user-prepared local branch
  survives `fetch` but must be understood as living in scratch space.
- **PR refs are not fetched by default.** `git fetch --all --tags` does
  not bring `refs/pull/*`, and fetching all of them is not viable at
  ggml-org's scale. They must be fetched one at a time, on demand.
- **`bN` tags *are* master's commit count.** That identity is the whole
  basis of `buildRank` ordering builds of different refs on one scale.
  It holds only on master; three commits on a PR branch off `b10717`
  count 10720 and would outrank a genuine `b10719`.
- **Build IDs are directory names** under `/data/builds/`, so any ref
  containing `/` has to be sanitized before it becomes one.

## Design

### Ref forms accepted by the Custom field

| input | resolves to |
|---|---|
| `pr/27836`, `#27836`, `27836` | `refs/pull/27836/head` |
| `master`, any branch name | remote branch |
| `a32af33`, full SHA | that commit |
| `b10717`, `v0.2.0` | tag (so pasting a tag still works) |

Bare digits are unambiguous: llama.cpp tags are `bNNNNN` or `vX.Y.Z`,
never a bare number. Resolution order is PR form → tag/branch → SHA.

### Fetching

- **PR:** `git fetch --force origin pull/N/head:refs/remotes/origin/pr/N`,
  then check out that remote-tracking ref. `--force` because PR heads
  are force-pushed routinely.
- **Branch / tag:** the existing `git fetch --all --tags`, unchanged.
- **SHA:** fetch first, then check out. A SHA reachable from any fetched
  ref works; one that is not is an error we surface plainly rather than
  a confusing checkout failure.

### Build identity

- **Tag refs — unchanged.** `<ref>-<profile>[-<tag>]`, with the existing
  flag-hash suffix on collision.
- **Non-tag refs** gain the resolved short SHA:
  `pr-27836-a32af33-rocm`. Sanitize `/` → `-` and restrict to
  `[A-Za-z0-9._-]`.
- New `BuildResult.RefKind` field (`tag` | `branch` | `pr` | `sha`) so
  ranking and the UI never re-parse a ref string. Absent on existing
  records, which is exactly the "legacy build" case `buildRank` already
  handles.

### Ranking

`buildRank` returns `(0, false)` for any `RefKind` other than `tag`.
This is not a new concept — the function's own comment already
describes a legacy branch-or-SHA build as unrankable, and
`LatestSuccessfulBuild` already sorts unrankable builds below ranked
ones, newest-`StartedAt` first among themselves. PR builds simply join
that class.

`CommitCount` is still recorded on every build; it just stops being the
rank for non-tags. Records predating `RefKind` keep today's behavior by
falling through to `refTagNumber(GitRef)`.

Regression test to write: a successful `pr` build with `CommitCount`
10720 must lose to a successful tag build of `b10719`.

### UI

- `web/templates/builds.html:27` — the `git_ref` select gains
  `<option value="__custom__">Custom…</option>`; choosing it reveals a
  text input. Only one of the two submits `git_ref`.
- `internal/api/build.go:139` `handleListRefs` emits the Custom option
  alongside the two existing optgroups.
- The **Release Notes** link builds a `releases/tag/<value>` URL from
  the selection; it needs a branch for custom refs — PR → the PR page,
  SHA/branch → the commit or tree page.
- **Refresh Tags** is unchanged.

### Touched call sites

| file | what |
|---|---|
| `internal/builder/builder.go:36` | `BuildResult` gains `RefKind` |
| `internal/builder/builder.go:157` | `buildRank` gates on `RefKind` |
| `internal/builder/builder.go:~317` | ID composition: SHA suffix |
| `internal/builder/builder.go:686` | `ensureRepo` — on-demand PR fetch |
| `internal/builder/builder.go:709` | `checkoutRef` — ref-form resolution |
| `internal/api/build.go:139` | Custom option in the ref list |
| `web/templates/builds.html:27` | picker + revealed text field |

## Phases

**Phase 1 — builder core, no UI.** Ref-form parsing, on-demand PR fetch,
`RefKind`, SHA-suffixed IDs, the ranking gate, and tests. Buildable via
the API with a hand-posted `git_ref` at the end of this phase.

**Phase 2 — UI.** Custom entry, revealed text field, release-notes link
handling.

**Phase 3 — visibility.** Surface `RefKind` in the builds table (a
`PR #27836` badge beside the ref) so an experimental build is obvious at
a glance, and note in the UI that such builds are not auto-selected.

## Risks

- **Unreachable SHA.** Some servers refuse fetching an arbitrary commit
  not reachable from an advertised ref. Mitigated by fetching tags and
  branches first, and by a clear error rather than a bare git failure.
- **A PR build outliving its PR.** We record the SHA, so the build stays
  meaningful — but `pr/27836` as an *input* is not reproducible over
  time. Documented, not solved.
- **Disk growth.** Every revision of a PR you build is another build
  directory. No auto-pruning in scope; the SHA in the ID at least makes
  them tellable apart.
- **Building unreviewed code.** The trust posture is unchanged in kind —
  the builder already compiles whatever upstream master contains — but a
  PR is weaker still. Worth a line of UI copy, not a gate.
