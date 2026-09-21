# Phase 04 notes — build environment stamp

Implemented and tested 21 September 2026.

## What passes

`go build ./...`, `go vet ./...` and `gofmt` are clean, with no new findings
beyond the three pre-existing unformatted files this work does not touch. The
whole test suite passes apart from the two `internal/builder` tests that fail on
this machine because of `tag.gpgsign`, which is a local git configuration issue
and not a regression.

**`internal/builder/buildenv_test.go`** — 26 subtests. Version-file parsing
(plain, trailing newline, surrounding whitespace, extra fields on the line,
empty, whitespace-only, missing, no paths at all), the path fall-through in both
directions, backends with no version, and a nine-case table for
`StampMismatch` including both malformed-stamp cases.

**`internal/api/build_stamp_test.go`** — three tests. The builds list with one
matching, one differing and one unstamped build; the info modal for the
differing and unstamped cases; and a pre-stamp `builds.json` loading without
complaint.

The render tests were mutation-checked rather than assumed: flagging unstamped
builds in `buildRowFor` makes two separate assertions fail — the count of
warning marks, and the explicit "an unstamped build must not be flagged". An
earlier attempted mutation, dropping the empty-stamp guard from
`StampMismatch`, did *not* fail, because the malformed-stamp guard below it
catches the same case. That is defence in depth rather than a gap, but it is
worth knowing the first guard is not the only thing holding that behaviour up.

## The testability seam, and why it is not optional

`CurrentBuildEnv` memoises per backend and reads real files, so a render test
using it would decide "matching" and "differing" from whichever ROCm happens to
be installed on the machine running `go test` — passing here and failing in CI,
or the reverse. `Server` therefore carries
`currentBuildEnv func(backend string) string`, nil in production and set by the
test. Without it these tests would be worthless rather than merely awkward.

## Design notes

- **Two version paths, in order.** `/opt/rocm/.info/version` for the RPM layout
  the Fedora image uses, then `/opt/rocm/core/.info/version` for the TheRock
  layout AMD's Ubuntu images use. Phase 01 and Phase 02 confirmed that each
  image has exactly one of the two, so both fall-through directions are load
  bearing and both are tested.
- **`hipconfig --version` is deliberately not a fallback**, and the comment in
  `buildenv.go` says why at length: it reports HIP's own component version,
  7.15.x on ROCm 10.0.0 but 7.14.x on ROCm 7.14.1. Using it would have agreed
  with the release on one image and disagreed on the other, so it would have
  tested clean and been silently wrong exactly where it mattered.
- **Unknown is never a mismatch.** Guarded twice, tested directly, and stated in
  the field's own doc comment. A build made before this change may work
  perfectly; flagging it would train the reader to ignore the flag.
- **One text field and one title field** on `buildRow`, both filled by
  `buildRowFor`, so the template carries no wording and the table and the info
  modal cannot drift. An earlier draft had a `MismatchTitle` that was empty when
  not flagged — which is precisely the state where the not-recorded tooltip is
  needed.
- **The em-dash is the unstamped rendering**, matching how absent measurements
  read elsewhere in the app, and it is produced in Go rather than in the
  template so there is one spelling of "unstamped" in the code.

## Not verified here

The stamp has never been written by a real build. Every value in these tests is
constructed. Phase 05 is where a build actually runs and records
`rocm 10.0.0`, and where the flag is exercised against two genuinely different
ROCm lines rather than against fixtures.
