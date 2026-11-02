# Phase 05 — Patch-set launch options (follow-up)

**Depends on:** 04, and the Phase 06 field test results · **Enables:** none

## Goal

Decide which launch options that only the patch set adds the toolchest should write, and gate them on the build's patch set instead of its version. This phase is a follow-up. The patch set's speed-up does not need it, because every feature is on by default. Do it only if the field test shows that an option helps on our hosts.

## Options the patch set adds (release r206)

| Option | What it does | Candidate use |
|---|---|---|
| `--spec-type draft-mtp-adaptive` | MTP drafting whose depth follows the acceptance rate | Instead of `draft-mtp` for models with an MTP head |
| `--spec-draft-n-min-adaptive`, `--spec-draft-n-start` | Bounds and start value for the adaptive depth | Tuning, with the option above |
| `--spec-draft-mtp-vocab N` | MTP drafting over a vocabulary subset (opt-in, off by default) | Not planned |
| `--host-experts` | Host-resident MoE experts | Compare with the toolchest's `-ncmoe` choice first |
| `-sm relaxed-tensor` | Tensor split for MoE architectures that upstream refuses | Not planned: the patch repo marks it temporary |

Read `patches/README.md` and `ENVIRONMENT.md` at the pinned release again before this phase. The table can be out of date.

## Files touched (if the phase goes ahead)

- `internal/models/launch.go` — a `PatchSet string` on `Target`, and a method per option (e.g. `AdaptiveMTPOption()`), true only for a build of a patch set that has it.
- `internal/patchsets/patchsets.go` — a list of option features per patch set (e.g. `Features: []string{"draft-mtp-adaptive"}`), so a new release that drops one can turn it off by changing the registry.
- `internal/api/build.go` — `buildTarget` sets `Target.PatchSet`.
- `internal/models/launch_test.go` — tests.

## Steps

1. Run the field test (Phase 06) with and without `--spec-type draft-mtp-adaptive` on one model with an MTP head. Add the option only if the adaptive mode is faster and the output is correct.
2. If it is: add the feature to the registry entry, the `Target` field and the gate. Write `draft-mtp-adaptive` where the toolchest now writes `draft-mtp`, only for builds whose patch set has the feature.
3. Show the option in the model's launch preview as for any option. No new UI control.

## Build gate

- `make build`
- `go vet ./...`
- `go test ./...`

## Test plan

- `launch_test.go`: a patched target writes `draft-mtp-adaptive`; an upstream target at the same version writes `draft-mtp`; a patched target whose patch set lacks the feature writes `draft-mtp`.

## Commit

`feat(models): use adaptive MTP drafting on builds that have it`

## Rollback

Revert the commit. Patched builds go back to upstream's `draft-mtp`.
