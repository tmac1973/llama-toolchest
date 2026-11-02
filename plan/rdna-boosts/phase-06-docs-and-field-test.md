# Phase 06 — Docs and field test

**Depends on:** 01–04 · **Enables:** a release with RDNA boosts support; Phase 05's decision

## Goal

Write the user docs, confirm the pinned release on asmodean (or move the pin to one that works), compare a patched build with upstream at the same base, check Autoconfigure's VRAM fit on a patched build, and check host mode. Record the results in the issue.

## Files touched

- `README.md` — a short "RDNA boosts" section.
- `web/templates/help.html` — a help section on patch sets.
- `internal/patchsets/patchsets.go` — the pinned release moved only if the asmodean test needs it.
- `internal/api/autoconfig*.go` or the fit code — only if step 5 shows the fit is wrong for patched builds (see step 5).

## Steps

1. **README section**, natural voice: what the patch set is, that it is upstream llama.cpp at an older commit plus patches, that it is ROCm only, that every feature is on by default, and that a patched build runs as the router's build like any other. Say that the base can be weeks behind upstream, so new upstream options and model support arrive only when the patch set re-bases. Link the patch repo.
2. **Help page section** in the project's short, plain style: how to build it, what "tested" and "untested" mean, how to switch the router to it and back, and that options newer than its base are not written.
3. **Pinned release check on asmodean** (container built with `--rocm-image 10.1.0-full`, R9700 gfx1201):
   - Build the pinned release and confirm the tree match in the log.
   - Select it as the router's build. Load one dense model and one MoE model, and send ten chat requests each (short prompts and a 20k-token prompt).
   - Restart the toolchest once and confirm the router comes back on the patched build.

   If the pinned release fails and a newer release works, move `Pinned` to that release in this phase's PR and write the reason in the PR body.
4. **Speed against upstream.** Build upstream at the same base (`b11887`, ROCm profile, same options). Run the toolchest's benchmark on both builds with the same two models and settings. Record prompt-processing and generation t/s for both in the field-test notes. Also run one model with an MTP head with `draft-mtp` and with `draft-mtp-adaptive` (in the model's extra arguments) for Phase 05's decision.
5. **VRAM fit.** Run Autoconfigure for the MoE model on the patched build. Compare the planned VRAM with the VRAM the router reports after warmup, and do the same on the upstream build.
   - If the patched build uses more than the plan allows (the 10% compute margin and the slab headroom are the likely causes), out-of-memory at warmup is the symptom.
   - In that case, add a per-patch-set margin to the fit, read from the registry entry (`FitMarginMiB` or a percentage). Make that change in this phase with a test, or open a follow-up issue if it needs more than one setting.
6. **Host mode.** On a host install (systemd unit with `ProtectHome`), build the pinned release once. This confirms that the patch repo clone and the `git am` environment (Phase 02's `HOME` override) work outside the container (issue #475's rule: test host mode before a release).
7. **Report.** Post the asmodean, speed, fit and host-mode results as a comment on the tracking issue.

## Build gate

- `make build`
- `go vet ./...`
- `go test ./...`
- `make js-test`

## Test plan

- The checks in steps 3, 5 and 6 pass.
- Step 4 shows a patched build faster than upstream b11887 on at least one of the two models. If it is slower on both, stop and report before release.
- The Help page renders the new section. Every tooltip added in Phases 01–04 follows the writing rules (short, plain, ASD-STE100 style).

## Commit

`docs(patchsets): document building llama.cpp with RDNA boosts`

## Rollback

Revert the commit. If the pin moved, the old pin comes back. Builds made from the new pin stay usable; they show "untested".
