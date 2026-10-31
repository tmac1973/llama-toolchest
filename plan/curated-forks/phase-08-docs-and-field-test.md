# Phase 08 — Docs, pinned commit check and field test

**Depends on:** 01–07 · **Enables:** a release with Avalon support, and hosting it for the 7900 XTX user

## Goal

Write the user docs, confirm the pinned Avalon commit on asmodean (or move the pin to one that works), check host mode, and run the final speed and stability check on the friend's RX 7900 XTX machine. Record the results in the issue.

## Files touched

- `README.md` — a short "llama.cpp forks (experimental)" section.
- `web/templates/help.html` — a help section on forks: what a fork instance is, the whole-GPU rule, the setting, and the known Avalon workarounds.
- `internal/forks/curated.go` — `PinnedSHA` and `PinnedDate` moved only if the asmodean test needs it.

## Steps

1. README section, natural voice: what curated forks are, that Avalon is the only one, that a fork instance takes one whole GPU and serves only its own model, that Autotune, Autoconfigure and benchmarks do not cover fork instances, and how to turn it on. Note that router models must be kept off the fork's GPU (a model split over all GPUs cannot load while the fork runs), and how to do that with the model's GPU setting.
2. Help page section in the project's short, plain style. List the known Avalon workarounds as `Extra environment` lines: `GGML_CUDA_MOE_PIN=0` when the host has less RAM than the model (Avalon#3), `GGML_CUDA_MOE_CACHE_MODE=off` on gfx1201 (Avalon#2). Also note that a fork server's GET endpoints (`/props`, `/slots`) are on its own port.
3. Pinned commit check on asmodean, container install built with `--rocm-image 10.1.0-full`: build the pinned ROCm commit, download the set, run the instance, send ten chat requests (short and 20k-token prompts), stop and start it twice. If the build or run fails at the pin and a newer Avalon `main` commit works, move `PinnedSHA`/`PinnedDate` to that commit in this phase's PR and write the reason in the PR body.
4. CUDA check on compute2 (binary swap of the branch build): build the pinned Avalon commit with the CUDA profile, start an instance on one A4000 with `-c 32768` in Extra arguments, and send one chat request. This confirms the CUDA build and launch path; its speed is not measured (16 GB cards, not a target). compute2 has known intermittent CUDA crashes in upstream llama-server too; a crash here counts against Avalon only if it repeats on a second try with the same command.
5. Host mode: on a host install (systemd unit with `ProtectHome`), build Avalon and start the instance once. This confirms the clone, build-state and models paths work outside the container (issue #475's rule: test host mode before a release).
6. Field test on the friend's RX 7900 XTX machine: install the branch build (binary swap, as on compute2), turn the setting on, build Avalon pinned for gfx1100, download the set, run one instance on one card. Record generation t/s with MTP for a short prompt and prefill t/s for a ~19k prompt, and compare with the Avalon README (about 91 and 1,600 t/s on a 7900 XT). Keep a router model on another card serving during the test.
7. Post the asmodean, compute2 and field results as a comment on issue #494.

## Build gate

- `make build`
- `go vet ./...`
- `go test ./...`
- `make js-test`

## Test plan

- The asmodean, compute2 and host-mode checks in steps 3–5 pass.
- The field test in step 6 runs without a crash for 30 minutes of mixed requests, and generation speed is within 20% of the Avalon README figure for its card.
- The Help page renders the new section, and every tooltip added in Phases 01–07 follows the writing rules (short, plain, ASD-STE100 style).

## Commit

`docs(forks): document llama.cpp fork instances`

(If the pin moves: a separate commit `fix(forks): pin Avalon to <sha7>`.)

## Rollback

Revert the commit. Docs only, plus at most a pin change; the previous pin is in git history.
