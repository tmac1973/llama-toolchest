# Phase 03 notes — setup.sh variant selection

Implemented and tested 21 September 2026.

## What passes

Thirteen checks, run as a single regression pass after the last edit, all green.
They cover file selection for both variants, the flag and environment surfaces,
tag validation, the invalid-value guard, non-ROCm backends, and all three
switch-warning cases:

| check | result |
|---|---|
| default resolves to `Dockerfile.rocm` / `docker-compose.rocm.yml` | pass |
| `next` resolves to the `rocm-next` pair | pass |
| `--rocm-image 10.0.0-full` selects and pins in one flag | pass |
| `--rocm-next` alone takes the default tag | pass |
| a full reference (`…/dev-ubuntu-26.04:10.0.0-full`) passes through unqualified | pass |
| `ROCM_VARIANT=next` in the environment is honoured | pass |
| a flag beats an environment value | pass |
| a nonexistent tag fails before the summary — **in 1 second** | pass |
| `ROCM_VARIANT=sideways` is rejected, naming the two valid values | pass |
| `GPU=cuda` and `GPU=cpu` are unaffected | pass |
| stable → next warns about rebuilding | pass |
| next → the same next is silent | pass |
| next → a different tag warns | pass |

The kernel pre-flight was verified by temporarily raising `ROCM_KERNEL_FLOOR` to
`9.0`: it warned, named the running kernel (`7.2.3-1-cachyos`), the floor, and
the amdgpu version as "(not reported)" — correct for an in-tree driver with no
`/sys/module/amdgpu/version` — and the run continued to the build confirmation
rather than aborting. The constant was restored to `6.14`.

The interactive prompt was exercised through a harness that extracts
`prompt_rocm_variant` and `rocm_base_image` from `setup.sh` verbatim and drives
them with scripted answers, since the install flow itself cannot be run
non-interactively without building. Six cases, all correct: a fresh machine
defaults to stable; a machine on `next` shows option 2 marked `(current)` and
keeps it on Enter; answering `1` from `next` returns to stable and clears the
pinned image; answering `2` from stable takes a tag; and a machine pinned to
another AMD repository keeps that repository on Enter.

## Two bugs found by running it

**The actions list disagreed with the summary.** `setup.sh:543` built its
"Build container image (Dockerfile.rocm)" line by interpolating `GPU_VENDOR`
rather than calling `dockerfile()`, so with the experimental variant selected
the summary said `Dockerfile.rocm-next` and the step list said `Dockerfile.rocm`.
The plan's Step 3 said every other caller "then follows automatically", which was
true only of callers that actually call the function. Now fixed to use
`$(dockerfile)`.

**Pressing Enter at the tag prompt could silently change repository.** The first
implementation defaulted the empty answer to the *tag* it had displayed, which
it derived by stripping everything before the last colon. For a machine pinned
to `docker.io/rocm/dev-ubuntu-26.04:10.0.0-full`, pressing Enter therefore
yielded the bare tag `10.0.0-full`, which `rocm_base_image` then re-qualified
against the default repository — moving the machine from 26.04 to 24.04 without
saying so. The prompt now keeps two values: the full reference, which is what an
empty answer preserves, and a display string that shows just the tag for the
usual repository and the whole reference for any other, so what is on offer is
never ambiguous.

## Not verified here — needs a terminal or a real install

Four checks could not be run in this environment, and none of them is a
code-reading substitute:

1. **The prompt in situ.** `read -rp` prints nothing when stdin is not a
   terminal, so the menu's appearance and wording have only been seen through
   the harness, not as a user sees them. The logic is verified; the presentation
   is not.
2. **`rebuild` does not prompt**, and **the `.env` round-trip** — both need a
   completed install to have written `.env` in the first place. These fall
   naturally into Phase 05, which installs for real.
3. **The host-mode message.** `./setup.sh install --host --rocm-next` prints the
   container-only notice and then calls `host_install`, which begins installing
   ROCm packages with no confirmation of its own — so it cannot be run here
   without changing the machine. Verified by reading the code only.

## Design notes worth keeping

- **A stored variant does not suppress the prompt.** It becomes the prompt's
  default. The first draft of this phase suppressed the prompt whenever `.env`
  held a value, which would have made it impossible to return an experimental
  machine to stable interactively — there would have been no question to
  answer. The prompt is gated on `command == install` plus an interactive
  terminal, so `rebuild`, `up` and `down` still never ask.
- **The three messages are deliberately grouped.** Tag validation, the kernel
  warning and the switch warning all run immediately before `print_summary()`,
  so everything a reader needs appears together, above the summary and above the
  build confirmation — before committing to a 20 GB pull.
- **Empty behaves as stable everywhere.** `compose_file()` and `dockerfile()`
  test for `next` rather than for `stable`, so a missing or unset variant can
  never select the experimental path by accident.
- `shellcheck` is not installed on this machine, so the plan's "no new findings"
  gate could not be run. `bash -n` is clean.
