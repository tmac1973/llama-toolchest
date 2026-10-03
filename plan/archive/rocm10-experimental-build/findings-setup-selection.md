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

The kernel pre-flight was verified by temporarily raising this card's floor to
`9.0`: it warned, named the running kernel (`7.2.3-1-cachyos`), the floor, and
the amdgpu version as "(not reported)" — correct for an in-tree driver with no
`/sys/module/amdgpu/version` — and the run continued to the build confirmation
rather than aborting. The table entry was restored.

## The kernel floor was wrong, and wrong in shape

The first implementation warned below a single `ROCM_KERNEL_FLOOR="6.14"`,
described in the prompt as "where the AMD driver gained support for RDNA 4
cards". Two errors, both caught by the user reading the prompt on a real run.

The number was invented. I had reasoned it from when the RX 9070 launched rather
than looked it up. The Gentoo AMDGPU wiki, which tracks this per generation,
gives RDNA 4 as kernel **6.12** (6.15+ recommended) — not 6.14.

The shape was worse. A single floor asserted an RDNA 4 requirement to everyone,
including RDNA 2 and RDNA 3 owners for whom it is simply false: RDNA 3 needs
6.0, RDNA 2 needs 5.9, RDNA 1 needs 5.3. An RDNA 2 owner on a 6.1 kernel would
have been warned their machine was too old when it was fine. And the wording
implied ROCm 10 is an RDNA 4 release, which Phase 01's own package inventory
contradicts: ROCm 10.0.0 ships code for gfx1010 through gfx1250 plus the CDNA
parts — RDNA 1 and newer.

The check is now card-aware. `rocm_gfx_kernel_floor` maps the detected gfx
target to the kernel version in which the AMD driver gained support for that
generation, and `ROCM10_GFX_TARGETS` holds the target list measured from the
base image's own packages. Two questions are asked instead of one: does ROCm 10
ship code for this card, and is the kernel new enough for *this* card. Where
there is no figure on record — the CDNA parts, gfx1250, an unreadable target —
the kernel and driver versions are reported and no comparison is made, rather
than a number being invented for them.

Verified across the whole table: gfx1201 → 6.12 RDNA 4, gfx1151 → 6.10,
gfx1100 → 6.0, gfx1032 → 5.9, gfx1010 → 5.3, gfx900 → 4.15, and gfx942 /
gfx1250 / unknown → no figure. A simulated gfx902 is correctly reported as
outside ROCm 10's target list. On this machine the check now reads
"Host kernel 7.2.3-1-cachyos is new enough for RDNA 4 (needs 6.12)".

The interactive prompt was exercised through a harness that extracts
`prompt_rocm_variant` and `rocm_base_image` from `setup.sh` verbatim and drives
them with scripted answers, since the install flow itself cannot be run
non-interactively without building. Six cases, all correct: a fresh machine
defaults to stable; a machine on `next` shows option 2 marked `(current)` and
keeps it on Enter; answering `1` from `next` returns to stable and clears the
pinned image; answering `2` from stable takes a tag; and a machine pinned to
another AMD repository keeps that repository on Enter.

## Three bugs found by running it

**The actions list disagreed with the summary — twice.** `setup.sh:543` built
its "Build container image (Dockerfile.rocm)" line by interpolating
`GPU_VENDOR` rather than calling `dockerfile()`, so with the experimental
variant selected the summary said `Dockerfile.rocm-next` and the step list said
`Dockerfile.rocm`. The plan's Step 3 claimed every other caller "then follows
automatically", which was true only of callers that actually call the function.

Switching it to `$(dockerfile)` did not fix it, which only a real install
revealed: `check_prerequisites` builds that list and ran *before* the variant
prompt, so it asked `dockerfile()` a question the user had not been asked yet
and got `Dockerfile.rocm` from an unset variant. The variant block now runs
first and `check_prerequisites` after it. Worth noting the failure mode — the
call was correct and the ordering made it lie, so reading the diff would not
have caught it.

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

## Added during implementation: `--from-source` for containers

Not in the plan, and it should have been. Running `./setup.sh install` and
selecting the experimental variant produces a container running the *last
release*, because every container Dockerfile downloads the released `.deb` or
`.rpm` from GitHub. Nothing in this branch is in it. Host mode has had
`--from-source` for exactly this, but container mode had no equivalent — so
there was no way to test local changes in a container at all.

That is not a nice-to-have for this project: Phase 05 cannot verify the build
stamp or the mismatch flag against a container that does not contain them.

`--from-source` now means "build from this tree" in both modes. On its own it
still implies `--host`, as it always has; with `--container` it builds the tree
into the image. Flag order does not matter, because the mode is only defaulted
when the user has not named one.

**It installs a binary, not a package.** The released package is still installed
in full — that is what brings in `cmake`, `ninja`, `git` and the compiler that
llama.cpp builds need, plus the systemd units — and then the locally built
binary replaces `/usr/bin/llama-toolchest`. Two properties make this work:
`CGO_ENABLED=0`, so the binary runs on Fedora, Ubuntu or Debian bases alike, and
`go:embed` for the templates and static files, so one file carries the UI as
well as the code. It needs nothing but the Go toolchain — no goreleaser (which
is not installed here anyway), no nfpm.

Wired through all four container Dockerfiles and all five compose files, since a
`LOCAL_BINARY` build argument the compose file never sets is no use. The
Makefile's `package-snapshot` comment already claimed to be "used by the dev
container rebuild flow"; that flow did not exist until now, and this is a
simpler one than the comment imagined.

Verified: the flag builds `dist/llama-toolchest-local` (statically linked,
version reported as `v2.29.4-7-gcb068c9-dirty`), the image carries it, `cmake`,
`ninja`, `git` and `hipcc` all survive, and the install summary gains a
`Program  built from this tree` line so it is obvious which one you are getting.

## Added during implementation: the container's own toolchain on the Builds page

Also not in the plan, and the missing half of Phase 04. The container's ROCm
version was computed in exactly one place and only ever appeared inside a
mismatch tooltip — so it was visible only when a mismatched build existed. The
page could say a build was stale without saying what it was stale against, and
after switching variants there are no builds at all, which is precisely when the
question matters most.

The Builds page now carries one line under the heading:

```
Running rocm 10.0.0 · built on docker.io/rocm/dev-ubuntu-24.04:10.0.0-full
```

The base image cannot be detected from inside a container, so
`Dockerfile.rocm-next` sets `LLAMA_TOOLCHEST_ROCM_BASE_IMAGE` alongside the
label it already had. The stable Fedora image sets neither, and then the line
reads `Running rocm 7.2.4` with no "built on" fragment rather than an empty one.

Two design corrections while building it. The first version asked
`activeBackend()`, which resolves the active build and panics with no config —
the banner describes the container, not a build, so it must not touch the
config. The second asked `DetectBackends()` for an *available* backend, which
reports whether the GPU is reachable: a container started without `/dev/kfd`
then said nothing at all, despite plainly having a ROCm SDK installed. It now
asks each GPU backend for its version and takes the first that answers, which
is the actual question — what this container would compile against.


## Losing .env silently moved model storage

Reported while switching between the variants: setup.sh had forgotten the host
model directory. It had not — `.env` was gone, and I had deleted it myself while
testing the variant flags, after the container was already installed.

The round-trip itself is correct, and was verified by driving `write_env_file`
and `load_env_ports` directly: install as experimental with a models directory,
read it back, switch to stable, read it back again — the directory and the
variant both survive in both directions.

But the failure mode this exposed is worth guarding regardless of who deleted
the file. `.env` is gitignored, so a fresh clone, a clean checkout or a stray
delete loses it — and losing it is not a harmless reset to defaults. Model
storage silently reverts to the container's internal volume, the bind mount
disappears on the next rebuild, and every model vanishes from the Models page
with nothing on screen explaining why. The user is left to remember a path they
typed once.

`recover_models_dir_from_container` now asks the installed container, which
knows the answer, when `.env` does not name a directory. It says what it found
and why it matters. Four paths tested: recovery when `.env` is missing and a
container is installed; silence when `.env` names a directory (which always
wins); a no-op when no container is installed; and a no-op when the recorded
mount no longer exists on disk.
