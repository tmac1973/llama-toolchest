# Phase 03 — setup.sh variant selection

**Depends on:** Phase 02 · **Enables:** Phase 05 (the choice can be exercised the
way a user would), Phase 06 (there are flags and prompts to document)

## Goal

Make the variant selectable from `setup.sh`: interactively when an AMD GPU is
detected in container mode, and non-interactively by flag or environment
variable. Selection persists to `.env` so `up`, `down` and `rebuild` reproduce
it. Before anything expensive happens, the base image tag is checked for
existence and the host kernel is checked against the floor RDNA 4 needs, and
switching between ROCm lines warns that existing `llama.cpp` builds must be
rebuilt. Accepting every default must produce exactly the install it produces
today.

## Files touched

- `setup.sh` — new globals; `compose_file()` and `dockerfile()` consult the
  variant; a new prompt; two new flags; tag validation; kernel pre-flight; switch
  warning; `.env` write and read-back; `--help` and `detect` output; a host-mode
  refusal message.
- `.env.example` — document `ROCM_VARIANT` and `ROCM_BASE_IMAGE`, which this
  phase is the first to write and read.

## Steps

1. **Globals.** Beside the existing `GPU_VENDOR` declaration (`setup.sh:14`) and
   `HOST_SDK_BACKENDS` (`:38`), add:
   - `ROCM_VARIANT=""` — empty means undecided; values `stable` and `next`.
   - `ROCM_VARIANT_EXPLICIT=false` — set by flag or environment variable, so a
     prompt is skipped and a stored `.env` value cannot override this run. Mirrors
     `INSTALL_MODE_EXPLICIT` and `SECURE_EXPLICIT`.
   - `ROCM_BASE_IMAGE=""` — the full base image reference for `next`.
   - Two readonly constants: `ROCM_NEXT_IMAGE_REPO="docker.io/rocm/dev-ubuntu-24.04"`
     and `ROCM_NEXT_DEFAULT_TAG="10.0.0-full"`.
   - `ROCM10_GFX_TARGETS`, the list of gfx targets ROCm 10.0.0 ships code for,
     taken from the base image's own package list. ROCm 10 does **not** require
     RDNA 4: it covers RDNA 1 and newer plus the CDNA parts. There is no single
     `ROCM_KERNEL_FLOOR`, because the kernel a machine needs depends on its
     card and not on ROCm — see `rocm_gfx_kernel_floor` in Step 9.
2. **Resolve the image reference in one place.** Add
   `rocm_base_image()`: if `ROCM_BASE_IMAGE` already contains a `/` treat it as a
   full reference and echo it unchanged; if it is a bare tag echo
   `${ROCM_NEXT_IMAGE_REPO}:${ROCM_BASE_IMAGE}`; if empty echo
   `${ROCM_NEXT_IMAGE_REPO}:${ROCM_NEXT_DEFAULT_TAG}`. This is what lets
   `--rocm-image 10.0.0-full` and `--rocm-image docker.io/rocm/dev-ubuntu-26.04:10.0.0-full`
   both work.
3. **File selection.** Change `compose_file()` (`setup.sh:1025`) and
   `dockerfile()` (`:1049`) to return the `rocm-next` names when
   `GPU_VENDOR` is `rocm` and `ROCM_VARIANT` is `next`, and the existing
   `docker-compose.${GPU_VENDOR}.yml` / `Dockerfile.${GPU_VENDOR}` otherwise.
   Keep both functions one-liners plus the conditional; every other caller
   (including the in-place rebuild path around `setup.sh:1331` and the file
   listing at `:1815`) then follows automatically.
4. **The prompt.** Add `prompt_rocm_variant()` modelled on
   `prompt_install_mode()` (`setup.sh:1110`) — numbered menu, `read -rp`, and an
   invalid choice re-prompting via the same
   `*) err …; prompt_rocm_variant; return` shape. Unlike
   `prompt_install_mode`, the default is not the literal `1`; see below. Text:
   - `1) Stable — ROCm 7.2.4 on Fedora (recommended)`
   - `2) Experimental — ROCm 10 on AMD's Ubuntu image`
   Under option 2, three lines, in plain language:
   - that ROCm 10 is published only as a container image, which is why this is
     the only way to get it;
   - that ROCm 10 supports RDNA 1 and newer and the CDNA cards, and that the
     kernel needed depends on the card rather than on ROCm, and is checked;
   - that `llama.cpp` builds made under one ROCm version must be rebuilt under
     another, and that nothing is deleted when switching.
   The menu's default is whatever is already installed, not a fixed `1`: the
   number in the `[N]` prompt is the stored choice's number, and that choice's
   line is suffixed `(current)`. On a machine with no previous selection the
   default is `1`. This is what lets someone press Enter to keep the variant they
   are on instead of being silently moved back to stable.
   On choosing 2, prompt for the tag in the style of `prompt_ports()`, with the
   stored tag as the default and `ROCM_NEXT_DEFAULT_TAG` when there is none —
   the bracketed value is that default, not a literal:
   `read -rp "$(echo -e "  ${BOLD}Base image tag${NC} [${default_tag}]: ")"`.
   An empty answer takes the default. Set `ROCM_VARIANT` accordingly.
5. **Call the prompt in the right place only.** Invoke it during the install flow
   after the GPU backend and install mode are known, and only when all of:
   `GPU_VENDOR` is `rocm`, `INSTALL_MODE` is `container`,
   `ROCM_VARIANT_EXPLICIT` is false, the command is `install`, and stdin is a
   terminal (`[[ -t 0 ]]`).
   A value restored from `.env` by Step 12 does **not** suppress the prompt — it
   becomes the prompt's default (Step 4). That distinction matters: suppressing
   the prompt whenever `.env` held a value would mean a machine already on `next`
   could never be returned to `stable` interactively, because there would be no
   question to answer. `rebuild`, `up` and `down` are not `install`, so they never
   prompt and simply reuse the stored value, which is what makes a rebuild keep
   the variant.
   After this point, if `ROCM_VARIANT` is still empty — a non-interactive run, a
   non-`install` command with no stored value, or a non-ROCm backend — default it
   to `stable` so every later branch has a definite value.
6. **Flags.** In the argument parser beside `--rocm` (`setup.sh:2064`):
   - `--rocm-next` — sets `ROCM_VARIANT=next`, `ROCM_VARIANT_EXPLICIT=true`.
   - `--rocm-image <tag>` — consumes the next argument into `ROCM_BASE_IMAGE`,
     and also sets `ROCM_VARIANT=next` and `ROCM_VARIANT_EXPLICIT=true`, so one
     flag both selects and pins. Error out if the value is missing or begins with
     `-`.
   Note in a comment that, unlike `--rocm`, neither flag implies `--host` —
   these are container-only and `--rocm` means something different (host SDK
   install).
7. **Environment overrides.** Where `GPU=` is honoured (`setup.sh:2217`), accept
   `ROCM_VARIANT` (`stable`/`next`) and `ROCM_BASE_IMAGE` from the environment,
   each setting `ROCM_VARIANT_EXPLICIT=true`. Reject any other `ROCM_VARIANT`
   value with a message naming the two valid ones. Precedence, stated in a
   comment: flags, then environment, then — for an interactive `install` — the
   answer to the prompt, whose own default is the value stored in `.env`; then
   the stored `.env` value for every other command and for non-interactive runs;
   then `stable`.
8. **Tag validation.** Add `validate_rocm_base_image()`. Call it from the install
   flow when the variant is `next`, immediately before `print_summary()`
   (`setup.sh:1779`) — so it runs before the build and before the confirmation
   prompt, which means it can be observed by starting `./setup.sh install` and
   declining at the confirmation. The check is:
   `$CONTAINER_CMD manifest inspect "$(rocm_base_image)" >/dev/null 2>&1`.
   Whatever `rocm_base_image()` resolved is what gets inspected, so a bare tag
   and a full reference to another repository are validated the same way — a
   reference such as `docker.io/rocm/dev-ubuntu-26.04:10.0.0-full` passes because
   that image exists, not because of the repository constant. On failure, `fatal`
   with the full reference that failed, a note that the tag may not exist or the
   registry may be unreachable, and a pointer to
   `https://hub.docker.com/r/rocm/dev-ubuntu-24.04/tags`. Use the container
   runtime already required in container mode rather than adding a `curl` or
   `skopeo` dependency. Skip the check when the runtime is unavailable, warning
   rather than failing, so a runtime problem is reported as itself.
9. **Kernel pre-flight.** Add `check_rocm_host_kernel()`, called from the same
   place as Step 8's validation — when the variant is `next`, immediately before
   `print_summary()` — so its warning appears above the summary and ahead of the
   confirmation prompt:
   - Warn when the detected gfx target is not in `ROCM10_GFX_TARGETS`.
   - Look the card's kernel floor up with `rocm_gfx_kernel_floor`, which maps
     gfx target to the kernel version where the AMD driver gained support for
     that generation (RDNA 4 → 6.12, RDNA 3.5 → 6.10, RDNA 3 → 6.0, RDNA 2 →
     5.9, RDNA 1 → 5.3, Vega → 4.15; figures from the Gentoo AMDGPU wiki).
     The CDNA parts are absent on purpose, so no figure is asserted for them.
   - Compare `uname -r`'s `major.minor` against that floor. When the target is
     unknown or has no figure on record, report the kernel and driver and make
     no comparison rather than inventing one.
   - If `/sys/module/amdgpu/version` exists (DKMS installs only) include its
     contents in the message; on an in-tree `amdgpu` it is absent, so treat that
     as "not reported" rather than a problem.
   - Warn — do not fail — naming the kernel found, the floor, and that the
     install will continue. Also warn if `/dev/kfd` is missing, since the
     container cannot work without it.
10. **Switch warning.** Add `warn_rocm_variant_switch()`, called alongside Steps
    8 and 9 — once the variant is resolved and immediately before
    `print_summary()`, so all three messages appear together above the summary
    and ahead of the confirmation prompt. Read the previous `ROCM_VARIANT` out of
    `.env`; treat absent as `stable`, which is correct for every install that
    predates this change. If it differs from the selection, or if the variant is
    `next` on both sides but `ROCM_BASE_IMAGE` differs, print that the ROCm
    version is changing, that `llama.cpp` builds in the data volume were compiled
    against the old one and must be rebuilt from the Builds page, and that
    nothing has been deleted so switching back restores them. Delete nothing.
11. **Persist.** In `write_env_file()` (`setup.sh:1053`), in the GPU-specific
    section next to `HSA_OVERRIDE_GFX_VERSION`, write `ROCM_VARIANT` whenever
    `GPU_VENDOR` is `rocm`, and `ROCM_BASE_IMAGE=$(rocm_base_image)` when the
    variant is `next`. The compose file reads `ROCM_BASE_IMAGE` for its build
    argument, so it has to be in `.env`, not merely exported.
12. **Read back.** In `load_env_ports()` (`setup.sh:870`), following the
    `SECURE_EXPLICIT` pattern exactly: when `ROCM_VARIANT_EXPLICIT` is false,
    restore `ROCM_VARIANT` and `ROCM_BASE_IMAGE` from `.env`. This is what makes
    `./setup.sh rebuild` keep the experimental variant without re-passing a flag.
    Read `ROCM_BASE_IMAGE` with `cut -d= -f2-` so a reference containing `:` and
    `/` survives.
13. **Host mode.** Where host SDK backends are resolved (`setup.sh:2253`–`2265`),
    if the variant is `next` and the mode is host, print that ROCm 10 is
    published only as a container image and cannot be installed on the host,
    that host mode will install the ROCm SDK from `repo.radeon.com` as usual, and
    that container mode is the way to get ROCm 10. Then continue with the normal
    host install — this is a message, not a failure.
14. **`--help`.** Four edits, all inside the help *text*: add both flags to the
    flag list near `--rocm` (`setup.sh:1929`), update the `detect` description in
    the subcommand list (`:2001`) to say it prints the ROCm variant too, add
    `ROCM_VARIANT` and `ROCM_BASE_IMAGE` to the environment-variable section
    (`:2009`), and add one example beside the existing ones (`:2025`):
    `./setup.sh install --rocm-image 10.0.0-full   # container install on ROCm 10 (experimental)`.
15. **`detect` output.** The subcommand is implemented at `setup.sh:2283`
    (`detect)    echo "$GPU_VENDOR" ;;`) — a different place from its help text
    in Step 14. When `GPU_VENDOR` is `rocm`, print a second line naming the
    variant that would be used and, for `next`, the resolved base image. Keep the
    first line exactly as it is: it is the machine-readable backend name and
    something may be parsing it.
16. **`.env.example`.** Document both variables with one-line comments: the
    variant selects which ROCm container is built (`stable` or `next`), and the
    base image pins the ROCm release used by `next`. Say that `setup.sh` writes
    both and that editing them by hand takes effect on the next
    `./setup.sh rebuild`. This belongs in this phase rather than Phase 02: it is
    the phase where the variables start existing and doing something.

## Build gate

```
bash -n setup.sh
shellcheck setup.sh                    # no new findings versus the pre-change run
./setup.sh --help                      # renders, shows both new flags
./setup.sh detect                      # prints backend and variant
```

Capture `shellcheck setup.sh` output before the change and compare, since the
script has pre-existing findings; the gate is "no new ones", not "clean".

## Test plan

1. **Default is unchanged.** With no flags and no `.env`, answer the variant
   prompt with Enter. Verify from the install summary: `print_summary()`
   (`setup.sh:1779`) already prints a `Dockerfile` and a `Compose file` line from
   `dockerfile()` and `compose_file()`, and it runs before the confirmation
   prompt — so start `./setup.sh install`, read those two lines, and answer no.
   They must read `Dockerfile.rocm` and `docker-compose.rocm.yml`. This is the
   whole regression check for the stable path and needs no new code to observe.
2. **Flag selects and pins.** `./setup.sh detect` with `--rocm-image 10.0.0-full`
   reports variant `next` and the full reference
   `docker.io/rocm/dev-ubuntu-24.04:10.0.0-full`. `--rocm-next` alone reports the
   default tag.
3. **Full reference accepted.** `--rocm-image docker.io/rocm/dev-ubuntu-26.04:10.0.0-full`
   is passed through unchanged rather than being prefixed.
4. **Environment override works and flags win.** `ROCM_VARIANT=next ./setup.sh
   detect` reports `next`; the same with `--rocm-image 7.14.1-full` on the command
   line reports the flag's tag.
5. **Bad tag fails fast.** `./setup.sh install --rocm-image definitely-not-a-tag`
   exits non-zero before the summary and the confirmation prompt, and the message
   names the failing reference. It must be `install`, not `detect`: per Step 8 the
   validation runs in the install flow. Time it — it must fail in seconds, not
   after a partial pull.
6. **Invalid variant rejected.** `ROCM_VARIANT=sideways ./setup.sh detect` fails
   with a message naming `stable` and `next`.
7. **Kernel warning fires.** Temporarily *raise* this card's floor above the
   running kernel — this host reports `7.2.3-1-cachyos`, which is already well
   past every floor in the table, so temporarily edit this card's entry in
   `rocm_gfx_kernel_floor` to `9.0` to force the warning. Run
   `./setup.sh install --rocm-next` (not `detect` — per Step 9 the check runs in
   the install flow), confirm the warning appears above the summary naming both
   the kernel found and the floor, and that the run reaches the confirmation
   prompt rather than aborting. Decline the confirmation, then restore the
   constant to `6.14`.
8. **Switch warning fires, and only on a switch.** With `ROCM_VARIANT=stable` in
   `.env`, run `./setup.sh install`, select `next`, and confirm the rebuild
   warning appears above the summary; select `stable` and confirm it does not.
   With no `.env` at all, selecting `stable` must not warn — the
   absent-means-stable case. Decline the confirmation each time.
9. **The prompt defaults to the installed variant.** With `ROCM_VARIANT=next` in
   `.env`, run `./setup.sh install` with no flags: the menu must appear (not be
   skipped), show option 2 marked `(current)` with `[2]` as the default, and
   pressing Enter must keep `next` — the summary then shows
   `Dockerfile.rocm-next`. Run it again and answer `1`: the summary must show
   `Dockerfile.rocm`, proving an interactive return to stable is possible. This
   is the pair of behaviours the prompt rule exists for.
10. **Non-install commands never prompt.** With `ROCM_VARIANT=next` in `.env`,
    `./setup.sh rebuild` must run without asking and use the `rocm-next` files.
11. **Round-trips through `.env`.** After an install selecting `next`, `.env`
    contains both variables; a subsequent `./setup.sh detect` with no flags still
    reports `next` with the same base image.
12. **Host mode says its piece.** `./setup.sh install --host --rocm-next` prints
    the container-only message and proceeds with a host install rather than
    aborting.
13. **Non-ROCm backends unaffected.** `GPU=cuda ./setup.sh detect` and
    `GPU=cpu ./setup.sh detect` show no variant line and resolve their usual
    compose files.

## Commit

```
feat(setup): choose between the stable and experimental ROCm containers

Adds a prompt when an AMD GPU is detected in container mode, plus
--rocm-next and --rocm-image <tag> for unattended runs, with
ROCM_VARIANT / ROCM_BASE_IMAGE as environment equivalents. The tag is
checked against the registry before anything is built, the host kernel
is checked against the floor RDNA 4 needs, and switching ROCm versions
warns that llama.cpp builds must be rebuilt. Nothing is deleted, and
accepting the default installs exactly what it did before.
```

## Rollback

Revert `setup.sh`. The `rocm-next` files from Phase 02 become unreachable but
harmless, and any `ROCM_VARIANT`/`ROCM_BASE_IMAGE` lines left in a user's `.env`
are ignored by the reverted script and by compose (the stable compose file has no
`ROCM_BASE_IMAGE` argument). An install already running the experimental image
keeps running it — the image is tagged `localhost/llama-toolchest:latest` either
way — so reverting does not break a machine; a `./setup.sh rebuild` afterwards
returns it to the stable image and its builds need rebuilding. Not safe to leave
half-applied: a `compose_file()` change without the matching `.env` persistence
would rebuild the stable image over an experimental install without warning.
