# ROCm Container Version Selector — Project Overview

## Problem

The ROCm container is pinned to one ROCm release. `Dockerfile.rocm` starts from
Fedora 43 and installs named RPMs from `https://repo.radeon.com/rocm/el9/7.2.4/main`,
so the only way to run a different ROCm is to edit the Dockerfile.

That pin has become a dead end. AMD released ROCm 10.0.0 on 26 August 2026 and
publishes it **only as container images** — `repo.radeon.com/rocm/el9`,
`el10`, `rhel9` and `rhel10` all stop at 7.2.4, and so does the `amdgpu-install`
path the install documentation still describes. The 7.14.x line has the same
problem: 7.14.1 shipped on 31 August 2026, after 10.0.0, and is likewise
container-only. Nothing newer than 7.2.4 can reach `Dockerfile.rocm` as written,
however long we wait, because the packages are not there to install.

Meanwhile AMD does ship `rocm/dev-ubuntu-24.04:10.0.0-full` — a stable-tagged,
8.22 GB image carrying the full ROCm 10.0.0 userspace. Anyone who wants to try
ROCm 10 on this project today has to hand-edit the Dockerfile and work out the
Fedora-to-Ubuntu packaging differences themselves.

## Goals

- Offer a choice, at install time, between the current ROCm build and a ROCm
  container built on an AMD-published base image.
- Let the chosen base-image tag be specified, so a new ROCm release can be tried
  without a repository change.
- State the host kernel requirement where the choice is made, and check it
  before a long build starts.
- Warn when switching between ROCm lines that existing `llama.cpp` builds must
  be rebuilt.
- Stamp every new build with the backend and version it was compiled against, and
  flag builds on the Builds page whose stamp does not match what is running.
- Keep the existing ROCm 7.2.4 path the default, unchanged in behaviour.

## Non-goals

- **Host installs are out of scope.** `./setup.sh install --rocm` installs the
  ROCm SDK natively from `repo.radeon.com`, which has no 10.x packages in any
  path. Host mode cannot install ROCm 10, and this work will not pretend
  otherwise — it will say so plainly if asked.
- **No CUDA, Vulkan or CPU changes.** The build stamp is written for every
  backend, but only the ROCm container gains a version selector.
- **`GGML_HIP_ROCWMMA_FATTN` is not carried forward.** Upstream removed the
  rocWMMA FlashAttention path at llama.cpp b10332 (PR #26046) and it is not
  expected back. Whether the ROCm 10 image ships rocWMMA headers does not affect
  whether this work succeeds.
- **No retroactive stamping.** Builds made before this change have no stamp and
  will not be guessed at; they are shown as unknown, not as mismatched.
- **The stable default does not move.** ROCm 7.2.4 on Fedora 43 stays the
  default in this work. Changing the default is a later decision, made when
  the alternative has been proven.
- **No new backend value.** This is a variant of the existing `rocm` backend,
  not a fifth backend alongside `cuda`/`rocm`/`vulkan`/`cpu`.

## Users & primary flow

The user is someone installing or re-running `./setup.sh` on a Linux machine
with an AMD GPU, in container mode.

1. `./setup.sh install` detects an AMD GPU and sets `GPU_VENDOR=rocm`, as today.
2. Because the backend is ROCm and the mode is container, setup.sh presents a
   numbered menu in the style of `prompt_install_mode` (`setup.sh:1110`): the
   stable ROCm 7.2.4 build or the experimental build on an AMD-published base
   image. The pre-selected option is whichever is already installed, so pressing
   Enter keeps the current choice; on a fresh machine that is the stable build.
   The menu text states that the experimental option needs a recent host
   `amdgpu`/KFD driver and that `llama.cpp` builds made under one ROCm line must
   be rebuilt under another.
3. Choosing the experimental option accepts a base-image tag, defaulting to
   `10.0.0-full`. The tag is checked against the registry before anything is
   built, so a typo fails in seconds rather than part-way through an 8 GB pull.
4. A pre-flight check compares the running kernel against a known-good minimum
   and warns loudly if it looks too old. The install continues — the warning
   informs, it does not block.
5. If an image is already installed and its ROCm line differs from the one being
   selected, setup.sh warns that existing builds must be rebuilt. Nothing is
   deleted, so switching back restores them.
6. The container builds from the chosen base and starts. The Builds page shows
   each build's backend and version, and flags any build whose stamp does not
   match the running container, with a tooltip saying what it was built against,
   what is running now, and what to do. Activating a flagged build is still
   allowed.
7. Re-running `./setup.sh install` with a flag or environment variable selects
   the same thing without prompting, for unattended and repeat installs.
   `rebuild`, `up` and `down` never prompt and reuse the stored choice.

## Constraints

- **Existing stack.** Bash `setup.sh` (2,425 lines), Docker/Podman via
  `docker-compose.<backend>.yml`, Go 1.x backend, htmx + Pico CSS templates.
  `compose_file()` (`setup.sh:1025`) maps a backend to exactly one compose file
  by name; a second ROCm variant has to fit that mapping or change it.
- **Packaging already supports Ubuntu.** `.goreleaser.yaml` builds `.deb` and
  `.rpm` from one nfpm template, with correct Debian dependency names
  (`build-essential`, `libssl-dev`, `pkg-config`). An Ubuntu-based image installs
  the existing `.deb`; no new packaging work is needed.
- **The base image supplies ROCm.** `rocm/dev-ubuntu-24.04:10.0.0-full` sets
  `ROCM_PATH=/opt/rocm` and puts `/opt/rocm/bin` on `PATH`, so the ROCm install
  block in `Dockerfile.rocm` has no equivalent in the experimental Dockerfile.
  Whether it carries `hipcc`, `hipblas-dev` and the rocWMMA headers under those
  names is unverified and must be confirmed by inspecting the pulled image, not
  assumed. The image contains no kernel components and relies on the host driver.
- **Version detection.** `/opt/rocm/.info/version` holds the ROCm version
  (`7.2.4` on the current host) and is the source for the build stamp.
- **Kernel detection is limited.** `/sys/module/amdgpu/version` exists only for
  DKMS installs; on an in-tree `amdgpu` (this host, `7.2.3-1-cachyos`) it is
  absent. `uname -r` is the only signal always available, so the check reads
  `/sys/module/amdgpu/version` when present and falls back to the kernel version.
- **Build records must degrade gracefully.** `BuildResult`
  (`internal/builder/builder.go:37`) is persisted to `builds.json` and read back
  across upgrades. A new field follows the `CommitCount` precedent: absent on old
  records, and the code treats absent as unknown rather than as a mismatch.
- **The data volume outlives the image.** `llama-toolchest-data` is mounted at
  `/data` independently of the image, so `builds/` and `llama.cpp/` survive a
  rebuild onto a different base. This is what makes the mismatch warning
  necessary: a binary linked against ROCm 7.2.4's `libhipblas` will not resolve
  under a ROCm 10 userspace.
- **Hardware for verification.** One machine: Radeon RX 9070 XT (gfx1201, RDNA 4,
  supported by ROCm 10) plus a gfx1036 integrated GPU, kernel `7.2.3-1-cachyos`,
  ROCm 7.2.4 host userspace, podman 6.1.1 and docker both available.
- **ROCm 10 is not in AMD's validated llama.cpp matrix**, which lists 7.0.0,
  6.4.3, 6.4.2 and 6.4.1. "Experimental" is the accurate label.

## Success criteria

- `./setup.sh install` on an AMD machine in container mode offers the choice. On
  a machine with no previous selection, accepting the default produces exactly
  the ROCm 7.2.4 install it produces today. On a machine already running the
  experimental variant the prompt's default is that variant, so a rebuild does
  not silently move it back — and answering the other option still returns it to
  stable.
- The experimental option can be selected interactively, by flag, and by
  environment variable, and all three reach the same result.
- A base-image tag can be given; a nonexistent tag is rejected before any build
  starts, naming the tag that failed.
- The kernel pre-flight warning appears on a too-old kernel and the install still
  proceeds.
- Switching between ROCm lines with an image already installed prints the
  rebuild warning, and no build directory is deleted.
- On the experimental image: the container builds, the app starts, `rocminfo`
  inside the container reports `gfx1201`, a `llama.cpp` build with `GGML_HIP=ON`
  completes, and a model loads and generates tokens.
- A build made after this change shows its backend and version on the Builds
  page; a build whose stamp differs from the running container is flagged with a
  tooltip explaining how to read it, and can still be activated.
- A build made before this change shows as unknown and is not flagged.
- `README.md`'s compatibility table and ROCm section describe the option, the
  version pin, the kernel requirement and the rebuild consequence.
- `go build ./...`, `go vet ./...` and `go test ./...` pass, except the two
  `internal/builder` tests that fail on this machine because of `tag.gpgsign`,
  which are a known local-environment issue. `bash -n setup.sh` passes and
  `shellcheck setup.sh` reports no findings beyond the ones already there before
  the change.
- If end-to-end verification fails, the documentation says what failed and the
  option is still shipped as experimental — the failure is recorded, not hidden.

## Decisions

- **Selection mechanism** → Interactive prompt plus a non-interactive path (CLI
  flag and environment override), matching how install mode, ports and secure
  mode already work.
- **Version pinning scope** → Accept any `rocm/dev-ubuntu-24.04` tag, defaulting
  to `10.0.0-full`, validated against the registry before building. This also
  makes `7.14.1-full` reachable as a middle step.
- **Host kernel requirement** → Note it at the prompt and add a soft pre-flight
  warning that does not block the install.
- **Host-install path** → Out of scope; container only. Host mode is unchanged
  and says plainly that ROCm 10 is unavailable there.
- **Definition of done** → Verified end-to-end on the RX 9070 XT: image builds,
  app starts, GPU visible in the container, `GGML_HIP` build succeeds, model
  loads and generates.
- **Existing builds when switching** → Warn at switch and leave builds alone;
  delete nothing, so switching back restores them.
- **Build stamp contents** → One field holding the backend and its version (for
  example `rocm 10.0.0`), read from `/opt/rocm/.info/version` at build time.
- **Mismatch behaviour** → Flag the build on the Builds page with an explanatory
  tooltip; do not block activation.
- **Documentation** → Update the `README.md` compatibility table and ROCm
  section; no separate page.
- **Longevity** → Build it as a permanent version selector: a stable default plus
  a pinnable alternative, so a new ROCm release can be tried without new code.
