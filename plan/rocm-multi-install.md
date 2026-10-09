# Multiple ROCm installs on one host

Status: builder and Builds page implemented (feat/rocm-multi-install); setup.sh and the wider distro matrix pending.

## Problem

A user on Ubuntu/Debian installed ROCm 10.1 from AMD's new repo next to an
existing ROCm 7.x. Every llama.cpp ROCm build then failed. Separately, host
mode in setup.sh offers no way to install 10.1 at all, while container mode
already does (Dockerfile.rocm-next).

## How ROCm installs now

Three packaging lines, which coexist to different degrees:

| Line | Source | Location | Side by side with itself |
|---|---|---|---|
| Distro | Ubuntu 26.04 (7.1), Fedora, Debian | `/usr` | n/a |
| AMD old (≤ 7.2.4) | repo.radeon.com | `/opt/rocm-X.Y.Z`, with `/opt/rocm` a link to the current one | yes |
| AMD new (10.x) | stable.repo.amd.com, `amdrocm-core-dev10.1-gfxNNNN` (`-devel` on RPM) | `/opt/rocm/core-X.Y` | yes (the repo carries 10.0 and 10.1) |

The 10.x packages' install script uses update-alternatives to point
`/opt/rocm/{bin,lib,include,llvm,amdgcn,libexec,share,core}` and
`/usr/bin/{hipcc,hipconfig,rocminfo,…}` at the newest 10.x. Where a real file
or directory is already in the way it warns and skips that one link, so
the system ends up **half switched**:

**Ubuntu 26.04, distro 7.1 + 10.1** (box `rocm-u2604`)
- `/usr/bin/hipconfig` → 10.1 (the distro's was a symlink, replaced silently)
- `/usr/bin/hipcc`, `rocminfo` → still 7.1 (real files, skipped)
- `/opt/rocm/*` → 10.1

**Ubuntu 24.04, AMD 7.2.4 + 10.1** (box `rocm-u2404`, the user's case)
- `/opt/rocm` → `/opt/rocm-7.2.4`, so 10.1 lands *inside* it at
  `/opt/rocm-7.2.4/core-10.1`
- `/opt/rocm/{bin,lib,include}` → still 7.2.4 (real dirs, skipped)
- `/opt/rocm/{llvm,amdgcn}` → **10.1**: they were symlinks in 7.2.4, so the
  7.2.4 install itself is now modified
- `/usr/bin/{hipconfig,hipcc,rocminfo}` → 10.1

AMD's install guide says to remove ≤ 7.2.4 before installing 10.x; this is why.

## Why builds fail today

The builder (`builder.go`, ROCm block) derives everything from the first
`hipconfig` found. Both mixes give it 10.1's `hipconfig` at `/usr/bin`, so it
takes `/usr` as the ROCm root but 10.1's clang. cmake then can't find
`hip-lang-config.cmake` for that compiler under `/usr` (reproduced exactly).
Separately, llama.cpp's `ggml-hip/CMakeLists.txt` uses `$ROCM_PATH`, else
`/opt/rocm` if it exists, else `/usr`, and the builder deliberately leaves
`ROCM_PATH` unset for `/usr` installs, so 10.1's `/opt/rocm` would win anyway.

## What works: pin the build to one install

Tested by hand in both boxes. For a chosen install root `R`:

1. **Compiler:** `R/lib/llvm/bin/clang++` (distro: `/usr/lib/llvm-N/bin/clang++`).
   Never `hipconfig --hipclangpath`, never `R/llvm` (can be a hijacked link).
2. **Device libraries:** the `amdgcn/bitcode` dir inside that compiler's own
   clang resource dir (`R/lib/llvm/lib/clang/N/lib/amdgcn/bitcode`), via
   `HIP_DEVICE_LIB_PATH`. Not `R/amdgcn`, for the same reason. Needed on the
   7.2.4 + 10.1 box, where 7.2.4's `amdgcn` now leads to 10.1.
3. **Env:** `ROCM_PATH=R`, `HIP_PATH=R`, `R/bin` first on `PATH`
   (distro `/usr` keeps today's handling: no `ROCM_PATH`).
4. **cmake:** `-DCMAKE_HIP_COMPILER_ROCM_ROOT=R`; for distro, also
   `-DCMAKE_PREFIX_PATH=/usr` so `/usr` is searched before `/opt/rocm`.
5. **Headers:** `-isystem R/include` in `CMAKE_HIP_FLAGS` and `CMAKE_CXX_FLAGS`.
   clang searches `/usr/include` *before* the ROCm include dir by design,
   so with distro HIP headers in `/usr/include/hip`, a 10.1 build compiled
   against 7.1's headers and failed (`__ocml_exp10_f32` undeclared).
6. **Runtime libraries:** bake `R/lib` into the binaries
   (`-DCMAKE_BUILD_RPATH=R/lib`). Not `--disable-new-dtags`: the run path
   also lists the shared build directory, which every build overwrites, and
   an RPATH that beats `LD_LIBRARY_PATH` would let old builds load the newest
   build's libggml. The cost: a user `LD_LIBRARY_PATH` naming another ROCm
   still wins. Essential: 7.x and 10.x ship the
   **same sonames** (`libhipblas.so.3`, `libamdhip64.so.7`, …), so without it
   a 10.1 build silently loads the distro's 7.1 libraries. 10.x also
   registers nothing with ldconfig. Baking it in covers the server,
   benchmarks, evals and jobs without touching each launch site.

Results (gfx1100, gemma-3-4b Q8_0, `llama-server -ngl 99`):

| Box | Install | Configure | Compile | Libraries resolve to | Runs |
|---|---|---|---|---|---|
| u2604 | distro 7.1 | ok | ok | `/usr/lib/x86_64-linux-gnu` | 95 tok/s |
| u2604 | 10.1 | ok | ok (with step 5) | `/opt/rocm/core-10.1/lib` | 101 tok/s |
| u2404 | AMD 7.2.4 | ok (with step 2) | ok | `/opt/rocm-7.2.4/lib` | 100 tok/s |
| u2404 | 10.1 | ok | ok | `/opt/rocm-7.2.4/core-10.1/lib` | 102 tok/s |

End to end through the app (feat/rocm-multi-install, form-encoded
`POST /api/builds` with `rocm_root`, launched with the build dir on
`LD_LIBRARY_PATH` as process.Manager does): all four combinations above
built, recorded `rocm <version> @ <root>`, loaded only their own install's
libraries and served gemma-3-4b on the GPU (93–101 tok/s).

## Design

### Detection (Go builder and setup.sh)
List installs by looking at disk, never through `PATH` or the shared links:
- `/usr`, if it has a HIP runtime (`/usr/lib*/**/cmake/hip-lang`)
- each `/opt/rocm-*` (old line)
- each `/opt/rocm/core-*` and `/opt/rocm-*/core-*` (new line)
- `$ROCM_PATH` if set
Resolve each to its real path and de-duplicate. Version comes from the
install's own `.info/version` (old line) or `core-X.Y/.info/version` (new line).
Keep the existing rule of never using `hipconfig --version`, which says 7.16 on 10.1.

### Builds page
One install: no picker. Two or more: a ROCm install picker on the build
form ("ROCm 10.1.0 — /opt/rocm/core-10.1", "ROCm 7.1 — distro (/usr)"),
defaulting to the newest. The builder applies steps 1–6 to that one root.

Pinned with two or more installs, and for a 10.x install even alone: its
packages point /usr/bin/hipconfig at themselves, so the old hipconfig-based
setup fails on any host-installed 10.x (Debian 13 and Rocky 10 with 10.1
alone both failed in cmake before this). A lone distro or old-line install
still builds the old way.

### Build record
`built_against` grows the install root (`rocm 10.1.0 @ /opt/rocm/core-10.1`).
The mismatch warning on the Builds page becomes "this build's install is no
longer present" rather than "the version differs from the one detected".
Older stamps without a root keep today's behaviour.

### Damaged-install warning
If an old-line install's `llvm` or `amdgcn` link points into a different
install (the 7.2.4 + 10.1 case), still allow builds (pinning copes) but
show a warning on the Builds page and in setup.sh: this setup is
unsupported by AMD, and the fix is to remove the old AMD packages or use
the distro's ROCm alongside 10.x.

### setup.sh host mode (AMD GPU)
Offer: distro ROCm (7.x), AMD ROCm 10.1, or both.
- 10.1: add stable.repo.amd.com for the distro (ubuntu2204/2404/2604,
  debian12/13, rhel8/9/10, sles), install `amdrocm-core-dev10.1-<gfx family>`
  (`-devel` on RPM) for the detected GPU.
- If old-line AMD packages (≤ 7.2.4) are present, say they must be removed
  before adding 10.1, and stop rather than create the damaged state.
- Fix `host_rocm_prefer_amd_packages`: an `/opt/rocm` made of 10.x
  alternatives links is not the old AMD repo.
- Fedora has no 10.x packages from AMD; test whether the rhel10 repo works
  before offering it there.

## Test matrix
Fresh distrobox per distro with its own home, gfx1100 (7900 XTX) passed
through. setup.sh host mode run interactively, then the app from this branch
built `latest` (b11537) against each install and served gemma-3-4b Q8_0.

| Distro | setup.sh offered | Chosen | Builds (each loads only its own libraries) |
|---|---|---|---|
| Ubuntu 26.04 | distro 7.1 / 10.1 / both | both | 7.1 ✓, 10.1 ✓ |
| Ubuntu 24.04 | distro 5.7 / 10.1 | 10.1 | (see the u2404 rows above) |
| Ubuntu 24.04 + AMD 7.2.4 | distro / 10.1 (with note) | 10.1 → refused | 7.2.4 kept |
| Debian 13 | distro 5.7 / 10.1 | 10.1 | 10.1 ✓ (failed before lone-10.x pinning) |
| Rocky Linux 10 | distro / 10.1 | 10.1 | 10.1 ✓ (failed before lone-10.x pinning) |
| Fedora 44 | distro 7.1 (10.1 via rhel10 added after) | distro, then 10.1 by hand | 7.1 ✓, 10.1 ✓ |

The ROCm 10.1 container image (dev-ubuntu-24.04:10.1.0-full), run with this
branch's binary and a scratch data dir: its lone 10.1 install is now pinned
(it built before only because the image puts /opt/rocm/bin first on PATH);
b11538 built, loaded only 10.1's libraries and served at 99 tok/s.

Not ROCm, found on the way: the .rpm requires ninja-build, which Rocky/RHEL
only carry in CRB, so a host install there fails at the package step.

## Unrelated, noticed while testing
- `POST /api/builds` with an empty `git_ref` builds the fresh clone's `HEAD`
  (a `backup/…` branch upstream). The UI always sends `latest`, so API-only.
- Build IDs containing `/` (branch refs) 404 on `/api/builds/{id}/logs`.
