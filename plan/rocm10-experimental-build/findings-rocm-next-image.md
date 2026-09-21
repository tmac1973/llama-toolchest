# Phase 02 notes — the rocm-next image

Built and tested 21 September 2026 with podman 6.1.1 on kernel `7.2.3-1-cachyos`.

## Result

All eight test-plan checks pass. `Dockerfile.rocm-next` and
`docker-compose.rocm-next.yml` are in place, the stable path is untouched, and
the image runs ROCm 10.0.0 with the GPU visible.

| check | result |
|---|---|
| 1 · image builds | pass — 21.1 GB |
| 2 · ROCm survives the `.deb` install | pass — `10.0.0`, `hipcc` at `/opt/rocm/bin/hipcc` |
| 3 · app installed and runnable | pass — `llama-toolchest --help` exits 0 |
| 4 · build toolchain from the `.deb`'s dependencies | pass — cmake 3.28.3, ninja 1.11.1, git 2.43.0, g++ 13.3.0, pkg-config 1.8.1 |
| 5 · GPU visible with devices attached | pass — `gfx1201`, "AMD Radeon RX 9070 XT" |
| 6 · base tag overridable | pass — 7.14.1-full base reports `7.14.1` |
| 7 · stable path untouched | pass — no diff to `Dockerfile.rocm` or `docker-compose.rocm.yml` |
| 8 · HIP cmake config in place | pass — no workaround needed |

Image sizes, for the Phase 06 documentation: **21.1 GB** on ROCm 10.0.0 and
20.4 GB on 7.14.1. Almost all of it is the base image — 20.8 GB and 20.2 GB
respectively — so the application layers add under a gigabyte.

No comparison against the stable Fedora image: it is not built on this machine
(this host runs a host-mode install, so `podman images` has no
`localhost/llama-toolchest:latest`). If Phase 06 wants that comparison, build
the stable image first and measure it — do not estimate it.

## The version source, confirmed twice over

Both AMD Ubuntu images use the TheRock layout and neither has the RPM path:

| image | `/opt/rocm/.info/version` | `/opt/rocm/core/.info/version` | `hipconfig --version` |
|---|---|---|---|
| 10.0.0-full | absent | `10.0.0` | `7.15.26333-0000000` |
| 7.14.1-full | absent | `7.14.1` | `7.14.60850-0000000` |
| Fedora, ROCm 7.2.4 | `7.2.4` | absent | — |

This is worth dwelling on, because it is the trap Phase 01 only half-revealed.
On the 7.14.1 image `hipconfig` reports `7.14.…`, which *matches* the ROCm
release. On the 10.0.0 image it reports `7.15.…`, which does not. Had Phase 04
kept `hipconfig --version` as its fallback, it would have looked perfectly
correct in any test run against 7.14 and been silently wrong on ROCm 10 — the
worst available failure mode for a version stamp whose whole job is to tell two
ROCm lines apart. The two-path file read is confirmed correct for all three
images in play.

## Two corrections to the plan, from running it

**Test commands need `--entrypoint`.** The built image sets
`ENTRYPOINT ["llama-toolchest", …]`, so `podman run <image> bash -lc '…'` passes
`bash` and its arguments to `llama-toolchest`, which ignores them and starts
serving on :3000. The command appears to hang rather than fail. Phase 01's
inventory commands did not need this because the AMD base image has no
entrypoint. The phase file's test plan is corrected.

**`--group-add video` does not work rootless.** Podman resolves group names
against the container's own `/etc/group`, which does not carry the host's
`video` and `render` groups. The numeric host GIDs are required — 983 and 987 on
this machine. Note that `docker-compose.rocm.yml` already does this correctly
through `${HOST_VIDEO_GID}` / `${HOST_RENDER_GID}`, which setup.sh resolves
numerically; only the ad-hoc test command was wrong.

## Implementation notes worth keeping

- The apt package lists are deliberately left in place by the prerequisites
  layer and removed only after the `.deb` install, because that install uses
  `apt-get` and needs them to resolve `cmake`, `ninja-build`, `git`,
  `build-essential`, `pkg-config` and `libssl-dev`. Deleting them earlier — which
  the first draft of the plan did — would have failed the build.
- Debian's `docker-clean` apt hook is removed so the `--mount=type=cache` on
  `/var/cache/apt` actually retains downloads. Phase 05 rebuilds the image
  several times; without this the toolchain is re-downloaded every time.
- Only `HIP_PATH`, `HIP_CLANG_PATH` and `HIP_DEVICE_LIB_PATH` are set. The base
  sets `ROCM_PATH` and `PATH` and nothing else, and re-setting them would be
  noise that later diverges from the base.
