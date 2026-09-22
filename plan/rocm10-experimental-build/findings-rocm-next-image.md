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

The stable Fedora image was not built at the time of writing, so no comparison
was possible then. It has since been built during the Phase 05 switch test and
**measures 14.1 GB** (Fedora Linux, ROCm 7.2.4), against 21.1 GB for the
experimental one. Worth recording that the first README draft guessed "~8 GB"
for it — wrong by 6 GB, and caught only because the note above said to measure
rather than estimate.

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


## The image was broken at runtime, and the test plan could not see it

Found by the first real use of the container: Autoconfigure reported "the helper
model could not read the model card", and the server log said

```
llama-server: error while loading shared libraries: libhipblas.so.3:
cannot open shared object file
==> Router exited with error: exit status 127
```

The library is present — Phase 01 recorded `libhipblas.so.3` in `/opt/rocm/lib`
— but AMD's image has **no `/etc/ld.so.conf.d` entry for ROCm**. `ldconfig -p`
did not know `libhipblas` existed. The ROCm tools inside the image do not need
one because they find their libraries through RPATH; anything built afterwards
does not inherit that.

The reason it is specifically a *transitive* failure is worth recording, because
the binary looks correctly linked. `llama-server`'s own RUNPATH is:

```
[/data/llama.cpp/build-rocm/bin:/opt/rocm/core-10.0/lib:/opt/rocm/lib:]
```

— it includes `/opt/rocm/lib`. But `libggml-hip.so`, which is what actually
needs hipBLAS, has:

```
[/data/llama.cpp/build-rocm/bin:]
```

— only the build directory, which the builder deletes after the build.
`DT_RUNPATH` applies only to the direct dependencies of the object that
declares it, so `llama-server`'s path was never consulted for hipBLAS. The
Fedora image never hit this because the ROCm RPMs ship the `ld.so.conf.d` entry
themselves.

Fixed by writing `/opt/rocm/lib` and `/opt/rocm/llvm/lib` to
`/etc/ld.so.conf.d/rocm.conf` and running `ldconfig`, with a
`ldconfig -p | grep -q libhipblas` assertion in the same layer so the image
fails to build if it ever stops working.

**The test-plan gap is the real lesson.** Eight checks passed on this image and
every one of them tested that it could *build* — hipcc present, cmake configs
present, toolchain present, a compile succeeding. Not one tested that a linked
binary could start. A ninth check now does, and it is the cheap one:
`ldconfig -p | grep -c libhipblas`.
