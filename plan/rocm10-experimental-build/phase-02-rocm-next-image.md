# Phase 02 — The rocm-next container image

**Depends on:** Phase 01 · **Enables:** Phase 03 (setup.sh has files to select),
Phase 05 (there is an image to verify)

## Goal

Add a second ROCm container image built on an AMD-published base, alongside the
existing Fedora one, with the base image tag as a build argument so a different
ROCm release can be tried without editing the file. The existing
`Dockerfile.rocm` and `docker-compose.rocm.yml` are not touched, so the stable
path cannot regress. At the end of this phase the image can be built and run by
hand with `podman compose`; wiring it into `setup.sh` is Phase 03.

The name is `rocm-next` rather than `rocm10` deliberately — the overview settles
this as a permanent version selector, so the file names must not carry a version
that will be wrong within a year.

## Files touched

- `Dockerfile.rocm-next` — new; Ubuntu base from an AMD ROCm image, installs the
  `.deb`, no ROCm SDK install of its own.
- `docker-compose.rocm-next.yml` — new; the ROCm compose service pointed at the
  new Dockerfile, with a `ROCM_BASE_IMAGE` build argument.

## Steps

1. Create `Dockerfile.rocm-next`, structured to mirror `Dockerfile.rocm` so the
   two can be read side by side:
   - Header comment stating: this is the experimental ROCm variant; the base
     image supplies ROCm, which is why there is no `repo.radeon.com` block; and
     that ROCm 10.x exists only as container images, which is the reason the
     variant exists at all.
   - `ARG ROCM_BASE_IMAGE=docker.io/rocm/dev-ubuntu-24.04:10.0.0-full`, then
     `FROM ${ROCM_BASE_IMAGE}`. Declare the `ARG` before `FROM` so it can be used
     there, and declare it a second time after `FROM`, because the `LABEL` in
     Step 7 reads it and a pre-`FROM` `ARG` is out of scope in the build stage.
   - Keep the existing three build arguments verbatim: `INSTALL_PACKAGE`,
     `LLAMA_TOOLCHEST_VERSION` and `TARGETARCH`.
2. Set the three HIP environment variables the base does not provide:
   ```
   ENV HIP_PATH=/opt/rocm \
     HIP_CLANG_PATH=/opt/rocm/llvm/bin \
     HIP_DEVICE_LIB_PATH=/opt/rocm/amdgcn/bitcode
   ```
   These are the same values `Dockerfile.rocm` sets. Do not set `ROCM_PATH` or
   `PATH`: the base image already sets both, and nothing else — Phase 01 Step 7
   confirmed the base sets exactly `ROCM_PATH` and `PATH`, so all three of the
   above are needed. It also confirmed all three directories exist at these
   paths, so no correction is required.
3. Install the runtime prerequisites with `apt-get`, not `dnf`:
   `curl` and `ca-certificates`. Not `libssl-dev` — the `.deb` already depends on
   it (Step 4), and naming it twice is the drift this step warns against. Use
   `apt-get update && apt-get install -y --no-install-recommends`. Do **not**
   delete `/var/lib/apt/lists/*` here: Step 4 installs the `.deb` with `apt-get`
   and needs those lists to resolve its dependencies. The cleanup belongs at the
   end of Step 4, after the last `apt-get`.
   Those two are the whole list. Phase 01 concluded plain **Go**, not "Go with
   additions", so the exception that rule allowed for does not apply and nothing
   further is added here. Everything else the image needs — the build toolchain (`cmake`,
   `ninja-build`, `git`, `build-essential`, `pkg-config`) and `libssl-dev` —
   arrives through the `.deb`'s declared dependencies in Step 4, so do not
   install any of that here and risk the two lists drifting apart.
4. Install the application as a `.deb` instead of an `.rpm`, mirroring the
   existing conditional in `Dockerfile.rocm` exactly — local snapshot via
   `INSTALL_PACKAGE`, else resolve `LLAMA_TOOLCHEST_VERSION` (`latest` resolves
   through the GitHub releases API) and download
   `llama-toolchest_${VERSION}_linux_${TARGETARCH}.deb`. Install with
   `apt-get install -y /tmp/llama-toolchest.deb` — not `dpkg -i` — so the
   package's dependencies (`cmake`, `ninja-build`, `git`, `build-essential`,
   `pkg-config`, `libssl-dev`, already declared for `deb` in `.goreleaser.yaml`)
   are resolved rather than left broken. Remove the downloaded file afterwards,
   and only now `rm -rf /var/lib/apt/lists/*` — this is the last `apt-get` in the
   image, so the lists can go without breaking the dependency resolution above.
5. Install `uv` exactly as `Dockerfile.rocm` does, with the same
   `UV_INSTALL_DIR=/usr/local/bin INSTALLER_NO_MODIFY_PATH=1` arguments. The
   benchmark runner shells out to `uvx llama-benchy`, so it must be on `PATH`
   system-wide.
6. Reproduce the tail of `Dockerfile.rocm` unchanged: create
   `/data/{config,builds,models,llama.cpp}`, write the same
   `/data/config/llama-toolchest.yaml`, `VOLUME ["/data"]`, `EXPOSE 3000`,
   `EXPOSE 8080`, and the same `ENTRYPOINT`. The container must be
   interchangeable with the stable one from the app's point of view.
7. Add one informational label,
   `LABEL org.opencontainers.image.base.name="${ROCM_BASE_IMAGE}"`, so
   `podman image inspect` on a built image says which base it came from. Note in
   a comment that this label is informational only and nothing reads it, so a
   reader does not mistake it for load-bearing.
8. Create `docker-compose.rocm-next.yml` by copying `docker-compose.rocm.yml` and
   changing only:
   - `build.dockerfile: Dockerfile.rocm-next`
   - add `ROCM_BASE_IMAGE: "${ROCM_BASE_IMAGE:-docker.io/rocm/dev-ubuntu-24.04:10.0.0-full}"`
     to `build.args`, keeping the existing `LLAMA_TOOLCHEST_VERSION` argument and
     its comment.
   Everything else stays byte-identical, in particular
   `image: localhost/llama-toolchest:latest` (the Quadlet unit generated by
   `generate_quadlet_app` expects that exact name), the `/dev/kfd` and `/dev/dri`
   devices, `group_add`, `ipc: host`, `seccomp=unconfined`, the `memlock`
   ulimits, and the `LLAMA_TOOLCHEST_MODELS_DIR: ""` override with its comment.

## Build gate

```
bash -n setup.sh                      # unchanged here, but keep it green
podman build -f Dockerfile.rocm-next -t llama-toolchest:rocm-next-test .
podman compose -f docker-compose.rocm-next.yml config
```

The build must complete, and `compose config` must render without an
unresolved-variable warning.

## Test plan

**Every `podman run` below needs `--entrypoint`.** The built image sets
`ENTRYPOINT ["llama-toolchest", …]`, so a bare `podman run <image> bash -lc '…'`
passes `bash` and its arguments to `llama-toolchest`, which ignores them and
starts the server listening on :3000 — the command appears to hang rather than
failing. Phase 01's inventory commands did not need this because the AMD base
image has no entrypoint; the moment the `.deb` is installed, it does.

1. **Image builds.** The `podman build` above completes. Record the final image
   size in this phase's notes.
2. **ROCm survives the build.** `podman run --rm --entrypoint bash
   llama-toolchest:rocm-next-test -lc 'cat /opt/rocm/core/.info/version; command -v hipcc'` reports
   `10.0.0` and a `hipcc` path — confirming the `.deb` install did not disturb
   the base's ROCm. Note the path: this image has no `/opt/rocm/.info/version`
   (Phase 01), which is why Phase 04 reads both layouts.
3. **The app is installed and runnable.** `podman run --rm --entrypoint
   llama-toolchest llama-toolchest:rocm-next-test --help` exits 0.
4. **The toolchain the Builds page needs is present.** `podman run --rm
   --entrypoint bash llama-toolchest:rocm-next-test -lc 'cmake --version &&
   ninja --version && git --version && c++ --version'` all succeed — this is what the `.deb`'s
   dependencies are for, and it is the thing most likely to differ from Fedora.
5. **GPU visible with devices attached.** Pass the host's numeric video and
   render GIDs, not the group names — rootless podman maps names in the
   container's own `/etc/group`, which does not have the host's:
   ```
   podman run --rm --device /dev/kfd --device /dev/dri \
     --group-add "$(getent group video | cut -d: -f3)" \
     --group-add "$(getent group render | cut -d: -f3)" \
     --security-opt seccomp=unconfined --entrypoint rocminfo \
     llama-toolchest:rocm-next-test | grep -E 'Marketing Name|gfx1201'
   ```
   finds the card. This is the first point the host driver is exercised.
6. **The base tag is overridable.** Re-run the build with
   `--build-arg ROCM_BASE_IMAGE=docker.io/rocm/dev-ubuntu-24.04:7.14.1-full` and
   confirm the version file reports a 7.14 version — check both
   `/opt/rocm/.info/version` and `/opt/rocm/core/.info/version`, since which
   layout a TheRock-era image uses is exactly what Phase 04 has to tolerate. Proves the argument
   is wired and gives Phase 05 a second ROCm line to test the mismatch flag
   against. Delete that test image afterwards.
7. **The stable path is untouched.** `git diff --stat` shows no change to
   `Dockerfile.rocm` or `docker-compose.rocm.yml`.
8. **A linked binary can actually RUN, not just compile.** This is the check
   whose absence let a broken image ship: every other test here proves the
   image can *build*, and none proves that what it builds can start.
   ```
   podman run --rm --entrypoint bash llama-toolchest:rocm-next-test \
     -lc 'ldconfig -p | grep -c libhipblas'
   ```
   must report a non-zero count. AMD's image has no `/etc/ld.so.conf.d` entry
   for ROCm, because its own tools find their libraries through RPATH — and
   `DT_RUNPATH` does not apply to transitive dependencies, so a freshly built
   `libggml-hip.so` cannot find `libhipblas` however well `llama-server` itself
   is linked. The symptom is a successful build followed by
   `error while loading shared libraries: libhipblas.so.3` at startup.
   Where a `llama-server` build already exists in the data volume, run it too:
   ```
   podman run --rm --entrypoint bash -v llama-toolchest-data:/data \
     llama-toolchest:rocm-next-test \
     -lc 'd=/data/builds/<id>; LD_LIBRARY_PATH=$d $d/llama-server --version'
   ```
   `LD_LIBRARY_PATH` is set here because that is what the app does when it
   launches the router (see `internal/api/jobs_env.go`), so the test matches
   how the binary is really started.
9. **No cmake workaround is needed.** Confirm the HIP cmake config is where
   Phase 01 Step 5 recorded it:
   `podman run --rm --entrypoint ls llama-toolchest:rocm-next-test /opt/rocm/lib/cmake/hip/hip-config.cmake`
   must succeed. That single check is the whole test — do not also probe with
   `cmake --find-package`, which reports a different thing and would give two
   pass criteria for one test. If the file is elsewhere, record it for Phase 05
   rather than changing the Dockerfile speculatively.

## Commit

```
feat(container): add an experimental ROCm image built on AMD's own base

ROCm 10 ships only as container images — repo.radeon.com's el9, el10,
rhel9 and rhel10 paths all stop at 7.2.4 — so Dockerfile.rocm cannot
reach it however long we wait. This adds a second image built on
rocm/dev-ubuntu-24.04, with the base tag as a build argument so a new
ROCm release can be tried without changing this file. The stable Fedora
image is untouched.
```

## Rollback

Delete `Dockerfile.rocm-next` and `docker-compose.rocm-next.yml`. Nothing else
references them at this point, so removal
is complete and cannot affect the stable path. Safe to leave partially applied:
until Phase 03, no code selects these files, so an incomplete Dockerfile sits
inert in the repository.
