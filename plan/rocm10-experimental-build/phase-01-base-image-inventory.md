# Phase 01 — Base image inventory

**Depends on:** nothing · **Enables:** Phase 02 (the Dockerfile is written against
these findings rather than against an assumption)

## Goal

Establish what `rocm/dev-ubuntu-24.04:10.0.0-full` actually contains before a line
of Dockerfile is written. The image is documented by AMD as a "ROCm user-space
runtime" but tagged `-full` in a `dev-` repository, and its 8.05 GB ROCm layer is
comparable to `7.2.4-complete`'s 7.37 GB — strong evidence of a complete SDK, but
evidence, not fact. If `hipcc` or the hipBLAS headers are absent there is nowhere
to `apt-get` them from, because no 10.x debs exist in any `repo.radeon.com` path.
That would change the shape of the whole project, so it is worth one download to
find out. This phase writes no code and changes no repository file outside the
plan folder; its output is a findings document the next phase reads.

## Files touched

- `plan/rocm10-experimental-build/findings-base-image.md` — new; the inventory and
  the go/no-go conclusion.

## Steps

1. Pull the image:
   `podman pull docker.io/rocm/dev-ubuntu-24.04:10.0.0-full`
   Record the pulled size and digest. Expect roughly 8.2 GB.
2. Confirm the ROCm version the image reports, which is also the source the build
   stamp will read in Phase 04:
   `podman run --rm docker.io/rocm/dev-ubuntu-24.04:10.0.0-full cat /opt/rocm/.info/version`
   Expect `10.0.0`. If the file is absent, record what `hipconfig --version` and
   `ls /opt/rocm/.info/` report instead. This is recorded for reference only —
   Phase 04 decides its own detection order and does not depend on this finding.
3. Inventory the compiler and tools. For each of `hipcc`, `hipconfig`,
   `amdclang++`, `rocminfo`, `rocm_agent_enumerator`, record the path and version:
   `podman run --rm <image> bash -lc 'for t in hipcc hipconfig amdclang++ rocminfo rocm_agent_enumerator; do printf "%s: " "$t"; command -v "$t" || echo MISSING; done'`
4. Inventory the math libraries and their headers — these are what
   `GGML_HIP=ON` links against:
   - libraries: `ls /opt/rocm/lib | grep -E 'hipblas|rocblas|hipblaslt|rccl|rocsolver'`
   - headers: `ls -d /opt/rocm/include/{hipblas,rocblas,hip,rccl} 2>&1`
   - rocWMMA headers: `ls -d /opt/rocm/include/rocwmma 2>&1` — record presence,
     but absence is **not** a blocker (see the overview's non-goals).
5. Inventory the CMake integration llama.cpp's HIP backend needs:
   `ls /opt/rocm/lib/cmake` and confirm `hip-config.cmake` /
   `hip-lang-config.cmake` and `hipblas-config.cmake` exist. `Dockerfile.rocm`
   already carries a comment about cmake deriving
   `CMAKE_HIP_LIBRARY_ARCHITECTURE` from the config location, so note the exact
   paths. Recorded for reference: Phase 02 deliberately does not apply that
   workaround, and if a build turns out to need it that is handled as a fix
   during Phase 05.
6. Record the base OS and the state of the build toolchain:
   `cat /etc/os-release`, then whether `cmake`, `ninja`, `git`, `g++` and
   `pkg-config` are present. They need not be — the `.deb`'s dependencies pull
   them in — but knowing avoids a surprise in Phase 02.
7. Record which environment variables the base already sets
   (`podman run --rm <image> env | sort`). Confirmed so far: `ROCM_PATH=/opt/rocm`
   and `/opt/rocm/bin` on `PATH`. Note specifically whether `HIP_PATH`,
   `HIP_CLANG_PATH` and `HIP_DEVICE_LIB_PATH` are set, because `Dockerfile.rocm`
   sets all three and Phase 02 must set whichever the base omits. Then confirm
   the three directories those variables point at actually exist, since Phase 02
   Step 2 hard-codes them:
   `podman run --rm <image> bash -lc 'ls -d /opt/rocm /opt/rocm/llvm/bin /opt/rocm/amdgcn/bitcode'`
   Record any that are missing or live elsewhere — that is the input Phase 02
   Step 2 needs in order to use a corrected path.
8. Write `findings-base-image.md` with a table of every item above, then a
   **Conclusion** section stating one of:
   - **Go** — the SDK is complete; Phase 02 proceeds as planned.
   - **Go with additions** — something is missing but obtainable (name the
     package and where it comes from).
   - **No-go** — the compiler (`hipcc`/`amdclang++`), the hipBLAS or rocBLAS
     headers, or the HIP cmake configs are missing, and no 10.x deb exists to
     install them from.
9. On a no-go conclusion, take exactly one further step before stopping: repeat
   Steps 2–7 against `docker.io/rocm/dev-ubuntu-26.04:10.0.0-full`, which carries
   the same ROCm 10.0.0 release on a newer Ubuntu and may be packaged
   differently. The rule for what follows:
   - **If the 26.04 image supplies everything the 24.04 image lacked**, it
     becomes the base: replace `rocm/dev-ubuntu-24.04` with
     `rocm/dev-ubuntu-26.04` at every occurrence in Phases 02, 03 and 06 —
     Dockerfile `ARG` default, compose build-argument default,
     `ROCM_NEXT_IMAGE_REPO`, the example build arguments in the test plans, and
     the Docker Hub link in the README text — and the project continues otherwise
     unchanged. Do not treat the list in this sentence as exhaustive; grep the
     plan folder for `dev-ubuntu-24.04` and change every hit.
   - **Otherwise** — whether both images fail the same way or the 26.04 image
     fails differently and still lacks something essential — the project stops
     here: nothing is merged, and `findings-base-image.md` records that AMD's
     published ROCm 10 images cannot build llama.cpp, which is the answer to the
     original question and worth keeping even though no code ships.

## Build gate

No repository code changes, so no build. The gate is that
`findings-base-image.md` exists, every item in Steps 2–7 has a recorded answer
(not a blank), and the Conclusion states go, go-with-additions, or no-go.

## Test plan

Verification is the inventory itself. Two checks that the findings are sound:

- Every "present" claim is backed by a pasted command output in the document, so
  a reader can tell what was actually run.
- Re-run the ROCm version command a second time and confirm it agrees, guarding
  against a typo in the recorded value that Phase 04 would then build on.

## Commit

```
docs(plan): record what the ROCm 10 base image actually contains
```

## Rollback

Nothing to roll back — the only new file is in the plan folder and no repository
code, image or configuration is touched. The pulled image can be reclaimed with
`podman rmi docker.io/rocm/dev-ubuntu-24.04:10.0.0-full` if the ~8.2 GB is
needed; that does not invalidate the findings.
