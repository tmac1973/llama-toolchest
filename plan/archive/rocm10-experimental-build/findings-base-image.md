# Phase 01 findings — `rocm/dev-ubuntu-24.04:10.0.0-full`

Inventoried 21 September 2026 with podman 6.1.1.

## Conclusion

**Go.** The image carries a complete ROCm 10.0.0 SDK: the HIP compiler, the
math libraries `GGML_HIP=ON` links against, their headers, and the CMake
configs llama.cpp's HIP backend needs. Nothing essential is missing, so
Phase 02 proceeds as planned and the 26.04 fallback in Phase 01 Step 9 is not
needed.

Two findings change the plan. Both are corrected in the phase files by the same
commit as this document:

1. **`/opt/rocm/.info/version` does not exist in this image.** The file is at
   `/opt/rocm/core/.info/version`. Phase 04's detection order has to include the
   second path.
2. **`hipconfig --version` must not be used as the version source.** It reports
   `7.15.26333-0000000`, which is the HIP component version, not the ROCm
   release. Stamping that would label a ROCm 10 build as a 7.x one and make the
   Builds page's mismatch comparison wrong in the most confusing possible way.
   Phase 04 Step 2 named it as a fallback; that is now removed.

## Step 1 — the image

| | |
|---|---|
| reference | `docker.io/rocm/dev-ubuntu-24.04:10.0.0-full` |
| digest | `sha256:a90cf047f615abe70fbef83c64def0a2d549ef37a39c8ea545430aba4981b374` |
| created | 2026-08-26 14:07:03 UTC — the day ROCm 10.0.0 was released |
| size | **20.8 GB on disk**, from an 8.22 GB compressed download |
| platform | amd64 / linux, 5 layers |

The 8.22 GB figure quoted while planning was the registry's compressed size. On
disk it is 20.8 GB, which is what actually matters for anyone deciding whether to
try this. Worth carrying into the Phase 06 documentation.

## Step 2 — ROCm version

```
/opt/rocm/.info/version                    (absent)
/opt/rocm/core/.info/version               10.0.0
/opt/rocm/core-10/.info/version            10.0.0
/opt/rocm/core-10.0/.info/version          10.0.0
```

`/opt/rocm/core` is a symlink to `/etc/alternatives/core`, so
`/opt/rocm/core/.info/version` is the stable, release-agnostic path — it does not
name a version and will keep working across ROCm releases. That is the path
Phase 04 should read second, after the Fedora image's `/opt/rocm/.info/version`.

Corroborated by the packages: `amdrocm-core-sdk10.0` and
`amdrocm-core-dev10.0-*` are all at `10.0.0-4`. The tag is honest.

**The HIP version is a different number.** `hipconfig --version` reports
`7.15.26333-0000000`, and `/opt/rocm/core-10.0/share/hip/version` contains
`HIP_PACKAGING_VERSION_PATCH=26333-0000000`. HIP is a component with its own
version line; ROCm 10.0.0 ships HIP 7.15. This is not a mislabelled image, and
the 7.15 number must never be used as the ROCm version.

## Step 3 — compiler and tools

| tool | path |
|---|---|
| `hipcc` | `/opt/rocm/bin/hipcc` |
| `hipconfig` | `/opt/rocm/bin/hipconfig` |
| `amdclang++` | `/opt/rocm/bin/amdclang++` |
| `amdclang` | `/opt/rocm/bin/amdclang` |
| `rocminfo` | `/opt/rocm/bin/rocminfo` |
| `rocm_agent_enumerator` | `/opt/rocm/bin/rocm_agent_enumerator` |
| `clang++` | absent (only `amdclang++`; not needed) |

The compiler question the whole gate existed to answer: **present.** "-full" does
mean the full SDK, not a runtime.

## Step 4 — math libraries and headers

Libraries in `/opt/rocm/lib`:

```
libhipblas.so.3.6   libhipblaslt.so.1.4   librocblas.so.5.6
librccl.so.1.0      librocsolver.so.0.11  (plus hipblaslt/ and rocblas/ dirs)
```

Headers: `hipblas`, `rocblas`, `hip`, `rccl` — all present under
`/opt/rocm/include`.

**`rocwmma` headers are also present** (`/opt/rocm/include/rocwmma`). Recorded
only because the question came up: it changes nothing here, since the overview
makes `GGML_HIP_ROCWMMA_FATTN` a non-goal and upstream removed that path at
llama.cpp b10332.

## Step 5 — CMake integration

| config | location |
|---|---|
| `hip-config.cmake` | `/opt/rocm/lib/cmake/hip/` — present |
| `hip-lang-config.cmake` | `/opt/rocm/lib/cmake/hip-lang/` — present |
| `hipblas-config.cmake` | `/opt/rocm/lib/cmake/hipblas/` — present |
| `rocblas-config.cmake` | `/opt/rocm/lib/cmake/rocblas/` — present |

`hip-lang-config.cmake` lives in its own `hip-lang/` directory, not under
`hip/`. That is the normal ROCm layout and the same as Fedora's; the first
inventory pass looked in the wrong directory and reported it missing.

`/opt/rocm/lib/cmake` holds 40 config directories in total, including
`composable_kernel`, `miopen`, `rccl` and `rocm-core`. Nothing suggests the
relocated-config workaround that Phase 02 test 8 guards against will be needed,
so Phase 02 proceeds without it.

## Step 6 — base OS and build toolchain

- Ubuntu **24.04.4 LTS (Noble Numbat)**
- `g++` 13.3.0 — present
- `cmake`, `ninja`, `git`, `pkg-config` — **absent**

Absent is expected and fine: these are declared dependencies of the `.deb` for
Debian targets in `.goreleaser.yaml`, so `apt-get install -y ./llama-toolchest.deb`
pulls them in. It does confirm Phase 02's design — the toolchain must come from
the package, and the apt lists must therefore still be present when that install
runs, which is why the cleanup belongs at the end of Step 4.

## Step 7 — environment and paths

The base sets only:

```
ROCM_PATH=/opt/rocm
PATH=/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
```

`HIP_PATH`, `HIP_CLANG_PATH` and `HIP_DEVICE_LIB_PATH` are **not** set, so
Phase 02 Step 2 is needed as written. All three directories it points at exist:

```
/opt/rocm                  present
/opt/rocm/llvm/bin         present
/opt/rocm/amdgcn/bitcode   present
```

No path correction is needed.

## Extra — hardware coverage

305 `rocm`-named dpkg entries. `gfx1201` is explicitly packaged, as both
`amdrocm-core-dev10.0-gfx1201` and `amdrocm-core-sdk10.0-gfx1201`, alongside
gfx1010–gfx1250 and the CDNA targets gfx908/90a/942/950. The RX 9070 XT this
will be verified on is a first-class target, not a "should work".
