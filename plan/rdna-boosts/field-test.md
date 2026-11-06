# RDNA boosts — Phase 06 field test

**Date:** 2026-11-05 · **Machine:** asmodean, one Radeon AI PRO R9700 (gfx1201, 32 GB), container on ROCm 10.1 · **Toolchest:** main cbc6702

Builds compared, same base and the same cmake flags (`GGML_HIP`, `GGML_CUDA_FORCE_MMQ`, `GGML_CUDA_FA_ALL_QUANTS`, `GGML_LTO`, `GGML_NATIVE`, `GPU_TARGETS=gfx1201`):

- `b11887-rocm-llama-optimized`: upstream `b11887` (6050b467c).
- `rdna-b11887-r206-rocm-llama-optimized`: the pinned release `v16-6050b467c-r206` (commit efaa60db, the release tip; the tree check passed).

## Step 3 — pinned release as the router's build

- The router started on the patched build (`system_fingerprint` b11903-efaa60db0).
- Ten chat requests each to gemma-4-12b-it UD-Q6_K_XL (dense) and Qwen3.6-35B-A3B-MTP UD-Q4_K_XL (MoE): nine short prompts and one 6.6K-token prompt (both models have an 8K context in the router). 20 of 20 answered; both counted the 416 items in the long prompt correctly.
- After `podman restart llama-toolchest` the router came back on the patched build.
- The 20K-token prompt ran in the step 4 benchmarks instead (`internal-long-ctx`, 24.6K real tokens, 36K context), so the saved router context did not need to change.

Changes from the plan: the dense model is gemma-4-12b, because Qwen3.6-27B is turned off on asmodean. The MoE tests ran with the ming-image instance stopped (its 11 GiB reservation left too little room for the MoE); it was started again after the test.

## Step 4 — speed against upstream

Toolchest benchmark, `internal-standard` (3 reps, 128 gen) and `internal-long-ctx` (1 rep, 512 gen), context 36864. t/s, mean of the reps.

gemma-4-12b-it UD-Q6_K_XL (dense):

| Prompt | b11887 pp | r206 pp | Change | b11887 tg | r206 tg | Change |
| --- | --- | --- | --- | --- | --- | --- |
| 128 | 831.6 | 970.1 | +17% | 45.4 | 46.9 | +3% |
| 512 | 1181.8 | 1538.8 | +30% | 44.8 | 45.2 | +1% |
| 2048 | 1273.3 | 1477.0 | +16% | 43.2 | 42.0 | −3% |
| 32K (24.6K) | 1163.6 | 1807.6 | +55% | 38.6 | 40.2 | +4% |

Qwen3.6-35B-A3B-MTP UD-Q4_K_XL (MoE, no drafting):

| Prompt | b11887 pp | r206 pp | Change | b11887 tg | r206 tg | Change |
| --- | --- | --- | --- | --- | --- | --- |
| 128 | 1333.4 | 1390.2 | +4% | 81.7 | 98.4 | +20% |
| 512 | 3111.1 | 3777.3 | +21% | 81.9 | 101.5 | +24% |
| 2048 | 3609.0 | 4301.2 | +19% | 81.3 | 101.9 | +25% |
| 32K (24.6K) | 3574.3 | 4268.6 | +19% | 74.6 | 93.6 | +25% |

The test plan's gate (faster on at least one of the two models) passes: the patched build is faster on both.

### MTP drafting (input for Phase 05)

Same MoE model, context 8192, `internal-standard` plus `autotune-code` (1.5K-token code edit, 512 gen).

| Prompt | b11887 `draft-mtp` tg | r206 `draft-mtp` tg | r206 `draft-mtp-adaptive` tg | adaptive vs draft-mtp |
| --- | --- | --- | --- | --- |
| 128 | 158.0 | 224.9 | 231.0 | +3% |
| 512 | 143.4 | 207.1 | 201.3 | −3% |
| 2048 | 141.8 | 193.5 | 197.4 | +2% |
| code edit | 165.6 | 232.0 | 231.7 | 0% |

`draft-mtp` on the patched build is 36–44% faster than on upstream. `draft-mtp-adaptive` (passed as an extra flag) is within ±3% of `draft-mtp`, which is run-to-run noise. Phase 05's rule is to add the option only if it is faster, so on this evidence Phase 05 is not needed.

## Step 5 — VRAM fit

Autoconfigure for the MoE model (medium context) with the patched build as the router's build proposed 32K context and `draft-mtp`, estimated at 25.0 GiB, and its own check load passed. The fit code does not depend on the build, so the plan on upstream is the same.

The proposal then ran as a benchmark on each build while `rocm-smi` sampled used VRAM once a second:

| Build | Peak above the idle baseline |
| --- | --- |
| b11887 | 24.38 GiB |
| r206 | 24.55 GiB |

The patched build uses 0.17 GiB more, and both stay under the 25.0 GiB plan. No per-patch-set fit margin is needed. The Autoconfigure result was discarded.

## Step 6 — host mode

On the local workstation (CachyOS, RX 9070 XT gfx1201, host ROCm 7.2.4 at `/opt/rocm`), the toolchest from this branch ran as a `systemd-run --user` unit with the system unit's protections: `ProtectHome=read-only`, `ProtectSystem=full`, `NoNewPrivileges=true`. Data and config were in `/var/tmp/lt-rdna`, so `HOME` was read-only. `PrivateTmp` was left off, because it hides `/var/tmp` from the unit; the real unit keeps its data in `/var/lib`.

Building `v16-6050b467c-r206` from a fresh data folder passed:

- the llama.cpp clone and the patch repo clone;
- the base checkout, `git am` of the 16 patches, and the tree check (commit efaa60db, the release tip);
- cmake and ninja, in 28 minutes; status `success`, commit count 11887, the patch fields set.

The built `llama-server --version` reports build 11903, commit efaa60db0, and its help lists `draft-mtp-adaptive`. The build also passed on ROCm 7.2.4, not only 10.1. A root system install was not tested (no sudo on this machine).

## State after the test

asmodean's router is back on `v0.8.0-rocm-llama-optimized`, ming-image is running again, and no model settings changed. Both builds stay on disk.
