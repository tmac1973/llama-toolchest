# RDNA Boosts Patch Set — Project Overview

## Problem

[llama-cpp-rdna-boosts](https://github.com/stew675/llama-cpp-rdna-boosts) (MIT) is a set of 16 patches on upstream llama.cpp that make RDNA2–RDNA4 GPUs faster for every model, not for one model only. It adds:

- adaptive MTP draft depth;
- chunked gated-delta-net prefill;
- a BF16 KV cache and WMMA flash attention;
- quantized prefill GEMM, k-quant decode boosts and fused kernels;
- host-resident MoE experts with an expert cache;
- an RDNA4 all-reduce for multi-GPU.

Every feature is on by default. The set needs no launch options or environment variables.

The toolchest cannot build it today. The curated-fork work (#494, `plan/curated-forks/`) does not fit it:

- A curated fork runs as a fork instance: one paired model, one whole GPU, no router. Its builds can never be the router's build, because the router writes launch options by upstream build number and a fork does not follow that number.
- rdna-boosts is upstream at a known commit plus patches. Its upstream build number is exact (release `v16-6050b467c-r206` is `b11887` plus 16 patches), so the router's option gating works for it. Running it as a fork instance would also lose the router, Autoconfigure, Autotune and benchmarks, which are where the speed-up matters.

So this plan builds rdna-boosts as a **patch set on the llama.cpp project**. A patched build is a llama.cpp build in every way the router, jobs and gating care about. The only differences are that its source has the patches applied and it records which patch set release it carries.

## Goals

- Add a short curated list of patch sets to the toolchest source, with rdna-boosts as the only entry. Each entry gives the patch repo, a pinned tested release tag and its expected git tree, the backends and the GPU architectures it is for.
- On the Builds page, offer each patch set as a choice in the project picker ("llama.cpp + RDNA boosts"), limited to ROCm profiles, with two releases: the pinned release (tested) and the newest release (untested). The build itself is a llama.cpp build.
- Build by checking out the release's upstream base commit, applying its patches with `git am`, and checking that the git tree equals the tree the release records. If the tree differs, the build fails.
- Record the upstream base's commit count as the build's version, so option gating, ranking and "newest build" see `b11887` and not the patched count.
- Let a patched build be the router's active build, and the build for benchmark, Autotune and Autoconfigure jobs, as any llama.cpp build is.
- Show which builds are patched, and with which release, wherever a build is listed or picked.

## Non-goals

- Patch sets for backends other than ROCm.
- Any patch repo or patch file the user gives. Only curated entries in the source.
- Beta patches (`apply-all.sh --include-beta`).
- Running the patch repo's own scripts. The toolchest applies the patches itself.
- Tracking the maintainer's fork branch (`stew675/llama.cpp`, branch `rdna-boosts`). It is force-pushed at every release and has no tags, so a pinned commit can stop being fetchable.
- Re-basing patches onto a newer upstream, or applying them with `git am -3` on a base that is not the one the release records.
- Launch options that only the patch set adds (`--spec-type draft-mtp-adaptive`, `--spec-draft-mtp-vocab`, `-sm relaxed-tensor`, `--host-experts`). Phase 05 lists them as a follow-up; they are not needed for the speed-up.
- Using the published containers (`ghcr.io/stew675/llama-cpp-rdna-boosts`).

## Users & primary flow

The user runs the toolchest on a Linux host with an RDNA GPU (asmodean: R9700, gfx1201) and serves models from the router.

1. On the Builds page the user chooses "llama.cpp + RDNA boosts" in the project picker and a ROCm profile. The ref field shows releases: "r206 (tested)" is selected; "Newest release (untested)" is the other choice.
2. The user builds. The log shows the base checkout, each patch applied, and "Tree matches release v16-6050b467c-r206". The build appears as `rdna-b11887-r206-rocm` with an "RDNA boosts r206" badge.
3. On the Server page the user selects that build as the router's build. The router starts with launch options for `b11887`.
4. Autoconfigure, Autotune and benchmarks can use that build. The results name it as they name any build.
5. To go back, the user selects an upstream build. Both kinds of build can stay on disk at the same time.

## Constraints

- Go backend in `internal/`, HTML templates with inline scripts in `web/templates/`, htmx. Follow the curated-fork and builder patterns (`internal/forks/`, `internal/builder/project.go`).
- Container and host installs must both work. Host mode runs under systemd with `ProtectHome`.
- `git am` needs a committer identity. The toolchest must set one (and the committer date) itself and must not depend on the host's git config. Fixed values also make the patched commit IDs match the release's `tip`.
- The patch repo publishes several releases a day (r202–r206 on 2026-11-01/02). The pinned release moves only on purpose, after a test.
- A release can move to a new upstream base (a re-base). Its `release.json` gives the base, so "newest release" must read the base from it and not assume the pinned one.
- The patches are validated on ROCm 7.14.1 and 10.1. asmodean's container uses ROCm 10.1.
- The set changes memory use: a 10% compute-buffer margin on HIP, a slab allocator for host-resident experts with about 1.3 GiB headroom, and an expert cache sized automatically. Autoconfigure's VRAM fit is measured on upstream builds and may be wrong for patched builds.
- The base is older than upstream `v0.8.0`. Options gated on a newer build (router memory, `b11930`) are off on patched builds, and that is correct.
- All UI text and tooltips follow the project's writing rules (short, ASD-STE100 style, no slang).

## Success criteria

- With the ROCm profile, the Builds page builds rdna-boosts at the pinned release. The build's git tree equals `release.json.tree` of that release, and its recorded version is the base's commit count (11887 for r206).
- A build with a tree that does not match fails with a clear message. No binary from it is installed.
- A patched build can be the router's build. The router starts, serves models, and gets only the launch options valid for its base version.
- Benchmark, Autotune and Autoconfigure jobs run on a patched build.
- Build cards, the router's build picker and job pickers show "RDNA boosts rN" on patched builds.
- Builds of upstream and patched source at the same base can exist together with different IDs. Building the same release and profile twice gives the existing rebuild prompt.
- On asmodean, a patched build is faster than upstream `b11887` on at least one MoE model and one dense model, measured with the toolchest's benchmark. The numbers are recorded in the field-test notes.
- `go build`, `go vet`, `go test ./...` and the JS tests pass in CI.

## Decisions

- **Fork or patch set** → a patch set on the llama.cpp project, not a curated fork. Its base is known exactly, so it can serve the router.
- **Where the source comes from** → upstream llama.cpp at the release's base, plus patches from the rdna-boosts repo at the release's tag. Not the maintainer's fork branch, which is force-pushed.
- **Proof of a correct apply** → the git tree after `git am` must equal `release.json.tree`. Strict `git am` only, no 3-way fallback.
- **Version for gating and ranking** → the base's commit count, not the patched count.
- **Which releases** → the pinned tested release, plus the newest release as an untested choice.
- **Where the choice is made** → a project picker entry per patch set. It reuses the picker's release list and backend filter; the build is recorded under the llama.cpp project.
- **Opt-in** → none. The picker entry allows only ROCm profiles. (Open: put it behind the "llama.cpp forks (experimental)" setting instead. Decide before Phase 03.)
- **Ranking against upstream** → a patched build ranks at its base, the same as an upstream build of that commit. "Auto" (no saved router build) skips patched builds while any upstream build can run, so building a patch set never changes the router's build unless the user chooses it.
- **Test host** → asmodean (R9700, gfx1201), the architecture the patch set is tuned on most.

## Phases

1. [Phase 01 — Patch set registry and build fields](phase-01-registry.md)
2. [Phase 02 — Apply and check a patch set in the builder](phase-02-builder-apply.md)
3. [Phase 03 — Builds page and build API](phase-03-builds-page.md)
4. [Phase 04 — Router, jobs and build pickers](phase-04-router-and-jobs.md)
5. [Phase 05 — Patch-set launch options (follow-up)](phase-05-launch-options.md)
6. [Phase 06 — Docs and field test](phase-06-docs-and-field-test.md)
