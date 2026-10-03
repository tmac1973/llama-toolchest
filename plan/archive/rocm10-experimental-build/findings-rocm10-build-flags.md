# Does llama.cpp need different build flags for ROCm 10?

Researched 22 September 2026 against the actual checkout being built —
`/data/llama.cpp` at `v0.4.1` (commit b29c606e2) — rather than against
documentation, which for HIP options tends to lag the source.

## Short answer

**No.** Nothing in this llama.cpp needs a different flag under ROCm 10, nothing
is version-gated off, and the flags this project already passes are the right
ones. One option is worth trying, and it is not ROCm-10-specific.

## Every HIP option this tree has

```
option(GGML_HIP                  "ggml: use HIP"                          OFF)
option(GGML_HIP_GRAPHS           "ggml: use HIP graph"                     ON)
option(GGML_HIP_RCCL             "ggml: use ROCm Collective Comm. Library" OFF)
option(GGML_HIP_NO_VMM           "ggml: do not try to use HIP VMM"         ON)
option(GGML_HIP_MMQ_MFMA         "ggml: enable MFMA MMA for CDNA in MMQ"   ON)
option(GGML_HIP_EXPORT_METRICS   "ggml: enable kernel perf metrics output" OFF)
```

What this project passes for the rocm profile: `GGML_HIP=ON`,
`GGML_NATIVE=ON`, `GPU_TARGETS=gfx1201`, `CMAKE_BUILD_TYPE=Release`,
`LLAMA_OPENSSL=ON`. Taking the rest in turn:

- **`GGML_HIP_GRAPHS` is already ON by default.** This is the one that would
  have mattered most for the speculative-decoding workload — HIP graphs cut
  per-launch overhead, and spec decoding is launch-heavy. Nothing to do.
- **`GGML_HIP_MMQ_MFMA` is CDNA-only.** MFMA is a matrix instruction the
  datacenter parts have and RDNA does not. Irrelevant on gfx1201 either way.
- **`GGML_HIP_RCCL` is already exposed** as a toggle on the Builds page. It is
  for multi-GPU collectives; one card, nothing to gain.
- **`GGML_HIP_EXPORT_METRICS`** emits kernel performance counters. A debugging
  aid, not something to ship on.
- **`GGML_HIP_NO_VMM` is the one worth trying.** See below.
- **`GGML_HIP_ROCWMMA_FATTN` no longer exists** in this tree, confirming what
  the build profile's own comment says: upstream removed it at b10332. The
  toggle the Builds page still offers is a no-op against any ref at or after
  that, which the profile description already states.

## Nothing is version-gated off under ROCm 10

ROCm 10.0.0 ships HIP 7.15, so `HIP_VERSION` is about 71500000. The gates in
the HIP backend are:

```
HIP_VERSION >= 60200000      (4 sites)
HIP_VERSION >= 60500000      (1 site)
```

Both are comfortably satisfied, so every version-gated path is active. There is
no upper bound anywhere, so nothing is disabled for being *too* new either.

One thing that looks like a bug and is not: `hip.h:171` reads
`#endif // HIP_VERSION >= 6050000` — a digit short of the `60500000` in the
matching `#if` on line 159. It is only in the comment. The directive itself is
correct.

RDNA 4 is properly handled, not an afterthought: 47 references to `RDNA4` in the
HIP/CUDA backend sources, alongside the CDNA generations.

## The one option worth an experiment

`GGML_HIP_NO_VMM` defaults to **ON**, which *disables* HIP's virtual-memory
allocator — `GGML_USE_VMM` is then undefined and the backend falls back to
ordinary allocation. The VMM path is a pooled allocator that grows virtual
address space rather than reallocating, so it mainly affects fragmentation and
large-context behaviour rather than raw throughput.

Two reasons it is interesting here and not elsewhere:

1. This machine runs a 9B model at 131,072 context in 16 GiB, which is exactly
   the fragmentation-sensitive case.
2. The default is conservative because VMM support has historically been patchy
   across ROCm versions. ROCm 10 is the newest runtime available, so if it is
   ever going to work well, it is here.

Building with `GGML_HIP_NO_VMM=OFF` would test it. It is not a ROCm 10
requirement and not a likely throughput win — it is a memory-behaviour
experiment, and the honest expectation is no measurable difference.

Worth noting that neither `GGML_HIP_GRAPHS` nor `GGML_HIP_NO_VMM` is reachable
from the Builds page today. Of the two, `NO_VMM` is the only one with a reason
to be changed, and exposing it would be a small addition to
`internal/builder/profiles.go` if the experiment turned up anything.
