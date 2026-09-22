# Phase 05 findings — end-to-end verification

In progress, 21–22 September 2026. Steps 3–8 are done; 9–11 need an interactive
install and are outstanding.

Machine: Radeon RX 9070 XT (gfx1201, RDNA 4), kernel `7.2.3-1-cachyos`, podman
6.1.1, 16 GiB VRAM.

## The headline

**ROCm 10 works on this machine.** llama.cpp builds against it, the router
starts, a model loads, and it generates. Nothing in the experimental path has
turned out to be fundamentally broken.

| step | result |
|---|---|
| 3 · install via the real route | pass, with `--from-source` |
| 4 · image and app start | pass — the Builds page confirms the tree it runs |
| 5 · GPU visible in the container | pass — gfx1201, "AMD Radeon RX 9070 XT" |
| 6 · `GGML_HIP` build of llama.cpp | **pass — about 106 seconds** |
| 7 · build stamp recorded | pass — see below; the mismatch *flag* is still untested |
| 8 · model loads and generates | pass, with throughput measured below |
| 9 · flagged build still activates | not yet run |
| 10 · a second ROCm line | not yet run |
| 11 · return to stable, nothing lost | not yet run |

## Throughput on ROCm 10

Model `Qwen3.5-4B-UD-Q4_K_XL`, llama.cpp `0.4.1-dev` (build 10964, commit
b29c606e2) compiled in-container with `GGML_HIP=ON`, context 8,192, flash
attention on, all layers on the GPU, 8 threads. Figures are llama.cpp's own
`timings`, not wall-clock.

| | median | runs |
|---|---|---|
| prompt processing | **5,732 tok/s** | 3,672 / 5,732 / 5,855 over ~2,430-token prompts |
| generation | **116.3 tok/s** | 115.56 / 116.32 / 116.30 over 256 tokens |

Two notes on method, because the first attempt produced nonsense:

- **Prompt caching has to be defeated.** Repeating the same long prompt made
  `prompt_n` collapse from 2,418 to 4 — the server was reusing the cache and the
  "measurement" was of nothing. Each run now uses a distinct prompt and passes
  `cache_prompt: false`.
- **The first prompt-processing run is a warm-up.** 3,672 tok/s against 5,732
  and 5,855 for the two after it. The median is reported rather than the mean
  for that reason.

Generation is remarkably stable — three runs within 0.8 tok/s of each other.

**There is no ROCm 7.2.4 comparison yet, and this is not a like-for-like number
without one.** Getting it needs step 11: switch back to stable, build llama.cpp
at the same ref, and measure the same model again. Until then this figure says
only "ROCm 10 performs reasonably", not "ROCm 10 is faster or slower than 7.2.4".

## The build stamp, verified for real

Everything in Phase 04's tests was constructed. This is the first time the
stamp has been written by an actual build:

```
Running rocm 10.0.0 · built on docker.io/rocm/dev-ubuntu-24.04:10.0.0-full
v0.4.1-rocm-rocm-optimized   built_against='rocm 10.0.0'   not flagged
```

The banner names the toolchain and the base image, the build recorded its own
stamp, and it is correctly *not* flagged because it matches what is running.
The mismatch flag itself has not fired yet — nothing on this machine is stamped
with a different ROCm line.

## A plan step was skipped, and it improved the order

Step 2 asked for a stamped ROCm 7.2.4 build *before* switching, so the mismatch
flag would have something to act on. The switch happened first, so no 7.2.4
build exists in this container.

That turns out to be better. The ROCm 10 build is now the thing that will be
flagged when the container switches to another line, so step 11 — returning to
stable — verifies steps 7, 9 and 11 at once: the flag fires on a real build, a
flagged build can still be activated, and nothing was deleted. One install
instead of two.

## Two failures on the way, both fixed

**The image could not run what it built.** The first container built llama.cpp
successfully and then could not start it:
`libhipblas.so.3: cannot open shared object file`. AMD's image ships no
`ld.so.conf.d` entry for ROCm, and `DT_RUNPATH` is not consulted for transitive
dependencies, so `libggml-hip.so` could not find hipBLAS however well
`llama-server` was linked. Fixed in `Dockerfile.rocm-next`; recorded in the
Phase 02 findings along with the test-plan gap that allowed it.

**The container ran the last release, not this tree.** Every container
Dockerfile installs the released package, so the first working container had
neither the variant selector nor the build stamp in it. `--from-source` now
works in container mode; step 3 of this phase requires it.
