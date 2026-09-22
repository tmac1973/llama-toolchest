# Phase 05 findings — end-to-end verification

Completed 21–22 September 2026. Every step except 10 was run; step 10 was
skipped deliberately, for the reason given at the end.

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
| 7 · build stamp recorded and the flag fires | pass |
| 8 · model loads and generates | pass, with throughput measured below |
| 9 · flagged build still activates | pass — allowed, then failed exactly as the tooltip predicted |
| 10 · a second ROCm line | not run; see below |
| 11 · return to stable, nothing lost | pass — the ROCm 10 build survived and is flagged |

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

## The comparison, after switching back to stable

Same model, same llama.cpp ref (`v0.4.1`, commit b29c606e2), same cmake flags,
same measurement method, same machine:

| | ROCm 10.0.0 | ROCm 7.2.4 |
|---|---|---|
| generation | 116.3 tok/s | **116.3 tok/s** |
| prompt processing | 5,732 tok/s | 5,306 tok/s |
| generation runs | 115.56 / 116.32 / 116.30 | 115.39 / 116.26 / 116.46 |
| prompt runs | 3,672 / 5,732 / 5,855 | 2,829 / 5,306 / 5,696 |

**Generation is identical** — 116.3 tok/s on both, and the individual runs
interleave. There is nothing to choose between them.

**Prompt processing looks ~8% better on ROCm 10, and that should not be
believed.** Look at the run spread: 2,829 to 5,696 on stable and 3,672 to 5,855
on ROCm 10. The first run of each set is a warm-up and the remaining two differ
by more than the gap between the two medians. Three samples cannot separate an
8% difference from that much variance. The honest reading is no measurable
difference in either direction.

One caveat that cannot be removed by more samples: the two containers use
different compilers — gcc 15.3.1 on Fedora, gcc 13.3.0 on Ubuntu 24.04. This is
a comparison of two *images*, not purely of two ROCm versions.

The useful conclusion for the documentation is the plain one: on this hardware
ROCm 10 is neither faster nor slower in any way these measurements can detect.

## The build stamp, verified for real

Everything in Phase 04's tests was constructed. This is the first time the
stamp has been written by an actual build:

```
Running rocm 10.0.0 · built on docker.io/rocm/dev-ubuntu-24.04:10.0.0-full
v0.4.1-rocm-rocm-optimized   built_against='rocm 10.0.0'   not flagged
```

The banner names the toolchain and the base image, the build recorded its own
stamp, and it is correctly *not* flagged because it matched what was running at
the time. The flag itself is covered further down, once a second ROCm line
existed to compare against.

## A plan step was skipped, and it improved the order

Step 2 asked for a stamped ROCm 7.2.4 build *before* switching, so the mismatch
flag would have something to act on. The switch happened first, so no 7.2.4
build exists in this container.

That turned out better. The ROCm 10 build became the thing that got flagged when
the container switched back, so returning to stable verified steps 7, 9 and 11
at once — the flag firing on a real build, a flagged build still activating, and
nothing being deleted. One install instead of two.

## Two failures on the way, both fixed and both found only by running it

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


## Steps 9 and 11, verified together

Switching back to stable did the work of three steps at once, as expected once
step 2 had been skipped:

**The flag fires on a real build.** With the container on ROCm 7.2.4, the
ROCm 10 build is marked, and the tooltip reads:

> Built against rocm 10.0.0; this container runs rocm 7.2.4. A llama-server
> built against one rocm version does not run under another, so this build needs
> rebuilding before it will load. Nothing has been deleted — it is still here if
> you switch back.

Exactly one build is flagged. The banner now reads `Running rocm 7.2.4` with no
"built on" fragment, because the Fedora image sets no base-image variable —
the case tested in Phase 04 and now confirmed in reality.

**A flagged build can still be activated, and fails as predicted.** Starting the
router with it produced:

```
llama-server: error while loading shared libraries: libhipblas.so.3
==> Router exited with error: exit status 127
```

The app did not block the choice, and the failure is the one the tooltip
described. That is the whole design: inform, do not prevent.

**Nothing was deleted.** `v0.4.1-rocm-rocm-optimized` is still listed with its
stamp intact after the switch.

## Why Dockerfile.rocm does not need the linker fix

Worth recording, because the same error string appears in two unrelated places
and the obvious explanation is wrong.

Neither image has an `/etc/ld.so.conf.d` entry for ROCm — the Fedora image's
directory is empty, and `ldconfig -p` knows no ROCm library in either. So the
earlier claim that "the Fedora image never had this problem because the ROCm
RPMs ship the ld.so.conf.d entry" was wrong about the mechanism.

The real difference is what cmake bakes into `libggml-hip.so`:

| image | RUNPATH of libggml-hip.so |
|---|---|
| Fedora, ROCm 7.2.4 | `[/data/llama.cpp/build-rocm/bin::/opt/rocm/lib]` |
| Ubuntu, ROCm 10 | `[/data/llama.cpp/build-rocm/bin:]` |

Under ROCm 10 the ROCm library directory is absent from the runpath — note the
empty entry where it should be, and that `/opt/rocm/lib` there is a symlink
through `/etc/alternatives`. So the stable image works by accident of what cmake
records, and `Dockerfile.rocm-next` needed the `ld.so.conf.d` entry to stop
depending on that. `Dockerfile.rocm` is left alone: it works, and adding the
entry there would be a change with no failure to justify it.

## Step 10 not run

Rebuilding on `7.14.1-full` was not exercised end to end. Phase 02's test 6
already built that base and confirmed it reports `7.14.1`, so the version
argument is proven; what step 10 would add is a third full install cycle for
little more information. Worth doing if the 7.14 line is ever recommended to
anyone.

## State left on the machine

- The container is on the **stable** variant, with the router running the
  `v0.4.1-rocm-stable-cmp` build.
- Two builds exist: `v0.4.1-rocm-rocm-optimized` (ROCm 10, flagged) and
  `v0.4.1-rocm-stable-cmp` (ROCm 7.2.4, in use). Both were made during
  verification; either can be deleted.
- `.env` records the experimental variant, so a bare `./setup.sh install` will
  offer `next` as its default.
