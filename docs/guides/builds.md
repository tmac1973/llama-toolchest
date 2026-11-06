# Builds

Builds compile `llama-server` from a `ggml-org/llama.cpp` ref for one backend. Several builds can be kept; choose one on the Server tab.

## Profiles { #profiles }

Only backends found on this system show. **CUDA**: NVIDIA; needs the CUDA toolkit. **ROCm**: AMD; toolchest detects the GPU arch (set `GPU_TARGETS` in Extra CMake Flags to change it). With more than one ROCm, a *ROCm Install* picker shows. **Vulkan**: any vendor; host installs only. **Metal**: macOS. **CPU**: always available.

## Ref { #ref }

A release tag (`b*`) or `latest`. rocWMMA FlashAttention was removed at `b10332`, but it is still the fastest prompt reading on RDNA4. To use it, pin an older ref.

## Options and Extra CMake Flags { #options-and-extra-cmake-flags }

Each toggle says when to use it. Extra CMake Flags takes anything else that CMake accepts; the preview shows the final flags. Vulkan tuning is in the runtime environment in Settings.

## Saved flag sets { #saved-flag-sets }

Enter a *Build Tag* and press *Save Flags* to keep toggles and extra flags under that name, then replay them on any ref.

## llama.cpp + RDNA boosts { #patch-sets }

[RDNA boosts](https://github.com/stew675/llama-cpp-rdna-boosts) is a set of patches on upstream llama.cpp. It makes AMD RDNA2, RDNA3 and RDNA4 GPUs faster. The patches apply to an older upstream commit, the *base*. All features are on, with no extra options. ROCm only.

1. **Build it**: on Builds, choose *llama.cpp + RDNA boosts* and a ROCm profile. Then choose a release. The build gets a name like `rdna-b11887-r206-rocm` and an *RDNA boosts* badge.
2. **Use it**: on the Server tab, choose the build as the router's build. To go back, choose an upstream build. *Auto* does not choose a patched build while an upstream build can run.

- **Tested**: the release that this toolchest version was tested with. **Untested**: a newer release. It usually works, but nobody tested it with this toolchest. The list shows the newest release and the ten most recent ones after the first build.
- The build fails if the patched source is not exactly the source of the release.
- The build counts as its base (for example `b11887`). Toolchest does not write launch options that are newer than the base. New upstream options and new model types arrive only when the patch set moves to a newer base.
- Benchmarks, Autotune and Autoconfigure can use a patched build like any build. To compare, run a benchmark on the patched build and on the upstream build of the same base.
