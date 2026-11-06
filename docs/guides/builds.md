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
