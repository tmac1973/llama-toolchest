package builder

import (
	"fmt"
	"strings"
)

// BuildProfile defines cmake flags for a build configuration.
type BuildProfile struct {
	Name       string            `json:"name"`
	Backend    string            `json:"backend"` // "rocm", "cuda", "vulkan", "metal", "cpu"
	CMakeFlags map[string]string `json:"cmake_flags"`
}

// BuildOption describes a toggleable cmake flag for a profile.
// Value is the cmake value set when the option is enabled; empty means "ON".
// OffValue is the cmake value set when the option is disabled; empty
// means the flag is left out, so cmake's own default applies. An option
// that llama.cpp turns on by default needs OffValue, or turning it off
// would change nothing.
type BuildOption struct {
	Flag        string `json:"flag"`
	Label       string `json:"label"`
	Description string `json:"description"`
	Default     bool   `json:"default"`
	Value       string `json:"value,omitempty"`
	OffValue    string `json:"off_value,omitempty"`
}

// CMakeValue returns the value assigned to Flag when the option is enabled.
func (o BuildOption) CMakeValue() string {
	if o.Value != "" {
		return o.Value
	}
	return "ON"
}

// ProfileOptions returns the toggleable build options for a given profile.
//
// The bar for a toggle: a real performance or compatibility payoff a
// non-expert might want on a ref they'd realistically build. Debug-only
// flags and flags removed upstream stay out (Extra CMake flags covers
// them); the one deliberate exception is GGML_HIP_ROCWMMA_FATTN, kept
// because it remains the fastest prompt-processing path on RDNA4 and its
// description carries the ref-pinning guidance. Removed toggles
// (GGML_CUDA_F16, LLAMA_HIP_UMA, the build-flag spelling of unified
// memory, GGML_CUDA_FORCE_CUBLAS_COMPUTE_16F, HIP fast-math) need no
// migration: ApplyOptionOverrides ignores stored overrides for flags no
// longer listed, and all of them were no-ops on current refs anyway.
func ProfileOptions(profile string) []BuildOption {
	common := []BuildOption{
		{
			Flag:        "GGML_NATIVE",
			Label:       "Native CPU Optimizations",
			Description: "Compiles with -march=native for this CPU. Turn it off to build for another CPU.",
			Default:     true,
		},
		{
			Flag:        "LLAMA_OPENSSL",
			Label:       "HTTPS Support (OpenSSL)",
			Description: "HTTPS for remote images and model downloads by URL. Needs the OpenSSL development libraries.",
			Default:     true,
		},
		{
			Flag:        "GGML_LTO",
			Label:       "Link-Time Optimization",
			Description: "Link-time optimization. A small CPU speedup and a longer build.",
			Default:     false,
		},
	}

	// Shared by the CUDA and HIP backends (the HIP backend compiles the
	// same kernel sources).
	faAllQuants := BuildOption{
		Flag:        "GGML_CUDA_FA_ALL_QUANTS",
		Label:       "FlashAttention for All KV Quants",
		Description: "Compiles FlashAttention kernels for all K/V cache type pairs. q8_0 and q4_0 are already compiled. Needed only for other or mixed types. Much longer build.",
		Default:     false,
	}
	forceMMQ := BuildOption{
		Flag:        "GGML_CUDA_FORCE_MMQ",
		Label:       "Force Custom Matrix Multiply (MMQ)",
		Description: "Uses ggml MMQ kernels for quantized weights, not BLAS. Less VRAM, often faster on consumer GPUs. Not with Force cuBLAS.",
		Default:     false,
	}

	switch profile {
	case "cuda":
		return append(common, []BuildOption{
			forceMMQ,
			{
				Flag:        "GGML_CUDA_FORCE_CUBLAS",
				Label:       "Force cuBLAS",
				Description: "Uses cuBLAS, not MMQ kernels. Can help large batches on datacenter GPUs. Not with Force MMQ.",
				Default:     false,
			},
			faAllQuants,
			{
				Flag:        "GGML_CUDA_GRAPHS",
				Label:       "CUDA Graphs",
				Description: "Groups kernel launches into CUDA graphs. On by default. Without it, Wan video models can crash in the decode. GGML_CUDA_DISABLE_GRAPHS turns it off at runtime.",
				// llama.cpp's top-level CMakeLists turns GGML_CUDA_GRAPHS on
				// when it is not set, so off must be written out.
				Default:  true,
				OffValue: "OFF",
			},
			{
				Flag:        "GGML_CUDA_NO_PEER_COPY",
				Label:       "Disable Peer-to-Peer Copies",
				Description: "Sends multi-GPU transfers through system memory. Fixes bad output on boards without real P2P support. Leave it off unless you see bad output.",
				Default:     false,
			},
		}...)
	case "rocm":
		opts := append(common, []BuildOption{
			forceMMQ,
			faAllQuants,
			{
				Flag:        "GGML_HIP_RCCL",
				Label:       "RCCL Collectives (Tensor Parallelism)",
				Description: "Builds with RCCL for the tensor-parallel AllReduce. Without it, two GPUs use the internal AllReduce and other counts a slower path. Needs rccl-devel.",
				Default:     false,
			},
			{
				Flag:        "GGML_HIP_ROCWMMA_FATTN",
				Label:       "rocWMMA FlashAttention (older refs only)",
				Description: "Builds FlashAttention with rocWMMA. Upstream removed it at b10332, so it works only on older refs. It was faster for prompt processing on RDNA4. Needs rocwmma-devel and flash attention.",
				Default:     false,
			},
		}...)
		if opt := igpuTargetOption(detectedROCmGPUs()); opt != nil {
			opts = append(opts, *opt)
		}
		return opts
	case "vulkan":
		// No Vulkan-specific toggles: upstream has no performance build
		// flags for Vulkan — its tuning options are runtime environment
		// variables (GGML_VK_DISABLE_COOPMAT, GGML_VK_FORCE_MAX_ALLOCATION_SIZE
		// under Settings). The debug flags (GGML_VULKAN_CHECK_RESULTS,
		// GGML_VULKAN_VALIDATE) are reachable via Extra CMake flags.
		return common
	case "metal":
		return common
	case "cpu":
		return common
	default:
		return common
	}
}

// detectedROCmGPUs returns the gfx targets rocminfo reports, in
// enumeration order (duplicates included — multi-GPU boxes repeat the
// arch).
func detectedROCmGPUs() []string {
	for _, b := range DetectBackends() {
		if b.Name == "rocm" {
			return b.GPUs
		}
	}
	return nil
}

// splitROCmTargets partitions detected gfx targets into discrete-GPU and
// iGPU lists, deduplicated, order preserved.
func splitROCmTargets(gpus []string) (dgpu, igpu []string) {
	seen := map[string]bool{}
	for _, g := range gpus {
		if seen[g] {
			continue
		}
		seen[g] = true
		if IsIGPUArch(g) {
			igpu = append(igpu, g)
		} else {
			dgpu = append(dgpu, g)
		}
	}
	return dgpu, igpu
}

// rocmGPUTargets returns the default GPU_TARGETS value: discrete GPUs
// only when any exist — the iGPU on such a box is a tiny memory
// carve-out most users don't want kernels for, and the Include iGPU
// Build Target toggle opts back in. APU-only boxes (a Strix Halo) keep
// their iGPU targets, since excluding them would leave nothing to build.
func rocmGPUTargets(gpus []string) string {
	dgpu, igpu := splitROCmTargets(gpus)
	if len(dgpu) > 0 {
		return strings.Join(dgpu, ";")
	}
	if len(igpu) > 0 {
		return strings.Join(igpu, ";")
	}
	return "gfx1100" // fallback when detection fails
}

// igpuTargetOption returns the Include iGPU Build Target toggle, or nil
// when the box doesn't have both a discrete GPU and an iGPU (nothing to
// opt into). When enabled it overrides the profile's dGPU-only
// GPU_TARGETS with the full detected list.
func igpuTargetOption(gpus []string) *BuildOption {
	dgpu, igpu := splitROCmTargets(gpus)
	if len(dgpu) == 0 || len(igpu) == 0 {
		return nil
	}
	return &BuildOption{
		Flag:  "GPU_TARGETS",
		Value: strings.Join(append(dgpu, igpu...), ";"),
		Label: "Include iGPU Build Target",
		Description: fmt.Sprintf("Also compiles kernels for the integrated GPU (%s). Off by default: the iGPU is much slower, and some ROCm libraries have no kernels for it.",
			strings.Join(igpu, ", ")),
		Default: false,
	}
}

// DefaultProfiles returns built-in profiles for each backend.
// The ROCm profile auto-detects GPU targets from rocminfo.
func DefaultProfiles() []BuildProfile {
	gpuTargets := rocmGPUTargets(detectedROCmGPUs())

	return []BuildProfile{
		{
			Name:    "rocm",
			Backend: "rocm",
			CMakeFlags: map[string]string{
				"GGML_HIP": "ON",
				// GPU_TARGETS is the current documented name; upstream
				// still forwards the older AMDGPU_TARGETS spelling to it.
				"GPU_TARGETS":      gpuTargets,
				"CMAKE_BUILD_TYPE": "Release",
			},
		},
		{
			Name:    "cuda",
			Backend: "cuda",
			CMakeFlags: map[string]string{
				"GGML_CUDA":        "ON",
				"CMAKE_BUILD_TYPE": "Release",
			},
		},
		{
			Name:    "vulkan",
			Backend: "vulkan",
			CMakeFlags: map[string]string{
				"GGML_VULKAN":      "ON",
				"CMAKE_BUILD_TYPE": "Release",
			},
		},
		{
			Name:    "metal",
			Backend: "metal",
			CMakeFlags: map[string]string{
				"GGML_METAL":               "ON",
				"GGML_METAL_EMBED_LIBRARY": "ON",
				"CMAKE_BUILD_TYPE":         "Release",
			},
		},
		{
			Name:    "cpu",
			Backend: "cpu",
			CMakeFlags: map[string]string{
				"CMAKE_BUILD_TYPE": "Release",
			},
		},
	}
}

// ApplyOptionOverrides folds a profile's build options into flags: each
// option is enabled per its Default unless toggled in overrides, enabled
// options are set to their CMakeValue ("ON" unless the option carries an
// explicit Value), and disabled options with an OffValue are set to it. Single source of truth shared by the
// actual build (Builder.Build) and the flag preview (api effectiveCMakeFlags)
// so the two can't diverge.
func ApplyOptionOverrides(flags map[string]string, options []BuildOption, overrides map[string]bool) {
	for _, opt := range options {
		enabled := opt.Default
		if overrides != nil {
			if v, ok := overrides[opt.Flag]; ok {
				enabled = v
			}
		}
		if enabled {
			flags[opt.Flag] = opt.CMakeValue()
		} else if opt.OffValue != "" {
			flags[opt.Flag] = opt.OffValue
		}
	}
}

// FindProfile returns the profile matching the given name.
func FindProfile(name string) (BuildProfile, bool) {
	for _, p := range DefaultProfiles() {
		if p.Name == name {
			return p, true
		}
	}
	return BuildProfile{}, false
}

// uniqueJoin deduplicates strings and joins with sep.
func uniqueJoin(items []string, sep string) string {
	seen := map[string]bool{}
	var out []string
	for _, s := range items {
		if !seen[s] {
			seen[s] = true
			out = append(out, s)
		}
	}
	result := ""
	for i, s := range out {
		if i > 0 {
			result += sep
		}
		result += s
	}
	return result
}
