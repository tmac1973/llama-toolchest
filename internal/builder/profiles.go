package builder

// BuildProfile defines cmake flags for a build configuration.
type BuildProfile struct {
	Name       string            `json:"name"`
	Backend    string            `json:"backend"` // "rocm", "cuda", "vulkan", "metal", "cpu"
	CMakeFlags map[string]string `json:"cmake_flags"`
}

// BuildOption describes a toggleable cmake flag for a profile.
// Value is the cmake value set when the option is enabled; empty means "ON".
type BuildOption struct {
	Flag        string `json:"flag"`
	Label       string `json:"label"`
	Description string `json:"description"`
	Default     bool   `json:"default"`
	Value       string `json:"value,omitempty"`
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
			Description: "Compile with -march=native for best performance on this machine. Disable if building for a different CPU.",
			Default:     true,
		},
		{
			Flag:        "LLAMA_OPENSSL",
			Label:       "HTTPS Support (OpenSSL)",
			Description: "Enable HTTPS for fetching remote images in vision models and downloading models by URL. Requires OpenSSL dev libraries.",
			Default:     true,
		},
		{
			Flag:        "GGML_LTO",
			Label:       "Link-Time Optimization",
			Description: "Optimize across compilation units at link time. Small speedup in CPU-side code at the cost of a noticeably longer build.",
			Default:     false,
		},
	}

	// Shared by the CUDA and HIP backends (the HIP backend compiles the
	// same kernel sources).
	faAllQuants := BuildOption{
		Flag:        "GGML_CUDA_FA_ALL_QUANTS",
		Label:       "FlashAttention for All KV Quants",
		Description: "Compile FlashAttention kernels for every KV-cache quantization combination. Enable if you run a quantized KV cache (cache type other than f16) — without it those combinations fall back to slower generic kernels. Significantly longer build time.",
		Default:     false,
	}
	forceMMQ := BuildOption{
		Flag:        "GGML_CUDA_FORCE_MMQ",
		Label:       "Force Custom Matrix Multiply (MMQ)",
		Description: "Always use llama.cpp's custom quantized matmul kernels instead of the BLAS library. Lower VRAM use; often faster on consumer GPUs, can be slower at large batch sizes on datacenter cards. Mutually exclusive with Force cuBLAS.",
		Default:     false,
	}

	switch profile {
	case "cuda":
		return append(common, []BuildOption{
			forceMMQ,
			{
				Flag:        "GGML_CUDA_FORCE_CUBLAS",
				Label:       "Force cuBLAS",
				Description: "Always use cuBLAS instead of the custom MMQ kernels. Can help prompt processing on datacenter GPUs with strong tensor cores. Mutually exclusive with Force MMQ.",
				Default:     false,
			},
			faAllQuants,
			{
				Flag:        "GGML_CUDA_GRAPHS",
				Label:       "CUDA Graphs",
				Description: "Batch kernel launches into CUDA graphs, cutting per-token launch overhead during generation. Recent refs leave this off by default for llama.cpp; if it misbehaves at runtime, the GGML_CUDA_DISABLE_GRAPHS variable under Settings switches it off without rebuilding.",
				Default:     false,
			},
			{
				Flag:        "GGML_CUDA_NO_PEER_COPY",
				Label:       "Disable Peer-to-Peer Copies",
				Description: "Route multi-GPU transfers through system memory instead of direct GPU-to-GPU copies. Fixes corrupt or garbage output on multi-GPU boards whose PCIe topology lacks real peer-to-peer support (common on consumer chipsets). Leave off unless you see that symptom.",
				Default:     false,
			},
		}...)
	case "rocm":
		return append(common, []BuildOption{
			forceMMQ,
			faAllQuants,
			{
				Flag:        "GGML_HIP_RCCL",
				Label:       "RCCL Collectives (Tensor Parallelism)",
				Description: "Build against RCCL, ROCm's collective communications library, for the AllReduce that fires every layer under tensor parallelism. Without it the build falls back to a slower generic path (the 'falling back to meta-backend butterfly' message in the logs). Only matters for split-mode tensor across 2+ GPUs; requires rccl-devel to build.",
				Default:     false,
			},
			{
				Flag:        "GGML_HIP_ROCWMMA_FATTN",
				Label:       "rocWMMA FlashAttention (older refs only)",
				Description: "Builds the FlashAttention kernel against rocWMMA. Upstream REMOVED this path at b10332 (PR #26046) — on b10332 or newer this toggle is a silent no-op and the native MMA kernel is always used. That native kernel regresses prompt processing on RDNA4 (gfx1201) by 25-49% at deep context (upstream issue #26220, still open); rocWMMA measured +24% prompt processing on Qwen3.6-27B-MTP here. To use it, pin a ref older than b10332 and enable this. Requires rocwmma-devel to build and --flash-attn at runtime.",
				Default:     false,
			},
		}...)
	case "vulkan":
		// No Vulkan-specific toggles: upstream has no performance build
		// flags for Vulkan — its tuning levers are runtime environment
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

// DefaultProfiles returns built-in profiles for each backend.
// The ROCm profile auto-detects GPU targets from rocminfo.
func DefaultProfiles() []BuildProfile {
	// Detect GPU targets for ROCm
	gpuTargets := "gfx1100" // fallback
	for _, b := range DetectBackends() {
		if b.Name == "rocm" && len(b.GPUs) > 0 {
			gpuTargets = uniqueJoin(b.GPUs, ";")
			break
		}
	}

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
// option is enabled per its Default unless toggled in overrides, and
// enabled options are set to their CMakeValue ("ON" unless the option
// carries an explicit Value). Single source of truth shared by the
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
