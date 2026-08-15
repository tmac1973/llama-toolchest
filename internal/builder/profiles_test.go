package builder

import "testing"

func optionFlags(profile string) map[string]BuildOption {
	out := map[string]BuildOption{}
	for _, o := range ProfileOptions(profile) {
		out[o.Flag] = o
	}
	return out
}

// Toggles removed in the audit must stay gone: they are no-ops on any
// ref users realistically build (removed upstream, or never a build flag
// at all — unified memory is a runtime environment variable).
func TestProfileOptionsExcludeRemovedFlags(t *testing.T) {
	removed := []string{
		"GGML_CUDA_F16",
		"LLAMA_HIP_UMA",
		"GGML_CUDA_ENABLE_UNIFIED_MEMORY",
		"GGML_CUDA_FORCE_CUBLAS_COMPUTE_16F",
		"CMAKE_HIP_FLAGS",
	}
	for _, profile := range []string{"cuda", "rocm", "vulkan", "metal", "cpu"} {
		opts := optionFlags(profile)
		for _, flag := range removed {
			if _, ok := opts[flag]; ok {
				t.Errorf("%s: removed toggle %s is still offered", profile, flag)
			}
		}
	}
}

// rocWMMA survives the audit deliberately: it's the fastest prompt
// processing on RDNA4 and the description carries the ref-pinning
// guidance (removed upstream at b10332).
func TestROCmKeepsRocWMMAToggle(t *testing.T) {
	opt, ok := optionFlags("rocm")["GGML_HIP_ROCWMMA_FATTN"]
	if !ok {
		t.Fatal("GGML_HIP_ROCWMMA_FATTN toggle missing from rocm profile")
	}
	if opt.Default {
		t.Error("rocWMMA must default off — it's a no-op on refs >= b10332")
	}
}

// Vulkan has no performance build flags upstream; its tuning is runtime
// env vars. Only the common toggles should appear — the old debug
// toggles live in Extra CMake flags now.
func TestVulkanOffersOnlyCommonToggles(t *testing.T) {
	vulkan := ProfileOptions("vulkan")
	common := ProfileOptions("cpu")
	if len(vulkan) != len(common) {
		t.Fatalf("vulkan should offer exactly the common toggles, got %d vs %d", len(vulkan), len(common))
	}
	for _, o := range vulkan {
		if o.Flag == "GGML_VULKAN_CHECK_RESULTS" || o.Flag == "GGML_VULKAN_VALIDATE" {
			t.Errorf("debug toggle %s should have been dropped", o.Flag)
		}
	}
}

// The audit's new toggles must actually be offered.
func TestAuditedTogglesPresent(t *testing.T) {
	cases := map[string][]string{
		"cuda": {"GGML_CUDA_FORCE_MMQ", "GGML_CUDA_FORCE_CUBLAS", "GGML_CUDA_FA_ALL_QUANTS",
			"GGML_CUDA_GRAPHS", "GGML_CUDA_NO_PEER_COPY", "GGML_LTO"},
		"rocm": {"GGML_CUDA_FORCE_MMQ", "GGML_CUDA_FA_ALL_QUANTS", "GGML_HIP_RCCL", "GGML_LTO"},
	}
	for profile, flags := range cases {
		opts := optionFlags(profile)
		for _, flag := range flags {
			if _, ok := opts[flag]; !ok {
				t.Errorf("%s: expected toggle %s", profile, flag)
			}
		}
	}
}

// GPU_TARGETS is the current documented name; upstream forwards
// AMDGPU_TARGETS to it, but new invocations should use the real one.
func TestROCmProfileUsesGPUTargets(t *testing.T) {
	prof, ok := FindProfile("rocm")
	if !ok {
		t.Fatal("rocm profile missing")
	}
	if _, ok := prof.CMakeFlags["GPU_TARGETS"]; !ok {
		t.Error("rocm profile should set GPU_TARGETS")
	}
	if _, ok := prof.CMakeFlags["AMDGPU_TARGETS"]; ok {
		t.Error("rocm profile should no longer set the legacy AMDGPU_TARGETS")
	}
}

// Saved overrides for removed toggles must be ignored harmlessly — the
// decision was to drop them silently, no migration.
func TestApplyOptionOverridesIgnoresStaleOverrides(t *testing.T) {
	flags := map[string]string{}
	ApplyOptionOverrides(flags, ProfileOptions("cuda"), map[string]bool{
		"GGML_CUDA_F16":       true, // removed toggle, stored by an old config
		"GGML_CUDA_FORCE_MMQ": true,
	})
	if _, ok := flags["GGML_CUDA_F16"]; ok {
		t.Error("override for a removed toggle must not reach the cmake flags")
	}
	if flags["GGML_CUDA_FORCE_MMQ"] != "ON" {
		t.Error("override for a live toggle should still apply")
	}
}
