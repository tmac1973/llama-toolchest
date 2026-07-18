package config

import (
	"fmt"
	"sort"
	"strings"
)

// RuntimeEnvOption is a curated environment variable that measurably
// affects llama-server. Deliberately a fixed list rather than a free-form
// map: most of the variables that circulate in forum threads are either
// no-ops for llama.cpp or workarounds for other stacks, and a free-form
// box invites pasting them in.
//
// These apply to the router process as a whole, not per model — the
// router hosts every model in one process, so per-model env is not
// expressible.
type RuntimeEnvOption struct {
	Name    string
	Label   string
	Help    string
	Values  []string // allowed values; empty means free text
	Backend string   // "rocm", "cuda", or "" for all
}

// RuntimeEnvOptions is the curated set.
//
// What is deliberately absent matters as much as what's here.
// HSA_NO_SCRATCH_RECLAIM and HSA_ENABLE_SDMA circulate as llama.cpp
// tuning knobs but have no supporting evidence — they come from PyTorch
// and container-stability threads. HSA_OVERRIDE_GFX_VERSION is a
// deployment concern for unsupported GPUs, already handled by setup.sh,
// and setting it on a natively supported card (gfx1030, gfx110x,
// gfx120x, gfx90a, gfx942) is actively harmful.
func RuntimeEnvOptions() []RuntimeEnvOption {
	return []RuntimeEnvOption{
		{
			Name:  "GGML_CUDA_CUBLAS_COMPUTE_TYPE",
			Label: "cuBLAS/hipBLAS compute type",
			Help: "Accumulator precision for GEMMs that fall through to cuBLAS/hipBLAS. " +
				"Replaces the old GGML_CUDA_FORCE_CUBLAS_COMPUTE_16F build flag, which current llama.cpp no longer defines. " +
				"Leave on auto unless you are measuring a specific difference.",
			Values: []string{"", "auto", "f16", "bf16", "f32"},
		},
		{
			Name:  "ROCBLAS_USE_HIPBLASLT",
			Label: "Prefer hipBLASLt (ROCm)",
			Help: "Routes rocBLAS calls through hipBLASLt where possible. " +
				"Note this is already the default on gfx12 (RDNA4), where setting it changes nothing, " +
				"and llama.cpp's own source reports a hipBLASLt crash on gfx942 (CDNA3). " +
				"Most quantized matmuls bypass rocBLAS entirely, so measure before keeping it.",
			Values:  []string{"", "1", "0"},
			Backend: "rocm",
		},
		{
			Name:  "GGML_CUDA_DISABLE_GRAPHS",
			Label: "Disable CUDA/HIP graphs",
			Help: "Graphs are enabled by default and generally help token generation. " +
				"This is a compatibility and debugging lever — expect a small slowdown, not a speedup.",
			Values: []string{"", "1"},
		},
		{
			Name:  "GGML_CUDA_ENABLE_UNIFIED_MEMORY",
			Label: "Unified memory (VRAM overflow)",
			Help: "Lets allocations spill into system RAM instead of failing when VRAM runs out. " +
				"A survivability option, not a performance one — spilled layers are much slower.",
			Values: []string{"", "1"},
		},
	}
}

// ValidateRuntimeEnv checks names are known and values are allowed.
func ValidateRuntimeEnv(env map[string]string) error {
	allowed := make(map[string]RuntimeEnvOption, len(RuntimeEnvOptions()))
	for _, o := range RuntimeEnvOptions() {
		allowed[o.Name] = o
	}

	names := make([]string, 0, len(env))
	for k := range env {
		names = append(names, k)
	}
	sort.Strings(names)

	for _, name := range names {
		opt, ok := allowed[name]
		if !ok {
			return fmt.Errorf("unknown runtime environment variable %q", name)
		}
		v := strings.TrimSpace(env[name])
		if v == "" || len(opt.Values) == 0 {
			continue
		}
		valid := false
		for _, allowedValue := range opt.Values {
			if v == allowedValue {
				valid = true
				break
			}
		}
		if !valid {
			return fmt.Errorf("%s: %q is not one of %s", opt.Label, v, strings.Join(opt.Values[1:], ", "))
		}
	}
	return nil
}

// RuntimeEnvPairs renders the configured variables as KEY=VALUE strings,
// skipping blanks so an unset option means "don't touch the environment"
// rather than "set it to empty". Sorted for a stable command line.
func (c *Config) RuntimeEnvPairs() []string {
	if len(c.RuntimeEnv) == 0 {
		return nil
	}
	out := make([]string, 0, len(c.RuntimeEnv))
	for k, v := range c.RuntimeEnv {
		if strings.TrimSpace(v) == "" {
			continue
		}
		out = append(out, k+"="+v)
	}
	sort.Strings(out)
	return out
}
