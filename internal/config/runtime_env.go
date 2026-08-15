package config

import (
	"fmt"
	"sort"
	"strings"
)

// RuntimeEnvOption is a curated environment variable that measurably
// affects llama-server. Deliberately a fixed list rather than a free-form
// map: most of the variables that circulate in forum threads either
// have no effect on llama.cpp or are workarounds for other software
// stacks. Anything outside this list goes through the free-form Extra
// environment entry, which warns about known-risky variables instead of
// refusing them.
//
// The environment applies to the router process and is inherited by
// every model instance it spawns identically — per-model env is not
// expressible (preset INI carries only CLI args).
type RuntimeEnvOption struct {
	Name     string
	Label    string
	Help     string
	Values   []string // allowed values; empty means free text
	Backends []string // build backends the variable affects; empty means all
}

// RuntimeEnvOptions is the curated set.
//
// What is deliberately absent matters as much as what's here.
// HSA_NO_SCRATCH_RECLAIM and HSA_ENABLE_SDMA circulate as llama.cpp
// tuning knobs but have no supporting evidence — they come from PyTorch
// and container-stability threads. HSA_OVERRIDE_GFX_VERSION is a
// deployment concern for unsupported GPUs, already handled by setup.sh,
// and setting it on a natively supported card (gfx1030, gfx110x,
// gfx120x, gfx90a, gfx942) is actively harmful. Both remain reachable
// through Extra environment, with a warning.
func RuntimeEnvOptions() []RuntimeEnvOption {
	return []RuntimeEnvOption{
		{
			Name:  "GGML_CUDA_CUBLAS_COMPUTE_TYPE",
			Label: "cuBLAS/hipBLAS compute type",
			Help: "Accumulator precision for GEMMs that fall through to cuBLAS/hipBLAS. " +
				"Replaces the old GGML_CUDA_FORCE_CUBLAS_COMPUTE_16F build flag, which current llama.cpp no longer defines. " +
				"Leave on auto unless you are measuring a specific difference.",
			Values:   []string{"", "auto", "f16", "bf16", "f32"},
			Backends: []string{"cuda", "rocm"},
		},
		{
			Name:  "ROCBLAS_USE_HIPBLASLT",
			Label: "Prefer hipBLASLt (ROCm)",
			Help: "Routes rocBLAS calls through hipBLASLt where possible. " +
				"Note this is already the default on gfx12 (RDNA4), where setting it changes nothing, " +
				"and llama.cpp's own source reports a hipBLASLt crash on gfx942 (CDNA3). " +
				"Most quantized matmuls bypass rocBLAS entirely, so measure before keeping it.",
			Values:   []string{"", "1", "0"},
			Backends: []string{"rocm"},
		},
		{
			Name:  "GGML_CUDA_DISABLE_GRAPHS",
			Label: "Disable CUDA/HIP graphs",
			Help: "Graphs are enabled by default and generally help token generation. " +
				"This is a compatibility and troubleshooting option — expect a small slowdown, not a speedup.",
			Values:   []string{"", "1"},
			Backends: []string{"cuda", "rocm"},
		},
		{
			Name:  "GGML_CUDA_ENABLE_UNIFIED_MEMORY",
			Label: "Unified memory (VRAM overflow)",
			Help: "Lets allocations spill into system RAM instead of failing when VRAM runs out. " +
				"A survivability option, not a performance one — spilled layers are much slower.",
			Values:   []string{"", "1"},
			Backends: []string{"cuda", "rocm"},
		},
		{
			Name:  "GGML_CUDA_P2P",
			Label: "Peer-to-peer transfers (CUDA)",
			Help: "Opt-in direct GPU-to-GPU transfers for multi-GPU splits. Needs driver P2P support " +
				"(usually workstation/datacenter cards); on consumer boards or with IOMMU enabled it can " +
				"crash or corrupt output. If you see garbage at long context on multi-GPU, leave this off " +
				"and consider the 'Disable Peer-to-Peer Copies' build toggle instead.",
			Values:   []string{"", "1"},
			Backends: []string{"cuda"},
		},
		{
			Name:  "CUDA_SCALE_LAUNCH_QUEUES",
			Label: "Scale CUDA launch queues",
			Help: "Grows the CUDA command buffers (e.g. 4x) so multi-GPU pipelines stall less on kernel " +
				"launches. Only relevant when several GPUs serve one model; measure before keeping.",
			Values:   []string{"", "2x", "4x"},
			Backends: []string{"cuda"},
		},
		{
			Name:  "GGML_VK_DISABLE_COOPMAT",
			Label: "Disable Vulkan matrix cores",
			Help: "Turns off cooperative-matrix use in the Vulkan backend. A troubleshooting option for " +
				"driver hangs and GPU resets (seen on some Intel Arc drivers); costs significant speed " +
				"when the driver is healthy — 'matrix cores: none' in the log confirms it took effect.",
			Values:   []string{"", "1"},
			Backends: []string{"vulkan"},
		},
		{
			Name:  "GGML_VK_FORCE_MAX_ALLOCATION_SIZE",
			Label: "Cap Vulkan allocation size",
			Help: "Caps a single Vulkan device allocation, in bytes (e.g. 1073741824 for 1 GiB). " +
				"Workaround for 'device memory allocation failed' on drivers with low allocation limits; " +
				"leave unset otherwise.",
			Backends: []string{"vulkan"},
		},
	}
}

// RuntimeEnvBackends returns the backends the curated set spans, in
// first-appearance order — the choices for the Settings backend
// selector. The selector is a view filter only: every variable is
// settable whether or not a build for its backend exists, since env is
// configuration for whatever the router launches next.
func RuntimeEnvBackends() []string {
	var out []string
	seen := map[string]bool{}
	for _, o := range RuntimeEnvOptions() {
		for _, b := range o.Backends {
			if !seen[b] {
				seen[b] = true
				out = append(out, b)
			}
		}
	}
	return out
}

// BackendsAttr renders the option's backends for the row's data
// attribute, space-separated ("cuda rocm"). Empty means all backends.
func (o RuntimeEnvOption) BackendsAttr() string {
	return strings.Join(o.Backends, " ")
}

// BackendsLabel renders the option's backends for display ("cuda, rocm").
func (o RuntimeEnvOption) BackendsLabel() string {
	if len(o.Backends) == 0 {
		return "all"
	}
	return strings.Join(o.Backends, ", ")
}

// EnvSet is one scope's worth of environment configuration: values for
// the curated options plus a free-form block of KEY=VALUE lines. It is
// scope-agnostic — Settings holds one global instance today; if
// llama.cpp ever supports per-instance environment, a per-model EnvSet
// slots in without changes here.
type EnvSet struct {
	Curated map[string]string
	Extra   string
}

// envPair is one parsed KEY=VALUE entry.
type envPair struct {
	Name  string
	Value string
}

// parseExtraEnv parses the free-form block: one KEY=VALUE per line,
// blank lines and #-comments ignored. Malformed lines are returned
// separately so validation can point at them.
func parseExtraEnv(raw string) (pairs []envPair, malformed []string) {
	for _, line := range strings.Split(raw, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		eq := strings.IndexByte(line, '=')
		if eq <= 0 || !validEnvName(line[:eq]) {
			malformed = append(malformed, line)
			continue
		}
		pairs = append(pairs, envPair{Name: line[:eq], Value: line[eq+1:]})
	}
	return pairs, malformed
}

func validEnvName(name string) bool {
	for i, r := range name {
		switch {
		case r == '_', r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z':
		case r >= '0' && r <= '9':
			if i == 0 {
				return false
			}
		default:
			return false
		}
	}
	return name != ""
}

// Validate checks the curated values against their allowed sets and the
// free-form block for well-formed KEY=VALUE lines. Unknown variable
// names in Extra are accepted by design — the free-form entry exists
// for variables outside the curated list; known-risky ones warn (see
// Warnings) but never block.
func (e EnvSet) Validate() error {
	allowed := make(map[string]RuntimeEnvOption, len(RuntimeEnvOptions()))
	for _, o := range RuntimeEnvOptions() {
		allowed[o.Name] = o
	}

	names := make([]string, 0, len(e.Curated))
	for k := range e.Curated {
		names = append(names, k)
	}
	sort.Strings(names)

	for _, name := range names {
		opt, ok := allowed[name]
		if !ok {
			return fmt.Errorf("unknown runtime environment variable %q", name)
		}
		v := strings.TrimSpace(e.Curated[name])
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

	if _, malformed := parseExtraEnv(e.Extra); len(malformed) > 0 {
		return fmt.Errorf("extra environment: not KEY=VALUE: %q", malformed[0])
	}
	return nil
}

// riskyEnvVars are variables that are legal to set but conflict with a
// toolchest mechanism or are commonly harmful. They warn — never block —
// because each has a legitimate use in the right situation.
var riskyEnvVars = map[string]string{
	"CUDA_DEVICE_ORDER":        "the process manager pins CUDA_DEVICE_ORDER=PCI_BUS_ID so the UI's GPU indices match llama-server; overriding it can silently remap which card \"GPU 0\" is",
	"CUDA_VISIBLE_DEVICES":     "hides GPUs from the router and every model instance, fighting the per-model GPU assignment (--device) set in model config",
	"HIP_VISIBLE_DEVICES":      "hides GPUs from the router and every model instance, fighting the per-model GPU assignment (--device) set in model config",
	"ROCR_VISIBLE_DEVICES":     "hides GPUs from the router and every model instance, fighting the per-model GPU assignment (--device) set in model config",
	"GGML_VK_VISIBLE_DEVICES":  "hides GPUs from the router and every model instance, fighting the per-model GPU assignment (--device) set in model config",
	"HSA_OVERRIDE_GFX_VERSION": "only needed for GPUs ROCm doesn't support natively (setup.sh already handles that case); on a natively supported card it selects the wrong kernels and hurts correctness and performance",
}

// Warnings returns one message per known-risky variable present in the
// set. The variables still apply — this is information, not enforcement.
func (e EnvSet) Warnings() []string {
	var names []string
	seen := map[string]bool{}
	note := func(name string) {
		if reason, ok := riskyEnvVars[name]; ok && !seen[name] {
			seen[name] = true
			names = append(names, fmt.Sprintf("%s: %s", name, reason))
		}
	}
	curated := make([]string, 0, len(e.Curated))
	for k, v := range e.Curated {
		if strings.TrimSpace(v) != "" {
			curated = append(curated, k)
		}
	}
	sort.Strings(curated)
	for _, k := range curated {
		note(k)
	}
	pairs, _ := parseExtraEnv(e.Extra)
	for _, p := range pairs {
		note(p.Name)
	}
	return names
}

// Pairs renders the set as KEY=VALUE strings, skipping blank curated
// values so an unset option means "don't touch the environment" rather
// than "set it to empty". Free-form entries override curated ones with
// the same name. Sorted for a stable command line.
func (e EnvSet) Pairs() []string {
	merged := map[string]string{}
	for k, v := range e.Curated {
		if strings.TrimSpace(v) != "" {
			merged[k] = v
		}
	}
	pairs, _ := parseExtraEnv(e.Extra)
	for _, p := range pairs {
		merged[p.Name] = p.Value
	}
	if len(merged) == 0 {
		return nil
	}
	out := make([]string, 0, len(merged))
	for k, v := range merged {
		out = append(out, k+"="+v)
	}
	sort.Strings(out)
	return out
}

// ValidateRuntimeEnv checks curated names and values (compat wrapper).
func ValidateRuntimeEnv(env map[string]string) error {
	return EnvSet{Curated: env}.Validate()
}

// EnvSet returns the global runtime environment as the reusable
// component type.
func (c *Config) EnvSet() EnvSet {
	return EnvSet{Curated: c.RuntimeEnv, Extra: c.RuntimeEnvExtra}
}

// RuntimeEnvPairs renders the configured variables as KEY=VALUE strings
// for the router launch.
func (c *Config) RuntimeEnvPairs() []string {
	return c.EnvSet().Pairs()
}
