package models

import (
	"fmt"
	"strconv"
	"strings"
)

// GPUOption represents one selectable GPU assignment choice for the dropdown.
type GPUOption struct {
	Value        string // "all", "0", "0-1", "0,2", "tensor-2", "tensor:0,2", "custom"
	Label        string // "All GPUs", "GPU 0", "GPUs 0-1"
	GPUs         []int  // which GPU indices (e.g., [0,1])
	IsTensor     bool   // true for tensor parallelism mode
	IsSpanAll    bool   // the "All (discrete) GPUs" option — the default when nothing is saved
	IncludesIGPU bool   // true when any member GPU is an integrated GPU
	Disabled     bool   // model doesn't fit
	Recommend    bool   // suggested option
}

// GPUAssignOptions generates all GPU assignment options. igpu marks
// which GPU indices are integrated GPUs (nil when none, or unknown).
//
// iGPU guard rails: single-GPU and range options that touch an iGPU
// stay selectable but say so in their label (someone who built with the
// iGPU target included may genuinely want it), MarkRecommended never
// recommends them, and the spanning options — "All GPUs" and the tensor
// parallelism variants — are generated from discrete GPUs only, since
// spreading a model onto the iGPU by default is never what anyone
// wants and hard-fails when the build excludes its kernels.
func GPUAssignOptions(numGPUs int, igpu []bool) []GPUOption {
	if numGPUs <= 0 {
		return nil
	}
	isIGPU := func(i int) bool { return i >= 0 && i < len(igpu) && igpu[i] }

	var dgpus []int
	for i := 0; i < numGPUs; i++ {
		if !isIGPU(i) {
			dgpus = append(dgpus, i)
		}
	}
	// APU-only box: the iGPU is all there is, so it stops being a
	// special case — span options include it and carry no warning label.
	if len(dgpus) == 0 {
		dgpus = make([]int, numGPUs)
		for i := range dgpus {
			dgpus[i] = i
		}
		isIGPU = func(int) bool { return false }
	}

	var opts []GPUOption
	mark := func(o GPUOption, suffix string) GPUOption {
		for _, g := range o.GPUs {
			if isIGPU(g) {
				o.IncludesIGPU = true
				o.Label += suffix
				break
			}
		}
		return o
	}

	// Single GPUs
	for i := 0; i < numGPUs; i++ {
		opts = append(opts, mark(GPUOption{
			Value: fmt.Sprintf("%d", i),
			Label: fmt.Sprintf("GPU %d", i),
			GPUs:  []int{i},
		}, " (iGPU)"))
	}

	// Contiguous pairs
	if numGPUs >= 2 {
		for i := 0; i <= numGPUs-2; i++ {
			opts = append(opts, mark(GPUOption{
				Value: fmt.Sprintf("%d-%d", i, i+1),
				Label: fmt.Sprintf("GPUs %d-%d", i, i+1),
				GPUs:  []int{i, i + 1},
			}, " (includes iGPU)"))
		}
	}

	// Contiguous triples
	if numGPUs >= 4 {
		for i := 0; i <= numGPUs-3; i++ {
			opts = append(opts, mark(GPUOption{
				Value: fmt.Sprintf("%d-%d", i, i+2),
				Label: fmt.Sprintf("GPUs %d-%d", i, i+2),
				GPUs:  []int{i, i + 1, i + 2},
			}, " (includes iGPU)"))
		}
	}

	// "All GPUs" — layer split across every discrete GPU. When an iGPU
	// exists, the option's value is the explicit index set rather than
	// "all", so llama-server placement matches the label exactly.
	if len(dgpus) == numGPUs {
		opts = append(opts, GPUOption{
			Value:     "all",
			Label:     fmt.Sprintf("All GPUs (%d, layer split)", numGPUs),
			GPUs:      append([]int(nil), dgpus...),
			IsSpanAll: true,
		})
	} else {
		opts = append(opts, GPUOption{
			Value:     joinInts(dgpus),
			Label:     fmt.Sprintf("All discrete GPUs (%d, layer split)", len(dgpus)),
			GPUs:      append([]int(nil), dgpus...),
			IsSpanAll: true,
		})
	}

	// Tensor parallelism variants over discrete GPUs — one per count.
	// The legacy "tensor-N" value means "first N GPUs"; when the first N
	// discrete GPUs aren't literally indices 0..N-1 (an iGPU sits among
	// them), the explicit-set form "tensor:i,j" is used instead.
	for n := 2; n <= len(dgpus); n++ {
		tpGPUs := append([]int(nil), dgpus[:n]...)
		value := fmt.Sprintf("tensor-%d", n)
		if tpGPUs[n-1] != n-1 {
			value = "tensor:" + joinInts(tpGPUs)
		}
		label := fmt.Sprintf("Tensor Parallelism (%d GPUs, experimental)", n)
		if n == len(dgpus) {
			label = fmt.Sprintf("Tensor Parallelism (all %d GPUs, experimental)", n)
		}
		opts = append(opts, GPUOption{
			Value:    value,
			Label:    label,
			GPUs:     tpGPUs,
			IsTensor: true,
		})
	}

	// "Custom" is always last
	opts = append(opts, GPUOption{
		Value: "custom",
		Label: "Custom (manual tensor split)",
	})

	return opts
}

// joinInts renders indices as "0,2,3".
func joinInts(xs []int) string {
	parts := make([]string, len(xs))
	for i, x := range xs {
		parts[i] = fmt.Sprintf("%d", x)
	}
	return strings.Join(parts, ",")
}

// ResolveGPUAssign converts a GPU assignment string into tensor-split,
// split-mode, and main-gpu values for llama-server.
//
//	"all"        → ("", "layer", 0)          — layer split across all GPUs
//	"tensor-N"   → tensor parallelism: split-mode=tensor; tensor-split="1,...,0"
//	               restricts to first N GPUs when N<numGPUs, else no split
//	"tensor:0,2" → tensor parallelism on an explicit GPU set (skips iGPUs)
//	"0"          → ("1,0,0,0", "layer", 0)   — single GPU
//	"0-1"        → ("1,1,0,0", "layer", 0)   — two GPUs (layer)
//	"2-3"        → ("0,0,1,1", "layer", 2)   — GPUs 2-3 (layer)
//	"0,2"        → ("1,0,1,0", "layer", 0)   — explicit set (skips iGPUs)
//	"custom"     → ("", "", 0)               — caller preserves raw TensorSplit
func ResolveGPUAssign(assign string, numGPUs int) (tensorSplit, splitMode string, mainGPU int) {
	if assign == "" || assign == "custom" || numGPUs <= 0 {
		return "", "", 0
	}

	// "all" — layer split across all GPUs
	if assign == "all" {
		return "", "layer", 0
	}

	// "tensor-N" (first N GPUs) or "tensor:i,j" (explicit set)
	if strings.HasPrefix(assign, "tensor") {
		gpus, ok := tensorAssignGPUs(assign, numGPUs)
		if !ok {
			return "", "", 0
		}
		if len(gpus) >= numGPUs {
			return "", "tensor", 0
		}
		parts := make([]string, numGPUs)
		for i := range parts {
			parts[i] = "0"
		}
		for _, g := range gpus {
			if g >= 0 && g < numGPUs {
				parts[g] = "1"
			}
		}
		return strings.Join(parts, ","), "tensor", 0
	}

	// Parse the assignment to get GPU indices
	gpus := parseGPURange(assign)
	if len(gpus) == 0 {
		return "", "", 0
	}

	// Build tensor-split string: 1 for active GPUs, 0 for inactive
	parts := make([]string, numGPUs)
	for i := range parts {
		parts[i] = "0"
	}
	for _, g := range gpus {
		if g >= 0 && g < numGPUs {
			parts[g] = "1"
		}
	}

	return strings.Join(parts, ","), "layer", gpus[0]
}

// backendDevicePrefix maps a build profile backend to the device-name
// prefix llama.cpp registers for it ("ROCm0", "CUDA0", ...). Backends
// absent here (cpu; metal, which is single-device) have no addressable
// multi-GPU device names, so no device list is emitted and placement
// falls back to the zero-padded tensor-split.
var backendDevicePrefix = map[string]string{
	"rocm":   "ROCm",
	"cuda":   "CUDA",
	"vulkan": "Vulkan",
	"sycl":   "SYCL",
}

// SplitDeviceIndices returns the GPU indices carrying non-zero weight in
// a zero-padded tensor-split string ("1,1,0,0" → [0 1]), or nil when the
// split doesn't restrict anything — empty, or every entry non-zero.
func SplitDeviceIndices(tensorSplit string) []int {
	if tensorSplit == "" {
		return nil
	}
	parts := strings.Split(tensorSplit, ",")
	var active []int
	for i, p := range parts {
		p = strings.TrimSpace(p)
		if p != "" && p != "0" && p != "0.0" {
			active = append(active, i)
		}
	}
	if len(active) == 0 || len(active) == len(parts) {
		return nil
	}
	return active
}

// DeviceList renders GPU indices as llama.cpp device names for a
// backend: ("rocm", [0 1]) → "ROCm0,ROCm1". Empty when the backend has
// no addressable device names.
func DeviceList(backend string, gpus []int) string {
	prefix := backendDevicePrefix[backend]
	if prefix == "" || len(gpus) == 0 {
		return ""
	}
	names := make([]string, len(gpus))
	for i, g := range gpus {
		names[i] = fmt.Sprintf("%s%d", prefix, g)
	}
	return strings.Join(names, ",")
}

// gpuPlacementParams returns the flags that place a model onto its GPUs,
// shared by the preset INI and the EffectiveFlags preview.
//
// When the assignment restricts the model to a subset of GPUs and the
// backend has addressable device names, placement is expressed as a
// "device" list rather than a zero-padded tensor-split. A zero split
// entry only keeps weights off a GPU — llama.cpp still initializes every
// visible device, and under split-mode tensor the excluded GPUs are
// enrolled in the collective ops, spinning at full utilization and
// forcing the slow butterfly all-reduce fallback ("internal AllReduce
// init failed (n_devices != N?)"). A device list keeps them out of the
// instance entirely, which is also what lets another model run on them
// concurrently.
//
// With a device list llama-server renumbers the visible devices 0..N-1,
// so the padded split is trimmed to its active entries and main-gpu —
// always the first active GPU, which becomes index 0 — is omitted.
//
// "custom" assignments keep the user's raw tensor-split untouched, and
// backends without device names keep the padded-split emission.
func gpuPlacementParams(c *ModelConfig, backend string) []specParam {
	if c.GPUAssign != "" && c.GPUAssign != "custom" {
		if gpus := SplitDeviceIndices(c.TensorSplit); gpus != nil {
			if devices := DeviceList(backend, gpus); devices != "" {
				out := []specParam{{"device", devices}}
				if ts := trimSplit(c.TensorSplit, gpus); ts != "" {
					out = append(out, specParam{"tensor-split", ts})
				}
				if c.SplitMode != "" {
					out = append(out, specParam{"split-mode", c.SplitMode})
				}
				return out
			}
		}
	}
	var out []specParam
	if c.TensorSplit != "" {
		out = append(out, specParam{"tensor-split", c.TensorSplit})
	}
	if c.SplitMode != "" {
		out = append(out, specParam{"split-mode", c.SplitMode})
	}
	if c.MainGPU > 0 {
		out = append(out, specParam{"main-gpu", strconv.Itoa(c.MainGPU)})
	}
	return out
}

// trimSplit keeps only the split entries at the given indices ("1,1,0,0"
// with [0 1] → "1,1"), matching the renumbering a device list applies.
func trimSplit(tensorSplit string, gpus []int) string {
	parts := strings.Split(tensorSplit, ",")
	out := make([]string, 0, len(gpus))
	for _, g := range gpus {
		if g < len(parts) {
			out = append(out, strings.TrimSpace(parts[g]))
		}
	}
	return strings.Join(out, ",")
}

// parseTensorAssign parses a "tensor-N" GPU assignment, returning N and
// whether the value was a valid tensor assignment (N > 0).
func parseTensorAssign(assign string) (int, bool) {
	var n int
	if _, err := fmt.Sscanf(assign, "tensor-%d", &n); err == nil && n > 0 {
		return n, true
	}
	return 0, false
}

// tensorAssignGPUs resolves either tensor form to GPU indices:
// "tensor-N" → the first N GPUs (legacy), "tensor:i,j" → the explicit
// set (used when an iGPU interrupts the discrete indices).
func tensorAssignGPUs(assign string, numGPUs int) ([]int, bool) {
	if rest, ok := strings.CutPrefix(assign, "tensor:"); ok {
		gpus := parseIntList(rest)
		return gpus, len(gpus) > 0
	}
	if n, ok := parseTensorAssign(assign); ok {
		if n > numGPUs {
			n = numGPUs
		}
		gpus := make([]int, n)
		for i := range gpus {
			gpus[i] = i
		}
		return gpus, true
	}
	return nil, false
}

// parseIntList parses "0,2,3" into indices; nil on any invalid entry.
func parseIntList(s string) []int {
	var out []int
	for _, p := range strings.Split(s, ",") {
		var n int
		if _, err := fmt.Sscanf(strings.TrimSpace(p), "%d", &n); err != nil || n < 0 {
			return nil
		}
		out = append(out, n)
	}
	return out
}

// parseGPURange parses GPU assignment values like "0", "0-1", "2-3",
// or an explicit set like "0,2" (used when an iGPU interrupts the
// discrete indices) into indices.
func parseGPURange(assign string) []int {
	// Try single GPU: "0", "1", etc.
	if len(assign) == 1 && assign[0] >= '0' && assign[0] <= '9' {
		return []int{int(assign[0] - '0')}
	}

	// Explicit set: "0,2,3"
	if strings.Contains(assign, ",") {
		return parseIntList(assign)
	}

	// Try range: "0-1", "2-3", etc.
	var start, end int
	if n, _ := fmt.Sscanf(assign, "%d-%d", &start, &end); n == 2 && end >= start {
		gpus := make([]int, 0, end-start+1)
		for i := start; i <= end; i++ {
			gpus = append(gpus, i)
		}
		return gpus
	}

	return nil
}

// GPUAllocation describes one model's footprint per GPU.
type GPUAllocation struct {
	ModelID   string
	ModelName string
	GPUs      []int
	PerGPUGB  float64
	TotalGB   float64
	Color     string
}

// allocationColors are the colors assigned to models in the GPU map.
var allocationColors = []string{
	"#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
	"#59a14f", "#edc948", "#b07aa1", "#ff9da7",
	"#9c755f", "#bab0ac",
}

// ComputeAllocations builds the GPU allocation list from enabled models.
// Uses peak VRAM (weights + KV cache at the configured context size and KV
// quantization) so the map reflects actual fit when a model is loaded.
func ComputeAllocations(modelList []*Model, configs map[string]*ModelConfig, numGPUs int) []GPUAllocation {
	if numGPUs <= 0 {
		return nil
	}

	var allocs []GPUAllocation
	colorIdx := 0

	for _, m := range modelList {
		cfg, ok := configs[m.ID]
		if !ok || !cfg.Enabled {
			continue
		}

		totalGB := VRAMEstimateForConfig(m, cfg)
		gpus := resolveModelGPUs(cfg, numGPUs)
		perGPU := totalGB
		if len(gpus) > 0 {
			perGPU = totalGB / float64(len(gpus))
		}

		allocs = append(allocs, GPUAllocation{
			ModelID:   m.ID,
			ModelName: ShortModelName(m.ModelID),
			GPUs:      gpus,
			PerGPUGB:  perGPU,
			TotalGB:   totalGB,
			Color:     allocationColors[colorIdx%len(allocationColors)],
		})
		colorIdx++
	}

	return allocs
}

// AssignedModelGPUs returns the GPU indices a model config places the
// model on — the exported face of resolveModelGPUs for callers outside
// the package (e.g. the iGPU assignment warning).
func AssignedModelGPUs(cfg *ModelConfig, numGPUs int) []int {
	return resolveModelGPUs(cfg, numGPUs)
}

// resolveModelGPUs determines which GPUs a model is assigned to.
func resolveModelGPUs(cfg *ModelConfig, numGPUs int) []int {
	if strings.HasPrefix(cfg.GPUAssign, "tensor") && cfg.GPUAssign != "tensor" {
		if gpus, ok := tensorAssignGPUs(cfg.GPUAssign, numGPUs); ok {
			return gpus
		}
	}

	if cfg.GPUAssign != "" && cfg.GPUAssign != "all" && cfg.GPUAssign != "custom" {
		if gpus := parseGPURange(cfg.GPUAssign); len(gpus) > 0 {
			return gpus
		}
	}

	// "all", "custom" with tensor-split, or default: assume all GPUs
	all := make([]int, numGPUs)
	for i := range all {
		all[i] = i
	}
	return all
}

// ShortModelName extracts a short display name from a model ID like
// "org/repo-GGUF": strips the org prefix and common GGUF suffixes.
func ShortModelName(modelID string) string {
	// Strip org prefix
	if idx := strings.LastIndex(modelID, "/"); idx >= 0 {
		modelID = modelID[idx+1:]
	}
	// Strip common suffixes
	modelID = strings.TrimSuffix(modelID, "-GGUF")
	modelID = strings.TrimSuffix(modelID, "-gguf")
	return modelID
}

// GPUAssignLabel returns a short human-readable label for a GPU assignment.
func GPUAssignLabel(assign string) string {
	switch {
	case assign == "" || assign == "all":
		return "all gpus"
	case assign == "custom":
		return "custom"
	case strings.HasPrefix(assign, "tensor-"):
		return "tp:" + strings.TrimPrefix(assign, "tensor-")
	case strings.HasPrefix(assign, "tensor:"):
		return "tp:" + strings.TrimPrefix(assign, "tensor:")
	default:
		return "gpu:" + assign
	}
}

// MarkRecommended marks the best GPU option: fewest GPUs where the model
// weights fit, preferring least-allocated GPUs. Does not disable any options
// since the router dynamically loads/unloads models — all enabled models
// won't be loaded simultaneously.
func MarkRecommended(options []GPUOption, modelVRAMGB float64, perGPUGB float64, existing []GPUAllocation) {
	if len(options) == 0 || perGPUGB <= 0 {
		return
	}

	// Calculate current weights usage per GPU
	gpuUsed := make(map[int]float64)
	for _, a := range existing {
		for _, g := range a.GPUs {
			gpuUsed[g] += a.PerGPUGB
		}
	}

	bestIdx := -1
	bestScore := -1.0

	for i := range options {
		opt := &options[i]
		if opt.Value == "custom" {
			continue
		}
		// Never recommend a placement touching an iGPU: it shares system
		// RAM, and the default build excludes its kernels entirely.
		if opt.IncludesIGPU {
			continue
		}

		gpuCount := len(opt.GPUs)
		if gpuCount == 0 {
			continue
		}

		perGPUNeed := modelVRAMGB / float64(gpuCount)

		// Check if model weights fit on these GPUs
		fits := true
		totalFree := 0.0
		for _, g := range opt.GPUs {
			free := perGPUGB - gpuUsed[g]
			if free < perGPUNeed {
				fits = false
			}
			totalFree += free
		}

		if !fits {
			continue
		}

		// Score: prefer fewer GPUs, then more free space
		score := totalFree - float64(gpuCount)*100 // heavily penalize more GPUs
		if bestIdx == -1 || score > bestScore {
			bestScore = score
			bestIdx = i
		}
	}

	if bestIdx >= 0 {
		options[bestIdx].Recommend = true
	}
}
