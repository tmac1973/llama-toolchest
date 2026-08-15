package api

import (
	"fmt"
	"strings"

	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/monitor"
)

// igpuFlags extracts the per-index iGPU markers from monitor metrics
// for models.GPUAssignOptions. Indexed by position in the metrics list,
// which matches the GPU indices the UI displays. Nil when no iGPU is
// present, so the no-iGPU path stays byte-identical to before.
func igpuFlags(gpus []monitor.GPUInfo) []bool {
	out := make([]bool, len(gpus))
	any := false
	for i, g := range gpus {
		out[i] = g.IsIGPU
		any = any || g.IsIGPU
	}
	if !any {
		return nil
	}
	return out
}

// migrateGPUAssign maps legacy and pre-iGPU-audit GPU assignments onto
// the current dropdown values, mutating cfg in place (the registry's
// shared struct, so preset emission sees the migrated values too; disk
// persists on the next save — the same lazy pattern the original
// legacy migration used).
//
// Legacy: GPUAssign "" or bare "tensor" predate the unified dropdown
// and are derived from the stored split fields.
//
// iGPU audit: on a box with an iGPU, "All GPUs" is no longer offered —
// the spanning option covers discrete GPUs only, under an explicit-set
// value. A stored "all" would match no option (the dropdown would
// silently display the first entry) while still spanning the iGPU at
// runtime, which fails to load under default builds that compile no
// iGPU kernels. It migrates to the discrete-set value, with the split
// fields re-resolved so display and emission agree immediately.
func migrateGPUAssign(cfg *models.ModelConfig, gpuOptions []models.GPUOption, numGPUs int) {
	if cfg.GPUAssign == "" || cfg.GPUAssign == "tensor" {
		switch {
		case cfg.SplitMode == "tensor":
			// Derive N from tensor-split (count of non-zero entries); fall
			// back to all GPUs if not set.
			n := countNonZeroSplit(cfg.TensorSplit)
			if n <= 0 || n > numGPUs {
				n = numGPUs
			}
			if n >= 2 && n < numGPUs {
				cfg.GPUAssign = fmt.Sprintf("tensor-%d", n)
			} else {
				cfg.GPUAssign = fmt.Sprintf("tensor-%d", numGPUs)
			}
		case cfg.TensorSplit != "":
			cfg.GPUAssign = "custom"
		}
	}

	if cfg.GPUAssign == "all" {
		for _, o := range gpuOptions {
			// On an iGPU-free box the spanning option's value is still
			// "all", so this is a no-op there.
			if o.IsSpanAll && o.Value != "all" {
				cfg.GPUAssign = o.Value
				cfg.TensorSplit, cfg.SplitMode, cfg.MainGPU = models.ResolveGPUAssign(o.Value, numGPUs)
				break
			}
		}
	}
}

// gpuAssignWarning reports why a model's current GPU assignment will
// fail on the active build: it places the model on an iGPU whose gfx
// target the build didn't compile kernels for. Empty when there is
// nothing to warn about (no iGPU in the assignment, targets unknown, or
// the build opted the iGPU in).
func (s *Server) gpuAssignWarning(cfg *models.ModelConfig, gpus []monitor.GPUInfo) string {
	build := s.resolveActiveBuild()
	if build == nil {
		return ""
	}
	targets := build.CMakeFlags["GPU_TARGETS"]
	if targets == "" {
		targets = build.CMakeFlags["AMDGPU_TARGETS"] // pre-rename builds
	}
	if targets == "" {
		return ""
	}
	for _, g := range models.AssignedModelGPUs(cfg, len(gpus)) {
		if g < 0 || g >= len(gpus) {
			continue
		}
		gi := gpus[g]
		if !gi.IsIGPU || gi.Arch == "" {
			continue
		}
		compiled := false
		for _, t := range strings.Split(targets, ";") {
			if strings.TrimSpace(t) == gi.Arch {
				compiled = true
				break
			}
		}
		if !compiled {
			return fmt.Sprintf(
				"This assignment places the model on GPU %d, an iGPU (%s) the active build has no kernels for (GPU_TARGETS=%s) — loading will fail. Pick a discrete GPU, or rebuild with 'Include iGPU Build Target' enabled.",
				g, gi.Arch, targets)
		}
	}
	return ""
}
