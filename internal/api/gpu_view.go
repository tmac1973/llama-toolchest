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
