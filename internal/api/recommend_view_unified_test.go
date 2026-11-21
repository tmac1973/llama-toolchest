package api

import (
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/huggingface"
	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/recommend"
)

// On a Mac the experts the GPU has no room for stay in the same memory
// and the CPU runs them: the card and the machine line say so, and do
// not speak of system memory or VRAM.
func TestRecommendWordsOnUnifiedMemory(t *testing.T) {
	pick := func(shared bool) *recommend.Pick {
		return &recommend.Pick{Quant: "Q4_K_M", Placement: models.PlacementExperts, CPURAMGiB: 2, SharedMemory: shared}
	}
	group := func(shared bool) *recommend.Group {
		return &recommend.Group{Name: "m", Repos: []huggingface.ListedModel{{ID: "a/m"}},
			Picks: map[models.ContextClass]*recommend.Pick{models.ContextMedium: pick(shared)}}
	}
	if c := newRecommendCardView(group(true), models.ContextMedium, "32k", 0); c.Placement != "Experts on the CPU" || strings.Contains(c.PlacementTip, "system memory") {
		t.Errorf("Mac card: %q, %q", c.Placement, c.PlacementTip)
	}
	if c := newRecommendCardView(group(false), models.ContextMedium, "32k", 0); c.Placement != "Experts in system memory" {
		t.Errorf("PC card: %q", c.Placement)
	}

	mac := models.Hardware{GPUs: []models.GPUSpec{{Index: 0, Name: "Apple M4 (10-core GPU)", VRAMTotalMiB: 18 * 1024, SharesRAM: true}}, RAMTotalMiB: 24 * 1024}
	line, _ := profileLine(recommend.Profile{Hardware: mac})
	if !strings.Contains(line, "18 GB of 24 GB unified memory for the GPU") || strings.Contains(line, "VRAM") || strings.Contains(line, "GB RAM") {
		t.Errorf("Mac machine line: %q", line)
	}
}
