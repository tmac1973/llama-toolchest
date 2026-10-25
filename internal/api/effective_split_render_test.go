package api

import (
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/models"
)

// The split balanced by each card's room is not in the saved config, so
// the panel lists it card by card; without one, nothing is shown.
func TestEffectiveSplitLine(t *testing.T) {
	cfg := &models.ModelConfig{Enabled: true, GPUAssign: "all"}
	if out := renderModelConfigWith(t, cfg, modelConfigPanelData{NumGPUs: 3}); strings.Contains(out, "Effective split") {
		t.Error("split line rendered without a balanced split")
	}
	out := renderModelConfigWith(t, cfg, modelConfigPanelData{NumGPUs: 3, EffectiveSplit: []models.CardSplit{
		{GPU: 0, Layers: 29, LoadGiB: 14.24, BudgetGiB: 15.1},
		{GPU: 1, Layers: 3, LoadGiB: 2.0, BudgetGiB: 5.6},
		{GPU: 2, Layers: 9, LoadGiB: 5.5, BudgetGiB: 7.6},
	}})
	want := "GPU 0: 29 layers, 14.2 of 15.1 GiB · GPU 1: 3 layers, 2.0 of 5.6 GiB · GPU 2: 9 layers, 5.5 of 7.6 GiB"
	if !strings.Contains(out, want) {
		t.Errorf("split line missing %q; output=\n%s", want, out)
	}
}
