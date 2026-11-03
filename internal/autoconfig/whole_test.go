package autoconfig

import (
	"context"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/models"
)

func threeGPUs() models.Hardware {
	hw := models.Hardware{LogicalCores: 16, RAMTotalMiB: 64 * 1024}
	for i := range 3 {
		hw.GPUs = append(hw.GPUs, models.GPUSpec{Index: i, VRAMTotalMiB: 24 * 1024})
	}
	return hw
}

func hasFieldNote(notes []models.ProfileNote, field, text string) bool {
	for _, n := range notes {
		if n.Field == field && strings.Contains(n.Reason, text) {
			return true
		}
	}
	return false
}

// Planned on the whole GPUs, the proposal writes an even split out as a
// custom one, so no launch rebalances it around the backends.
func TestRunOnWholeGPUsPinsAnEvenSplit(t *testing.T) {
	reg := runRegistry(t)
	res, err := Run(context.Background(), Deps{Registry: reg, Hardware: threeGPUs(), WholeGPUs: true}, runModelID, models.ContextShort)
	if err != nil {
		t.Fatal(err)
	}
	if res.Proposed.GPUAssign != "custom" || res.Proposed.TensorSplit != "1,1,1" || res.Proposed.SplitMode != "layer" {
		t.Errorf("placement = %q %q %q, want custom 1,1,1 layer", res.Proposed.GPUAssign, res.Proposed.TensorSplit, res.Proposed.SplitMode)
	}
	if !hasFieldNote(res.Notes, "gpu_assign", "split 1,1,1") {
		t.Errorf("no note explains the split: %+v", res.Notes)
	}
	if !hasFieldNote(res.Notes, "", "Planned on the whole GPUs") {
		t.Errorf("no note says the plan uses the whole GPUs: %+v", res.Notes)
	}

	res, err = Run(context.Background(), Deps{Registry: reg, Hardware: threeGPUs()}, runModelID, models.ContextShort)
	if err != nil {
		t.Fatal(err)
	}
	if res.Proposed.GPUAssign == "custom" {
		t.Errorf("without the option the split is pinned: %q", res.Proposed.TensorSplit)
	}
}

// Reservations of backends kept running stay in the plan, and the note
// says so.
func TestWholeGPUsNoteCountsKeptBackends(t *testing.T) {
	hw := threeGPUs()
	hw.GPUs[1].ReservedMiB = 6144
	if n := wholeGPUsNote(keptReservedGiB(hw)); !strings.Contains(n.Reason, "6.0 GiB reserved by backends set to keep running") {
		t.Errorf("note = %q", n.Reason)
	}
	if n := wholeGPUsNote(keptReservedGiB(threeGPUs())); strings.Contains(n.Reason, "keep running") {
		t.Errorf("note mentions kept backends when there are none: %q", n.Reason)
	}
}

// Only a layer split over every GPU is pinned: a custom split cannot
// leave a card out, and a tensor split is even already.
func TestPinWholeSplitLeavesOtherPlacementsAlone(t *testing.T) {
	reg := runRegistry(t)
	m, _ := reg.Get(runModelID)
	for name, cfg := range map[string]models.ModelConfig{
		"two of three": {GPUAssign: "0,1", TensorSplit: "1,1,0"},
		"tensor":       {GPUAssign: "tensor-3", SplitMode: "tensor"},
		"custom":       {GPUAssign: "custom", TensorSplit: "3,2,3"},
	} {
		if got, _, ok := pinWholeSplit(m, cfg, threeGPUs()); ok || got.GPUAssign != cfg.GPUAssign || got.TensorSplit != cfg.TensorSplit {
			t.Errorf("%s: pinned to %q %q", name, got.GPUAssign, got.TensorSplit)
		}
	}
	one := models.Hardware{GPUs: []models.GPUSpec{{Index: 0, VRAMTotalMiB: 24 * 1024}}}
	if _, _, ok := pinWholeSplit(m, models.ModelConfig{GPUAssign: "all"}, one); ok {
		t.Error("pinned a split on one GPU")
	}
	if got, _, ok := pinWholeSplit(m, models.ModelConfig{GPUAssign: "all"}, threeGPUs()); !ok || got.TensorSplit != "1,1,1" {
		t.Errorf("all GPUs: %q, %v", got.TensorSplit, ok)
	}
}
