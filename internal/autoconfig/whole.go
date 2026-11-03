package autoconfig

import (
	"fmt"
	"strings"

	"github.com/tmac1973/llama-toolchest/internal/models"
)

// wholeGPUsNote says why a plan made on the whole GPUs (Deps.WholeGPUs)
// leaves the image and music backends' reservations out, and what
// loading the model then does. keptGiB is what backends kept running
// still reserve on the cards, which no load stops.
func wholeGPUsNote(keptGiB float64) models.ProfileNote {
	reason := "Planned on the whole GPUs, as asked: the memory reserved for image and music backends was left out. Auto-evict is on, so loading the model stops the backends in its way, and each starts again for its next job."
	if keptGiB > 0 {
		reason += fmt.Sprintf(" The %.1f GiB reserved by backends set to keep running still counts, because they are never stopped.", keptGiB)
	}
	return models.ProfileNote{Origin: "hardware fit", Reason: reason}
}

// keptReservedGiB is what stays reserved on the cards a plan uses after
// WholeCards: the reservations of backends set to keep running.
func keptReservedGiB(hw models.Hardware) float64 {
	var n int
	for _, g := range models.PlanCards(hw) {
		n += g.ReservedMiB
	}
	return float64(n) / 1024
}

// pinWholeSplit writes cfg's layer split out as a custom one balanced on
// the whole GPUs (hw, with the evictable reservations already given
// back), so every launch keeps it. Without it the launch balances the
// split by the room beside the reservations whenever the model fits
// there (models.CardRoom.For), and the cards with backends on them get
// fewer layers. With the split written out, a load whose share of a card
// does not fit beside a backend stops the backend instead.
//
// The split is even, or for a model with expert layers in system memory
// (or cards of different sizes) the size-balanced one MoESplitConfig
// works out for the whole cards. It applies only to a layer split over
// every GPU in the machine: a custom split cannot leave a card out, and
// one with a share of nothing would still bring that card into the
// process (see moveLayersOff). ok is false when nothing was written.
func pinWholeSplit(m *models.Model, cfg models.ModelConfig, hw models.Hardware) (models.ModelConfig, models.ProfileNote, bool) {
	n := len(hw.GPUs)
	if n < 2 || cfg.GPUAssign == "custom" || strings.HasPrefix(cfg.GPUAssign, "tensor") ||
		cfg.SplitMode == "tensor" || cfg.SplitMode == "row" {
		return cfg, models.ProfileNote{}, false
	}
	if len(models.AssignedModelGPUs(&cfg, n)) != n || models.SplitDeviceIndices(cfg.TensorSplit) != nil {
		return cfg, models.ProfileNote{}, false
	}
	split := strings.Repeat("1,", n-1) + "1"
	if launch := models.MoESplitConfig(&cfg, m, models.GPUMiB(hw)); launch != &cfg {
		split = launch.TensorSplit
	}
	out := cfg
	out.GPUAssign = "custom"
	out.TensorSplit = split
	if out.SplitMode == "" {
		out.SplitMode = "layer"
	}
	return out, models.ProfileNote{
		Field:  "gpu_assign",
		Origin: "hardware fit",
		Reason: fmt.Sprintf("The layers are split %s over the whole GPUs and kept that way, so cards with image or music backends on them get the same share as the others. Without a fixed split, each load would give those cards fewer layers whenever the model fits beside the backends.", split),
	}, true
}
