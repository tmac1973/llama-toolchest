package api

import (
	"net/http"

	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/monitor"
)

// handleGPUMap renders the GPU allocation map partial.
func (s *Server) handleGPUMap(w http.ResponseWriter, r *http.Request) {
	respondHTML(w)

	metrics := s.monitor.Current()
	numGPUs := len(metrics.GPU)
	if numGPUs == 0 {
		return // render nothing when no GPUs detected
	}
	allocations := s.gpuMapAllocations(numGPUs)
	bars := gpuMapBars(metrics.GPU, allocations)

	data := struct {
		Bars        []gpuBar
		Allocations []models.GPUAllocation
	}{
		Bars:        bars,
		Allocations: allocations,
	}
	s.renderPartial(w, "gpu_map", data)
}

// gpuSegment is one allocation's part of a GPU's bar in the map.
type gpuSegment struct {
	ModelName string
	Color     string
	WidthPct  float64 // of the card's total VRAM
	Reserved  bool
}

// gpuBar is one GPU's bar in the map.
type gpuBar struct {
	Index      int
	Name       string
	TotalGB    float64
	UsedGB     float64
	FreeGB     float64
	Link       *monitor.PCIeLink
	Segments   []gpuSegment
	Overcommit bool
}

// gpuMapBars builds each GPU's bar from the allocations that use it. The
// i-th GPU of gpus is GPU index i of the allocations. A card that reports
// no VRAM gets segments of width 0, not a division by zero.
func gpuMapBars(gpus []monitor.GPUInfo, allocations []models.GPUAllocation) []gpuBar {
	bars := make([]gpuBar, len(gpus))
	for i, gpu := range gpus {
		totalGB := float64(gpu.VRAMTotalMB) / 1024.0
		bars[i] = gpuBar{
			Index:   gpu.Index,
			Name:    gpu.Name,
			Link:    gpu.PCIe,
			TotalGB: totalGB,
		}

		// Collect segments for this GPU
		var usedGB float64
		for _, a := range allocations {
			for _, g := range a.GPUs {
				if g == i {
					pct := 0.0
					if totalGB > 0 {
						pct = (a.LoadOn(i) / totalGB) * 100
					}
					bars[i].Segments = append(bars[i].Segments, gpuSegment{
						ModelName: a.ModelName,
						Color:     a.Color,
						WidthPct:  pct,
						Reserved:  a.Reserved,
					})
					usedGB += a.LoadOn(i)
					break
				}
			}
		}
		bars[i].UsedGB = usedGB
		bars[i].FreeGB = totalGB - usedGB
		bars[i].Overcommit = usedGB > totalGB
	}
	return bars
}

// gpuMapAllocations is every enabled LLM as its launch splits it, then
// each enabled generation instance's reservation, in the map's colors.
// The sidebar's map picks the ones loaded now from the same list, so a
// model has the same color in both.
func (s *Server) gpuMapAllocations(numGPUs int) []models.GPUAllocation {
	// Helper models are loaded only for a moment during Autoconfigure, so
	// they are not part of what is planned onto the GPUs.
	allModels := s.registry.ListServing()
	allConfigs := make(map[string]*models.ModelConfig)
	for _, m := range allModels {
		if c, err := s.registry.GetConfig(m.ID); err == nil {
			allConfigs[m.ID] = c
		}
	}

	// Each model as its launch splits it: by the cards' room, beside the
	// generation reservations or, for a model that does not fit beside
	// them with auto-evict on, on the whole cards, as the preset was
	// written (targetFor).
	room := models.RoomOf(s.idleHardware())
	if len(room.Shared) != numGPUs {
		room = models.CardRoom{Shared: make([]int, numGPUs)}
	}
	allocations := models.ComputeAllocationsOn(allModels, allConfigs, room)
	allocations = append(allocations, s.genReservationAllocations(len(allocations))...)
	return append(allocations, s.forkReservationAllocations(len(allocations))...)
}
