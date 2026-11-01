package api

import (
	"net/http"

	"github.com/tmac1973/llama-toolchest/internal/models"
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

	// Build per-GPU data for the template
	type GPUSegment struct {
		ModelName string
		Color     string
		WidthPct  float64
		Reserved  bool
	}
	type GPUBar struct {
		Index      int
		Name       string
		TotalGB    float64
		UsedGB     float64
		FreeGB     float64
		Segments   []GPUSegment
		Overcommit bool
	}

	bars := make([]GPUBar, numGPUs)
	for i, gpu := range metrics.GPU {
		totalGB := float64(gpu.VRAMTotalMB) / 1024.0
		bars[i] = GPUBar{
			Index:   gpu.Index,
			Name:    gpu.Name,
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
					bars[i].Segments = append(bars[i].Segments, GPUSegment{
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

	data := struct {
		Bars        []GPUBar
		Allocations []models.GPUAllocation
	}{
		Bars:        bars,
		Allocations: allocations,
	}
	s.renderPartial(w, "gpu_map", data)
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
