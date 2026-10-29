package api

import (
	"fmt"
	"slices"
	"strings"

	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/monitor"
)

// Kinds of entry on the sidebar's map.
const (
	sidebarLLM     = "llm"     // an LLM the router has loaded
	sidebarLoading = "loading" // an LLM the router is still loading
	sidebarGen     = "gen"     // a running image or music backend
	sidebarStopped = "stopped" // a stopped backend's reservation
)

// sidebarSegment is one entry's share of one GPU on the sidebar's map.
type sidebarSegment struct {
	Name     string
	Color    string
	Kind     string
	WidthPct float64
	Title    string
}

// sidebarMapGPU is one GPU's bar on the sidebar's map.
type sidebarMapGPU struct {
	Index    int
	Segments []sidebarSegment
}

// sidebarLegendEntry names one entry under the sidebar's map.
type sidebarLegendEntry struct {
	Name  string
	Color string
	Kind  string
	Title string
}

// sidebarMap is the small map above the sidebar's meters: only what is
// on the GPUs now (loaded LLMs, running backends, and what stopped
// backends keep reserved), where the Server page's map shows every
// enabled model.
type sidebarMap struct {
	GPUs   []sidebarMapGPU
	Legend []sidebarLegendEntry
}

// sidebarMapData builds the sidebar's map from the GPU map's entries.
// Empty when there are no GPUs or nothing is on them.
func (s *Server) sidebarMapData(m monitor.Metrics) sidebarMap {
	if len(m.GPU) == 0 {
		return sidebarMap{}
	}
	loaded := map[string]string{} // registry ID → router status
	if s.process.IsRunning() {
		if list, err := s.process.ListModels(); err == nil {
			for _, rm := range list {
				if rm.Status.Value != "loaded" && rm.Status.Value != "loading" {
					continue
				}
				if reg, _ := s.resolveRouterModel(rm); reg != nil {
					loaded[reg.ID] = rm.Status.Value
					if rm.Status.Value == "loaded" {
						s.noteModelLoaded(reg)
					}
				}
			}
		}
	}
	running := map[string]bool{} // by allocation ID, as genReservationAllocations names them
	if s.gen != nil && s.genMgr != nil {
		for _, inst := range s.gen.Instances() {
			running["gen:"+inst.ID] = s.genMgr.IsActive(inst.ID)
		}
	}
	allocs := s.gpuMapAllocations(len(m.GPU))
	if s.genEvictOn() {
		// A stopped backend gives its memory up, so it is not on the card.
		allocs = slices.DeleteFunc(allocs, func(a models.GPUAllocation) bool { return a.Reserved && !running[a.ModelID] })
	}
	return buildSidebarMap(m.GPU, allocs, loaded, running)
}

// buildSidebarMap keeps the allocations that are on the GPUs now: LLMs
// in loaded (registry ID → router status) and every generation
// reservation, solid while its backend runs (running, by allocation ID)
// and outlined while it is stopped. Widths are shares of each card's
// total memory, as on the Server page's map.
func buildSidebarMap(gpus []monitor.GPUInfo, allocs []models.GPUAllocation, loaded map[string]string, running map[string]bool) sidebarMap {
	var out sidebarMap
	for i, gpu := range gpus {
		bar := sidebarMapGPU{Index: gpu.Index}
		totalGB := float64(gpu.VRAMTotalMB) / 1024
		for _, a := range allocs {
			kind, ok := sidebarKind(a, loaded, running)
			if !ok || !onGPU(a, i) || totalGB <= 0 {
				continue
			}
			gb := a.LoadOn(i)
			bar.Segments = append(bar.Segments, sidebarSegment{
				Name: sidebarName(a), Color: a.Color, Kind: kind,
				WidthPct: min(100, gb/totalGB*100),
				Title:    fmt.Sprintf("%s: about %.1f GiB on GPU %d (%s)", sidebarName(a), gb, gpu.Index, sidebarKindText(kind, loaded[a.ModelID])),
			})
		}
		if len(bar.Segments) > 0 {
			out.GPUs = append(out.GPUs, bar)
		}
	}
	if len(out.GPUs) == 0 {
		return sidebarMap{}
	}
	for _, a := range allocs {
		kind, ok := sidebarKind(a, loaded, running)
		if !ok {
			continue
		}
		name := sidebarName(a)
		if kind == sidebarLoading {
			name += " (loading)"
		}
		out.Legend = append(out.Legend, sidebarLegendEntry{
			Name: name, Color: a.Color, Kind: kind,
			Title: fmt.Sprintf("%s: about %.1f GiB (%s)", sidebarName(a), a.TotalGB, sidebarKindText(kind, loaded[a.ModelID])),
		})
	}
	return out
}

// sidebarKind says how the sidebar's map shows an allocation, and false
// when it is an LLM that is not loaded.
func sidebarKind(a models.GPUAllocation, loaded map[string]string, running map[string]bool) (string, bool) {
	switch {
	case !a.Reserved:
		// A loading model is drawn apart: its planned size shows at once,
		// while the card fills up only as the weights are read, which
		// takes half a minute for a large one.
		status, ok := loaded[a.ModelID]
		if status == "loading" {
			return sidebarLoading, ok
		}
		return sidebarLLM, ok
	case running[a.ModelID]:
		return sidebarGen, true
	default:
		return sidebarStopped, true
	}
}

// sidebarKindText is the tooltip's word for what an entry is.
func sidebarKindText(kind, routerStatus string) string {
	switch kind {
	case sidebarGen:
		return "generation backend, running"
	case sidebarStopped:
		return "reserved for a generation backend that is stopped"
	case sidebarLoading:
		return "LLM, loading: the planned size; the VRAM bars below fill up as it loads"
	}
	return "LLM, loaded"
}

// sidebarName is an allocation's name without the Server map's
// " (reserved)", which the sidebar's tooltips say in words instead.
func sidebarName(a models.GPUAllocation) string {
	return strings.TrimSuffix(a.ModelName, " (reserved)")
}

// onGPU reports whether allocation a puts anything on GPU index i.
func onGPU(a models.GPUAllocation, i int) bool {
	for _, g := range a.GPUs {
		if g == i {
			return true
		}
	}
	return false
}
