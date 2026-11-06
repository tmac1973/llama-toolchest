package api

import (
	"math"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/monitor"
)

func near(a, b float64) bool { return math.Abs(a-b) < 1e-9 }

// Each allocation draws a segment on the cards it uses, as a share of the
// card's VRAM; used and free add up to the card, and a card holding more
// than it has is overcommitted.
func TestGPUMapBars(t *testing.T) {
	gpus := []monitor.GPUInfo{
		{Index: 0, Name: "A", VRAMTotalMB: 16384},
		{Index: 1, Name: "B", VRAMTotalMB: 8192},
	}
	allocs := []models.GPUAllocation{
		{ModelName: "split", GPUs: []int{0, 1}, GPULoadGB: map[int]float64{0: 4, 1: 6}, Color: "#1"},
		{ModelName: "gen", GPUs: []int{1}, PerGPUGB: 3, Color: "#2", Reserved: true},
	}
	bars := gpuMapBars(gpus, allocs)
	if len(bars) != 2 {
		t.Fatalf("got %d bars, want 2", len(bars))
	}

	a := bars[0]
	if len(a.Segments) != 1 || !near(a.Segments[0].WidthPct, 25) || !near(a.UsedGB, 4) || !near(a.FreeGB, 12) || a.Overcommit {
		t.Errorf("GPU 0 bar = %+v; want one 25%% segment, 4 used, 12 free", a)
	}
	b := bars[1]
	if len(b.Segments) != 2 || !near(b.UsedGB, 9) || !near(b.FreeGB, -1) || !b.Overcommit {
		t.Errorf("GPU 1 bar = %+v; want two segments, 9 of 8 GiB used, overcommitted", b)
	}
	if !b.Segments[1].Reserved || b.Segments[1].ModelName != "gen" || !near(b.Segments[1].WidthPct, 37.5) {
		t.Errorf("GPU 1 reservation segment = %+v", b.Segments[1])
	}
}

// A card that reports no VRAM gets segments of width 0.
func TestGPUMapBarsZeroVRAM(t *testing.T) {
	bars := gpuMapBars([]monitor.GPUInfo{{Index: 0}},
		[]models.GPUAllocation{{ModelName: "m", GPUs: []int{0}, PerGPUGB: 2}})
	if len(bars[0].Segments) != 1 || bars[0].Segments[0].WidthPct != 0 || !bars[0].Overcommit {
		t.Errorf("zero-VRAM bar = %+v; want a 0%% segment and overcommit", bars[0])
	}
}

// With no models and no instances the map has no allocations, also when
// the hardware the server sees has another GPU count than the monitor.
func TestGPUMapAllocationsEmpty(t *testing.T) {
	s := newTestServer(t)
	if got := s.gpuMapAllocations(3); len(got) != 0 {
		t.Errorf("allocations = %+v, want none", got)
	}
}
