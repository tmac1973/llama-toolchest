package api

import (
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/benchmark"
	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/monitor"
)

// The board in compute2: GPU 0 behind the chipset, GPUs 1 and 2 on the
// CPU.
func compute2Links() []monitor.GPUInfo {
	return []monitor.GPUInfo{
		{Index: 0, PCIe: &monitor.PCIeLink{Gen: 4, Lanes: 4, CardLanes: 16, Shared: true}},
		{Index: 1, PCIe: &monitor.PCIeLink{Gen: 4, Lanes: 8, CardLanes: 16}},
		{Index: 2, PCIe: &monitor.PCIeLink{Gen: 4, Lanes: 8, CardLanes: 16}},
	}
}

func TestWithGPULinksNamesEachGPUsLanes(t *testing.T) {
	opts := withGPULinks(models.GPUAssignOptions(3, nil), compute2Links())
	want := map[string]string{
		"0":        "x4 shared",
		"1-2":      "x8, x8",
		"0-1":      "x4 shared, x8",
		"all":      "x4 shared, x8, x8",
		"tensor-2": "x4 shared, x8",
		"custom":   "",
	}
	seen := 0
	for _, o := range opts {
		if w, ok := want[o.Value]; ok {
			seen++
			if o.LinkLabel != w {
				t.Errorf("%s: link label = %q, want %q", o.Value, o.LinkLabel, w)
			}
		}
	}
	if seen != len(want) {
		t.Errorf("found %d of the %d options checked", seen, len(want))
	}
}

// A GPU whose link is not known leaves the option without a label, so the
// list does not read as if that GPU had no lanes.
func TestWithGPULinksNeedsEveryGPU(t *testing.T) {
	gpus := compute2Links()
	gpus[1].PCIe = nil
	for _, o := range withGPULinks(models.GPUAssignOptions(3, nil), gpus) {
		if o.Value == "0-1" && o.LinkLabel != "" {
			t.Errorf("0-1: link label = %q, want none", o.LinkLabel)
		}
		if o.Value == "0" && o.LinkLabel != "x4 shared" {
			t.Errorf("0: link label = %q, want x4 shared", o.LinkLabel)
		}
	}
}

// A benchmark keeps the link the card had when it ran.
func TestGPUSnapshotsKeepTheLink(t *testing.T) {
	snaps := benchmark.GPUSnapshotsFromMetrics(monitor.Metrics{GPU: append(compute2Links(), monitor.GPUInfo{Index: 3})})
	if snaps[0].PCIe != "PCIe 4.0 x4, shared" || snaps[1].PCIe != "PCIe 4.0 x8" || snaps[3].PCIe != "" {
		t.Errorf("snapshot links = %q, %q, %q", snaps[0].PCIe, snaps[1].PCIe, snaps[3].PCIe)
	}
}
