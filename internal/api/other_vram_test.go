package api

import (
	"slices"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/memreport"
	"github.com/tmac1973/llama-toolchest/internal/process"
)

func modelStatus(id, state string) process.ModelStatus {
	var ms process.ModelStatus
	ms.ID = id
	ms.Status.Value = state
	return ms
}

func reports(r map[string]memreport.Report) func(string) (memreport.Report, bool) {
	return func(name string) (memreport.Report, bool) {
		rep, ok := r[name]
		return rep, ok
	}
}

func TestAttributeUse(t *testing.T) {
	used := []int{12000, 50}
	flash := memreport.Report{Entries: []memreport.Entry{
		{Device: "ROCm0", Kind: memreport.KindModel, MiB: 9000},
		{Device: "ROCm0", Kind: memreport.KindKV, MiB: 1000},
		{Device: "ROCm_Host", Kind: memreport.KindCompute, MiB: 400}, // system memory: not ours on the card
		{Device: "CPU_Mapped", Kind: memreport.KindModel, MiB: 30000},
	}}
	split := memreport.Report{Entries: []memreport.Entry{{Device: "Meta(ROCm0,ROCm1)", Kind: memreport.KindModel, MiB: 9000}}}

	for _, tt := range []struct {
		name      string
		running   bool
		loaded    []process.ModelStatus
		reports   map[string]memreport.Report
		wantOther []int
		wantIdle  bool
	}{
		{"router stopped: all of it is someone else's", false, nil, nil, used, true},
		{"router running with nothing loaded", true, []process.ModelStatus{modelStatus("a", "unloaded")}, nil, used, true},
		{"a model loaded: less what llama.cpp reported for it", true,
			[]process.ModelStatus{modelStatus("a", "loaded")}, map[string]memreport.Report{"a": flash}, []int{2000, 50}, false},
		{"still loading: not known from this reading", true, []process.ModelStatus{modelStatus("a", "loading")}, nil, nil, false},
		{"loaded with no report (verbosity below 4)", true, []process.ModelStatus{modelStatus("a", "loaded")}, nil, nil, false},
		{"a tensor-parallel aggregate cannot be put on a card", true,
			[]process.ModelStatus{modelStatus("a", "loaded")}, map[string]memreport.Report{"a": split}, nil, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			other, idle := attributeUse(used, tt.running, tt.loaded, reports(tt.reports))
			if idle != tt.wantIdle || !slices.Equal(other, tt.wantOther) {
				t.Errorf("got %v idle=%v, want %v idle=%v", other, idle, tt.wantOther, tt.wantIdle)
			}
		})
	}
}

func TestDeviceIndex(t *testing.T) {
	for dev, want := range map[string]int{"CUDA0": 0, "CUDA2": 2, "ROCm1": 1, "Vulkan0": 0} {
		if got, ok := deviceIndex(dev); !ok || got != want {
			t.Errorf("deviceIndex(%q) = %d, %v", dev, got, ok)
		}
	}
	for _, dev := range []string{"CPU", "ROCm_Host", "Meta(CUDA0,CUDA1)"} {
		if _, ok := deviceIndex(dev); ok {
			t.Errorf("deviceIndex(%q) gave an index", dev)
		}
	}
}
