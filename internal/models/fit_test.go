package models

import (
	"reflect"
	"strings"
	"testing"
)

const fitGiB = int64(1 << 30)

// Synthetic models sized like real ones, using the uniform KV estimate.
var (
	dense8B = &Model{NLayers: 36, NEmbd: 4096, NHead: 32, NKVHead: 8,
		ContextLength: 131072, SizeBytes: 49 * fitGiB / 10}
	moe30B = &Model{NLayers: 48, NEmbd: 2048, NHead: 32, NKVHead: 4,
		ContextLength: 262144, SizeBytes: 177 * fitGiB / 10,
		ExpertCount: 128, ExpertUsedCount: 8, ExpertBytes: 165 * fitGiB / 10, ExpertLayerFirst: 0, ExpertLayers: 48}
	dense70B = &Model{NLayers: 80, NEmbd: 8192, NHead: 64, NKVHead: 8,
		ContextLength: 131072, SizeBytes: 40 * fitGiB}
)

func oneGPU(mib int) Hardware {
	return Hardware{GPUs: []GPUSpec{{Index: 0, Name: "dGPU", VRAMTotalMiB: mib}}, LogicalCores: 16, RAMTotalMiB: 64 * 1024}
}

func defaultBase() ModelConfig {
	return ModelConfig{Enabled: true, GPULayers: 999, ContextSize: 8192, Threads: 8, FlashAttention: true, Jinja: true}
}

// fieldJSONNames maps the ModelConfig fields the planner may change to the
// JSON names its notes use.
var plannedFields = map[string]string{
	"ContextSize": "context_size", "KVCacheQuant": "kv_cache_quant", "CPUMoE": "cpu_moe",
	"GPULayers": "gpu_layers", "GPUAssign": "gpu_assign", "FlashAttention": "flash_attention",
	"UBatchSize": "ubatch_size", "Parallel": "parallel", "Threads": "threads",
}

// assertOneNotePerChange checks that every setting the plan changed has
// exactly one note, and that no note describes an unchanged one.
func assertOneNotePerChange(t *testing.T, base ModelConfig, r FitResult) {
	t.Helper()
	count := map[string]int{}
	for _, n := range r.Notes {
		count[n.Field]++
		if n.Origin != "hardware fit" {
			t.Errorf("note %q has origin %q", n.Field, n.Origin)
		}
	}
	bv, rv := reflect.ValueOf(base), reflect.ValueOf(r.Config)
	for field, key := range plannedFields {
		changed := !reflect.DeepEqual(bv.FieldByName(field).Interface(), rv.FieldByName(field).Interface())
		if field == "UBatchSize" {
			changed = base.BatchSize != r.Config.BatchSize || base.UBatchSize != r.Config.UBatchSize
		}
		switch {
		case changed && count[key] != 1:
			t.Errorf("%s changed but has %d notes", key, count[key])
		case !changed && count[key] != 0:
			t.Errorf("%s unchanged but has a note", key)
		}
	}
}

func TestPlanFit(t *testing.T) {
	cases := []struct {
		name  string
		m     *Model
		hw    Hardware
		class ContextClass
		check func(t *testing.T, r FitResult)
	}{
		{"8B dense, 24 GiB card, medium", dense8B, oneGPU(24 * 1024), ContextMedium, func(t *testing.T, r FitResult) {
			c := r.Config
			if !r.Fits || c.ContextSize != 32768 || c.KVCacheQuant != "" || c.GPULayers != 999 || c.CPUMoE != 0 {
				t.Errorf("want everything on the GPU at 32K with f16: %+v", c)
			}
			if c.Threads != 8 {
				t.Errorf("threads changed to %d with nothing on the CPU", c.Threads)
			}
		}},
		{"8B dense, 12 GiB card, max", dense8B, oneGPU(12 * 1024), ContextMax, func(t *testing.T, r FitResult) {
			c := r.Config
			if !r.Fits || c.ContextSize >= 131072 || c.KVCacheQuant != "q8_0" || c.GPULayers != 999 {
				t.Errorf("want a reduced context with q8_0, all layers on the GPU: %+v", c)
			}
			if !notesMention(r, "context_size", "did not fit") {
				t.Error("the reduced context is not explained")
			}
		}},
		{"30B-A3B MoE, 16 GiB card, medium", moe30B, oneGPU(16 * 1024), ContextMedium, func(t *testing.T, r FitResult) {
			c := r.Config
			if !r.Fits || c.CPUMoE == 0 || c.GPULayers != 999 || c.ContextSize != 32768 {
				t.Errorf("want experts in system memory, all layers on the GPU, 32K: %+v", c)
			}
			if c.Threads != 8 {
				t.Errorf("threads = %d, want 8 (half of 16 logical cores)", c.Threads)
			}
			// The smallest number that fits: one fewer must not.
			less := c
			less.CPUMoE--
			if VRAMEstimateForConfigOn(moe30B, &less, 1) <= r.BudgetGiB {
				t.Errorf("cpu_moe %d is not the smallest that fits", c.CPUMoE)
			}
			if r.CPURAMGiB <= 0 {
				t.Error("no system memory reported for the moved experts")
			}
		}},
		{"70B dense, 8 GiB card", dense70B, oneGPU(8 * 1024), ContextMedium, func(t *testing.T, r FitResult) {
			c := r.Config
			if !r.Fits || c.GPULayers <= 0 || c.GPULayers >= 80 || c.CPUMoE != 0 {
				t.Errorf("want a partial GPU offload: %+v", c)
			}
		}},
		{"too large for RAM and VRAM", &Model{NLayers: 80, NEmbd: 8192, NHead: 64, NKVHead: 8,
			ContextLength: 131072, SizeBytes: 200 * fitGiB}, oneGPU(8 * 1024), ContextShort, func(t *testing.T, r FitResult) {
			if r.Fits {
				t.Errorf("a 200 GiB model fits in 8 GiB of VRAM and 64 GiB of RAM: %+v", r.Config)
			}
			if !notesMention(r, "", "too large") {
				t.Error("no note says the model is too large")
			}
		}},
		{"two cards", dense8B, Hardware{GPUs: []GPUSpec{{Index: 0, VRAMTotalMiB: 24 * 1024}, {Index: 1, VRAMTotalMiB: 24 * 1024}},
			LogicalCores: 16}, ContextLong, func(t *testing.T, r FitResult) {
			if r.Config.GPUAssign != "all" || r.BudgetGiB < 44 || r.BudgetGiB > 45 {
				t.Errorf("GPUAssign = %q, budget %.1f; want all, about 44", r.Config.GPUAssign, r.BudgetGiB)
			}
		}},
		{"integrated GPU left out", dense8B, Hardware{GPUs: []GPUSpec{{Index: 0, VRAMTotalMiB: 16 * 1024}, {Index: 1, VRAMTotalMiB: 2 * 1024, IsIGPU: true}},
			LogicalCores: 16}, ContextShort, func(t *testing.T, r FitResult) {
			if r.Config.GPUAssign != "0" || r.BudgetGiB > 16 {
				t.Errorf("GPUAssign = %q, budget %.1f; want the dedicated GPU alone", r.Config.GPUAssign, r.BudgetGiB)
			}
		}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			base := defaultBase()
			r := PlanFit(c.m, base, c.hw, c.class)
			c.check(t, r)
			assertOneNotePerChange(t, base, r)
			if r.Fits && r.EstimateGiB > r.BudgetGiB {
				t.Errorf("reported as fitting at %.1f GiB over a %.1f GiB budget", r.EstimateGiB, r.BudgetGiB)
			}
		})
	}
}

// The planner keeps what it does not own: sampling, aliases, speculative
// decoding.
func TestPlanFitKeepsOtherSettings(t *testing.T) {
	base := defaultBase()
	temp := 0.6
	base.Temperature = &temp
	base.Aliases = []string{"x"}
	base.SpecType = "draft-mtp"
	r := PlanFit(dense8B, base, oneGPU(24*1024), ContextShort)
	if r.Config.Temperature == nil || *r.Config.Temperature != 0.6 || r.Config.SpecType != "draft-mtp" || len(r.Config.Aliases) != 1 {
		t.Errorf("planner changed settings it does not own: %+v", r.Config)
	}
}

func notesMention(r FitResult, field, text string) bool {
	for _, n := range r.Notes {
		if n.Field == field && strings.Contains(n.Reason, text) {
			return true
		}
	}
	return false
}

func TestGroupDigits(t *testing.T) {
	for in, want := range map[int]string{0: "0", 999: "999", 4096: "4,096", 131072: "131,072", 1048576: "1,048,576"} {
		if got := groupDigits(in); got != want {
			t.Errorf("groupDigits(%d) = %q, want %q", in, got, want)
		}
	}
}

// The planner offered a 27B model its full 262,144-token context on
// three 16 GiB cards. It did not load: the estimate was 6 GiB short of
// what the machine actually used, because a layer split pays for several
// copies of the compute graph, a hybrid model keeps recurrent state, and
// speculative decoding runs a second context with a cache of its own.
// None of the three were modelled. Measured on that machine: 40.27 GiB
// at 131,072 tokens.
func TestPlanFitDoesNotOfferAContextThatCannotLoad(t *testing.T) {
	hw := Hardware{
		GPUs: []GPUSpec{
			{Index: 0, Name: "NVIDIA RTX A4000", VRAMTotalMiB: 16376},
			{Index: 1, Name: "NVIDIA RTX A4000", VRAMTotalMiB: 16376},
			{Index: 2, Name: "NVIDIA RTX A4000", VRAMTotalMiB: 16376},
		},
		LogicalCores: 16, RAMTotalMiB: 64225,
	}
	m := &Model{
		ID: "qwen3.8-27b", NLayers: 65, AttnLayers: 16, NEmbd: 5120, NHead: 24,
		NKVHead: 4, KVFullPerTok: 32768, ContextLength: 262144, NextNLayers: 1,
		SizeBytes: gibBytes(29.30), TokenEmbdBytes: gibBytes(1.258),
	}

	res := PlanFit(m, ModelConfig{Enabled: true, SpecType: "draft-mtp"}, hw, ContextMax)
	if !res.Fits {
		t.Fatalf("no context fitted at all; the planner proposed %d", res.Config.ContextSize)
	}
	if res.Config.ContextSize >= 262144 {
		t.Errorf("planned %d tokens, which ran out of memory on the machine this is from",
			res.Config.ContextSize)
	}
	if res.Config.ContextSize < 32768 {
		t.Errorf("planned only %d tokens; the machine runs this model at 131072",
			res.Config.ContextSize)
	}

	// The estimate has to cover what the machine really used, or the
	// planner is back to proposing a config that cannot load.
	cfg := ModelConfig{ContextSize: 131072, KVCacheQuant: "q8_0", GPULayers: 999,
		SpecType: "draft-mtp", SplitMode: "layer", UBatchSize: 512}
	if got := VRAMEstimateForConfigOn(m, &cfg, 3); got < 40.27 {
		t.Errorf("estimate at 131072 is %.2f GiB against 40.27 measured", got)
	}
}

// A tensor-parallel split shares one set of graph buffers, so it must not
// be charged for the copies a layer split pays for.
func TestLayerSplitCostsMoreGraphScratchThanTensorParallel(t *testing.T) {
	m := &Model{NLayers: 65, AttnLayers: 16, NEmbd: 5120, NHead: 24, NKVHead: 4,
		KVFullPerTok: 32768, ContextLength: 262144, SizeBytes: gibBytes(29.30)}
	base := ModelConfig{ContextSize: 131072, UBatchSize: 512, GPULayers: 999}

	tensor, layer, one := base, base, base
	tensor.SplitMode = "tensor"
	layer.SplitMode = "layer"
	one.SplitMode = "layer"

	if a, b := VRAMBreakdownForConfigOn(m, &tensor, 3).Compute, VRAMBreakdownForConfigOn(m, &layer, 3).Compute; a >= b {
		t.Errorf("tensor-parallel scratch %.2f GiB is not less than a layer split's %.2f", a, b)
	}
	// One card cannot pipeline against itself, whatever the mode says.
	if a, b := VRAMBreakdownForConfigOn(m, &one, 1).Compute, VRAMBreakdownForConfigOn(m, &tensor, 1).Compute; a != b {
		t.Errorf("a single card was charged for a split: %.2f vs %.2f", a, b)
	}
}
