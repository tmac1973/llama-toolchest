package models

import (
	"strings"
	"testing"
)

// --moe-cache-mib is written only beside experts kept in system memory,
// and only for a build that splits the cache the way the estimate does.
func TestPresetWritesTheMoECache(t *testing.T) {
	write := func(cfg *ModelConfig, tgt Target) string {
		var b strings.Builder
		writeConfigParams(&b, cfg, false, tgt)
		return b.String()
	}
	cfg := &ModelConfig{GPULayers: 999, CPUMoE: 20, MoECacheMiB: 4096}
	if got := write(cfg, Target{}); !strings.Contains(got, "moe-cache-mib = 4096\n") {
		t.Errorf("current build: cache not written:\n%s", got)
	}
	if got := write(cfg, Target{Version: versionMoECache - 1}); strings.Contains(got, "moe-cache-mib") {
		t.Errorf("older build: cache written, which stops the router at startup:\n%s", got)
	}
	if got := write(&ModelConfig{GPULayers: 999, MoECacheMiB: 4096}, Target{}); strings.Contains(got, "moe-cache-mib") {
		t.Errorf("no experts in system memory: cache written:\n%s", got)
	}
}

// The cache is GPU memory up to the size asked for, never more than the
// experts it caches, and not part of the buffers llama.cpp itemises.
func TestVRAMEstimateCountsTheMoECache(t *testing.T) {
	m := flashNext()
	cfg := &ModelConfig{GPUAssign: "all", GPULayers: 999, CPUMoE: 32, ContextSize: 32768, KVCacheQuant: "q8_0", FlashAttention: true}
	without := VRAMBreakdownForConfigOn(m, cfg, 3)

	withCache := *cfg
	withCache.MoECacheMiB = 4096
	b := VRAMBreakdownForConfigOn(m, &withCache, 3)
	if b.MoECache != 4 {
		t.Errorf("MoECache = %.2f GiB, want 4", b.MoECache)
	}
	if d := b.Total() - without.Total(); d < 3.99 || d > 4.01 {
		t.Errorf("total grew by %.2f GiB, want 4", d)
	}
	if b.Reported() != without.Reported() {
		t.Errorf("Reported changed (%.2f -> %.2f): llama.cpp does not report the cache as a buffer", without.Reported(), b.Reported())
	}

	withCache.MoECacheMiB = 1 << 20 // 1 TiB: far more than the experts
	if b := VRAMBreakdownForConfigOn(m, &withCache, 3); b.MoECache > BytesToGiB(cpuExpertBytes(m, cfg))+0.01 {
		t.Errorf("MoECache = %.1f GiB, more than the %.1f GiB of experts it caches", b.MoECache, BytesToGiB(cpuExpertBytes(m, cfg)))
	}
}

// The planner owns GPU memory, so it plans without a cache and says so.
func TestPlanFitLeavesTheMoECacheOff(t *testing.T) {
	m := flashNext()
	hw := Hardware{LogicalCores: 32, RAMTotalMiB: 64 * 1024}
	for i := range 3 {
		hw.GPUs = append(hw.GPUs, GPUSpec{Index: i, Name: "RTX A4000", VRAMTotalMiB: 16376})
	}
	base := DefaultConfig()
	base.MoECacheMiB = 8192
	r := PlanFit(m, base, hw, ContextMedium)
	if r.Config.MoECacheMiB != 0 {
		t.Errorf("the plan kept a %d MiB cache", r.Config.MoECacheMiB)
	}
	noted := false
	for _, n := range r.Notes {
		noted = noted || n.Field == "moe_cache_mib"
	}
	if !noted {
		t.Error("dropping the cache is not explained")
	}
}

// The largest cache that fits does fit, on every card, and one step more
// does not.
func TestMaxMoECacheMiB(t *testing.T) {
	m := flashNext()
	hw := Hardware{LogicalCores: 32, RAMTotalMiB: 64 * 1024}
	for i := range 3 {
		hw.GPUs = append(hw.GPUs, GPUSpec{Index: i, Name: "RTX A4000", VRAMTotalMiB: 16376})
	}
	plan := PlanFit(m, DefaultConfig(), hw, ContextMedium)
	if !plan.Fits || plan.Config.CPUMoE == 0 {
		t.Fatalf("plan: %+v", plan.Config)
	}
	got := MaxMoECacheMiB(m, plan.Config, hw)
	if got <= 0 {
		t.Fatalf("MaxMoECacheMiB = %d, want room for a cache beside a plan with %d CPU expert layers", got, plan.Config.CPUMoE)
	}
	budget, ram := FitBudgets(hw)
	at := func(mib int) bool {
		c := plan.Config
		c.MoECacheMiB = mib
		_, ok := fitsOn(m, &c, 3, budget, ram, GPUMiB(hw))
		return ok
	}
	if !at(got) {
		t.Errorf("%d MiB was offered but does not fit", got)
	}
	if int64(got)<<20 < cpuExpertBytes(m, &plan.Config) && at(got+moeCacheStepMiB) {
		t.Errorf("%d MiB is not the largest: %d fits too", got, got+moeCacheStepMiB)
	}

	noExperts := plan.Config
	noExperts.CPUMoE = 0
	if got := MaxMoECacheMiB(m, noExperts, hw); got != 0 {
		t.Errorf("no experts in system memory: MaxMoECacheMiB = %d, want 0", got)
	}
}
