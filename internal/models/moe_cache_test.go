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
		writeConfigParams(&b, cfg, KindChat, tgt)
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

// --moe-cache-cpu-share is written only with the cache it shares out,
// only when set, and only for a build that takes it.
func TestPresetWritesTheMoECacheCPUShare(t *testing.T) {
	write := func(cfg *ModelConfig, tgt Target) string {
		var b strings.Builder
		writeConfigParams(&b, cfg, KindChat, tgt)
		return b.String()
	}
	cfg := &ModelConfig{GPULayers: 999, CPUMoE: 20, MoECacheMiB: 1024, MoECacheCPUShare: 0.9}
	if got := write(cfg, Target{}); !strings.Contains(got, "moe-cache-cpu-share = 0.9\n") {
		t.Errorf("current build: share not written:\n%s", got)
	}
	if got := cfg.EffectiveFlagsFor(KindChat, Target{}); !strings.Contains(got, "--moe-cache-cpu-share 0.9") {
		t.Errorf("current build: share not in the effective flags: %s", got)
	}
	if got := write(cfg, Target{Version: versionMoECacheCPUShare - 1}); strings.Contains(got, "moe-cache-cpu-share") {
		t.Errorf("older build: share written, which stops the router at startup:\n%s", got)
	}
	if got := write(&ModelConfig{GPULayers: 999, CPUMoE: 20, MoECacheCPUShare: 0.9}, Target{}); strings.Contains(got, "moe-cache-cpu-share") {
		t.Errorf("cache off: share written:\n%s", got)
	}
	if got := write(&ModelConfig{GPULayers: 999, CPUMoE: 20, MoECacheMiB: 1024}, Target{}); strings.Contains(got, "moe-cache-cpu-share") {
		t.Errorf("share not set: written anyway:\n%s", got)
	}
}

// The cache is GPU memory up to the size asked for, never more than the
// experts it caches, and part of what llama.cpp itemises.
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
	if d := b.Reported() - without.Reported(); d < 3.99 || d > 4.01 {
		t.Errorf("Reported grew by %.2f GiB, want 4: llama.cpp reports the cache's size", d)
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
		// The old 8% margin (1,311 MiB) leaves room for a cache.
		hw.GPUs = append(hw.GPUs, GPUSpec{Index: i, Name: "RTX A4000", VRAMTotalMiB: 16376, MarginMiB: 1311})
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

// With auto-evict on, a profile that fits only once the backends in its
// way are stopped was planned on the whole cards, and its cache is sized
// there too. Beside the reservations nothing fitted, so Autotune skipped
// the expert-cache stage on compute2 for every such model. With
// auto-evict off the reservations stay, and so does the answer.
func TestMaxMoECacheMiBUsesTheWholeCardsAPlanNeeds(t *testing.T) {
	m := flashNext()
	hw := evictAll(compute2Reserved())
	plan := PlanFit(m, DefaultConfig(), hw, ContextMedium)
	if !plan.Fits || plan.Config.CPUMoE == 0 {
		t.Fatalf("plan: %+v", plan.Config)
	}
	// The plan keeps the fewest expert layers in system memory that fit,
	// which leaves the cards full. A profile keeping a few more there, as
	// Autotune's starting profiles often do, leaves room to cache them.
	plan.Config.CPUMoE += 4
	if _, fits := maxMoECacheOn(m, plan.Config, hw); fits {
		t.Fatal("the plan fits beside the reservations; the fixture does not test the whole cards")
	}
	if got := MaxMoECacheMiB(m, plan.Config, hw); got <= 0 {
		t.Errorf("MaxMoECacheMiB = %d with auto-evict on, want room on the whole cards", got)
	}
	if got := MaxMoECacheMiB(m, plan.Config, compute2Reserved()); got != 0 {
		t.Errorf("MaxMoECacheMiB = %d with auto-evict off, want 0: the reservations stay", got)
	}
}
