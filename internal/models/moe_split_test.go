package models

import (
	"slices"
	"strconv"
	"strings"
	"testing"
)

// flashNext is Qwen3.8-Flash-Next UD-Q4_K_XL as the registry records it
// on compute2: 48 layers, all with experts, a 26.8 GiB embedding table.
func flashNext() *Model {
	return &Model{
		SizeBytes: 111334654784, NLayers: 48, NEmbd: 2560, NHead: 24, NKVHead: 2,
		ContextLength: 262144, KVFullPerTok: 12288, AttnLayers: 12, IndexerKeyLength: 128,
		PLEBytes: 28800138240, TokenEmbdBytes: 675430400,
		ExpertCount: 512, ExpertUsedCount: 10, ExpertBytes: 77017907200, ExpertLayers: 48,
	}
}

func a4000s(n int) []int {
	out := make([]int, n)
	for i := range out {
		out[i] = 16376
	}
	return out
}

// The case that failed on compute2: experts of 32 layers in system
// memory over three 16 GiB cards. llama.cpp's own split by layer count
// asked one card for 24.9 GiB; a size-balanced split (-ts 36,6,6) loaded
// at 14.5 / 10.7 / 10.5 GiB.
func TestMoESplitBalancesFlashNext(t *testing.T) {
	m := flashNext()
	cfg := &ModelConfig{GPUAssign: "all", SplitMode: "layer", GPULayers: 999, CPUMoE: 32,
		ContextSize: 131072, KVCacheQuant: "q8_0", FlashAttention: true}

	// What llama.cpp does without a split: equal counts.
	b := VRAMBreakdownForConfigOn(m, cfg, 3)
	layers := layerGPUBytes(m, cfg, b)
	even := cardLoads(layers, []int{16, 16, 16}, []float64{0, 0, 0})
	if even[2] < 24 || even[0] > 4 {
		t.Errorf("equal split loads = %.1f, want the last card over 24 GiB of layers", even)
	}

	out := MoESplitConfig(cfg, m, a4000s(3))
	if out == cfg || out.SplitMode != "layer" || len(strings.Split(out.TensorSplit, ",")) != 3 {
		t.Fatalf("split = %+v", out)
	}
	if cfg.TensorSplit != "" {
		t.Error("MoESplitConfig changed its argument; the config form would undo a saved split")
	}
	loads, _, ok := PlanCardLoads(m, cfg, a4000s(3))
	if !ok {
		t.Fatal("PlanCardLoads did not apply")
	}
	// Balanced: no card carries more than a heavy layer or two above another.
	lo, hi := loads[0], loads[0]
	for _, l := range loads {
		lo, hi = min(lo, l), max(hi, l)
	}
	if hi-lo > 3.5 {
		t.Errorf("split %s gives %.1f GiB: unbalanced", out.TensorSplit, loads)
	}
}

// The planner checks each card, so the plan it returns fits every card
// under the split the preset writes.
func TestPlanFitChecksEachCard(t *testing.T) {
	m := flashNext()
	hw := Hardware{LogicalCores: 32, RAMTotalMiB: 64 * 1024}
	for i := range 3 {
		hw.GPUs = append(hw.GPUs, GPUSpec{Index: i, Name: "RTX A4000", VRAMTotalMiB: 16376})
	}
	for _, class := range []ContextClass{ContextMedium, ContextLong} {
		r := PlanFit(m, DefaultConfig(), hw, class)
		if !r.Fits || r.Config.CPUMoE == 0 {
			t.Fatalf("%s: %+v", class, r.Config)
		}
		loads, budgets, ok := PlanCardLoads(m, &r.Config, GPUMiB(hw))
		if !ok {
			t.Fatalf("%s: no per-card plan", class)
		}
		for i := range loads {
			if loads[i] > budgets[i] {
				t.Errorf("%s: card %d holds %.1f of %.1f GiB (cpu_moe %d)", class, i, loads[i], budgets[i], r.Config.CPUMoE)
			}
		}
	}
}

// llama.cpp counts n_layer+1 layers (the output layer last) and puts layer
// il on the first card whose cumulative share exceeds il/(n_layer+1).
func TestLayerDevicesFollowsLlamaCpp(t *testing.T) {
	got := layerDevices([]int{36, 6, 6}, 48)
	count := map[int]int{}
	for _, d := range got {
		count[d]++
	}
	// 36/48 = 0.75: layers 0..36 have il/49 < 0.75, so card 0 takes 37.
	if count[0] != 37 || count[1] != 6 || count[2] != 5 {
		t.Errorf("layers per card = %v", count)
	}
}

func TestMoESplitLeavesOtherConfigsAlone(t *testing.T) {
	m := flashNext()
	base := ModelConfig{GPUAssign: "all", SplitMode: "layer", GPULayers: 999, CPUMoE: 32, ContextSize: 32768}
	for name, mut := range map[string]func(*ModelConfig){
		"no expert offload": func(c *ModelConfig) { c.CPUMoE = 0 },
		"custom split":      func(c *ModelConfig) { c.GPUAssign, c.TensorSplit = "custom", "1,1,1" },
		"tensor split mode": func(c *ModelConfig) { c.SplitMode = "tensor" },
		"one GPU":           func(c *ModelConfig) { c.GPUAssign, c.TensorSplit = "0", "1,0,0" },
	} {
		cfg := base
		mut(&cfg)
		if out := MoESplitConfig(&cfg, m, a4000s(3)); out != &cfg {
			t.Errorf("%s: changed to %q", name, out.TensorSplit)
		}
	}
	if out := MoESplitConfig(&base, m, nil); out != &base {
		t.Error("changed without knowing the GPUs")
	}
}

// An excluded GPU (an integrated one) keeps a zero, so the preset still
// restricts the model to the dedicated cards.
func TestMoESplitKeepsExcludedGPUs(t *testing.T) {
	cfg := &ModelConfig{GPUAssign: "0-2", TensorSplit: "1,1,1,0", SplitMode: "layer", GPULayers: 999, CPUMoE: 32, ContextSize: 32768, KVCacheQuant: "q8_0"}
	out := MoESplitConfig(cfg, flashNext(), append(a4000s(3), 2048))
	parts := strings.Split(out.TensorSplit, ",")
	if len(parts) != 4 || parts[3] != "0" || parts[0] == "1" {
		t.Errorf("TensorSplit = %q", out.TensorSplit)
	}
}

// The preset carries the balanced split; without the GPUs' sizes it
// leaves llama.cpp to split by count, as before.
func TestPresetWritesTheBalancedSplit(t *testing.T) {
	m := flashNext()
	m.ID, m.FilePath = "flash", "/models/flash.gguf"
	cfg := &ModelConfig{Enabled: true, GPUAssign: "all", SplitMode: "layer", GPULayers: 999, CPUMoE: 32, ContextSize: 131072, KVCacheQuant: "q8_0"}
	cfgs := map[string]*ModelConfig{"flash": cfg}

	ini := GeneratePresetINI("/models", []*Model{m}, cfgs, Target{Backend: "cuda", GPUMiB: a4000s(3)})
	want := "tensor-split = " + MoESplitConfig(cfg, m, a4000s(3)).TensorSplit
	if !strings.Contains(ini, want) || !strings.Contains(ini, "split-mode = layer") {
		t.Errorf("preset lacks %q:\n%s", want, ini)
	}
	if ini := GeneratePresetINI("/models", []*Model{m}, cfgs, Target{Backend: "cuda"}); strings.Contains(ini, "tensor-split") {
		t.Errorf("a split was written without the GPU sizes:\n%s", ini)
	}
}

// dense20 is a dense 20 GiB model: every layer the same size.
func dense20() *Model {
	return &Model{SizeBytes: 20 << 30, NLayers: 48, NEmbd: 5120, NHead: 40, NKVHead: 8,
		ContextLength: 131072, KVFullPerTok: 8192}
}

// compute2Reserved is compute2 on 2026-10-25: three 16 GiB A4000s with
// FLUX.2 klein reserving 9.5 GiB on GPU 1 and ACE-Step 7.5 GiB on GPU 2.
func compute2Reserved() Hardware {
	hw := Hardware{LogicalCores: 32, RAMTotalMiB: 64 * 1024}
	for i, res := range []int{0, 9728, 7680} {
		hw.GPUs = append(hw.GPUs, GPUSpec{Index: i, Name: "RTX A4000", VRAMTotalMiB: 16376, ReservedMiB: res})
	}
	return hw
}

// On cards with different room, a plan is split by room, so it fits each
// card: the even split it used to get asked the reserved cards for more
// than they had left.
func TestPlanFitSplitsByRoomBesideReservations(t *testing.T) {
	m, hw := dense20(), compute2Reserved()
	r := PlanFit(m, DefaultConfig(), hw, ContextMedium)
	if !r.Fits {
		t.Fatalf("plan does not fit: %+v", r.Config)
	}
	gpuMiB := GPUMiB(hw)
	loads := CardLoadsMiB(m, &r.Config, gpuMiB)
	for _, g := range hw.GPUs {
		if loads[g.Index] > CardBudgetMiB(g) {
			t.Errorf("GPU %d holds %d MiB of %d MiB", g.Index, loads[g.Index], CardBudgetMiB(g))
		}
	}
	if loads[0] <= loads[1] || loads[0] <= loads[2] {
		t.Errorf("loads %v: the card without a reservation should take the most", loads)
	}
	// What an even split would have asked of GPU 1.
	even := int(VRAMEstimateForConfig(m, &r.Config) * 1024 / 3)
	if even <= CardBudgetMiB(hw.GPUs[1]) {
		t.Errorf("test does not cover the bug: an even split (%d MiB) fits GPU 1 (%d MiB)", even, CardBudgetMiB(hw.GPUs[1]))
	}
	if out := MoESplitConfig(&r.Config, m, gpuMiB); out == &r.Config || out.TensorSplit == "" {
		t.Error("the launch flags would not carry the balanced split")
	}
}

// BalancedSplit lists the split the launch flags carry, card by card, so
// the Configure panel can show it; nil when the flags carry none.
func TestBalancedSplitMatchesTheFlags(t *testing.T) {
	m, hw := dense20(), compute2Reserved()
	r := PlanFit(m, DefaultConfig(), hw, ContextMedium)
	gpuMiB := GPUMiB(hw)
	cards := BalancedSplit(m, &r.Config, gpuMiB)
	if len(cards) != 3 {
		t.Fatalf("cards = %+v, want all three", cards)
	}
	parts := strings.Split(MoESplitConfig(&r.Config, m, gpuMiB).TensorSplit, ",")
	layers := 0
	for i, c := range cards {
		if c.GPU != i || parts[i] != strconv.Itoa(c.Layers) {
			t.Errorf("card %+v does not match the flags' split %v", c, parts)
		}
		if c.LoadGiB <= 0 || c.LoadGiB > c.BudgetGiB {
			t.Errorf("card %+v: load should be above zero and within the budget", c)
		}
		layers += c.Layers
	}
	if layers != m.NLayers {
		t.Errorf("layers add up to %d, want %d", layers, m.NLayers)
	}

	cfg := &ModelConfig{GPUAssign: "all", SplitMode: "layer", GPULayers: 999, ContextSize: 32768}
	if cards := BalancedSplit(dense20(), cfg, a4000s(3)); cards != nil {
		t.Errorf("equal cards: got %+v, want nil", cards)
	}
}

// Cards with the same room keep llama.cpp's own split for a dense model.
func TestSplitLeftAloneOnEqualCards(t *testing.T) {
	cfg := &ModelConfig{GPUAssign: "all", SplitMode: "layer", GPULayers: 999, ContextSize: 32768}
	if out := MoESplitConfig(cfg, dense20(), a4000s(3)); out != cfg {
		t.Errorf("split changed to %q on equal cards", out.TensorSplit)
	}
	// Less than unequalRoomMiB apart counts as equal.
	if out := MoESplitConfig(cfg, dense20(), []int{16376, 16376 - 512, 16376}); out != cfg {
		t.Errorf("split changed to %q for a 512 MiB difference", out.TensorSplit)
	}
}

// A split with weights of its own is followed when working out each
// card's load.
func TestCardLoadsFollowExplicitSplit(t *testing.T) {
	m := dense20()
	cfg := &ModelConfig{GPUAssign: "custom", TensorSplit: "2,1,1", SplitMode: "layer", GPULayers: 999, ContextSize: 32768}
	loads := CardLoadsMiB(m, cfg, a4000s(3))
	if loads[0] < loads[1]*19/10 || loads[1] != loads[2] {
		t.Errorf("loads %v, want GPU 0 twice GPUs 1 and 2", loads)
	}
}

// The GPU map shows each card's own share where the launch splits by
// room, and the shares add up to the whole model.
func TestAllocationsFollowTheSplit(t *testing.T) {
	m := dense20()
	m.ID, m.ModelID = "dense", "org/dense"
	hw := compute2Reserved()
	cfg := PlanFit(m, DefaultConfig(), hw, ContextMedium).Config
	cfg.Enabled = true
	allocs := ComputeAllocationsOn([]*Model{m}, map[string]*ModelConfig{m.ID: &cfg}, RoomOf(hw))
	if len(allocs) != 1 || allocs[0].GPULoadGB == nil {
		t.Fatalf("allocations = %+v, want per-card shares", allocs)
	}
	a := allocs[0]
	if a.LoadOn(0) <= a.LoadOn(1) {
		t.Errorf("GPU 0 shows %.1f GiB, GPU 1 %.1f GiB; the card without a reservation should hold more", a.LoadOn(0), a.LoadOn(1))
	}
	sum := a.LoadOn(0) + a.LoadOn(1) + a.LoadOn(2)
	if sum < a.TotalGB-0.01 || sum > a.TotalGB+0.01 {
		t.Errorf("shares add up to %.1f GiB, but the total shown is %.1f", sum, a.TotalGB)
	}
	// Every card the model spans gets a compute buffer of its own, so
	// the shares are more than the one-card estimate.
	if a.TotalGB <= VRAMEstimateForConfig(m, &cfg) {
		t.Errorf("total %.1f GiB is not more than the one-card estimate %.1f", a.TotalGB, VRAMEstimateForConfig(m, &cfg))
	}
	// Equal cards: an even spread, as before.
	even := ComputeAllocations([]*Model{m}, map[string]*ModelConfig{m.ID: &cfg}, 3)
	if even[0].GPULoadGB != nil {
		t.Error("an allocation without card budgets got per-card shares")
	}
}

// dense32 is a dense 32 GiB model, the size of a 27B at 8 bits: too big
// for compute2 beside its reservations, small enough for the whole cards.
func dense32() *Model {
	return &Model{SizeBytes: 32 << 30, NLayers: 64, NEmbd: 5120, NHead: 40, NKVHead: 8,
		ContextLength: 131072, KVFullPerTok: 8192}
}

// evictAll marks every reservation on hw as one a loading model may take
// back, as auto-evict does for backends not kept running.
func evictAll(hw Hardware) Hardware {
	for i := range hw.GPUs {
		hw.GPUs[i].EvictableMiB = hw.GPUs[i].ReservedMiB
	}
	return hw
}

// With auto-evict on, a model that would run partly from system memory
// beside the reservations is planned on the whole cards, and its split
// and load check use the whole cards too.
func TestPlanFitTakesWholeCardsWhenSharingSpills(t *testing.T) {
	m := dense32()
	shared := PlanFit(m, DefaultConfig(), compute2Reserved(), ContextMedium)
	if shared.CPURAMGiB < 0.5 {
		t.Fatalf("without auto-evict the model was expected to spill to system memory; plan keeps %.1f GiB there", shared.CPURAMGiB)
	}

	hw := evictAll(compute2Reserved())
	r := PlanFit(m, DefaultConfig(), hw, ContextMedium)
	if !r.Fits || r.CPURAMGiB > 0.05 || r.Config.ContextSize < shared.Config.ContextSize {
		t.Fatalf("with auto-evict: fits %v, %.1f GiB in system memory, context %d; want all on the GPUs",
			r.Fits, r.CPURAMGiB, r.Config.ContextSize)
	}
	said := false
	for _, n := range r.Notes {
		said = said || strings.Contains(n.Reason, "use the whole GPUs")
	}
	if !said {
		t.Errorf("no note says the plan takes the whole GPUs: %+v", r.Notes)
	}

	room := RoomOf(hw)
	if got := room.For(m, &r.Config); !slices.Equal(got, room.Whole) {
		t.Fatalf("room for the plan = %v, want the whole cards %v", got, room.Whole)
	}
	for g, mib := range CardLoadsMiB(m, &r.Config, room.Whole) {
		if mib > room.Whole[g] {
			t.Errorf("GPU %d holds %d MiB of the whole-card split, more than its %d", g, mib, room.Whole[g])
		}
	}
	if loads := CardLoadsMiB(m, &r.Config, room.Shared); loads[1] <= room.Shared[1] {
		t.Errorf("the plan fits GPU 1 beside flux (%d of %d MiB); the test model is too small", loads[1], room.Shared[1])
	}
}

// At the longest context, a shared plan can keep it only by running
// layers on the CPU. A whole-card plan with every layer on the GPUs at a
// shorter context wins: the CPU is the last resort, as in PlanFit.
func TestPlanFitPrefersGPUOverContextOnWholeCards(t *testing.T) {
	// 36 GiB with a large KV cache: on the whole cards every layer fits
	// only at a shortened context.
	m := &Model{SizeBytes: 36 << 30, NLayers: 64, NEmbd: 5120, NHead: 40, NKVHead: 8,
		ContextLength: 262144, KVFullPerTok: 98304}
	hw := evictAll(compute2Reserved())
	shared := PlanFit(m, DefaultConfig(), compute2Reserved(), ContextMax)
	r := PlanFit(m, DefaultConfig(), hw, ContextMax)
	if shared.CPURAMGiB < 0.5 {
		t.Fatalf("the shared plan was expected to use system memory; it keeps %.1f GiB there", shared.CPURAMGiB)
	}
	if whole, _ := WholeCards(hw); planFit(m, DefaultConfig(), whole, ContextMax).Config.ContextSize >= m.ContextLength {
		t.Fatal("the test model fits the whole cards at its full context; it cannot tell the rules apart")
	}
	if !r.Fits || r.CPURAMGiB > 0.05 || r.Config.GPULayers != 999 {
		t.Errorf("with auto-evict at the longest context: layers %d, %.1f GiB in system memory, context %d; want every layer on the GPUs",
			r.Config.GPULayers, r.CPURAMGiB, r.Config.ContextSize)
	}
}

// A model that fits beside the reservations keeps sharing the cards with
// auto-evict on: no switch between chat and images stops anything.
func TestPlanFitSharesWhenItFits(t *testing.T) {
	m := dense20()
	hw := evictAll(compute2Reserved())
	r := PlanFit(m, DefaultConfig(), hw, ContextMedium)
	want := PlanFit(m, DefaultConfig(), compute2Reserved(), ContextMedium)
	if r.Config.TensorSplit != want.Config.TensorSplit || r.Config.ContextSize != want.Config.ContextSize {
		t.Errorf("plan with auto-evict = split %q context %d, want the shared plan's %q %d",
			r.Config.TensorSplit, r.Config.ContextSize, want.Config.TensorSplit, want.Config.ContextSize)
	}
	for _, n := range r.Notes {
		if strings.Contains(n.Reason, "whole GPUs") {
			t.Errorf("a shared plan says it takes the whole GPUs: %s", n.Reason)
		}
	}
	room := RoomOf(hw)
	if got := room.For(m, &r.Config); !slices.Equal(got, room.Shared) {
		t.Errorf("room for a model that fits beside the backends = %v, want shared %v", got, room.Shared)
	}
}

// Only the reservations auto-evict may take back are given to the model:
// a backend kept running keeps its memory.
func TestWholeCardsKeepsKeptRunningReservations(t *testing.T) {
	hw := compute2Reserved()
	if _, ok := WholeCards(hw); ok {
		t.Error("WholeCards reported room to take back with auto-evict off")
	}
	if RoomOf(hw).Whole != nil {
		t.Error("RoomOf has whole cards with auto-evict off")
	}
	hw.GPUs[1].EvictableMiB = 9728 // flux may be stopped; ace is kept running
	whole, ok := WholeCards(hw)
	if !ok || whole.GPUs[1].ReservedMiB != 0 || whole.GPUs[2].ReservedMiB != 7680 {
		t.Errorf("whole cards = %+v, want GPU 1 freed and GPU 2 still reserved", whole.GPUs)
	}
	if hw.GPUs[1].ReservedMiB != 9728 {
		t.Error("WholeCards changed the hardware it was given")
	}
}

// qwen38Hybrid27B is Qwen3.8-27B at UD-Q8_K_XL as compute2's registry has
// it: a hybrid model, 16 of 65 layers attention, the rest linear
// attention with 0.146 GiB of state per conversation, and an MTP layer.
func qwen38Hybrid27B() *Model {
	return &Model{SizeBytes: 31457991680, NLayers: 65, AttnLayers: 16, NEmbd: 5120, NHead: 24, NKVHead: 4,
		ContextLength: 262144, KVFullPerTok: 32768, RecurrentStateBytes: 156893184, NextNLayers: 1,
		TokenEmbdBytes: 1350860800}
}

// With MTP's default of 16 drafted tokens, each keeping a copy of the
// recurrent state, the 27B does not fit compute2's whole cards at 128K.
// The plan drafts fewer tokens rather than running layers on the CPU.
func TestPlanFitLowersDraftCountToFit(t *testing.T) {
	SetVRAMBackend("cuda")
	defer SetVRAMBackend("")
	m := qwen38Hybrid27B()
	base := DefaultConfig()
	base.SpecType = "draft-mtp"
	hw := evictAll(compute2Reserved())

	r := PlanFit(m, base, hw, ContextLong)
	if !r.Fits || r.Config.GPULayers != 999 || r.CPURAMGiB > 0.05 {
		t.Fatalf("plan: layers %d, %.1f GiB in system memory; want every layer on the GPUs", r.Config.GPULayers, r.CPURAMGiB)
	}
	if r.Config.DraftMax <= 0 || r.Config.DraftMax >= specDraftTokens {
		t.Errorf("draft max = %d, want fewer than llama.cpp's default %d", r.Config.DraftMax, specDraftTokens)
	}
	if r.Config.ContextSize != ContextClassTokens[ContextLong] {
		t.Errorf("context = %d, want the requested %d kept", r.Config.ContextSize, ContextClassTokens[ContextLong])
	}
	said := false
	for _, n := range r.Notes {
		said = said || (n.Field == "draft_max" && strings.Contains(n.Reason, "instead of 16"))
	}
	if !said {
		t.Errorf("no note explains the draft count: %+v", r.Notes)
	}
	assertOneNotePerChange(t, base, r)

	// A draft count the user set that already fits is kept.
	base.DraftMax = 4
	if r := PlanFit(m, base, hw, ContextLong); r.Config.DraftMax != 4 {
		t.Errorf("draft max 4 that fits became %d", r.Config.DraftMax)
	}
	// Without speculative decoding nothing is set.
	base.SpecType, base.DraftMax = "", 0
	if r := PlanFit(m, base, hw, ContextLong); r.Config.DraftMax != 0 {
		t.Errorf("a plan without a drafter set draft max %d", r.Config.DraftMax)
	}
}
