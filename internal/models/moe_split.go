package models

import (
	"sort"
	"strconv"
	"strings"
)

// Expert offload and the layer split.
//
// --n-cpu-moe N keeps the expert weights of layers 0..N-1 in system
// memory, so those layers become small while the rest keep their experts.
// llama.cpp's layer split divides layers between GPUs by count (in
// proportion to --tensor-split, or to free memory without one), not by
// size, so the heavy layers all land on the last cards. On Qwen3.8-Flash-
// Next with 32 of 48 layers' experts offloaded over three 16 GiB cards,
// the last card was asked for 24.9 GiB of weights while the other two
// held 1.5 GiB each, and the load failed.
//
// The cards can also differ in what they have room for, with the layers
// all the same size: memory reserved for image and music backends on some
// of them, other programs' memory, or cards of different sizes. llama.cpp
// splits by the memory free when it loads, which does not see a
// reservation an idle backend is not using; on three 16 GiB cards with 9.5
// and 7.5 GiB reserved on two of them, an even split asked those two for
// more than they had left.
//
// MoESplitConfig gives such a config a --tensor-split that balances the
// cards by size and room, and PlanCardLoads is what the fit planner, and
// the reservation check on loads, check each card against.

// unequalRoomMiB is how much the cards' room must differ before the split
// is balanced by it when the layers are all the same size. Below it,
// llama.cpp's own split is left alone.
const unequalRoomMiB = 1024

// moeSplitApplies reports whether cfg runs m over a layer split that
// needs balancing on cards with gpuMiB room: with expert offload, where
// the layers differ in size, or with cards whose room differs.
func moeSplitApplies(cfg *ModelConfig, m *Model, gpuMiB []int) bool {
	if m == nil || cfg == nil || m.NLayers <= 0 {
		return false
	}
	if cfg.GPUAssign == "custom" || cfg.SplitMode == "tensor" || cfg.SplitMode == "row" {
		return false // the user's own split, or not a layer split
	}
	if cfg.GPULayers > 0 && cfg.GPULayers < m.NLayers {
		return false
	}
	if cfg.CPUMoE > 0 && m.ExpertLayers > 0 && m.ExpertBytes > 0 {
		return true
	}
	lo, hi := -1, 0
	for _, g := range activeGPUs(cfg, len(gpuMiB)) {
		if g >= len(gpuMiB) {
			continue
		}
		if lo < 0 || gpuMiB[g] < lo {
			lo = gpuMiB[g]
		}
		hi = max(hi, gpuMiB[g])
	}
	return lo >= 0 && hi-lo >= unequalRoomMiB
}

// activeGPUs are the GPU indices a config spreads layers over: the
// non-zero entries of its tensor split, or every GPU without one.
func activeGPUs(cfg *ModelConfig, numGPUs int) []int {
	if idx := SplitDeviceIndices(cfg.TensorSplit); idx != nil {
		return idx
	}
	out := make([]int, numGPUs)
	for i := range out {
		out[i] = i
	}
	return out
}

// layerGPUBytes is what each layer puts on its GPU: its share of the
// weights that are not experts, its experts unless --n-cpu-moe keeps them
// in system memory, and its share of the KV cache, recurrent state and
// indexer cache, which llama.cpp places with the layer.
func layerGPUBytes(m *Model, cfg *ModelConfig, b VRAMBreakdown) []float64 {
	resident := m.SizeBytes - m.PLEBytes - m.TokenEmbdBytes
	if resident <= 0 {
		resident = m.SizeBytes
	}
	experts := min(m.ExpertBytes, resident)
	base := float64(resident-experts) / float64(m.NLayers)
	perExpertLayer := float64(experts) / float64(m.ExpertLayers)
	perLayerState := (b.KVCache + b.Recurrent + b.IndexerCache) * (1 << 30) / float64(m.NLayers)
	// llama.cpp gives the MoE cache to the cards that hold the layers
	// whose experts it caches. Spreading it over those layers puts it on
	// the same cards, roughly in the same proportion.
	var perCachedLayer float64
	if n := cpuExpertLayers(m, cfg); n > 0 {
		perCachedLayer = b.MoECache * (1 << 30) / float64(n)
	}

	out := make([]float64, m.NLayers)
	for il := range out {
		out[il] = base + perLayerState
		inSpan := il >= m.ExpertLayerFirst && il < m.ExpertLayerFirst+m.ExpertLayers
		if inSpan && il >= cfg.CPUMoE {
			out[il] += perExpertLayer
		}
		if inSpan && il < cfg.CPUMoE {
			out[il] += perCachedLayer
		}
	}
	return out
}

// layerDevices assigns each layer to a card the way llama.cpp's layer
// split does: with every layer offloaded it counts n_layer+1 layers (the
// output layer last), and layer il goes to the first card whose share of
// the cumulative split exceeds il/(n_layer+1).
func layerDevices(split []int, nLayers int) []int {
	total := 0
	for _, s := range split {
		total += s
	}
	cum := make([]float64, len(split))
	acc := 0
	for i, s := range split {
		acc += s
		cum[i] = float64(acc) / float64(total)
	}
	act := float64(nLayers + 1)
	out := make([]int, nLayers)
	for il := range out {
		x := float64(il) / act
		d := sort.Search(len(cum), func(i int) bool { return cum[i] > x })
		out[il] = min(d, len(cum)-1)
	}
	return out
}

// cardLoads sums the layer bytes each card receives under split, in GiB,
// adding each card's fixed costs.
func cardLoads(layers []float64, split []int, fixed []float64) []float64 {
	loads := append([]float64(nil), fixed...)
	for il, d := range layerDevices(split, len(layers)) {
		loads[d] += layers[il] / (1 << 30)
	}
	return loads
}

// balanceSplit chooses layer counts per card that keep each card's load
// within the same share of its budget: a first cut in proportion to the
// budgets, then single-layer moves between neighbouring cards while they
// lower the fullest card's share.
func balanceSplit(layers []float64, budgets, fixed []float64) []int {
	n, k := len(layers), len(budgets)
	var totalBytes, totalBudget float64
	for _, l := range layers {
		totalBytes += l / (1 << 30)
	}
	for i := range budgets {
		totalBudget += max(budgets[i]-fixed[i], 0.1)
	}

	// First cut: fill each card up to its share of the total, leaving at
	// least one layer for every card after it.
	split := make([]int, k)
	il := 0
	acc, target := 0.0, 0.0
	for i := 0; i < k-1; i++ {
		target += max(budgets[i]-fixed[i], 0.1) / totalBudget * totalBytes
		start := il
		for il < n-(k-1-i) && (il == start || acc+layers[il]/(1<<30) <= target) {
			acc += layers[il] / (1 << 30)
			il++
		}
		split[i] = il - start
	}
	split[k-1] = n - il

	worst := func(s []int) float64 {
		w := 0.0
		for i, l := range cardLoads(layers, s, fixed) {
			w = max(w, l/budgets[i])
		}
		return w
	}
	best := worst(split)
	for range 4 * n {
		improved := false
		for i := 0; i+1 < k; i++ {
			for _, d := range []int{1, -1} {
				if split[i]-d < 1 || split[i+1]+d < 1 {
					continue
				}
				split[i] -= d
				split[i+1] += d
				if w := worst(split); w < best-1e-9 {
					best, improved = w, true
				} else {
					split[i] += d
					split[i+1] -= d
				}
			}
		}
		if !improved {
			break
		}
	}
	return split
}

// moeSplitInputs gathers what balancing needs for cfg on the given cards:
// the per-layer bytes and each card's budget and fixed costs, in GiB.
func moeSplitInputs(m *Model, cfg *ModelConfig, gpuMiB []int, active []int) (layers, budgets, fixed []float64) {
	b := VRAMBreakdownForConfigOn(m, cfg, len(active))
	layers = layerGPUBytes(m, cfg, b)
	perCard := (b.Compute + b.Overhead + b.IndexerScratch) / float64(len(active))
	for i, g := range active {
		// gpuMiB is already each card's budget, after the safety margin.
		budgets = append(budgets, float64(gpuMiB[g])/1024)
		f := perCard
		if i == 0 {
			// The first card also holds what is not per layer: draft
			// model caches and auxiliary files such as the image reader.
			f += b.SpecKV + b.Aux
		}
		fixed = append(fixed, f)
	}
	return layers, budgets, fixed
}

// MoESplitConfig returns cfg with a --tensor-split that balances the
// cards by size and room, when cfg spreads layers over two or more GPUs
// and either keeps experts in system memory or the cards' room differs;
// otherwise cfg itself. gpuMiB is each GPU's room by index (GPUMiB: its
// memory less other programs' and generation reservations); without it
// nothing changes. The returned config
// keeps cfg's GPU assignment, so the saved config, and the form, are
// untouched: the split is worked out each time the flags are written.
func MoESplitConfig(cfg *ModelConfig, m *Model, gpuMiB []int) *ModelConfig {
	if len(gpuMiB) < 2 || !moeSplitApplies(cfg, m, gpuMiB) {
		return cfg
	}
	active := activeGPUs(cfg, len(gpuMiB))
	if len(active) < 2 {
		return cfg
	}
	for _, g := range active {
		if g >= len(gpuMiB) || gpuMiB[g] <= 0 {
			return cfg
		}
	}
	layers, budgets, fixed := moeSplitInputs(m, cfg, gpuMiB, active)
	split := balanceSplit(layers, budgets, fixed)

	parts := make([]string, len(gpuMiB))
	for i := range parts {
		parts[i] = "0"
	}
	for i, g := range active {
		parts[g] = strconv.Itoa(split[i])
	}
	out := *cfg
	out.TensorSplit = strings.Join(parts, ",")
	if out.SplitMode == "" {
		out.SplitMode = "layer"
	}
	return &out
}

// PlanCardLoads is the GPU memory, in GiB, each card in use would hold
// under cfg with the split MoESplitConfig writes, and each card's budget
// after the planner's safety margin. ok is false when cfg does not use a
// size-balanced split (no expert offload, or a single card), in which case
// the pooled estimate stands.
func PlanCardLoads(m *Model, cfg *ModelConfig, gpuMiB []int) (loads, budgets []float64, ok bool) {
	split := MoESplitConfig(cfg, m, gpuMiB)
	if split == cfg {
		return nil, nil, false
	}
	active := activeGPUs(split, len(gpuMiB))
	layers, budgets, fixed := moeSplitInputs(m, cfg, gpuMiB, active)
	counts := make([]int, len(active))
	parts := strings.Split(split.TensorSplit, ",")
	for i, g := range active {
		counts[i], _ = strconv.Atoi(parts[g])
	}
	return cardLoads(layers, counts, fixed), budgets, true
}

// CardLoadsMiB is the GPU memory, in MiB, each GPU would hold with m
// loaded under cfg on cards with gpuMiB room, by GPU index: under the
// balanced split MoESplitConfig writes where it applies, and otherwise the
// estimate spread evenly over the GPUs cfg uses.
func CardLoadsMiB(m *Model, cfg *ModelConfig, gpuMiB []int) map[int]int {
	out := map[int]int{}
	if loads, _, ok := PlanCardLoads(m, cfg, gpuMiB); ok {
		for i, g := range activeGPUs(MoESplitConfig(cfg, m, gpuMiB), len(gpuMiB)) {
			out[g] += int(loads[i] * 1024)
		}
		return out
	}
	total := VRAMEstimateForConfig(m, cfg) * 1024
	// A layer split with weights of its own (a hand-written one, or the
	// one Autoconfigure writes when it moves layers off a card) is split
	// in that proportion.
	if cfg.SplitMode != "tensor" && cfg.SplitMode != "row" {
		if w := splitWeights(cfg.TensorSplit, len(gpuMiB)); w != nil {
			var sum float64
			for _, x := range w {
				sum += x
			}
			for g, x := range w {
				if x > 0 {
					out[g] += int(total * x / sum)
				}
			}
			return out
		}
	}
	gpus := AssignedModelGPUs(cfg, len(gpuMiB))
	for _, g := range gpus {
		out[g] += int(total / float64(len(gpus)))
	}
	return out
}

// splitWeights reads a --tensor-split ("3,1,0") as one weight per GPU, or
// nil when it is empty, unreadable, all zero or names more GPUs than
// there are.
func splitWeights(tensorSplit string, numGPUs int) []float64 {
	if strings.TrimSpace(tensorSplit) == "" {
		return nil
	}
	parts := strings.Split(tensorSplit, ",")
	if len(parts) > numGPUs {
		return nil
	}
	w := make([]float64, len(parts))
	var sum float64
	for i, p := range parts {
		x, err := strconv.ParseFloat(strings.TrimSpace(p), 64)
		if err != nil || x < 0 {
			return nil
		}
		w[i], sum = x, sum+x
	}
	if sum == 0 {
		return nil
	}
	return w
}
