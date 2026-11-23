package models

import (
	"fmt"
	"slices"
	"strconv"
	"strings"
)

// The loop core on one GPU.
//
// A looped model (inkling2_loop) runs blocks LoopStart..LoopEnd-1 several
// times per token. llama.cpp gives every extra pass a cache slot of its
// own after the file's blocks, and each slot keeps its KV cache on the
// card the layer split gives that slot, while the weights it runs with
// stay on the card of the block it repeats. A layer split puts the slots
// on the last cards, away from those weights, so every token crosses
// between cards hundreds of times: Inkling 2 Mini over three A4000s with
// Autoconfigure's split 9,10,29 had 592 graph splits and generated 16
// tokens per second.
//
// With LoopCoreOneGPU the launch keeps the loop core's blocks and every
// cache slot from LoopStart on together on one card (the loop card), and
// moves the weights of the blocks after the loop core, whose slots fall
// there too, to the other cards with --override-tensor. On compute2 this
// gave 126 graph splits and 29 tokens per second (2026-11-23), at the
// cost of less context: the loop card holds the most.
//
// The loop card is the last card or the one before it. Cards before it
// take the blocks before the loop core; a card after it takes only the
// output layer (the position past every layer in llama.cpp's split), and
// with it the moved blocks' weights. No other placement keeps the slots
// in one piece: llama.cpp's split gives each card one run of positions.

// LoopLayoutApplies reports whether cfg runs m with its loop core on one
// card: LoopCoreOneGPU is set, m is looped, and cfg spreads every layer
// over two or more cards with a layer split of the launch's own choosing.
// Experts in system memory, fewer GPU layers, a hand-written split or
// a tensor override in Extra Flags leave the plain split in place.
func LoopLayoutApplies(m *Model, cfg *ModelConfig, numGPUs int) bool {
	if m == nil || cfg == nil || !cfg.LoopCoreOneGPU || !m.IsLooped() {
		return false
	}
	if cfg.GPUAssign == "custom" || cfg.SplitMode == "tensor" || cfg.SplitMode == "row" || strings.HasPrefix(cfg.GPUAssign, "tensor") {
		return false
	}
	if cfg.CPUMoE > 0 || (cfg.GPULayers > 0 && cfg.GPULayers < m.NLayers) || extraFlagsPlaceTensors(cfg.ExtraFlags) {
		return false
	}
	return len(activeGPUs(cfg, numGPUs)) >= 2
}

// IsLooped reports whether m repeats a run of its blocks (LoopSlots).
func (m *Model) IsLooped() bool {
	return m.LoopSlots > 0 && m.LoopEnd > m.LoopStart && m.LoopEnd <= m.NLayers
}

// extraFlagsPlaceTensors reports whether Extra Flags place tensors
// themselves, which the layout's own --override-tensor would clash with.
func extraFlagsPlaceTensors(raw string) bool {
	if raw == "" {
		return false
	}
	for _, f := range parseExtraFlags(raw) {
		switch f.key {
		case "ot", "override-tensor", "cmoe", "cpu-moe", "ncmoe", "n-cpu-moe":
			return true
		}
	}
	return false
}

// loopLayout is where the loop-core layout puts things, over the active
// cards (indices into active, not GPU numbers).
type loopLayout struct {
	shares  []int       // the --tensor-split entry of each active card
	moved   map[int]int // block → active card its weights are moved to
	loads   []float64   // GiB each active card holds
	budgets []float64   // each active card's budget, GiB
}

// planLoopLayout works out the loop-core layout of cfg over active, with
// each card's budget and fixed costs from moeSplitInputs. Of the two
// places the loop card can be, it keeps the one whose fullest card is the
// least full against its budget. ok is false when the cards cannot take
// the layout: more cards before the loop card than blocks before the
// loop core.
func planLoopLayout(m *Model, cfg *ModelConfig, gpuMiB []int, active []int) (loopLayout, bool) {
	layers, budgets, fixed := moeSplitInputs(m, cfg, gpuMiB, active)
	b := VRAMBreakdownForConfigOn(m, cfg, len(active))
	slots := len(layers)
	state := (b.KVCache + b.Recurrent + b.IndexerCache) * (1 << 30) / float64(slots)

	k := len(active)
	var best loopLayout
	bestWorst := -1.0
	for _, loop := range []int{k - 2, k - 1} {
		if loop < 0 || loop > m.LoopStart {
			continue
		}
		after := k - 1 - loop // 0 or 1 card after the loop card
		shares := make([]int, k)
		// The blocks before the loop core go to the cards before the loop
		// card, by their budgets, at least one each.
		pre := 0
		if loop > 0 {
			pre = m.LoopStart
			var room float64
			for i := range loop {
				room += max(budgets[i]-fixed[i], 0.1)
			}
			left := pre
			for i := range loop {
				n := left - (loop - 1 - i) // leave one for each card after
				if i < loop-1 {
					n = min(n, max(1, int(float64(pre)*max(budgets[i]-fixed[i], 0.1)/room+0.5)))
				}
				shares[i] = n
				left -= n
			}
		}
		// llama.cpp counts slots+1 positions, the output layer last.
		shares[loop] = slots + 1 - pre - after
		if after == 1 {
			shares[loop+1] = 1
		}
		devices := layerDevices(shares, slots)
		loads := addLayerLoads(fixed, layers, devices)

		// Blocks outside the loop core whose slots fell on the loop card
		// keep their KV cache there; their weights go to the other
		// cards, the heaviest first, each to the card with the most room
		// left.
		var outer []int
		for il := 0; il < m.NLayers; il++ {
			if devices[il] == loop && (il < m.LoopStart || il >= m.LoopEnd) {
				outer = append(outer, il)
			}
		}
		slices.SortStableFunc(outer, func(a, b int) int {
			switch {
			case layers[a] > layers[b]:
				return -1
			case layers[a] < layers[b]:
				return 1
			}
			return 0
		})
		moved := map[int]int{}
		for _, il := range outer {
			w := (layers[il] - state) / (1 << 30)
			to := -1
			for i := range k {
				if i == loop {
					continue
				}
				if to < 0 || loads[i]/budgets[i] < loads[to]/budgets[to] {
					to = i
				}
			}
			if to < 0 {
				break
			}
			loads[loop] -= w
			loads[to] += w
			moved[il] = to
		}

		worst := 0.0
		for i := range loads {
			worst = max(worst, loads[i]/budgets[i])
		}
		if bestWorst < 0 || worst < bestWorst {
			best, bestWorst = loopLayout{shares: shares, moved: moved, loads: loads, budgets: budgets}, worst
		}
	}
	return best, bestWorst >= 0
}

// loopLayoutConfig is cfg with the loop-core layout written out: the
// split, padded with zeros for the cards cfg leaves out, and the moved
// blocks by GPU number ("18:2,19:2", ModelConfig.LoopBlocks). nil when the layout does
// not apply or the cards cannot take it.
func loopLayoutConfig(cfg *ModelConfig, m *Model, gpuMiB []int) (*ModelConfig, loopLayout) {
	if !LoopLayoutApplies(m, cfg, len(gpuMiB)) {
		return nil, loopLayout{}
	}
	active := activeGPUs(cfg, len(gpuMiB))
	for _, g := range active {
		if g >= len(gpuMiB) || gpuMiB[g] <= 0 {
			return nil, loopLayout{}
		}
	}
	lay, ok := planLoopLayout(m, cfg, gpuMiB, active)
	if !ok {
		return nil, loopLayout{}
	}
	parts := make([]string, len(gpuMiB))
	for i := range parts {
		parts[i] = "0"
	}
	for i, g := range active {
		parts[g] = strconv.Itoa(lay.shares[i])
	}
	out := *cfg
	out.TensorSplit = strings.Join(parts, ",")
	out.SplitMode = "layer"
	ils := make([]int, 0, len(lay.moved))
	for il := range lay.moved {
		ils = append(ils, il)
	}
	slices.Sort(ils)
	pairs := make([]string, len(ils))
	for i, il := range ils {
		pairs[i] = fmt.Sprintf("%d:%d", il, active[lay.moved[il]])
	}
	out.LoopBlocks = strings.Join(pairs, ",")
	return &out, lay
}

// loopOverride renders the moved blocks ("18:2,19:2", block:GPU) as an
// --override-tensor value,
// one rule per card: blk\.(18|19)\.=CUDA2. "" when nothing moves or the
// backend has no device names.
func loopOverride(blocks, backend string) string {
	prefix := backendDevicePrefix[backend]
	if prefix == "" || blocks == "" {
		return ""
	}
	byGPU := map[int][]int{}
	for _, pair := range strings.Split(blocks, ",") {
		a, b, ok := strings.Cut(pair, ":")
		il, err1 := strconv.Atoi(a)
		g, err2 := strconv.Atoi(b)
		if !ok || err1 != nil || err2 != nil {
			continue
		}
		byGPU[g] = append(byGPU[g], il)
	}
	gpus := make([]int, 0, len(byGPU))
	for g := range byGPU {
		gpus = append(gpus, g)
	}
	slices.Sort(gpus)
	rules := make([]string, 0, len(gpus))
	for _, g := range gpus {
		ils := byGPU[g]
		slices.Sort(ils)
		names := make([]string, len(ils))
		for i, il := range ils {
			names[i] = strconv.Itoa(il)
		}
		rules = append(rules, fmt.Sprintf(`blk\.(%s)\.=%s%d`, strings.Join(names, "|"), prefix, BackendDeviceIndex(backend, g)))
	}
	return strings.Join(rules, ",")
}
