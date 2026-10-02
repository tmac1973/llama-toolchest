package recommend

import (
	"cmp"
	"slices"

	"github.com/tmac1973/llama-toolchest/internal/models"
)

// Intent is one of the feed's orders.
type Intent string

const (
	IntentQuality Intent = "quality"
	IntentFastest Intent = "fastest"
	IntentContext Intent = "context"
	IntentNewest  Intent = "newest"
)

// Intents are the orders, in the order the feed shows their buttons.
var Intents = []Intent{IntentQuality, IntentFastest, IntentContext, IntentNewest}

// qualityFactor discounts a model's size for the quant it is suggested at.
// A much larger model at 4 bits usually beats a smaller one at 8, so
// there is little discount down to goodBPW. Below it the loss shows, and
// most on mixture-of-experts models with small experts: a 3-bit quant has
// to be of a much larger model to win.
func qualityFactor(bpw float64) float64 {
	switch {
	case bpw >= 5.5:
		return 1
	case bpw >= goodBPW:
		return 0.97
	case bpw >= 3.5:
		return 0.8
	}
	return 0.6
}

// systemMemorySlowdown is about how much slower a byte is to read from
// system memory than from GPU memory: a few hundred GB/s on a graphics
// card against a few tens on a desktop's memory.
const systemMemorySlowdown = 8

// readCost is about how long a model takes per generated token, in GiB
// read from GPU memory: reading the weights is what limits generation
// speed in llama.cpp. A dense model reads its whole file; a
// mixture-of-experts model only the experts in use. Experts kept in
// system memory cost systemMemorySlowdown times as much to read.
func readCost(p *Pick) float64 {
	size := float64(p.SizeBytes)
	if p.ExpertCount <= 0 || p.ExpertUsedCount <= 0 || p.ExpertBytes <= 0 {
		return gib(size)
	}
	experts := float64(min(p.ExpertBytes, p.SizeBytes))
	used := float64(p.ExpertUsedCount) / float64(p.ExpertCount)
	inRAM := min(experts, p.CPURAMGiB*(1<<30))
	onGPU := (size - experts) + (experts-inRAM)*used
	return gib(onGPU + systemMemorySlowdown*inRAM*used)
}

func headroom(p *Pick) float64 {
	if p.BudgetGiB <= 0 {
		return 0
	}
	return max(0, (p.BudgetGiB-p.EstimateGiB)/p.BudgetGiB)
}

func boolScore(b bool) float64 {
	if b {
		return 1
	}
	return 0
}

// maxContext is the longest context any pick of the group holds.
func (g *Group) maxContext() int {
	n := 0
	for _, p := range g.Picks {
		n = max(n, p.Context)
	}
	return n
}

// orderAll computes every order at every class. Switching category or
// context in the feed then only reads a slice.
func orderAll(groups []*Group) map[Intent]map[models.ContextClass][]*Group {
	out := map[Intent]map[models.ContextClass][]*Group{}
	for _, in := range Intents {
		out[in] = map[models.ContextClass][]*Group{}
	}
	for _, class := range Classes {
		var gs []*Group
		for _, g := range groups {
			if g.Picks[class] != nil {
				gs = append(gs, g)
			}
		}
		if len(gs) == 0 {
			continue
		}
		size := make([]float64, len(gs))
		speed := make([]float64, len(gs))
		params := make([]float64, len(gs))
		downloads := make([]float64, len(gs))
		for i, g := range gs {
			p := g.Picks[class]
			size[i] = float64(g.Params) * qualityFactor(p.BPW)
			speed[i] = 1 / max(readCost(p), 0.01)
			params[i] = float64(g.Params)
			downloads[i] = float64(g.Downloads)
		}
		pSize, pSpeed, pParams, pDownloads := percentiles(size), percentiles(speed), percentiles(params), percentiles(downloads)
		quality := map[*Group]float64{}
		fastest := map[*Group]float64{}
		for i, g := range gs {
			p := g.Picks[class]
			onGPU := p.Placement == models.PlacementGPU
			// Downloads are a small part: how widely a model is used says
			// something about it that its size does not, and keeps a
			// week-old fine-tune from outranking the model it was made from.
			quality[g] = 0.65*pSize[i] + 0.15*pDownloads[i] + 0.10*headroom(p) + 0.10*boolScore(onGPU)
			// The read cost already counts experts in system memory; the
			// small extra penalty is for what it leaves out (the CPU's
			// share of the work, transfers between the two).
			fastest[g] = min(1, max(0, 0.60*pSpeed[i]+0.25*pParams[i]+0.15*headroom(p)-0.15*boolScore(!onGPU)))
		}

		byScore := func(score map[*Group]float64) []*Group {
			s := slices.Clone(gs)
			slices.SortStableFunc(s, func(a, b *Group) int {
				if c := cmp.Compare(score[b], score[a]); c != 0 {
					return c
				}
				return tieBreak(a, b)
			})
			return s
		}
		out[IntentQuality][class] = byScore(quality)
		out[IntentFastest][class] = byScore(fastest)

		ctx := slices.Clone(gs)
		slices.SortStableFunc(ctx, func(a, b *Group) int {
			if c := cmp.Compare(b.maxContext(), a.maxContext()); c != 0 {
				return c
			}
			if c := cmp.Compare(b.TrainedCtx, a.TrainedCtx); c != 0 {
				return c
			}
			if c := cmp.Compare(quality[b], quality[a]); c != 0 {
				return c
			}
			return tieBreak(a, b)
		})
		out[IntentContext][class] = ctx

		newest := slices.Clone(gs)
		slices.SortStableFunc(newest, func(a, b *Group) int {
			if c := b.Released.Compare(a.Released); c != 0 {
				return c
			}
			return tieBreak(a, b)
		})
		out[IntentNewest][class] = newest
	}
	return out
}

// tieBreak orders equal scores: more downloads first, then by name.
func tieBreak(a, b *Group) int {
	if c := cmp.Compare(b.Downloads, a.Downloads); c != 0 {
		return c
	}
	return cmp.Compare(a.Key, b.Key)
}
