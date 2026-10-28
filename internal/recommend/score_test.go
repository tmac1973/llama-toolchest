package recommend

import (
	"slices"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/models"
)

// qualityFactor has three steps. The boundaries belong to the higher
// step, so a quant exactly at goodBPW (about IQ4_XS) gets almost no
// discount, and the factor never rises as the bits per weight fall.
func TestQualityFactorSteps(t *testing.T) {
	cases := []struct {
		bpw  float64
		want float64
	}{
		{16, 1},
		{8.5, 1},
		{5.5, 1},
		{5.49, 0.97},
		{4.85, 0.97},
		{goodBPW, 0.97},
		{goodBPW - 0.01, 0.8},
		{3.5, 0.8},
		{3.49, 0.6},
		{smallestBPW, 0.6},
		{0, 0.6},
	}
	for _, c := range cases {
		if got := qualityFactor(c.bpw); got != c.want {
			t.Errorf("qualityFactor(%v) = %v, want %v", c.bpw, got, c.want)
		}
	}
	for i := 1; i < len(cases); i++ {
		if qualityFactor(cases[i].bpw) > qualityFactor(cases[i-1].bpw) {
			t.Errorf("factor rises from %v to %v bpw", cases[i-1].bpw, cases[i].bpw)
		}
	}
}

// headroom is the share of the budget a plan leaves free, from 0 to 1.
// A plan over its budget, or a pick without a known budget, has none;
// it must never become negative, which would lower a score below what a
// pick that just fits gets.
func TestHeadroom(t *testing.T) {
	cases := []struct {
		name             string
		budget, estimate float64
		want             float64
	}{
		{"half used", 20, 10, 0.5},
		{"nothing used", 20, 0, 1},
		{"exactly full", 20, 20, 0},
		{"over budget", 20, 30, 0},
		{"no budget", 0, 10, 0},
		{"negative budget", -5, 1, 0},
	}
	for _, c := range cases {
		if got := headroom(&Pick{BudgetGiB: c.budget, EstimateGiB: c.estimate}); got != c.want {
			t.Errorf("%s: headroom = %v, want %v", c.name, got, c.want)
		}
	}
}

// tieBreak orders equal scores by downloads, most first, then by key, so
// the feed order is the same on every refresh.
func TestTieBreak(t *testing.T) {
	gs := []*Group{
		{Key: "c", Downloads: 10},
		{Key: "b", Downloads: 500},
		{Key: "a", Downloads: 10},
		{Key: "d", Downloads: 500},
	}
	slices.SortFunc(gs, tieBreak)
	var keys []string
	for _, g := range gs {
		keys = append(keys, g.Key)
	}
	if want := []string{"b", "d", "a", "c"}; !slices.Equal(keys, want) {
		t.Errorf("order = %v, want %v", keys, want)
	}
	same := &Group{Key: "x", Downloads: 1}
	if got := tieBreak(same, &Group{Key: "x", Downloads: 1}); got != 0 {
		t.Errorf("tieBreak of equal groups = %d, want 0", got)
	}
}

// Groups that score the same in every order must come out in tieBreak
// order in all four orders, not in input order.
func TestOrderAllUsesTieBreakForEqualScores(t *testing.T) {
	class := Classes[0]
	mk := func(key string) *Group {
		return &Group{Key: key, Params: 8e9, Downloads: 100, TrainedCtx: 32768, Picks: map[models.ContextClass]*Pick{
			class: {SizeBytes: 5e9, BPW: 4.85, Context: 32768, Placement: models.PlacementGPU, BudgetGiB: 20, EstimateGiB: 8},
		}}
	}
	// A group without a pick at this class is left out of its orders.
	missing := &Group{Key: "aa-missing", Picks: map[models.ContextClass]*Pick{}}
	orders := orderAll([]*Group{mk("zeta"), missing, mk("alpha"), mk("mid")})
	for _, in := range Intents {
		var keys []string
		for _, g := range orders[in][class] {
			keys = append(keys, g.Key)
		}
		if want := []string{"alpha", "mid", "zeta"}; !slices.Equal(keys, want) {
			t.Errorf("%s order = %v, want %v", in, keys, want)
		}
	}
	// No group has a pick at the other classes, so they have no order.
	if got := orders[IntentQuality][Classes[1]]; got != nil {
		t.Errorf("class without picks = %v, want no order", got)
	}
}
