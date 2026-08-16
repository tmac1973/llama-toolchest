package benchmark

import (
	"math"
	"sort"
)

// ComputeSummary aggregates results into a summary.
func ComputeSummary(results []BenchmarkResult) *BenchmarkSummary {
	if len(results) == 0 {
		return nil
	}

	var sumPP, sumTG, sumTTFT float64
	minTG := math.MaxFloat64
	maxTG := 0.0

	for _, r := range results {
		sumPP += r.PromptTokPerSec
		sumTG += r.GenTokPerSec
		sumTTFT += r.TTFTMs
		if r.GenTokPerSec < minTG {
			minTG = r.GenTokPerSec
		}
		if r.GenTokPerSec > maxTG {
			maxTG = r.GenTokPerSec
		}
	}

	n := float64(len(results))
	return &BenchmarkSummary{
		AvgPromptTokPerSec: sumPP / n,
		AvgGenTokPerSec:    sumTG / n,
		AvgTTFTMs:          sumTTFT / n,
		MinGenTokPerSec:    minTG,
		MaxGenTokPerSec:    maxTG,
		PerSize:            computePerSize(results),
	}
}

// computePerSize groups results by prompt length and reports mean and
// standard deviation per group, which is how llama-bench and everything
// that quotes it report: one figure per fixed prompt size (pp512,
// pp2048), never averaged across sizes.
//
// Prompt-processing throughput rises steeply with prompt length, so an
// average across sizes is comparable to nothing — not to llama-bench,
// not to another run using a different preset, not to the same run if
// its preset changes. The standard deviation matters too: llama-bench
// reports ±10% at pp512 and ±0.1% at pp2048 on the same hardware, and a
// bare mean hides which differences are real.
func computePerSize(results []BenchmarkResult) []SizeSummary {
	byTokens := map[int][]BenchmarkResult{}
	var order []int
	for _, r := range results {
		if _, seen := byTokens[r.PromptTokens]; !seen {
			order = append(order, r.PromptTokens)
		}
		byTokens[r.PromptTokens] = append(byTokens[r.PromptTokens], r)
	}
	sort.Ints(order)

	out := make([]SizeSummary, 0, len(order))
	for _, tok := range order {
		group := byTokens[tok]
		pp := make([]float64, 0, len(group))
		tg := make([]float64, 0, len(group))
		var sumTTFT float64
		for _, r := range group {
			pp = append(pp, r.PromptTokPerSec)
			tg = append(tg, r.GenTokPerSec)
			sumTTFT += r.TTFTMs
		}
		ppMean, ppStd := meanStd(pp)
		tgMean, tgStd := meanStd(tg)
		out = append(out, SizeSummary{
			PromptTokens: tok,
			GenTokens:    group[0].GenTokens,
			Count:        len(group),
			PPMean:       ppMean,
			PPStd:        ppStd,
			TGMean:       tgMean,
			TGStd:        tgStd,
			AvgTTFTMs:    sumTTFT / float64(len(group)),
		})
	}
	return out
}

// meanStd returns the mean and population standard deviation.
func meanStd(v []float64) (float64, float64) {
	if len(v) == 0 {
		return 0, 0
	}
	var sum float64
	for _, x := range v {
		sum += x
	}
	mean := sum / float64(len(v))
	var sq float64
	for _, x := range v {
		d := x - mean
		sq += d * d
	}
	return mean, math.Sqrt(sq / float64(len(v)))
}

// ComparisonData holds data for comparing multiple benchmark runs.
type ComparisonData struct {
	Runs          []BenchmarkRun
	MaxGenTPS     float64
	MaxPromptTPS  float64
	HasLlamaBench bool
	// HasEval is true when at least one compared run carries capability
	// scores; the compare view then shows the Score column. Mixed
	// comparisons (some performance, some capability) show both — each
	// row renders its own metric (the view template calls the
	// mode-appropriate renderer per row), and the mode is part of the
	// row identity so no cross-metric math happens.
	HasEval bool
}

// BuildComparison prepares data for the comparison view.
func BuildComparison(runs []BenchmarkRun) ComparisonData {
	c := ComparisonData{Runs: runs}
	for _, r := range runs {
		if r.Eval != nil {
			c.HasEval = true
		}
		if r.Summary == nil {
			continue
		}
		if r.Summary.AvgGenTokPerSec > c.MaxGenTPS {
			c.MaxGenTPS = r.Summary.AvgGenTokPerSec
		}
		if r.Summary.AvgPromptTokPerSec > c.MaxPromptTPS {
			c.MaxPromptTPS = r.Summary.AvgPromptTokPerSec
		}
		if r.LlamaBench != nil {
			c.HasLlamaBench = true
		}
	}
	return c
}
