package benchmark

import "math"

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
	}
}

// ComparisonData holds data for comparing multiple benchmark runs.
type ComparisonData struct {
	Runs          []BenchmarkRun
	MaxGenTPS     float64
	MaxPromptTPS  float64
	HasLlamaBench bool
}

// BuildComparison prepares data for the comparison view.
func BuildComparison(runs []BenchmarkRun) ComparisonData {
	c := ComparisonData{Runs: runs}
	for _, r := range runs {
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
