package benchmark

import (
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
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

	// Labels names each run by what makes it DIFFERENT from the others,
	// keyed by run ID. Without it a swept comparison renders every bar
	// with the same text and the winner cannot be identified — which is
	// the whole purpose of the view. See BuildRunLabels.
	Labels map[string]RunLabel

	// BestGenRunID and BestPromptRunID are the winning runs on each bar
	// chart, so the view can mark them instead of leaving the reader to
	// compare bar lengths by eye.
	BestGenRunID    string
	BestPromptRunID string

	// Varies says, per descriptive column of the details table, whether
	// the compared runs actually differ on it. A column every run agrees
	// on carries no information about which run won, and a wide table
	// that has to be scrolled sideways hides the columns that do. The
	// view marks such columns constant, folds their shared value into
	// Common above the table, and offers a control to show them anyway.
	//
	// Keys are the column names used in the template. A key that is
	// absent means "show it": a comparison of fewer than two runs has
	// nothing to collapse, and a column not listed here was never a
	// candidate.
	Varies map[string]bool

	// Common lists the columns every compared run agrees on, with the
	// value they share, in table order. This is where the collapsed
	// columns go: dropping them without saying what they held would
	// leave the reader to assume rather than read.
	Common []CommonColumn

	// MixedPromptSizes is true when the compared runs did not all
	// measure the same prompt lengths. The bar charts show each run's
	// average across whatever it measured, and prompt throughput climbs
	// steeply with prompt length, so averages over different size sets
	// are not comparable with one another. The view says so rather than
	// presenting them side by side without comment.
	MixedPromptSizes bool

	// NoResultRuns are selected runs that produced neither timings nor a
	// capability score — a failed run, most often. They used to vanish
	// from both the chart and the table, so a comparison could silently
	// contain fewer runs than the user picked.
	NoResultRuns []BenchmarkRun

	// MissingRunIDs are selected runs that no longer exist in the store,
	// deleted between selecting them and asking for the comparison. Set
	// by the handler, not by BuildComparison, which only sees the runs
	// that resolved.
	MissingRunIDs []string
}

// BuildComparison prepares data for the comparison view.
func BuildComparison(runs []BenchmarkRun) ComparisonData {
	c := ComparisonData{Runs: runs, Labels: BuildRunLabels(runs)}
	sizeSets := map[string]bool{}
	for _, r := range runs {
		if r.Eval != nil {
			c.HasEval = true
		}
		if r.Summary == nil {
			if r.Eval == nil {
				c.NoResultRuns = append(c.NoResultRuns, r)
			}
			continue
		}
		if r.Summary.AvgGenTokPerSec > c.MaxGenTPS {
			c.MaxGenTPS = r.Summary.AvgGenTokPerSec
			c.BestGenRunID = r.ID
		}
		if r.Summary.AvgPromptTokPerSec > c.MaxPromptTPS {
			c.MaxPromptTPS = r.Summary.AvgPromptTokPerSec
			c.BestPromptRunID = r.ID
		}
		if r.LlamaBench != nil {
			c.HasLlamaBench = true
		}
		sizeSets[promptSizeKey(r)] = true
	}
	c.MixedPromptSizes = len(sizeSets) > 1
	c.Varies, c.Common = compareColumnVariance(runs)
	return c
}

// CommonColumn is one descriptive column the compared runs all agree on.
type CommonColumn struct {
	Name  string
	Value string
}

// compareColumns are the details table's descriptive columns: name, and
// the text the cell shows.
//
// The formatting is duplicated from the template on purpose, and it has
// to stay in step — a shared value stated one way in the summary line
// and another in the cell that the "show every column" control reveals
// would read as two different facts. compare_columns_test.go pins each
// one against the template's own fallbacks.
//
// The measured columns are deliberately absent: throughput, score and
// VRAM are what the reader came for, and a comparison where two runs
// scored identically is still a comparison about those numbers. Model is
// absent too — it anchors each row and carries the run's full label and
// any unverified warning, so it stays even when every run shares it.
func compareColumns() []struct {
	Name  string
	Value func(BenchmarkRun) string
} {
	type col = struct {
		Name  string
		Value func(BenchmarkRun) string
	}
	return []col{
		{"quant", func(r BenchmarkRun) string { return r.Quant }},
		{"sweep", func(r BenchmarkRun) string { return sweepCellText(r) }},
		{"prompt sizes", func(r BenchmarkRun) string { return r.PromptSizesText() }},
		{"size", func(r BenchmarkRun) string { return fmt.Sprintf("%.1f GiB", r.SizeGiB) }},
		{"GPUs", func(r BenchmarkRun) string {
			if r.Config.GPUAssign == "" {
				return "all"
			}
			return r.Config.GPUAssign
		}},
		{"context", func(r BenchmarkRun) string { return strconv.Itoa(r.Config.ContextSize) }},
		{"batch", func(r BenchmarkRun) string { return batchCellText(r) }},
		{"KV cache", func(r BenchmarkRun) string {
			if r.Config.KVCacheQuant == "" {
				return "f16"
			}
			return r.Config.KVCacheQuant
		}},
		{"flash attention", func(r BenchmarkRun) string {
			if r.Config.FlashAttention {
				return "yes"
			}
			return "no"
		}},
		{"build", func(r BenchmarkRun) string { return compareBuildCellText(r) }},
	}
}

// compareColumnVariance splits the descriptive columns into those that
// differ across the compared runs and those that do not.
//
// Below two runs there is nothing to compare, so nothing is collapsed:
// a single run's table is a statement of what it was, and every column
// of it is worth reading.
func compareColumnVariance(runs []BenchmarkRun) (map[string]bool, []CommonColumn) {
	if len(runs) < 2 {
		return nil, nil
	}
	varies := map[string]bool{}
	var common []CommonColumn
	for _, c := range compareColumns() {
		first := c.Value(runs[0])
		differs := false
		for _, r := range runs[1:] {
			if c.Value(r) != first {
				differs = true
				break
			}
		}
		varies[c.Name] = differs
		// A column that is blank in every row is hidden with no note.
		// "build —" or "sweep —" in the summary states an absence as
		// though it were a shared setting; there is nothing to say.
		if !differs && first != "" && first != "—" {
			common = append(common, CommonColumn{Name: c.Name, Value: first})
		}
	}
	return varies, common
}

// sweepCellText, batchCellText and compareBuildCellText mirror the three
// cells whose text the template assembles rather than prints.
func sweepCellText(r BenchmarkRun) string {
	if len(r.SweepValues) == 0 {
		return "—"
	}
	keys := make([]string, 0, len(r.SweepValues))
	for k := range r.SweepValues {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		parts = append(parts, k+"="+r.SweepValues[k])
	}
	return strings.Join(parts, " ")
}

func batchCellText(r BenchmarkRun) string {
	if r.Config.BatchSize == 0 && r.Config.UBatchSize == 0 {
		return "—"
	}
	part := func(n int) string {
		if n == 0 {
			return "—"
		}
		return strconv.Itoa(n)
	}
	return part(r.Config.BatchSize) + "/" + part(r.Config.UBatchSize)
}

func compareBuildCellText(r BenchmarkRun) string {
	b := r.EffectiveBuild()
	switch {
	case b.Profile != "" && b.GitRef != "":
		return b.Profile + " · " + b.GitRef
	case b.Profile != "":
		return b.Profile
	case b.GitRef != "":
		return b.GitRef
	}
	return "—"
}

// promptSizeKey renders the set of prompt lengths a run measured, so
// two runs that measured the same lengths compare equal. A run with no
// per-size data reports "mixed": its average is over an unknown set and
// cannot be assumed to match anything.
func promptSizeKey(r BenchmarkRun) string {
	rows := r.SizeRows()
	parts := make([]string, 0, len(rows))
	for _, s := range rows {
		if s.PromptTokens == 0 {
			return "mixed"
		}
		parts = append(parts, strconv.Itoa(s.PromptTokens))
	}
	sort.Strings(parts)
	return strings.Join(parts, ",")
}
