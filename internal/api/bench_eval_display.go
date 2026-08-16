package api

import (
	"fmt"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/benchmark"
	"github.com/tmac1973/llama-toolchest/internal/evaluate"
)

// The display and export layer for capability scores. Every surface
// (results list, job cell matrix, compare table, run detail, CSV) goes
// through the same two renderers here so a score reads identically
// everywhere:
//
//	PPL 6.234 ±0.04 (100 chunks)   perplexity, quick
//	PPL 6.198 ±0.03 (full)         perplexity, full (requested cap 0)
//	KLD 0.012 ±0.001 · same top token 97.4% (vs UD-Q8_K_XL)
//	HellaSwag 77.2% [75.9–78.5] (400)   the tool's asymmetric 95% CI
//	Winogrande 74.1% ±2.2 (400)
//
// The chunk/task count comes from the run's EvalScores: the count the
// tool announced it would score, or the requested cap when the output
// carried none, and "full" only when neither was known. The tooltip
// copy on each column explains the metric in plain language.

// evalScoreText renders the run's score cell. Returns "" for runs that
// are not capability runs (the caller renders the em-dash), and a
// generic note when a capability run somehow has no parseable score
// (should not happen — the runner fails the cell before storing a
// scoreless run).
func evalScoreText(e *benchmark.EvalScores) string {
	if e == nil || e.Mode == "" {
		return ""
	}
	switch e.Mode {
	case string(evaluate.ModePerplexity):
		return fmt.Sprintf("PPL %.3f ±%.2f (%s)", e.Perplexity, e.PerplexityErr, evalChunkOrTaskText(e, "chunks"))
	case string(evaluate.ModeKLDiv):
		s := fmt.Sprintf("KLD %.3f ±%.3f", e.KLMean, e.KLMeanErr)
		if e.SameTopPct > 0 {
			s += fmt.Sprintf(" · same top token %.1f%%", e.SameTopPct)
		}
		if ref := evalReferenceText(e); ref != "" {
			s += " (vs " + ref + ")"
		}
		return s
	case string(evaluate.ModeHellaSwag):
		return fmt.Sprintf("HellaSwag %.1f%% [%.1f–%.1f] (%d)", e.Accuracy, e.AccuracyCILow, e.AccuracyCIHigh, e.Tasks)
	case string(evaluate.ModeWinogrande):
		return fmt.Sprintf("Winogrande %.1f%% ±%.1f (%d)", e.Accuracy, (e.AccuracyCIHigh-e.AccuracyCILow)/2, e.Tasks)
	default:
		return "score unavailable"
	}
}

// evalComparabilityNote returns the note explaining why a capability
// score is not comparable to figures measured elsewhere, or "" when it
// is. It mirrors benchmark.evalComparabilityWarning, which records the
// same fact on the run itself; this is the short form for a tooltip
// next to the number, so the caveat travels with the score instead of
// living only in the run detail.
func evalComparabilityNote(cfg benchmark.ConfigSnapshot) string {
	if cfg.KVCacheQuant == "" {
		return ""
	}
	return "Measured with a compressed short-term memory cache (kv_cache_quant = " + cfg.KVCacheQuant +
		"), not the uncompressed f16 these tests default to. That changes the answer rather than the speed, so this number is not comparable to published figures or to runs that used f16."
}

// evalNoScoreRank is the sort rank of a run with no capability score.
// A finite sentinel, not math.Inf: the value is rendered into a
// data-sort attribute and read back with parseFloat, which turns "+Inf"
// into NaN and NaN comparisons into an arbitrary order.
const evalNoScoreRank = 1e18

// evalReferenceText names the KL reference the way a person reads it:
// the label the runner recorded, falling back to the registry ID for
// runs stored before the label existed.
func evalReferenceText(e *benchmark.EvalScores) string {
	if e == nil {
		return ""
	}
	if e.ReferenceLabel != "" {
		return e.ReferenceLabel
	}
	return e.Reference
}

// evalScoreValue returns a RANK for sorting, not the raw score: every
// mode is normalised so that a smaller number is a better result.
//
// Perplexity and KL divergence are already lower-is-better and pass
// through. Accuracy is higher-is-better, so it is inverted (100 − acc)
// — otherwise one ascending sort would order the perplexity rows best
// first and the HellaSwag rows worst first in the same table.
//
// Performance runs sort LAST rather than first: they have no score, so
// evalNoScoreRank keeps them out of the way of the rows the button is
// about instead of parking them all at the top on a zero.
func evalScoreValue(e *benchmark.EvalScores) float64 {
	if e == nil || e.Mode == "" {
		return evalNoScoreRank
	}
	switch e.Mode {
	case string(evaluate.ModePerplexity):
		return e.Perplexity
	case string(evaluate.ModeKLDiv):
		return e.KLMean
	default:
		return 100 - e.Accuracy
	}
}

// evalChunkOrTaskText renders the count in parentheses: the chunk count
// (perplexity/KL) or the task count (the accuracy modes), or "full"
// when neither is recorded — which now means only that the run predates
// chunk-count parsing, since a current run records the real count even
// for an uncapped one.
func evalChunkOrTaskText(e *benchmark.EvalScores, unit string) string {
	if e.Chunks > 0 {
		return fmt.Sprintf("%d %s", e.Chunks, unit)
	}
	if e.Tasks > 0 {
		return fmt.Sprintf("%d tasks", e.Tasks)
	}
	return "full"
}

// hasEvalRuns reports whether any run carries capability scores. The
// results list and compare view show the Score column only when at
// least one compared run has them — a pure performance comparison
// renders exactly as before.
func hasEvalRuns(runs []benchmark.BenchmarkRun) bool {
	for i := range runs {
		if runs[i].Eval != nil {
			return true
		}
	}
	return false
}

// evalHas is the template-side form of hasEvalRuns over a single run.
func evalHas(e *benchmark.EvalScores) bool { return e != nil && e.Mode != "" }

// evalSizeText renders a capability preset's run size: the task or
// chunk count, or "full" when the preset has no cap (0 = everything).
func evalSizeText(p benchmark.Preset) string {
	switch p.EvalMode {
	case evaluate.ModeKLDiv, evaluate.ModePerplexity:
		if p.EvalChunks > 0 {
			return fmt.Sprintf("%d chunks", p.EvalChunks)
		}
		return "all chunks (full)"
	case evaluate.ModeHellaSwag, evaluate.ModeWinogrande:
		if p.EvalTasks > 0 {
			return fmt.Sprintf("%d tasks", p.EvalTasks)
		}
		return "all tasks (full)"
	}
	return "—"
}

// evalMeaningText renders what the mode's score means, in one line —
// the capability section of the About modal and the help page's
// quick-vs-full guidance share these sentences.
func evalMeaningText(p benchmark.Preset) string {
	switch p.EvalMode {
	case evaluate.ModePerplexity:
		return "Perplexity on the wikitext-2 test set — lower is better; the ± error bar is the uncertainty of the estimate."
	case evaluate.ModeKLDiv:
		return "How far the model's token probabilities deviate from the reference quant's — 0 is identical, lower is better."
	case evaluate.ModeHellaSwag:
		return "Commonsense accuracy — the share of candidate sentence endings the model ranks first; higher is better."
	case evaluate.ModeWinogrande:
		return "Pronoun-resolution accuracy — the share of sentences completed with the correct referent; higher is better."
	}
	return "—"
}

// fmtDurationSince renders a duration as a coarse relative age for the
// logits cache list: minutes, hours, or days (never "0 seconds ago").
func fmtDurationSince(d time.Duration) string {
	switch {
	case d < 0:
		return "just now"
	case d < time.Minute:
		return "just now"
	case d < time.Hour:
		m := int(d.Minutes())
		return fmt.Sprintf("%d min ago", m)
	case d < 24*time.Hour:
		h := int(d.Hours())
		return fmt.Sprintf("%d h ago", h)
	default:
		days := int(d.Hours() / 24)
		return fmt.Sprintf("%d day%s ago", days, pluralS(days))
	}
}

func pluralS(n int) string {
	if n == 1 {
		return ""
	}
	return "s"
}
