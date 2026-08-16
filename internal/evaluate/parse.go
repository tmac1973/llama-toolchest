package evaluate

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
)

// ParseError is returned when llama-perplexity's output does not match
// the exact format strings the parsers are pinned to (anchored to
// tools/perplexity/perplexity.cpp). It names the missing line so a
// llama.cpp format change fails loudly and is identifiable with
// errors.As, rather than surfacing as a silent zero score.
//
// It carries the tail of the tool's output. A missing score line is NOT
// always a format change: llama-perplexity exits 0 without printing a
// final score when the input is too short for a single chunk
// (perplexity.cpp:597-601) and when Winogrande scored fewer than 100
// tasks (perplexity.cpp:1296). Those cases are only distinguishable
// from the output itself, so the error carries it instead of sending
// the reader to re-run the tool by hand.
type ParseError struct {
	// Line is the format line the parser was looking for, in the
	// source's spelling (placeholders in <brackets>).
	Line string
	// Tail is the last errorTailBytes of the tool's combined output,
	// empty only when the parser was called without one.
	Tail string
}

func (e *ParseError) Error() string {
	msg := fmt.Sprintf("llama-perplexity printed no %q line — the run may have stopped before scoring, or the upstream output format changed", e.Line)
	if e.Tail != "" {
		msg += " — output tail: " + e.Tail
	}
	return msg
}

// newParseError builds a ParseError for the named line, attaching the
// tail of the output the parse ran over. Every parser goes through it
// so no missing-score error can be raised without its evidence.
func newParseError(line, output string) *ParseError {
	return &ParseError{Line: line, Tail: tailSnippet(strings.TrimSpace(output))}
}

// The pinned output formats, one regex per score line, anchored to the
// source's exact printf formats:
//
//	Final estimate: PPL = %.4lf +/- %.5lf                perplexity.cpp:654
//	Final Winogrande score(%d tasks): %.4lf +/- %.4lf    perplexity.cpp:1300
//	%zu\t%3.8lf%%\t[%3.4lf%%, %3.4lf%%]                  perplexity.cpp:1006
//	Mean    KLD: %10.6lf ± %10.6lf                       perplexity.cpp:1949
//	Maximum KLD: %10.6f                                  perplexity.cpp:1961
//	99.9%%   KLD: %10.6f                                 perplexity.cpp:1962
//	Same top p: %6.3lf ± %5.3lf %%                       perplexity.cpp:2005
//
// The decimal counts are pinned to the source (%.4lf → 4 decimals,
// %3.8lf → 8, %3.4lf/%6.3lf → 4/3): if llama.cpp changes a precision,
// the parser must fail loudly, not quietly re-score at a different
// shape. Only the padding whitespace around values is matched loosely
// (the %10.6f-style widths are right-aligned and value-dependent).
var (
	// Note: the plus sign in "+/-" is escaped — an unescaped "+" would
	// quantify the preceding space in RE2 and never match.
	pplFinalRe  = regexp.MustCompile(`^Final estimate: PPL = (\d+\.\d{4}) \+/- (\d+\.\d{5})\s*$`)
	winoFinalRe = regexp.MustCompile(
		`^Final Winogrande score\((\d+) tasks\): (\d+\.\d{4}) \+/- (\d+\.\d{4})\s*$`)
	// HellaSwag prints one line per task; the LAST line is the score.
	// Column 1 is the task count, column 2 the cumulative accuracy
	// percent, the brackets an ASYMMETRIC 95% confidence interval.
	hsTaskRe    = regexp.MustCompile(`^(\d+)\t(\d+\.\d{8})%\t\[(\d+\.\d{4})%, (\d+\.\d{4})%\]\s*$`)
	klMeanRe    = regexp.MustCompile(`^Mean\s+KLD:\s*([-+]?\d+\.\d{6}) ±\s*([-+]?\d+\.\d{6})\s*$`)
	klMaxRe     = regexp.MustCompile(`^Maximum KLD:\s*([-+]?\d+\.\d{6})\s*$`)
	klP999Re    = regexp.MustCompile(`^99\.9%\s+KLD:\s*([-+]?\d+\.\d{6})\s*$`)
	klSameTopRe = regexp.MustCompile(`^Same top p:\s*([-+]?\d+\.\d{3}) ±\s*([-+]?\d+\.\d{3}) %\s*$`)
	// The chunk count the tool actually scored, from its own header
	// line ("perplexity: calculating perplexity over 2 chunks,
	// n_ctx=512, batch_size=2048, n_seq=4", perplexity.cpp:606). It is
	// the only place the count appears — the final-score lines carry
	// none — so a full run can report "650 chunks" instead of "full",
	// and a short dataset shows the count it really got.
	pplChunkCountRe = regexp.MustCompile(`^perplexity: calculating perplexity over (\d+) chunks`)
)

// logPrefixRe matches llama.cpp's log line prefix so it can be stripped
// before the score patterns run.
//
// common_init turns BOTH the prefix and the timestamps on
// (common/common.cpp:391-392), and the logger then writes
// "<M>.<SS>.<mmm>.<uuu> I " ahead of every LOG_INF / LOG_WRN / LOG_ERR
// line (common/log.cpp:96-115); plain LOG lines (level NONE) get
// nothing. Two of the four score lines are LOG_INF — the final
// perplexity estimate (perplexity.cpp:654) and the final Winogrande
// score (perplexity.cpp:1300) — so an anchored ^Final… pattern matches
// neither, while the HellaSwag and KL lines (plain LOG) are unprefixed.
// Stripping here rather than passing --no-log-prefix keeps older builds
// working: those flags are not accepted by every llama.cpp version the
// builder can produce.
//
// The colour escapes are optional: the logger emits them only when the
// stream is a terminal (--log-colors auto), which a captured pipe is
// not, but matching them costs nothing and makes the parser independent
// of how the process was launched. Every part is optional, so a line
// with no prefix passes through untouched.
var logPrefixRe = regexp.MustCompile(
	`^(?:\x1b\[[0-9;]*m)*` + // leading colour
		`(?:\d+\.\d{2}\.\d{3}\.\d{3}(?:\x1b\[[0-9;]*m)*[ ]+)?` + // timestamp
		`(?:(?:\x1b\[[0-9;]*m)*[IWED][ ]+(?:\x1b\[[0-9;]*m)*)?`) // level letter

// stripLogPrefix removes the log prefix and the trailing CR from one
// output line.
func stripLogPrefix(line string) string {
	return logPrefixRe.ReplaceAllString(strings.TrimRight(line, "\r"), "")
}

// lastLineMatching returns the submatch of the LAST output line
// matching re, or nil. Scores live in final lines (or, for HellaSwag,
// in the last of the per-task lines), so the last match wins; earlier
// duplicates are noise.
func lastLineMatching(output string, re *regexp.Regexp) []string {
	lines := strings.Split(output, "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		if m := re.FindStringSubmatch(stripLogPrefix(lines[i])); m != nil {
			return m
		}
	}
	return nil
}

// firstLineMatching is lastLineMatching's forward twin, for lines the
// tool prints once before scoring (the chunk-count header).
func firstLineMatching(output string, re *regexp.Regexp) []string {
	for _, line := range strings.Split(output, "\n") {
		if m := re.FindStringSubmatch(stripLogPrefix(line)); m != nil {
			return m
		}
	}
	return nil
}

// parseChunkCount returns the chunk count the tool announced it would
// score, or 0 when the header line is absent (an older format, or a run
// that failed before it). Callers treat 0 as "unknown" and fall back to
// the requested cap.
func parseChunkCount(output string) int {
	m := firstLineMatching(output, pplChunkCountRe)
	if m == nil {
		return 0
	}
	n, err := strconv.Atoi(m[1])
	if err != nil {
		return 0
	}
	return n
}

// parsePerplexity extracts the final estimate from the combined output.
// The line is LOG_INF (stderr) — the combined buffer is what carries it.
func parsePerplexity(output string) (value, errVal float64, err error) {
	m := lastLineMatching(output, pplFinalRe)
	if m == nil {
		return 0, 0, newParseError(`Final estimate: PPL = <f> +/- <f>`, output)
	}
	v, _ := strconv.ParseFloat(m[1], 64)
	e, _ := strconv.ParseFloat(m[2], 64)
	return v, e, nil
}

// parseWinogrande extracts the final score. The tool prints a symmetric
// +/- error; it is mapped onto the CI bounds as acc−err / acc+err.
func parseWinogrande(output string) (acc, low, high float64, tasks int, err error) {
	m := lastLineMatching(output, winoFinalRe)
	if m == nil {
		return 0, 0, 0, 0, newParseError(`Final Winogrande score(<n> tasks): <f> +/- <f>`, output)
	}
	tasks, _ = strconv.Atoi(m[1])
	acc, _ = strconv.ParseFloat(m[2], 64)
	sigma, _ := strconv.ParseFloat(m[3], 64)
	return acc, acc - sigma, acc + sigma, tasks, nil
}

// parseHellaSwag extracts the LAST per-task line — the cumulative
// accuracy over all scored tasks with its asymmetric CI bounds.
func parseHellaSwag(output string) (acc, low, high float64, tasks int, err error) {
	m := lastLineMatching(output, hsTaskRe)
	if m == nil {
		return 0, 0, 0, 0, newParseError(`<n>\t<acc>%\t[<low>%, <high>%]`, output)
	}
	tasks, _ = strconv.Atoi(m[1])
	acc, _ = strconv.ParseFloat(m[2], 64)
	low, _ = strconv.ParseFloat(m[3], 64)
	high, _ = strconv.ParseFloat(m[4], 64)
	return acc, low, high, tasks, nil
}

// klStats is the KL divergence block parsed from the comparison output.
type klStats struct {
	Mean       float64
	MeanErr    float64
	Max        float64
	P999       float64
	SameTop    float64
	SameTopErr float64
}

// parseKLDivergence extracts the KL block. Each line is named
// individually in the ParseError so a partial format change points at
// the exact line.
func parseKLDivergence(output string) (klStats, error) {
	var s klStats

	if m := lastLineMatching(output, klMeanRe); m != nil {
		s.Mean, _ = strconv.ParseFloat(m[1], 64)
		s.MeanErr, _ = strconv.ParseFloat(m[2], 64)
	} else {
		return s, newParseError(`Mean    KLD: <f> ± <f>`, output)
	}

	if m := lastLineMatching(output, klMaxRe); m != nil {
		s.Max, _ = strconv.ParseFloat(m[1], 64)
	} else {
		return s, newParseError(`Maximum KLD: <f>`, output)
	}

	if m := lastLineMatching(output, klP999Re); m != nil {
		s.P999, _ = strconv.ParseFloat(m[1], 64)
	} else {
		return s, newParseError(`99.9%   KLD: <f>`, output)
	}

	if m := lastLineMatching(output, klSameTopRe); m != nil {
		s.SameTop, _ = strconv.ParseFloat(m[1], 64)
		s.SameTopErr, _ = strconv.ParseFloat(m[2], 64)
	} else {
		return s, newParseError(`Same top p: <f> ± <f> %`, output)
	}

	return s, nil
}

// parse dispatches to the per-mode parser and fills the Result. The
// engine fills every field except Reference (the runner records the KL
// reference identity there); Dataset is derived from the mode, which
// fixes the dataset the tool consumed.
//
// Chunks is the count the tool ANNOUNCED it would score, taken from its
// own header line; the requested cap is the fallback for outputs that
// carry no header. That distinction matters twice: a full run (cap 0)
// records the real ~650 instead of the placeholder "full", and a run
// over a short dataset records what it actually scored rather than the
// cap it never reached.
func parse(spec Spec, output string) (Result, error) {
	res := Result{
		Mode:        string(spec.Mode),
		Dataset:     spec.Mode.DatasetName(),
		ContextSize: EvalContextSize,
	}
	chunks := parseChunkCount(output)
	if chunks == 0 {
		chunks = spec.Chunks
	}
	switch spec.Mode {
	case ModePerplexity:
		res.Chunks = chunks
		v, e, err := parsePerplexity(output)
		if err != nil {
			return res, err
		}
		res.Perplexity = v
		res.PerplexityErr = e
	case ModeKLDiv:
		res.Chunks = chunks
		s, err := parseKLDivergence(output)
		if err != nil {
			return res, err
		}
		res.KLMean = s.Mean
		res.KLMeanErr = s.MeanErr
		res.KLMax = s.Max
		res.KLP999 = s.P999
		res.SameTopPct = s.SameTop
		res.SameTopPctErr = s.SameTopErr
	case ModeHellaSwag:
		acc, low, high, tasks, err := parseHellaSwag(output)
		if err != nil {
			return res, err
		}
		res.Accuracy = acc
		res.AccuracyCILow = low
		res.AccuracyCIHigh = high
		res.Tasks = tasks
	case ModeWinogrande:
		acc, low, high, tasks, err := parseWinogrande(output)
		if err != nil {
			return res, err
		}
		res.Accuracy = acc
		res.AccuracyCILow = low
		res.AccuracyCIHigh = high
		res.Tasks = tasks
	default:
		return res, fmt.Errorf("unknown evaluation mode %q", spec.Mode)
	}
	return res, nil
}
