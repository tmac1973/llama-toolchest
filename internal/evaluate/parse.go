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
type ParseError struct {
	// Line is the format line the parser was looking for, in the
	// source's spelling (placeholders in <brackets>).
	Line string
}

func (e *ParseError) Error() string {
	return fmt.Sprintf("llama-perplexity output missing %q line — upstream output format may have changed", e.Line)
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
)

// lastLineMatching returns the submatch of the LAST output line
// matching re, or nil. Scores live in final lines (or, for HellaSwag,
// in the last of the per-task lines), so the last match wins; earlier
// duplicates are noise.
func lastLineMatching(output string, re *regexp.Regexp) []string {
	lines := strings.Split(output, "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		if m := re.FindStringSubmatch(strings.TrimRight(lines[i], "\r")); m != nil {
			return m
		}
	}
	return nil
}

// parsePerplexity extracts the final estimate from the combined output.
// The line is LOG_INF (stderr) — the combined buffer is what carries it.
func parsePerplexity(output string) (value, errVal float64, err error) {
	m := lastLineMatching(output, pplFinalRe)
	if m == nil {
		return 0, 0, &ParseError{Line: `Final estimate: PPL = <f> +/- <f>`}
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
		return 0, 0, 0, 0, &ParseError{Line: `Final Winogrande score(<n> tasks): <f> +/- <f>`}
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
		return 0, 0, 0, 0, &ParseError{Line: `<n>\t<acc>%\t[<low>%, <high>%]`}
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
		return s, &ParseError{Line: `Mean    KLD: <f> ± <f>`}
	}

	if m := lastLineMatching(output, klMaxRe); m != nil {
		s.Max, _ = strconv.ParseFloat(m[1], 64)
	} else {
		return s, &ParseError{Line: `Maximum KLD: <f>`}
	}

	if m := lastLineMatching(output, klP999Re); m != nil {
		s.P999, _ = strconv.ParseFloat(m[1], 64)
	} else {
		return s, &ParseError{Line: `99.9%   KLD: <f>`}
	}

	if m := lastLineMatching(output, klSameTopRe); m != nil {
		s.SameTop, _ = strconv.ParseFloat(m[1], 64)
		s.SameTopErr, _ = strconv.ParseFloat(m[2], 64)
	} else {
		return s, &ParseError{Line: `Same top p: <f> ± <f> %`}
	}

	return s, nil
}

// parse dispatches to the per-mode parser and fills the Result. The
// engine fills every field except Reference (the runner records the KL
// reference identity there); Dataset is derived from the mode, which
// fixes the dataset the tool consumed.
func parse(spec Spec, output string) (Result, error) {
	res := Result{
		Mode:        string(spec.Mode),
		Dataset:     spec.Mode.DatasetName(),
		ContextSize: EvalContextSize,
	}
	switch spec.Mode {
	case ModePerplexity:
		res.Chunks = spec.Chunks
		v, e, err := parsePerplexity(output)
		if err != nil {
			return res, err
		}
		res.Perplexity = v
		res.PerplexityErr = e
	case ModeKLDiv:
		res.Chunks = spec.Chunks
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
