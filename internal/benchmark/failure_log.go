package benchmark

import (
	"regexp"
	"strings"
)

// Why a run failed, as llama-server said it.
//
// The router answers a request for a model that did not load with one
// sentence — "model ... failed to load" — and the reason is only ever in
// its log: which buffer could not be allocated, on which device. That log
// is held in memory and emptied on every restart, so by the time anyone
// looks at a failed cell the reason is gone. These helpers take it out of
// the log at the moment of the failure, so it can be stored on the run.

// maxFailureLogLines bounds what is kept on a run. A failed load prints a
// handful of error lines; a dozen covers the cause and what followed from
// it without turning the results file into a log archive.
const maxFailureLogLines = 12

// maxFailureReasonLen bounds the one line quoted in the error text.
const maxFailureReasonLen = 300

// serverReasonSep joins a run's own error to the line llama-server
// printed. failureHeadline cuts at it, so the separator must not occur in
// the errors this package writes itself.
const serverReasonSep = ". llama-server reported: "

// serverLogRE matches a llama-server log line: an optional "[port] "
// prefix naming the model instance, the timestamp, and the level letter.
// The prefix allows leading spaces for the same reason memreport's does —
// the router pads a port below 10000.
var serverLogRE = regexp.MustCompile(`^(?:\[\s*\d+\] )?[\d.]+ ([A-Z]) (.*)$`)

// instancePrefixRE is the "[port] " prefix on its own, for lines that
// carry no timestamp or level.
var instancePrefixRE = regexp.MustCompile(`^\[\s*\d+\] `)

// rawFailureWords mark a line that reports a failure without going
// through llama.cpp's logger. A process that aborts prints straight to
// its error stream — "CUDA error: out of memory", "terminate called after
// throwing" — and those lines have no level to filter on.
var rawFailureWords = []string{"error", "out of memory", "failed", "abort", "terminate called", "exception", "segmentation fault"}

// linesAfter returns the lines that follow the last occurrence of mark,
// or all of them when mark is empty or no longer held. Log lines carry a
// timestamp, so the last line seen before a run started identifies where
// that run's own output begins.
func linesAfter(log []string, mark string) []string {
	if mark == "" {
		return log
	}
	for i := len(log) - 1; i >= 0; i-- {
		if log[i] == mark {
			return log[i+1:]
		}
	}
	return log
}

// serverErrorLines returns the error lines in a stretch of llama-server
// log, oldest first, with the instance prefix, timestamp and level
// removed. A line that repeats is kept once: a warm-up is retried, and
// every retry prints the same failure again.
//
// The first lines are the ones kept when there are more than fit. In a
// failed load the first error is the cause — the allocation that did not
// succeed — and the rest are each layer above it reporting that it could
// not carry on.
func serverErrorLines(log []string) []string {
	var out []string
	seen := map[string]bool{}
	for _, line := range log {
		line = strings.TrimSpace(strings.TrimPrefix(line, "data: "))
		var text string
		if m := serverLogRE.FindStringSubmatch(line); m != nil {
			if m[1] != "E" {
				continue
			}
			text = strings.TrimSpace(m[2])
		} else {
			raw := strings.TrimSpace(instancePrefixRE.ReplaceAllString(line, ""))
			if !hasFailureWord(raw) {
				continue
			}
			text = raw
		}
		if text == "" || seen[text] {
			continue
		}
		seen[text] = true
		out = append(out, text)
		if len(out) == maxFailureLogLines {
			break
		}
	}
	return out
}

func hasFailureWord(line string) bool {
	lower := strings.ToLower(line)
	for _, w := range rawFailureWords {
		if strings.Contains(lower, w) {
			return true
		}
	}
	return false
}

// withServerReason appends the first error line llama-server printed to
// a run's own error text. With no lines the text is returned unchanged.
func withServerReason(headline string, lines []string) string {
	if len(lines) == 0 {
		return headline
	}
	reason := lines[0]
	if len(reason) > maxFailureReasonLen {
		reason = reason[:maxFailureReasonLen] + "…"
	}
	return headline + serverReasonSep + reason
}

// failureHeadline is a run's error without the line quoted from
// llama-server. Two failures are "the same" when their headlines match:
// the quoted line can differ in detail between two cells that fail for
// one reason — the size of the buffer that did not fit changes with the
// split mode — and that must not make them look unrelated.
func failureHeadline(err string) string {
	head, _, _ := strings.Cut(err, serverReasonSep)
	return head
}
