package benchmark

import (
	"strings"

	"github.com/tmac1973/llama-toolchest/internal/models"
)

// Why a run failed, as llama-server said it.
//
// The router answers a request for a model that did not load with one
// sentence — "model ... failed to load" — and the reason is only ever in
// its log: which buffer could not be allocated, on which device. That log
// is held in memory and emptied on every restart, so by the time anyone
// looks at a failed cell the reason is gone. These helpers take it out of
// the log at the moment of the failure, so it can be stored on the run;
// serverlog.ErrorLines picks the lines that say what went wrong.

// maxFailureReasonLen bounds the one line quoted in the error text.
const maxFailureReasonLen = 300

// serverReasonSep joins a run's own error to the line llama-server
// printed. failureHeadline cuts at it, so the separator must not occur in
// the errors this package writes itself.
const serverReasonSep = ". llama-server reported: "

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

// memoryWords mark a line that says memory ran out. When one is present
// it is the cause, wherever it sits among the other error lines.
var memoryWords = []string{"out of memory", "failed to allocate", "cudamalloc", "not enough memory", "insufficient memory"}

// saysOutOfMemory reports whether a log line says memory ran out.
func saysOutOfMemory(line string) bool {
	lower := strings.ToLower(line)
	for _, w := range memoryWords {
		if strings.Contains(lower, w) {
			return true
		}
	}
	return false
}

// OutOfMemory reports whether the lines llama-server printed about a
// failure say that memory ran out.
func OutOfMemory(lines []string) bool {
	for _, l := range lines {
		if saysOutOfMemory(l) {
			return true
		}
	}
	return false
}

// failureReason picks the one line that best says why a run failed: a
// line that says memory ran out when there is one, and otherwise the
// first line.
//
// The first line alone is not reliable. The log is a fixed number of
// recent lines, and a warm-up is retried, so what is held can begin in
// the middle of an earlier attempt — with that attempt's last line, the
// router reporting a lost connection, ahead of the next attempt's cause.
func failureReason(lines []string) string {
	for _, l := range lines {
		if saysOutOfMemory(l) {
			return l
		}
	}
	if len(lines) == 0 {
		return ""
	}
	return lines[0]
}

// withServerReason appends the line that best explains the failure, of
// those llama-server printed, to a run's own error text. With no lines
// the text is returned unchanged.
func withServerReason(headline string, lines []string) string {
	if len(lines) == 0 {
		return headline
	}
	reason := failureReason(lines)
	reason = models.TruncateText(reason, maxFailureReasonLen)
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
