// Package serverlog reads the lines llama-server's router writes: its own
// lines, and those of each model instance, which it forwards with a
// "[port] " prefix. It is the one place that knows their shape, for the
// benchmark runner (why a run failed), the memory report (when a load
// starts and ends) and anything else that reads the router's log.
//
// It imports nothing else from this project, so any package can use it.
package serverlog

import (
	"regexp"
	"strings"
)

// MaxErrorLines bounds what ErrorLines returns. A failed load prints a
// handful of error lines; a dozen covers the cause and what followed from
// it without turning a stored record into a log archive.
const MaxErrorLines = 12

// LineRE matches a llama-server log line: an optional "[port] " prefix
// naming the model instance, the timestamp, and the level letter. The
// prefix allows leading spaces because the router pads a port below
// 10000.
var LineRE = regexp.MustCompile(`^(?:\[\s*\d+\] )?[\d.]+ ([A-Z]) (.*)$`)

// InstancePrefixRE is the "[port] " prefix on its own, for lines that
// carry no timestamp or level.
var InstancePrefixRE = regexp.MustCompile(`^\[\s*(\d+)\] `)

// spawnRE matches the router's announcement that it has started a child
// process for a model, which is the only line tying an instance back to
// the model it serves:
//
//	0.05.123.456 I srv          load: spawning server instance with name=Qwen3.6-27B on port 46231
//
// The router logs this itself rather than through a model instance, at
// LOG_INF — verbosity 3, llama.cpp's default.
//
// The name is matched greedily up to the port so a model whose id
// contains " on port " still resolves.
var spawnRE = regexp.MustCompile(`spawning server instance with name=(.+) on port (\d+)\s*$`)

// readyRE matches the line a child writes when it has finished loading
// and is ready to serve:
//
//	0.14.275.900 D [46231] cmd_child_to_router:state:{"state":"ready","payload":{...}}
//
// The router logs these control messages itself, at debug level, so the
// line carries the router's own timestamp and level prefix — unlike the
// child's ordinary output, which it forwards bare. The prefix is optional
// here so a build that logs it without one still matches.
var readyRE = regexp.MustCompile(`^(?:[\d.]+ )?(?:[A-Z] )?\[\s*(\d+)\] cmd_child_to_router:state:.*"state"\s*:\s*"ready"`)

// loadedRE matches the line a model instance prints through its own
// logger when its model is loaded and it is about to serve:
//
//	[45221] 0.07.999.369 I srv  server_run_i: model loaded
//
// It is logged at info level, so it is there at llama.cpp's default
// verbosity. The router's own "ready" line (readyRE) is logged at debug
// level and only appears at verbosity 5.
var loadedRE = regexp.MustCompile(`^\[\s*(\d+)\] (?:[\d.]+ )?(?:[A-Z] )?srv\s+\S+: model loaded\s*$`)

// LoadedPort reports the port of the instance whose own line says its
// model is loaded. Unlike ReadyPort it needs no more than the default
// verbosity.
func LoadedPort(line string) (port string, ok bool) {
	m := loadedRE.FindStringSubmatch(line)
	if m == nil {
		return "", false
	}
	return m[1], true
}

// ParseSpawn reports the model name and port in the router's line
// announcing a new model instance.
func ParseSpawn(line string) (name, port string, ok bool) {
	m := spawnRE.FindStringSubmatch(line)
	if m == nil {
		return "", "", false
	}
	return m[1], m[2], true
}

// ReadyPort reports the port of the instance whose line says it has
// finished loading.
func ReadyPort(line string) (port string, ok bool) {
	m := readyRE.FindStringSubmatch(line)
	if m == nil {
		return "", false
	}
	return m[1], true
}

// InstancePort reports the port in a line's "[port] " prefix, without the
// padding: the instance that wrote the line. The router's own lines have
// no prefix.
func InstancePort(line string) (port string, ok bool) {
	m := InstancePrefixRE.FindStringSubmatch(line)
	if m == nil {
		return "", false
	}
	return m[1], true
}

// rawFailureWords mark a line that reports a failure without going
// through llama.cpp's logger. A process that aborts prints straight to
// its error stream — "CUDA error: out of memory", "terminate called after
// throwing" — and those lines have no level to filter on.
var rawFailureWords = []string{"error", "out of memory", "failed", "abort", "terminate called", "exception", "segmentation fault"}

// ErrorLines returns the error lines in a stretch of llama-server log,
// oldest first, with the instance prefix, timestamp and level removed. A
// line that repeats is kept once: a warm-up is retried, and every retry
// prints the same failure again.
//
// The first lines are the ones kept when there are more than fit. In a
// failed load the first error is the cause — the allocation that did not
// succeed — and the rest are each layer above it reporting that it could
// not carry on. Backtrace frames are left out: they are most of what an
// aborting process prints and none of the explanation.
func ErrorLines(log []string) []string {
	var out []string
	seen := map[string]bool{}
	for _, line := range log {
		line = strings.TrimSpace(strings.TrimPrefix(line, "data: "))
		var text string
		if m := LineRE.FindStringSubmatch(line); m != nil {
			if m[1] != "E" {
				continue
			}
			text = strings.TrimSpace(m[2])
		} else {
			raw := strings.TrimSpace(InstancePrefixRE.ReplaceAllString(line, ""))
			if !hasFailureWord(raw) {
				continue
			}
			text = raw
		}
		if text == "" || seen[text] || stackFrameRE.MatchString(text) {
			continue
		}
		seen[text] = true
		out = append(out, text)
		if len(out) == MaxErrorLines {
			break
		}
	}
	return out
}

// stackFrameRE matches one frame of the backtrace a process prints when
// it aborts: a library, a symbol and an address. The frames say where the
// code was, not what went wrong, and their addresses differ on every run.
var stackFrameRE = regexp.MustCompile(`\(.*\+0x[0-9a-f]+\)\s*\[0x[0-9a-f]+\]|^\S+\s*\[0x[0-9a-f]+\]$`)

func hasFailureWord(line string) bool {
	lower := strings.ToLower(line)
	for _, w := range rawFailureWords {
		if strings.Contains(lower, w) {
			return true
		}
	}
	return false
}
