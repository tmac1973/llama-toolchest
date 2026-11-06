package generation

import (
	"fmt"
	"regexp"
	"strings"
)

// lastLinesSearched is how far back from the end LastErrorLine looks.
const lastLinesSearched = 50

// maxErrorLineLen caps the log line added to an error message.
const maxErrorLineLen = 200

// sdSourceSuffix is the " --- file.cpp:123" that sd.cpp adds to its log
// lines.
var sdSourceSuffix = regexp.MustCompile(` --- [\w./-]+:\d+$`)

// LastErrorLine returns the line of a backend's log that most likely says
// why it failed. Of the last 50 lines it takes the first sd.cpp error
// line ("[E]"): sd.cpp logs the cause first and then each caller that
// gave up ("[E] get sd version from file failed", then "[E] new_sd_ctx_t
// failed"). Without one, it takes the last line with the word "error",
// and else the last line. Toolchest's own "==> ..." lines and empty lines
// are skipped. It returns "" when nothing is left.
func LastErrorLine(lines []string) string {
	firstE, lastError, last := "", "", ""
	for _, raw := range lines[max(0, len(lines)-lastLinesSearched):] {
		line := errorLine(raw)
		if line == "" || strings.HasPrefix(line, "==>") {
			continue
		}
		last = line
		if firstE == "" && strings.Contains(line, "[E]") {
			firstE = line
		}
		if strings.Contains(strings.ToLower(line), "error") {
			lastError = line
		}
	}
	switch {
	case firstE != "":
		return firstE
	case lastError != "":
		return lastError
	}
	return last
}

// CleanLogLine returns the text of a backend log line: trimmed, without
// an SSE "data: " prefix or sd.cpp's " --- file.cpp:123" source suffix. A
// progress bar redraws itself with carriage returns and an erase-line
// code, and a line printed while it shows lands on the same line, after
// the last of them; only that part is kept. The "[E] " level is kept:
// LastErrorLine looks for it.
func CleanLogLine(line string) string {
	line = strings.TrimRight(line, "\r\n")
	if i := strings.LastIndex(line, "\r"); i >= 0 {
		line = line[i+1:]
	}
	if i := strings.LastIndex(line, "\x1b[K"); i >= 0 {
		line = line[i+len("\x1b[K"):]
	}
	line = strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(line), "data: "))
	return sdSourceSuffix.ReplaceAllString(line, "")
}

// errorLine is CleanLogLine cut to maxErrorLineLen, for an error message.
func errorLine(line string) string {
	line = CleanLogLine(line)
	if r := []rune(line); len(r) > maxErrorLineLen {
		line = string(r[:maxErrorLineLen]) + "…"
	}
	return line
}

// withLastLogLine adds the backend's last error line to a start failure,
// keeping err for errors.Is. It returns err unchanged when there is no
// line to add.
func withLastLogLine(err error, lines []string) error {
	if err == nil {
		return nil
	}
	line := LastErrorLine(lines)
	if line == "" {
		return err
	}
	return fmt.Errorf("%w. Last log line: %s", err, line)
}
