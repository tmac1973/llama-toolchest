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
		line := cleanLogLine(raw)
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

// cleanLogLine trims a log line, drops an SSE "data: " prefix and sd.cpp's
// source location, and cuts it to maxErrorLineLen.
func cleanLogLine(line string) string {
	line = strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(line), "data: "))
	line = sdSourceSuffix.ReplaceAllString(line, "")
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
