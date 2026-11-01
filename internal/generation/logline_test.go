package generation

import (
	"errors"
	"strings"
	"testing"
)

func TestLastErrorLine(t *testing.T) {
	for name, tc := range map[string]struct {
		lines []string
		want  string
	}{
		// The log of a set with an upscaler file as its diffusion model
		// (asmodean, 2026-11-01): the first [E] line is the cause.
		"sd.cpp error chain": {[]string{
			"[I] load /data/models/x.safetensors using safetensors format --- model_loader.cpp:222",
			"",
			"[E] get sd version from file failed: '' --- diffusion_engine.cpp:972",
			"[E] new_sd_ctx_t failed --- main.cpp:105",
			"==> sd-server ct-e9 exited with error: exit status 1",
		}, "[E] get sd version from file failed: ''"},
		"SSE prefix":     {[]string{"data: [E] out of memory --- ggml.c:1"}, "[E] out of memory"},
		"error word":     {[]string{"loading", "error: model not found", "bye"}, "error: model not found"},
		"no error lines": {[]string{"one", "two", "==> x stopped"}, "two"},
		"empty":          {nil, ""},
		"only toolchest": {[]string{"==> x exited with error: exit status 1"}, ""},
	} {
		if got := LastErrorLine(tc.lines); got != tc.want {
			t.Errorf("%s: got %q, want %q", name, got, tc.want)
		}
	}
}

// Only the last 50 lines count.
func TestLastErrorLineLooksAtTheEnd(t *testing.T) {
	lines := []string{"[E] old failure"}
	for range lastLinesSearched {
		lines = append(lines, "info")
	}
	if got := LastErrorLine(lines); got != "info" {
		t.Errorf("got %q, want the last line", got)
	}
}

func TestLastErrorLineIsCut(t *testing.T) {
	got := LastErrorLine([]string{"[E] " + strings.Repeat("x", 500)})
	if n := len([]rune(got)); n != maxErrorLineLen+1 || !strings.HasSuffix(got, "…") {
		t.Errorf("got %d runes: %q", n, got)
	}
}

func TestWithLastLogLine(t *testing.T) {
	base := errors.New("sd-server x failed to start: exit status 1")
	err := withLastLogLine(base, []string{"[E] boom"})
	if !errors.Is(err, base) || !strings.Contains(err.Error(), "exit status 1. Last log line: [E] boom") {
		t.Errorf("got %v", err)
	}
	if err := withLastLogLine(base, nil); err != base {
		t.Errorf("no log lines: got %v, want the error unchanged", err)
	}
	if withLastLogLine(nil, []string{"[E] boom"}) != nil {
		t.Error("nil error got a log line")
	}
}
