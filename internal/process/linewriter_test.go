package process

import (
	"slices"
	"strings"
	"testing"
)

// Lines are emitted whole however the writes split them, a CR before the
// newline is dropped, a last line without a newline waits for flush, and a
// line over maxLogLine comes out in pieces.
func TestLineWriter(t *testing.T) {
	var got []string
	w := &lineWriter{emit: func(l string) { got = append(got, l) }}
	for _, p := range []string{"one\ntw", "o\r\nthr", "ee"} {
		w.Write([]byte(p))
	}
	if want := []string{"one", "two"}; !slices.Equal(got, want) {
		t.Fatalf("before flush: %q, want %q", got, want)
	}
	w.flush()
	if want := []string{"one", "two", "three"}; !slices.Equal(got, want) {
		t.Fatalf("after flush: %q, want %q", got, want)
	}

	got = nil
	w.Write([]byte(strings.Repeat("x", maxLogLine+10) + "\n"))
	if len(got) != 2 || len(got[0]) != maxLogLine || len(got[1]) != 10 {
		t.Errorf("long line came out as %d pieces", len(got))
	}
}
