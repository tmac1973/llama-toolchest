package models

import "testing"

// Text cut for display (model card quotes, llama-server output, labels)
// keeps whole characters: a byte slice could split a multi-byte one and
// leave invalid UTF-8 on the page.
func TestTruncateText(t *testing.T) {
	cases := []struct {
		in   string
		max  int
		want string
	}{
		{"short", 80, "short"},
		{"abcdef", 6, "abcdef"},
		{"abcdef", 3, "abc…"},
		// Each "é" is two bytes; a byte slice at 3 would split one.
		{"éééé", 3, "ééé…"},
	}
	for _, c := range cases {
		if got := TruncateText(c.in, c.max); got != c.want {
			t.Errorf("TruncateText(%q, %d) = %q, want %q", c.in, c.max, got, c.want)
		}
	}
}
