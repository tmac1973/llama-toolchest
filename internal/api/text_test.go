package api

import "testing"

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
		if got := truncateText(c.in, c.max); got != c.want {
			t.Errorf("truncateText(%q, %d) = %q, want %q", c.in, c.max, got, c.want)
		}
	}
}
