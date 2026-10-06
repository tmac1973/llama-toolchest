package api

import "unicode/utf8"

// plural returns one when n is 1 and many otherwise.
func plural(n int, one, many string) string {
	if n == 1 {
		return one
	}
	return many
}

// truncateText shortens s to at most limit characters (runes, not bytes, so
// a multi-byte character is never cut in half) and adds "…" when it cut
// something off.
func truncateText(s string, limit int) string {
	if utf8.RuneCountInString(s) <= limit {
		return s
	}
	return string([]rune(s)[:limit]) + "…"
}
