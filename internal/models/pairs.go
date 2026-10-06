package models

import (
	"maps"
	"slices"
	"strings"
	"unicode/utf8"
)

// JoinSorted renders a map as "k=v" pairs sorted by key and joined with
// sep: the stable text of a set of settings, as a sweep point's export
// cell or an Autotune candidate's identity. An empty map gives "".
func JoinSorted(m map[string]string, sep string) string {
	keys := slices.Sorted(maps.Keys(m))
	pairs := make([]string, len(keys))
	for i, k := range keys {
		pairs[i] = k + "=" + m[k]
	}
	return strings.Join(pairs, sep)
}

// TruncateText shortens s to at most limit characters (runes, not bytes, so
// a multi-byte character is never cut in half) and adds "…" when it cut
// something off. Model card quotes and llama-server output, which this
// trims for display, are often not ASCII.
func TruncateText(s string, limit int) string {
	if utf8.RuneCountInString(s) <= limit {
		return s
	}
	return string([]rune(s)[:limit]) + "…"
}
