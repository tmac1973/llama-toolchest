package models

import (
	"maps"
	"slices"
	"strings"
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
