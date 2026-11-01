package generation

import (
	"slices"
	"strings"
)

// FilesKey lists a set's files as "slot=path", sorted by slot, so a
// measurement can tell whether the set still has the files it was
// measured with.
func FilesKey(set ModelSet) []string {
	out := make([]string, 0, len(set.Components))
	for _, c := range set.Components {
		out = append(out, c.Slot+"="+c.Path)
	}
	slices.SortFunc(out, func(a, b string) int {
		sa, _, _ := strings.Cut(a, "=")
		sb, _, _ := strings.Cut(b, "=")
		return strings.Compare(sa, sb)
	})
	return out
}
