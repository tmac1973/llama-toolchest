package generation

import (
	"slices"
	"testing"
)

// JoinArgs writes flags that ValidateExtraArgs reads back unchanged: every
// catalog entry's, and words with spaces and quotes.
func TestJoinArgsRoundTrip(t *testing.T) {
	cases := [][]string{
		{"--negative-prompt", `say "hi" it's`, "--x", ""},
		{"--a", "two words", "--b", `"quoted"`, "--c", "it's"},
	}
	for _, e := range Catalog() {
		cases = append(cases, e.LaunchArgs)
	}
	for _, args := range cases {
		if args == nil {
			continue
		}
		line := JoinArgs(args)
		got, err := ValidateExtraArgs(line)
		if err != nil || !slices.Equal(got, args) {
			t.Errorf("JoinArgs(%q) = %s, read back as %q (%v)", args, line, got, err)
		}
	}
	if got := JoinArgs([]string{"--steps", "4", "--backend", "te=cpu"}); got != "--steps 4 --backend te=cpu" {
		t.Errorf("plain words were quoted: %s", got)
	}
}
