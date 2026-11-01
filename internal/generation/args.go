package generation

import "strings"

// JoinArgs writes flags as one line that ValidateExtraArgs (splitWords)
// reads back as the same words, for a form field. splitWords has no
// escapes: a quote groups everything up to the matching quote. So a word
// with a space or a quote goes in double quotes, and each double quote in
// it is closed, given in single quotes, and opened again.
func JoinArgs(args []string) string {
	words := make([]string, len(args))
	for i, a := range args {
		words[i] = joinWord(a)
	}
	return strings.Join(words, " ")
}

func joinWord(w string) string {
	if w != "" && !strings.ContainsAny(w, " \t\n\"'") {
		return w
	}
	return `"` + strings.ReplaceAll(w, `"`, `"'"'"`) + `"`
}
