package builder

import "testing"

// The clone log names the repository without any token a custom URL
// carries; a URL with no credentials is logged as it is.
func TestRedactURL(t *testing.T) {
	for in, want := range map[string]string{
		"https://user:ghp_secret@github.com/org/repo.git": "https://user:xxxxx@github.com/org/repo.git",
		"https://ghp_secret@github.com/org/repo.git":      "https://xxxxx@github.com/org/repo.git",
		"https://github.com/ggml-org/llama.cpp":           "https://github.com/ggml-org/llama.cpp",
		"git@github.com:org/repo.git":                     "git@github.com:org/repo.git",
	} {
		if got := redactURL(in); got != want {
			t.Errorf("redactURL(%q) = %q, want %q", in, got, want)
		}
	}
}
