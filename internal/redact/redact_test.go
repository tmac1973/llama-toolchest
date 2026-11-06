package redact

import (
	"errors"
	"fmt"
	"net/url"
	"strings"
	"testing"
)

func TestURL(t *testing.T) {
	for in, want := range map[string]string{
		"https://user:ghp_secret@github.com/org/repo.git":        "https://user:xxxxx@github.com/org/repo.git",
		"https://ghp_secret@github.com/org/repo.git":             "https://xxxxx@github.com/org/repo.git",
		"https://github.com/ggml-org/llama.cpp":                  "https://github.com/ggml-org/llama.cpp",
		"git@github.com:org/repo.git":                            "git@github.com:org/repo.git",
		"https://cdn.example.com/f.gguf?X-Amz-Signature=abc&e=1": "https://cdn.example.com/f.gguf?xxxxx",
		"not a url \x7f": "not a url \x7f",
	} {
		if got := URL(in); got != want {
			t.Errorf("URL(%q) = %q, want %q", in, got, want)
		}
	}
}

// A *url.Error loses its signature, also once wrapped, and keeps
// matching.
func TestError(t *testing.T) {
	inner := &url.Error{Op: "Get", URL: "https://cdn.example.com/f?sig=secret", Err: errors.New("EOF")}
	err := fmt.Errorf("downloading: %w", Error(inner))
	if strings.Contains(err.Error(), "secret") {
		t.Errorf("signature still in %q", err)
	}
	var ue *url.Error
	if !errors.As(err, &ue) {
		t.Error("the url.Error is no longer in the chain")
	}
}
