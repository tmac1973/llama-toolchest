package huggingface

import "testing"

func TestProviderFileURL(t *testing.T) {
	hf := Provider{URL: hfDownloadURL, RevisionURL: hfRevisionURL}
	if got, want := hf.fileURL("org/repo", "", "a/b.gguf"), "https://huggingface.co/org/repo/resolve/main/a/b.gguf"; got != want {
		t.Errorf("no revision: %s, want %s", got, want)
	}
	if got, want := hf.fileURL("org/repo", "abc123", "a/b.gguf"), "https://huggingface.co/org/repo/resolve/abc123/a/b.gguf"; got != want {
		t.Errorf("revision: %s, want %s", got, want)
	}
	// A source that cannot name a revision uses its default branch.
	other := Provider{URL: func(m, f string) string { return "x/" + m + "/" + f }}
	if got := other.fileURL("org/repo", "abc123", "f"); got != "x/org/repo/f" {
		t.Errorf("no RevisionURL: %s", got)
	}
}
