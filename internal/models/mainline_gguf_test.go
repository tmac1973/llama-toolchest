package models

import (
	"slices"
	"testing"
)

// A publisher's renamed GGUF ("*.gguf_file") is listed as a GGUF and
// saved with ".gguf" again; other names are left as they are.
func TestMainlineGGUFNames(t *testing.T) {
	cases := []struct {
		name     string
		listed   bool
		mainline bool
		local    string
	}{
		{"Model-Q4_K_M.gguf", true, false, "Model-Q4_K_M.gguf"},
		{"Model-Q8_0.GGUF", true, false, "Model-Q8_0.GGUF"},
		{"Mainline_Support/Model-UD-Q4_K_XL.gguf_file", true, true, "Mainline_Support/Model-UD-Q4_K_XL.gguf"},
		{"Mainline_Support/Model.GGUF_FILE", true, true, "Mainline_Support/Model.GGUF"},
		{"README.md", false, false, "README.md"},
		{"notes.gguf_file.txt", false, false, "notes.gguf_file.txt"},
	}
	for _, c := range cases {
		if got := IsGGUFName(c.name); got != c.listed {
			t.Errorf("IsGGUFName(%q) = %v, want %v", c.name, got, c.listed)
		}
		if got := IsMainlineGGUF(c.name); got != c.mainline {
			t.Errorf("IsMainlineGGUF(%q) = %v, want %v", c.name, got, c.mainline)
		}
		if got := LocalGGUFName(c.name); got != c.local {
			t.Errorf("LocalGGUFName(%q) = %q, want %q", c.name, got, c.local)
		}
	}
}

// A split renamed GGUF expands to shards that keep the renamed suffix,
// because the shards are fetched by their names in the repository.
func TestExpandShardsKeepsTheRenamedSuffix(t *testing.T) {
	got := ExpandShards("M/Big-00001-of-00002.gguf_file")
	want := []string{"M/Big-00001-of-00002.gguf_file", "M/Big-00002-of-00002.gguf_file"}
	if !slices.Equal(got, want) {
		t.Errorf("ExpandShards = %v, want %v", got, want)
	}
	if got := ExpandShards("Big-00001-of-00002.gguf"); got[1] != "Big-00002-of-00002.gguf" {
		t.Errorf("plain shards = %v", got)
	}
}
