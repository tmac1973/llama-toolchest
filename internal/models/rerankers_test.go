package models

import (
	"strings"
	"testing"
)

func TestCuratedRerankModelsComplete(t *testing.T) {
	repos := map[string]bool{}
	for _, p := range CuratedRerankModels() {
		if p.Name == "" || p.Repo == "" || p.Filename == "" || p.SizeMB <= 0 || p.Context <= 0 ||
			p.License == "" || p.Desc == "" {
			t.Errorf("curated %q has an empty field: %+v", p.Name, p)
		}
		if !strings.HasSuffix(strings.ToLower(p.Filename), ".gguf") {
			t.Errorf("curated %s: filename %q is not a GGUF file", p.Name, p.Filename)
		}
		if repos[p.Repo] {
			t.Errorf("curated repo %s is listed twice", p.Repo)
		}
		repos[p.Repo] = true
		// A download is classified by its name until the file is read.
		if KindFromName(p.Repo) != KindRerank {
			t.Errorf("curated %s: repo %q does not look like a reranker by name", p.Name, p.Repo)
		}
	}
}

func TestRerankPresetSizeLabel(t *testing.T) {
	if got := (RerankModelPreset{SizeMB: 305}).SizeLabel(); got != "305 MB" {
		t.Errorf("305 MB: got %q", got)
	}
	if got := (RerankModelPreset{SizeMB: 4280}).SizeLabel(); got != "4.3 GB" {
		t.Errorf("4280 MB: got %q", got)
	}
}
