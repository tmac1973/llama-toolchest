package models

import (
	"strings"
	"testing"
)

func TestCuratedSpeechModelsComplete(t *testing.T) {
	repos := map[string]bool{}
	for _, p := range CuratedSpeechModels() {
		if p.Name == "" || p.Repo == "" || p.Filename == "" || p.MmprojFilename == "" || p.SizeMB <= 0 ||
			p.MinBuild <= 0 || p.VRAMMiB <= 0 || p.Languages == "" || p.License == "" || p.Desc == "" || p.Voice == "" {
			t.Errorf("curated %q has an empty field: %+v", p.Name, p)
		}
		for _, f := range []string{p.Filename, p.MmprojFilename} {
			if !strings.HasSuffix(strings.ToLower(f), ".gguf") {
				t.Errorf("curated %s: filename %q is not a GGUF file", p.Name, f)
			}
		}
		// The projector is told from the model by its name, both when it
		// lands and when the folder is scanned.
		if IsMMProjFile(p.Filename) || !IsMMProjFile(p.MmprojFilename) {
			t.Errorf("curated %s: %q and %q are not a model and a projector by name", p.Name, p.Filename, p.MmprojFilename)
		}
		if repos[p.Repo] {
			t.Errorf("curated repo %s is listed twice", p.Repo)
		}
		repos[p.Repo] = true
	}
}

func TestSpeechPresetLabels(t *testing.T) {
	p := SpeechModelPreset{SizeMB: 2688, VRAMMiB: 3584}
	if got := p.SizeLabel(); got != "2.7 GB" {
		t.Errorf("size: got %q", got)
	}
	if got := p.VRAMLabel(); got != "3.5 GiB" {
		t.Errorf("VRAM: got %q", got)
	}
}
