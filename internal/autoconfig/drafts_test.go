package autoconfig

import (
	"context"
	"errors"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/modelsource"
)

func TestFamilyAndSize(t *testing.T) {
	cases := []struct {
		name   string
		family string
		params float64
		ok     bool
	}{
		{"Qwen3.5-9B-Instruct", "Qwen3.5", 9e9, true},
		{"gemma-4-27b-it", "gemma-4", 27e9, true},
		{"Llama-3.3-70B-Instruct", "Llama-3.3", 70e9, true},
		{"Qwen3.5-0.8B", "Qwen3.5", 0.8e9, true},
		{"SmolLM-360M", "SmolLM", 360e6, true},
		{"Mistral-Nemo-Instruct", "", 0, false},
	}
	for _, c := range cases {
		fam, p, ok := familyAndSize(c.name)
		if fam != c.family || p != c.params || ok != c.ok {
			t.Errorf("familyAndSize(%q) = %q, %g, %v", c.name, fam, p, ok)
		}
	}
}

type fakeHub struct {
	search []modelsource.SearchResult
	repos  map[string][]modelsource.File
}

func (h fakeHub) Search(context.Context, string) ([]modelsource.SearchResult, error) {
	return h.search, nil
}

func (h fakeHub) GetModel(_ context.Context, repo string) (*modelsource.Detail, error) {
	files, ok := h.repos[repo]
	if !ok {
		return nil, errors.New("404")
	}
	return &modelsource.Detail{ID: repo, Files: files}, nil
}

func TestDraftSuggestionFromNamedRepo(t *testing.T) {
	hub := fakeHub{repos: map[string][]modelsource.File{
		"o/draft-GGUF": {{Filename: "draft-Q4_K_M.gguf", Quant: "Q4_K_M", Size: 1}, {Filename: "draft-Q8_0.gguf", Quant: "Q8_0", Size: 2}},
	}}
	got := FindDraftSuggestions(context.Background(), hub, &models.Model{ModelID: "o/m-GGUF"},
		Checked{WantDraft: "draft", DraftRepo: "o/draft-GGUF"}, nil)
	if len(got) != 1 || got[0].Filename != "draft-Q8_0.gguf" {
		t.Errorf("suggestions = %+v, want the Q8_0 file", got)
	}
}

func TestDraftSuggestionMTPHeadInSameRepo(t *testing.T) {
	hub := fakeHub{repos: map[string][]modelsource.File{
		"o/m-GGUF": {{Filename: "m-Q4_K_M.gguf", Quant: "Q4_K_M"}, {Filename: "m-MTP-Q8_0.gguf", Quant: "Q8_0"}},
	}}
	got := FindDraftSuggestions(context.Background(), hub, &models.Model{ModelID: "o/m-GGUF"}, Checked{WantDraft: "draft-mtp"}, nil)
	if len(got) != 1 || got[0].Filename != "m-MTP-Q8_0.gguf" {
		t.Errorf("suggestions = %+v", got)
	}
	installed := func(repo, file string) bool { return file == "m-MTP-Q8_0.gguf" }
	if got := FindDraftSuggestions(context.Background(), hub, &models.Model{ModelID: "o/m-GGUF"}, Checked{WantDraft: "draft-mtp"}, installed); len(got) != 0 {
		t.Errorf("an installed file was suggested: %+v", got)
	}
}

func TestDraftSuggestionFamilySearch(t *testing.T) {
	hub := fakeHub{
		search: []modelsource.SearchResult{
			{ID: "unsloth/Qwen3.5-9B-GGUF"},
			{ID: "unsloth/Qwen3.5-4B-GGUF"},      // not below a quarter of 9B
			{ID: "someone/Qwen3.5-0.5B-GGUF"},    // another publisher
			{ID: "unsloth/Qwen3.5-0.8B-GGUF"},    // the one
			{ID: "unsloth/Qwen3.5-VL-0.8B-GGUF"}, // another family
		},
		repos: map[string][]modelsource.File{
			"unsloth/Qwen3.5-0.8B-GGUF": {{Filename: "Qwen3.5-0.8B-Q8_0.gguf", Quant: "Q8_0", Size: 900}},
		},
	}
	m := &models.Model{ModelID: "unsloth/Qwen3.5-9B-GGUF", BaseModelRepo: "Qwen/Qwen3.5-9B"}
	got := FindDraftSuggestions(context.Background(), hub, m, Checked{WantDraft: "draft"}, nil)
	if len(got) != 1 || got[0].Repo != "unsloth/Qwen3.5-0.8B-GGUF" {
		t.Errorf("suggestions = %+v", got)
	}
	if got := FindDraftSuggestions(context.Background(), hub, m, Checked{}, nil); got != nil {
		t.Error("suggestions without a recommended draft method")
	}
}
