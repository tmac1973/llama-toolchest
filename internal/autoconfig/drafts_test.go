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

// A card that names its head without an owner ("we also release
// MiniCPM-V-4.7-35B-A3B-DSpark") means the publisher's repository, and its
// GGUF conversion is the one that can be downloaded.
func TestDraftSuggestionCompletesAnOwnerlessHeadName(t *testing.T) {
	m := &models.Model{ModelID: "quantizer/Model-35B-GGUF", BaseModelRepo: "pub/Model-35B"}
	hub := fakeHub{repos: map[string][]modelsource.File{
		"pub/Model-35B-DSpark":      {}, // the original weights: no GGUF
		"pub/Model-35B-DSpark-gguf": {{Filename: "dspark-Model-Q4_K_M.gguf", Quant: "Q4_K_M"}, {Filename: "dspark-Model-F16.gguf", Quant: "F16"}},
	}}
	for _, named := range []string{"Model-35B-DSpark", "pub/Model-35B-DSpark"} {
		got := FindDraftSuggestions(context.Background(), hub, m, Checked{WantDraft: "draft-dspark", DraftRepo: named}, nil)
		if len(got) != 1 || got[0].Repo != "pub/Model-35B-DSpark-gguf" || got[0].Filename != "dspark-Model-F16.gguf" {
			t.Errorf("named %q: suggestions = %+v, want the publisher's GGUF head at full precision over 4 bits", named, got)
		}
	}
}

// Without a name on the card, a head is found by searching for the
// model's name and the method's. Only repositories named for both count,
// and the publisher's own comes before anyone else's.
func TestDraftSuggestionSearchesForAHeadByName(t *testing.T) {
	m := &models.Model{ModelID: "quantizer/Model-35B-GGUF", BaseModelRepo: "pub/Model-35B"}
	hub := fakeHub{
		search: []modelsource.SearchResult{
			{ID: "other/Other-9B-DSpark-GGUF"},    // another model's head
			{ID: "someone/Model-35B-DFlash-GGUF"}, // this model, another method
			{ID: "someone/Model-35B-DSpark-GGUF"}, // more downloads, not the publisher
			{ID: "pub/Model-35B-DSpark-gguf"},
		},
		repos: map[string][]modelsource.File{
			"other/Other-9B-DSpark-GGUF":    {{Filename: "o-Q8_0.gguf", Quant: "Q8_0"}},
			"someone/Model-35B-DFlash-GGUF": {{Filename: "f-Q8_0.gguf", Quant: "Q8_0"}},
			"someone/Model-35B-DSpark-GGUF": {{Filename: "s-Q8_0.gguf", Quant: "Q8_0"}},
			"pub/Model-35B-DSpark-gguf":     {{Filename: "p-Q8_0.gguf", Quant: "Q8_0"}},
		},
	}
	got := FindDraftSuggestions(context.Background(), hub, m, Checked{WantDraft: "draft-dspark"}, nil)
	if len(got) != 1 || got[0].Repo != "pub/Model-35B-DSpark-gguf" || got[0].Mode != "draft-dspark" {
		t.Errorf("suggestions = %+v, want the publisher's DSpark head", got)
	}

	// Without the publisher's, the next one named for this model and method.
	delete(hub.repos, "pub/Model-35B-DSpark-gguf")
	got = FindDraftSuggestions(context.Background(), hub, m, Checked{WantDraft: "draft-dspark"}, nil)
	if len(got) != 1 || got[0].Repo != "someone/Model-35B-DSpark-GGUF" {
		t.Errorf("suggestions = %+v, want the other DSpark head for this model", got)
	}
}
