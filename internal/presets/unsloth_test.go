package presets

import (
	"testing"
)

const qwen38PageMD = `# Qwen 3.8: How to Run Locally

Qwen 3.8 is the latest release.

### Recommended Settings

#### Qwen3.8-**27B Settings:**

Deeper sub-headings must stay inside the section (the real page has them).

| Parameter | Thinking Mode | Instruct (non-thinking) Mode |
|---|---|---|
| ` + "`temperature`" + ` | 1.0 | 0.7 |
| ` + "`top_p`" + ` | 0.95 | 0.80 |
| ` + "`top_k`" + ` | 20 | 20 |
| ` + "`min_p`" + ` | 0.0 | 0.0 |
| ` + "`presence_penalty`" + ` | 0.0 | 1.5 |

### Running with llama.cpp

Some other content with numbers like 4096 that must not be parsed.
`

const singleColumnMD = `# Some Model

## Recommended Inference Settings

| Parameter | Value |
|---|---|
| temperature | 0.8 |
| top_p | 0.9 |
`

const inlineOnlyMD = `# Another Model

### Recommended Settings

We suggest ` + "`temperature`" + ` = 0.6 and ` + "`min_p`" + ` = 0.01 for this model.

## Next section

temperature = 99 should not be picked up here.
`

func TestParseRecommendedSettingsTable(t *testing.T) {
	presets := parseRecommendedSettings(qwen38PageMD, "https://unsloth.ai/docs/models/qwen3.8")
	if len(presets) != 2 {
		t.Fatalf("got %d presets, want 2: %+v", len(presets), presets)
	}
	byName := map[string]int{}
	for i, p := range presets {
		byName[p.Name] = i
	}
	th, ok1 := byName["thinking"]
	nt, ok2 := byName["non-thinking"]
	if !ok1 || !ok2 {
		t.Fatalf("missing variants: %+v", presets)
	}
	if v := presets[th].Temperature; v == nil || *v != 1.0 {
		t.Errorf("thinking temperature = %v, want 1.0", v)
	}
	if v := presets[nt].Temperature; v == nil || *v != 0.7 {
		t.Errorf("non-thinking temperature = %v, want 0.7", v)
	}
	if v := presets[nt].PresencePenalty; v == nil || *v != 1.5 {
		t.Errorf("non-thinking presence_penalty = %v, want 1.5", v)
	}
	if v := presets[th].MinP; v == nil || *v != 0.0 {
		t.Errorf("thinking min_p = %v, want explicit 0.0", v)
	}
	if v := presets[th].TopK; v == nil || *v != 20 {
		t.Errorf("thinking top_k = %v, want 20", v)
	}
	if presets[th].Source != "unsloth-docs" || presets[th].SourceURL != "https://unsloth.ai/docs/models/qwen3.8" {
		t.Errorf("bad provenance: %+v", presets[th])
	}
}

func TestParseRecommendedSettingsSingleColumn(t *testing.T) {
	presets := parseRecommendedSettings(singleColumnMD, "u")
	if len(presets) != 1 || presets[0].Name != "default" {
		t.Fatalf("got %+v, want one default preset", presets)
	}
	if v := presets[0].Temperature; v == nil || *v != 0.8 {
		t.Errorf("temperature = %v, want 0.8", v)
	}
}

func TestParseRecommendedSettingsInlineFallback(t *testing.T) {
	presets := parseRecommendedSettings(inlineOnlyMD, "u")
	if len(presets) != 1 {
		t.Fatalf("got %+v, want one preset", presets)
	}
	if v := presets[0].Temperature; v == nil || *v != 0.6 {
		t.Errorf("temperature = %v, want 0.6 (and never 99 from the next section)", v)
	}
	if v := presets[0].MinP; v == nil || *v != 0.01 {
		t.Errorf("min_p = %v, want 0.01", v)
	}
}

func TestParseRecommendedSettingsNoSection(t *testing.T) {
	if got := parseRecommendedSettings("# Page\n\nNo settings here. temperature = 0.5\n", "u"); got != nil {
		t.Errorf("expected nil without a Recommended Settings section, got %+v", got)
	}
}

func TestFamilyKey(t *testing.T) {
	cases := map[string]string{
		"Qwen3.8-27B-GGUF":                 "qwen3.8",
		"Qwen 3.8":                         "qwen3.8",
		"qwen3.8":                          "qwen3.8",
		"Qwen3-30B-A3B-Instruct-2507-GGUF": "qwen3",
		"DeepSeek-R1-0528-GGUF":            "deepseekr1",
		"Llama-4-Scout-17B-16E-Instruct":   "llama4scout",
		"gemma-3-27b-it-GGUF":              "gemma3",
	}
	for in, want := range cases {
		if got := familyKey(in); got != want {
			t.Errorf("familyKey(%q) = %q, want %q", in, got, want)
		}
	}
}

const llmsTxt = `# Unsloth Documentation

- [Qwen 3.8: How to Run Locally](https://unsloth.ai/docs/models/qwen3.8.md): Run Qwen 3.8
- [Qwen3: How to Run & Fine-tune](https://unsloth.ai/docs/models/tutorials/qwen3-how-to-run-and-fine-tune.md): Run Qwen3
- [DeepSeek-R1-0528: How to Run Locally](https://unsloth.ai/docs/models/tutorials/deepseek-r1-0528-how-to-run-locally.md): desc
- [Llama 4: How to Run & Fine-tune](https://unsloth.ai/docs/models/tutorials/llama-4-how-to-run-and-fine-tune.md): desc
- [Fine-tuning Guide](https://unsloth.ai/docs/get-started/fine-tuning-guide.md): not a model page
`

func TestMatchDocsPage(t *testing.T) {
	base := "https://unsloth.ai/docs"
	cases := map[string]string{
		// Exact family match — and crucially qwen3.8 must NOT match the qwen3 page.
		"unsloth/Qwen3.8-27B-GGUF":      "https://unsloth.ai/docs/models/qwen3.8",
		"unsloth/Qwen3-32B-GGUF":        "https://unsloth.ai/docs/models/tutorials/qwen3-how-to-run-and-fine-tune",
		"unsloth/DeepSeek-R1-0528-GGUF": "https://unsloth.ai/docs/models/tutorials/deepseek-r1-0528-how-to-run-locally",
		// Letter-boundary prefix match: llama4scout → llama4 page.
		"unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF": "https://unsloth.ai/docs/models/tutorials/llama-4-how-to-run-and-fine-tune",
		// No match at all.
		"someone/TotallyUnknown-7B-GGUF": "",
	}
	for repo, want := range cases {
		if got := matchDocsPage(llmsTxt, base, repo); got != want {
			t.Errorf("matchDocsPage(%q) = %q, want %q", repo, got, want)
		}
	}
}
