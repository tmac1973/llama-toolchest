package presets

import (
	"context"
	"encoding/json"
	"strings"

	"github.com/tmac1973/llama-toolchest/internal/models"
)

// resolveBaseRepo finds the upstream model a GGUF quant repo derives from via
// the repo's HF tags, for quants old enough to lack general.base_model.* in
// the GGUF header. Plain "base_model:org/repo" tags are preferred over
// "base_model:quantized:org/repo" (the former names the true upstream).
func (f *Fetcher) resolveBaseRepo(ctx context.Context, repoID string) string {
	body, err := f.get(ctx, f.hfBase()+"/api/models/"+repoID, true)
	if err != nil {
		debugMiss("base-repo", repoID, err)
		return ""
	}
	var info struct {
		Tags []string `json:"tags"`
	}
	if err := json.Unmarshal(body, &info); err != nil {
		return ""
	}
	quantized := ""
	for _, tag := range info.Tags {
		rest, ok := strings.CutPrefix(tag, "base_model:")
		if !ok {
			continue
		}
		if q, ok := strings.CutPrefix(rest, "quantized:"); ok {
			if quantized == "" {
				quantized = q
			}
			continue
		}
		if strings.Count(rest, "/") == 1 {
			return rest
		}
	}
	return quantized
}

// fetchGenerationConfig pulls the base model's generation_config.json —
// authoritative but sparse (vendors rarely set min_p, and some publish only
// token IDs, which counts as a miss). Gated repos (meta-llama, google) return
// 401/403 without an accepted license; the unsloth mirror of the same repo
// name is tried before giving up.
func (f *Fetcher) fetchGenerationConfig(ctx context.Context, baseRepo string) *models.SamplingPreset {
	url := f.hfBase() + "/" + baseRepo + "/raw/main/generation_config.json"
	body, err := f.get(ctx, url, true)
	if err != nil && !strings.HasPrefix(baseRepo, "unsloth/") {
		mirror := "unsloth/" + repoName(baseRepo)
		url = f.hfBase() + "/" + mirror + "/raw/main/generation_config.json"
		body, err = f.get(ctx, url, true)
	}
	if err != nil {
		debugMiss("generation_config", baseRepo, err)
		return nil
	}

	var gc struct {
		Temperature       *float64 `json:"temperature"`
		TopP              *float64 `json:"top_p"`
		TopK              *int     `json:"top_k"`
		MinP              *float64 `json:"min_p"`
		RepetitionPenalty *float64 `json:"repetition_penalty"`
	}
	if err := json.Unmarshal(body, &gc); err != nil {
		return nil
	}
	p := models.SamplingPreset{
		Name:        "default",
		Label:       "Upstream default",
		Description: "From " + baseRepo + "/generation_config.json",
		Source:      "generation_config.json",
		SourceURL:   url,
	}
	setIfValid(&p, "temperature", gc.Temperature)
	setIfValid(&p, "top_p", gc.TopP)
	setIfValid(&p, "min_p", gc.MinP)
	setIfValid(&p, "repeat_penalty", gc.RepetitionPenalty)
	if gc.TopK != nil {
		assignParam(&p, "top_k", float64(*gc.TopK))
	}
	if emptyPreset(p) {
		return nil
	}
	return &p
}

// fetchParamsFile reads the Ollama-style params JSON that pre-2026 unsloth
// GGUF repos ship — the richest single file when present, since it carries
// the publisher's own min_p / repeat_penalty recommendations.
func (f *Fetcher) fetchParamsFile(ctx context.Context, repoID string) *models.SamplingPreset {
	url := f.hfBase() + "/" + repoID + "/raw/main/params"
	body, err := f.get(ctx, url, true)
	if err != nil {
		return nil
	}
	var pf struct {
		Temperature   *float64 `json:"temperature"`
		TopP          *float64 `json:"top_p"`
		TopK          *float64 `json:"top_k"` // sometimes written as 20.0
		MinP          *float64 `json:"min_p"`
		RepeatPenalty *float64 `json:"repeat_penalty"`
	}
	if err := json.Unmarshal(body, &pf); err != nil {
		return nil
	}
	p := models.SamplingPreset{
		Name:        "default",
		Label:       "Publisher recommendation",
		Description: "From the repo's params file",
		Source:      "params",
		SourceURL:   url,
	}
	setIfValid(&p, "temperature", pf.Temperature)
	setIfValid(&p, "top_p", pf.TopP)
	setIfValid(&p, "min_p", pf.MinP)
	setIfValid(&p, "repeat_penalty", pf.RepeatPenalty)
	if pf.TopK != nil {
		assignParam(&p, "top_k", *pf.TopK)
	}
	if emptyPreset(p) {
		return nil
	}
	return &p
}

func setIfValid(p *models.SamplingPreset, param string, v *float64) {
	if v != nil {
		assignParam(p, param, *v)
	}
}
