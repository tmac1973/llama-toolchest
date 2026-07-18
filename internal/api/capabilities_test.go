package api

import (
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/models"
)

func ptrF(v float64) *float64 { return &v }
func ptrI(v int) *int         { return &v }

// TestBuildSamplingOverrideWins verifies that a per-model config override takes
// precedence over the card recommendation, and that unset params fall through.
func TestBuildSamplingOverrideWins(t *testing.T) {
	m := &models.Model{ModelID: "does/not-exist-GGUF"} // no presets
	cfg := &models.ModelConfig{Temperature: ptrF(0.42), TopK: ptrI(7)}

	s := buildSampling(m, cfg)
	def := s["default"].(map[string]any)

	if def["temperature"] != 0.42 {
		t.Errorf("temperature = %v, want 0.42", def["temperature"])
	}
	if def["top_k"] != 7 {
		t.Errorf("top_k = %v, want 7", def["top_k"])
	}
	// Unset params with no card recommendation must be explicit null, not absent.
	if v, ok := def["top_p"]; !ok || v != nil {
		t.Errorf("top_p = %v (present=%v), want explicit nil", v, ok)
	}
	// presets is always an array, never nil, so clients can iterate safely.
	if _, ok := s["presets"].([]models.SamplingPreset); !ok {
		t.Errorf("presets should be a []SamplingPreset, got %T", s["presets"])
	}
}

// TestBuildSamplingFallsBackToCard verifies the card recommendation fills in
// params the user didn't override, and provenance is surfaced.
func TestBuildSamplingFallsBackToCard(t *testing.T) {
	repo := pickRepoWithPreset(t)
	m := &models.Model{ModelID: repo}

	s := buildSampling(m, nil)
	def := s["default"].(map[string]any)

	presets := models.LookupSamplingPresets(repo)
	if len(presets) == 0 {
		t.Fatalf("expected presets for %s", repo)
	}
	card := presets[0]
	if card.Temperature != nil && def["temperature"] != *card.Temperature {
		t.Errorf("temperature = %v, want card value %v", def["temperature"], *card.Temperature)
	}
	if s["source"] != card.Source {
		t.Errorf("source = %v, want %v", s["source"], card.Source)
	}
}

// pickRepoWithPreset returns any repo id that has a sampling preset with a
// temperature, so the fallback assertion has something to check.
func pickRepoWithPreset(t *testing.T) string {
	t.Helper()
	candidates := []string{
		"unsloth/DeepSeek-R1-0528-GGUF",
		"unsloth/Apertus-8B-Instruct-2509-GGUF",
	}
	for _, c := range candidates {
		ps := models.LookupSamplingPresets(c)
		if len(ps) > 0 && ps[0].Temperature != nil {
			return c
		}
	}
	t.Skip("no known repo with a temperature preset in this build")
	return ""
}
