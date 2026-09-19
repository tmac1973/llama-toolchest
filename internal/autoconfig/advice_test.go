package autoconfig

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/llmcall"
	"github.com/tmac1973/llama-toolchest/internal/models"
)

func f64(v float64) *float64 { return &v }
func ip(v int) *int          { return &v }
func sp(v string) *string    { return &v }

func proposal(c Checked, field string) (Proposal, bool) {
	for _, p := range c.Proposals {
		if p.Field == field {
			return p, true
		}
	}
	return Proposal{}, false
}

func hasNote(c Checked, text string) bool {
	for _, n := range c.Notes {
		if strings.Contains(n.Reason, text) {
			return true
		}
	}
	return false
}

func baseInputs() Inputs {
	return Inputs{
		Model: &models.Model{ModelID: "o/m-GGUF", ContextLength: 131072},
		Class: models.ContextMedium,
		Fit:   models.FitResult{Config: models.ModelConfig{ContextSize: 32768}},
	}
}

func TestValidateSamplingFromCard(t *testing.T) {
	adv := Advice{Temperature: f64(0.6), TopP: f64(0.95), TopK: ip(20), MinP: f64(3), SamplingQuote: "use temperature=0.6"}
	c := Validate(adv, baseInputs())
	if p, ok := proposal(c, "temperature"); !ok || p.Value != 0.6 || p.Origin != "model card" || !strings.Contains(p.Reason, "temperature=0.6") {
		t.Errorf("temperature proposal = %+v, %v", p, ok)
	}
	if _, ok := proposal(c, "min_p"); ok {
		t.Error("an out-of-range min_p was proposed")
	}
	if !hasNote(c, "min_p of 3 is outside") {
		t.Error("the dropped min_p is not explained")
	}
}

func TestValidatePublisherPresetWins(t *testing.T) {
	in := baseInputs()
	in.Model.SamplingPresets = []models.SamplingPreset{{Name: "default", Label: "Default", Source: "generation_config.json", Temperature: f64(1.0), TopK: ip(40)}}
	c := Validate(Advice{Temperature: f64(0.6)}, in)
	if p, _ := proposal(c, "temperature"); p.Value != 1.0 || p.Origin != "publisher preset" {
		t.Errorf("temperature = %+v, want the publisher's 1.0", p)
	}
	if p, _ := proposal(c, "sampling_preset"); p.Value != "default" {
		t.Errorf("sampling_preset = %v", p.Value)
	}
	if !hasNote(c, "different sampling values") {
		t.Error("the disagreement with the card text is not mentioned")
	}
}

func TestValidateSpeculative(t *testing.T) {
	// MTP recommended, and the model has built-in MTP layers.
	in := baseInputs()
	in.Model.NextNLayers = 1
	c := Validate(Advice{DraftMethod: sp("draft-mtp"), SpeculativeQuote: "supports MTP"}, in)
	if p, ok := proposal(c, "spec_type"); !ok || p.Value != "draft-mtp" {
		t.Errorf("built-in MTP not proposed: %+v", c.Proposals)
	}

	// MTP recommended, but nothing on this machine can serve it.
	in = baseInputs()
	c = Validate(Advice{DraftMethod: sp("draft-mtp"), DraftRepo: sp("o/m-MTP-GGUF")}, in)
	if _, ok := proposal(c, "spec_type"); ok || c.WantDraft != "draft-mtp" || c.DraftRepo != "o/m-MTP-GGUF" {
		t.Errorf("missing MTP head should become a suggestion: %+v", c)
	}

	// A draft model recommended and one installed.
	in = baseInputs()
	in.DraftCandidates = func(mode string) []models.DraftCandidate {
		return []models.DraftCandidate{{ID: "small", Filename: "small-Q8_0.gguf", FilePath: "/models/small-Q8_0.gguf"}}
	}
	c = Validate(Advice{DraftMethod: sp("draft")}, in)
	if p, _ := proposal(c, "draft_model_path"); p.Value != "/models/small-Q8_0.gguf" {
		t.Errorf("installed draft model not used: %+v", c.Proposals)
	}

	// An n-gram assist is never set here, only mentioned.
	c = Validate(Advice{AssistMode: sp("ngram-mod")}, baseInputs())
	if _, ok := proposal(c, "spec_assist"); ok || !hasNote(c, "Autotune measures") {
		t.Errorf("assist advice = %+v", c)
	}
}

func TestValidateContextOnlyLowersUnderMaximum(t *testing.T) {
	c := Validate(Advice{RecommendedContext: ip(16384)}, baseInputs())
	if _, ok := proposal(c, "context_size"); ok || !hasNote(c, "size you chose is kept") {
		t.Errorf("medium class: context changed or not explained: %+v", c)
	}
	in := baseInputs()
	in.Class = models.ContextMax
	in.Fit.Config.ContextSize = 131072
	c = Validate(Advice{RecommendedContext: ip(32768)}, in)
	if p, ok := proposal(c, "context_size"); !ok || p.Value != 32768 {
		t.Errorf("maximum class: recommended context not proposed: %+v", c)
	}
	c = Validate(Advice{RecommendedContext: ip(1 << 30)}, in)
	if _, ok := proposal(c, "context_size"); ok {
		t.Error("an out-of-range context was proposed")
	}
}

func TestValidateOtherNotesNeverBecomeSettings(t *testing.T) {
	c := Validate(Advice{OtherNotes: []string{"Use --override-kv foo=bar for best results."}}, baseInputs())
	if len(c.Proposals) != 0 || !hasNote(c, "also mentions: Use --override-kv") {
		t.Errorf("other notes = %+v", c)
	}
}

func TestApplyProposalSetsDraftDefaults(t *testing.T) {
	var cfg models.ModelConfig
	ApplyProposal(&cfg, Proposal{Field: "spec_type", Value: "draft-mtp"})
	ApplyProposal(&cfg, Proposal{Field: "temperature", Value: 0.7})
	ApplyProposal(&cfg, Proposal{Field: "extra_flags", Value: "--danger"})
	if cfg.SpecType != "draft-mtp" || cfg.DraftMax == 0 || cfg.Temperature == nil || *cfg.Temperature != 0.7 {
		t.Errorf("config = %+v", cfg)
	}
	if cfg.ExtraFlags != "" {
		t.Error("a field outside the proposal set was applied")
	}
}

type backend struct{ url string }

func (b backend) Busy() string      { return "" }
func (b backend) RouterURL() string { return b.url }
func (b backend) Prepare(context.Context, string) (llmcall.Target, error) {
	return llmcall.Target{RouterName: "helper"}, nil
}

func TestAskDecodesTheForm(t *testing.T) {
	var sent map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&sent)
		answer := `{"temperature": 0.6, "top_p": null, "top_k": 20, "min_p": null, "presence_penalty": null,
			"repeat_penalty": null, "sampling_quote": "temperature=0.6", "thinking": null, "thinking_quote": "",
			"draft_method": "draft-mtp", "draft_repo": null, "assist_mode": null, "speculative_quote": "MTP",
			"recommended_context": null, "context_quote": "", "other_notes": []}`
		json.NewEncoder(w).Encode(map[string]any{"choices": []map[string]any{{"message": map[string]any{"content": answer}, "finish_reason": "stop"}}})
	}))
	defer srv.Close()
	llm := &llmcall.Client{Backend: backend{srv.URL}}
	adv, err := Ask(context.Background(), llm, "h", &models.Model{ModelID: "o/m"}, Card{Text: "card text"})
	if err != nil {
		t.Fatal(err)
	}
	if adv.Temperature == nil || *adv.Temperature != 0.6 || adv.DraftMethod == nil || *adv.DraftMethod != "draft-mtp" {
		t.Errorf("advice = %+v", adv)
	}
	msgs := sent["messages"].([]any)
	if !strings.Contains(msgs[1].(map[string]any)["content"].(string), "card text") {
		t.Error("the card was not sent")
	}

	// No card, no request.
	sent = nil
	if _, err := Ask(context.Background(), llm, "h", &models.Model{}, Card{}); err != nil || sent != nil {
		t.Errorf("an empty card was sent: %v", err)
	}
}
