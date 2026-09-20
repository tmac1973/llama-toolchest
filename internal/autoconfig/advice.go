package autoconfig

import (
	"context"
	"fmt"
	"sort"
	"strings"

	"github.com/tmac1973/llama-toolchest/internal/llmcall"
	"github.com/tmac1973/llama-toolchest/internal/models"
)

// Advice is what the helper model reads out of a model card. Every value
// is nullable: null means the card does not say. Each group carries the
// card's own words (a quote), so a reviewer can check the value against
// the card rather than trust the helper.
//
// The fields are deliberately flat and few. A small model fills a short
// form reliably; nested optional objects it fills less reliably.
type Advice struct {
	Temperature     *float64 `json:"temperature"`
	TopP            *float64 `json:"top_p"`
	TopK            *int     `json:"top_k"`
	MinP            *float64 `json:"min_p"`
	PresencePenalty *float64 `json:"presence_penalty"`
	RepeatPenalty   *float64 `json:"repeat_penalty"`
	SamplingQuote   string   `json:"sampling_quote"`

	// Thinking is "on", "off" or "model default".
	Thinking      *string `json:"thinking"`
	ThinkingQuote string  `json:"thinking_quote"`

	// DraftMethod is one of models.DraftModes() or "none"; DraftRepo is a
	// Hugging Face repository the card names for a draft model or head.
	DraftMethod      *string `json:"draft_method"`
	DraftRepo        *string `json:"draft_repo"`
	AssistMode       *string `json:"assist_mode"`
	SpeculativeQuote string  `json:"speculative_quote"`

	RecommendedContext *int   `json:"recommended_context"`
	ContextQuote       string `json:"context_quote"`

	// OtherNotes are anything else the card says about running the
	// model. They are shown to the user and never turned into settings.
	OtherNotes []string `json:"other_notes"`
}

// adviceSchema is the JSON Schema the helper's answer is constrained to.
func adviceSchema() map[string]any {
	num := map[string]any{"type": []string{"number", "null"}}
	integer := map[string]any{"type": []string{"integer", "null"}}
	str := map[string]any{"type": "string"}
	enumOrNull := func(values []string) map[string]any {
		vals := make([]any, 0, len(values)+1)
		for _, v := range values {
			vals = append(vals, v)
		}
		vals = append(vals, nil)
		return map[string]any{"enum": vals}
	}
	var draftModes, assistModes []string
	for _, m := range models.DraftModes() {
		draftModes = append(draftModes, m.Name)
	}
	draftModes = append(draftModes, "none")
	for _, m := range models.AssistModes() {
		assistModes = append(assistModes, m.Name)
	}
	assistModes = append(assistModes, "none")

	props := map[string]any{
		"temperature":         num,
		"top_p":               num,
		"top_k":               integer,
		"min_p":               num,
		"presence_penalty":    num,
		"repeat_penalty":      num,
		"sampling_quote":      str,
		"thinking":            enumOrNull([]string{"on", "off", "model default"}),
		"thinking_quote":      str,
		"draft_method":        enumOrNull(draftModes),
		"draft_repo":          map[string]any{"type": []string{"string", "null"}},
		"assist_mode":         enumOrNull(assistModes),
		"speculative_quote":   str,
		"recommended_context": integer,
		"context_quote":       str,
		"other_notes":         map[string]any{"type": "array", "items": str, "maxItems": 5},
	}
	required := make([]string, 0, len(props))
	for k := range props {
		required = append(required, k)
	}
	sort.Strings(required) // the same request every time
	return map[string]any{
		"type":                 "object",
		"properties":           props,
		"required":             required,
		"additionalProperties": false,
	}
}

const adviceInstructions = `You read the documentation of a language model (its model card) and fill in a form about how to run it with llama.cpp.

Rules:
- Use only what the model card says. Do not use what you know about other models, and do not guess.
- When the card does not state a value, answer null for it, and an empty string for its quote.
- For every value you fill in, copy into the matching quote field the sentence from the card that states it.
- Sampling values: when the card gives more than one set (for example one for thinking and one for non-thinking use), use the general or default set.
- draft_method: "draft-mtp" when the card says the model has multi-token prediction (MTP) layers or an MTP head for speculative decoding; "draft" when it recommends a separate smaller draft model; "draft-eagle3", "draft-dflash" or "draft-dspark" when it names that kind of head; otherwise null.
- draft_repo: the Hugging Face repository (owner/name) the card names for that draft model or head, or null.
- recommended_context: a context length the card recommends running with, in tokens, or null.
- other_notes: at most five short sentences about anything else the card says about running this model with llama.cpp. Leave out instructions for other servers, such as vLLM, SGLang or TGI, and anything about training or fine-tuning. Leave it empty when there is nothing.`

// Ask has the helper model with registry ID helperID read card and fill
// in the advice form. An empty card is not sent: there is nothing to read.
func Ask(ctx context.Context, llm *llmcall.Client, helperID string, m *models.Model, card Card) (Advice, error) {
	var adv Advice
	if strings.TrimSpace(card.Text) == "" {
		return adv, nil
	}
	user := fmt.Sprintf("Model repository: %s\nModel file: %s\n\nModel card:\n<<<\n%s\n>>>", m.ModelID, m.Filename, card.Text)
	err := llm.JSON(ctx, helperID, "model_card_advice", adviceSchema(), []llmcall.Message{
		{Role: "system", Content: adviceInstructions},
		{Role: "user", Content: user},
	}, &adv)
	return adv, err
}

// Proposal is one setting autoconfigure proposes, with where it came from.
type Proposal struct {
	Field  string // ModelConfig JSON key
	Value  any    // float64, int or string, as ApplyProposal expects for Field
	Reason string
	Origin string // "model card", "publisher preset", "model file", "hardware fit"
}

// ApplyProposal writes one proposal into cfg. Unknown fields are ignored:
// only the fields Validate produces can be applied, which is what keeps
// the helper's answer from reaching anything else.
func ApplyProposal(cfg *models.ModelConfig, p Proposal) {
	f := func() *float64 {
		v, ok := p.Value.(float64)
		if !ok {
			return nil
		}
		return &v
	}
	switch p.Field {
	case "temperature":
		cfg.Temperature = f()
	case "top_p":
		cfg.TopP = f()
	case "min_p":
		cfg.MinP = f()
	case "presence_penalty":
		cfg.PresencePenalty = f()
	case "repeat_penalty":
		cfg.RepeatPenalty = f()
	case "top_k":
		if v, ok := p.Value.(int); ok {
			cfg.TopK = &v
		}
	case "sampling_preset":
		cfg.SamplingPreset, _ = p.Value.(string)
	case "spec_type":
		mode, _ := p.Value.(string)
		if mode != cfg.SpecType {
			cfg.SpecType = mode
			cfg.ApplyDraftDefaults()
		}
	case "draft_model_path":
		cfg.DraftModelPath, _ = p.Value.(string)
	}
}

// Inputs is what Validate checks the helper's answer against.
type Inputs struct {
	Model *models.Model
	Base  models.ModelConfig // the live config
	Class models.ContextClass
	Fit   models.FitResult
	// DraftCandidates lists installed files usable by a draft method.
	DraftCandidates func(mode string) []models.DraftCandidate
}

// Checked is the helper's answer after checking: the settings to propose,
// notes for the review screen, and draft files worth downloading.
type Checked struct {
	Proposals []Proposal
	Notes     []models.ProfileNote
	// WantDraft is set when the card recommends a draft method whose file
	// is not installed; DraftRepo is the repository it names, if any.
	WantDraft string
	DraftRepo string
}

func (c *Checked) note(field, origin, reason string) {
	c.Notes = append(c.Notes, models.ProfileNote{Field: field, Reason: reason, Origin: origin})
}

// quoted renders a reason with the card's own words.
func quoted(text, quote string) string {
	quote = strings.TrimSpace(quote)
	if quote == "" {
		return text
	}
	if len(quote) > 240 {
		quote = quote[:240] + "…"
	}
	return fmt.Sprintf("%s The model card says: “%s”", text, quote)
}

// Validate turns the helper's answer into proposals, keeping only values
// that are in range and usable on this machine. Everything else becomes a
// note, never a setting.
func Validate(adv Advice, in Inputs) Checked {
	var out Checked
	m := in.Model

	// Sampling. A preset the publisher wrote down in a structured form
	// (generation_config.json, GGUF metadata, the Unsloth guides) beats a
	// reading of prose, so it is used when there is one.
	if presets := m.EffectiveSamplingPresets(); len(presets) > 0 {
		p := presets[0]
		reason := fmt.Sprintf("From the publisher's %q sampling settings (%s).", p.Label, p.Source)
		out.Proposals = append(out.Proposals, samplingProposals(p, reason)...)
		out.Proposals = append(out.Proposals, Proposal{Field: "sampling_preset", Value: p.Name, Reason: reason, Origin: "publisher preset"})
		if cardDiffers(adv, p) {
			out.note("temperature", "model card", quoted("The model card's text gives different sampling values from the publisher's settings; the publisher's settings are used.", adv.SamplingQuote))
		}
	} else {
		out.Proposals = append(out.Proposals, cardSampling(adv, &out)...)
	}

	if adv.Thinking != nil && *adv.Thinking != "model default" {
		out.note("", "model card", quoted(fmt.Sprintf("The model card recommends thinking %s. Clients choose this per request; the model's own default is kept.", *adv.Thinking), adv.ThinkingQuote))
	}

	// Speculative decoding: only a method whose file is on this machine
	// becomes a setting.
	if adv.DraftMethod != nil && *adv.DraftMethod != "none" && models.IsDraftMode(*adv.DraftMethod) {
		mode := *adv.DraftMethod
		repo := ""
		if adv.DraftRepo != nil {
			repo = strings.TrimSpace(*adv.DraftRepo)
		}
		switch {
		case mode == "draft-mtp" && (m.NextNLayers > 0 || in.Base.MtpPath != ""):
			out.Proposals = append(out.Proposals, Proposal{Field: "spec_type", Value: mode, Origin: "model card",
				Reason: quoted("Speculative decoding with the model's multi-token prediction (MTP) layers. It speeds up generation and does not change the answers.", adv.SpeculativeQuote)})
		case mode != "draft-mtp" && in.DraftCandidates != nil && len(in.DraftCandidates(mode)) > 0:
			c := in.DraftCandidates(mode)[0]
			out.Proposals = append(out.Proposals,
				Proposal{Field: "spec_type", Value: mode, Origin: "model card",
					Reason: quoted("Speculative decoding with a draft model the model card recommends. It speeds up generation and does not change the answers.", adv.SpeculativeQuote)},
				Proposal{Field: "draft_model_path", Value: c.FilePath, Origin: "model card",
					Reason: "The installed draft model " + c.Filename + "."})
		default:
			out.WantDraft, out.DraftRepo = mode, repo
			reason := "The model card recommends speculative decoding (" + mode + "), but the file it needs is not installed."
			if mode == "draft-mtp" {
				reason = "The model card mentions multi-token prediction (MTP), but this model file carries no MTP layers, so llama.cpp cannot use it. A separate MTP head published for this model would be needed."
			}
			out.note("spec_type", "model card", quoted(reason, adv.SpeculativeQuote))
		}
	}
	if adv.AssistMode != nil && *adv.AssistMode != "none" && models.IsAssistMode(*adv.AssistMode) {
		out.note("spec_assist", "model card", quoted("The model card mentions the "+*adv.AssistMode+" n-gram assist. Autotune measures whether it helps on this machine.", adv.SpeculativeQuote))
	}

	// Context: the planner owns it, whichever size was asked for. A card
	// that names a different number becomes a note and nothing more.
	//
	// It used to lower the context when the user had asked for the
	// maximum, on the reasoning that a publisher knows where its model
	// stops being reliable. That reading was wrong twice over. "Maximum"
	// is an instruction, not a preference to be weighed against a
	// paragraph of prose; and the number the helper model reads out of
	// the prose is often not the one the card states — a card saying
	// "262,144 tokens by default, extensible to 1,010,000 with YaRN" has
	// been read as a recommendation of 131,072, which then silently
	// halved a context that fitted on the machine with room to spare.
	if adv.RecommendedContext != nil {
		rc := *adv.RecommendedContext
		switch {
		case rc < 2048 || rc > m.ContextLength && m.ContextLength > 0:
			// Out of range: not worth mentioning.
		case rc != in.Fit.Config.ContextSize:
			out.note("context_size", "model card", quoted(fmt.Sprintf("The model card recommends a context of %d tokens. The size you chose is kept.", rc), adv.ContextQuote))
		}
	}

	for _, n := range adv.OtherNotes {
		if n = strings.TrimSpace(n); n != "" {
			out.note("", "model card", "The model card also mentions: "+n)
		}
	}
	return out
}

// samplingProposals turns a publisher preset into proposals.
func samplingProposals(p models.SamplingPreset, reason string) []Proposal {
	var out []Proposal
	add := func(field string, v any) {
		out = append(out, Proposal{Field: field, Value: v, Reason: reason, Origin: "publisher preset"})
	}
	if p.Temperature != nil {
		add("temperature", *p.Temperature)
	}
	if p.TopP != nil {
		add("top_p", *p.TopP)
	}
	if p.TopK != nil {
		add("top_k", *p.TopK)
	}
	if p.MinP != nil {
		add("min_p", *p.MinP)
	}
	if p.PresencePenalty != nil {
		add("presence_penalty", *p.PresencePenalty)
	}
	if p.RepeatPenalty != nil {
		add("repeat_penalty", *p.RepeatPenalty)
	}
	return out
}

// cardSampling range-checks the sampling values the helper read.
func cardSampling(adv Advice, out *Checked) []Proposal {
	var props []Proposal
	reason := quoted("Sampling settings the model card recommends.", adv.SamplingQuote)
	floatIn := func(field string, v *float64, lo, hi float64) {
		if v == nil {
			return
		}
		if *v < lo || *v > hi {
			out.note(field, "model card", fmt.Sprintf("The model card's %s of %g is outside the usual range (%g to %g) and was not used.", field, *v, lo, hi))
			return
		}
		props = append(props, Proposal{Field: field, Value: *v, Reason: reason, Origin: "model card"})
	}
	floatIn("temperature", adv.Temperature, 0, 2)
	floatIn("top_p", adv.TopP, 0, 1)
	floatIn("min_p", adv.MinP, 0, 1)
	floatIn("presence_penalty", adv.PresencePenalty, 0, 2)
	floatIn("repeat_penalty", adv.RepeatPenalty, 0, 2)
	if adv.TopK != nil {
		if *adv.TopK < 0 || *adv.TopK > 1000 {
			out.note("top_k", "model card", fmt.Sprintf("The model card's top_k of %d is outside the usual range (0 to 1000) and was not used.", *adv.TopK))
		} else {
			props = append(props, Proposal{Field: "top_k", Value: *adv.TopK, Reason: reason, Origin: "model card"})
		}
	}
	return props
}

// cardDiffers reports whether the helper read sampling values from the
// card that disagree with the publisher preset.
func cardDiffers(adv Advice, p models.SamplingPreset) bool {
	diff := func(a, b *float64) bool {
		return a != nil && b != nil && fmt.Sprintf("%.3f", *a) != fmt.Sprintf("%.3f", *b)
	}
	if diff(adv.Temperature, p.Temperature) || diff(adv.TopP, p.TopP) || diff(adv.MinP, p.MinP) {
		return true
	}
	return adv.TopK != nil && p.TopK != nil && *adv.TopK != *p.TopK
}
