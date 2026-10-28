package autoconfig

import (
	"context"
	"fmt"

	"github.com/tmac1973/llama-toolchest/internal/llmcall"
	"github.com/tmac1973/llama-toolchest/internal/models"
)

// Deps is what Run needs from the server around it.
type Deps struct {
	Registry *models.Registry
	Hardware models.Hardware
	Fetcher  Fetcher
	HFBase   string
	Hub      Hub
	LLM      *llmcall.Client
	// HelperID is the helper model's registry ID, "" when none is
	// installed; HelperContext is its context size, for the card budget.
	HelperID      string
	HelperContext int
	// Progress receives a plain-language line as each step starts.
	Progress func(string)
	// Verify loads a proposal and sends it a request. Nil leaves the
	// proposal unchecked: an estimate, as it was before there was a
	// check.
	Verify VerifyFunc
}

// Result is a proposed starting profile and everything the review screen
// shows about it.
type Result struct {
	ModelID  string
	Class    models.ContextClass
	Base     models.ModelConfig // the live config when the run started
	Proposed models.ModelConfig
	// Notes explain the proposal: one per changed setting, plus general
	// notes (what the card also says, what could not be used).
	Notes []models.ProfileNote
	Fit   models.FitResult
	// CardSources are the repositories whose model cards were read.
	CardSources []string
	Suggestions []DraftSuggestion
	// Check is what the test load of the proposal found.
	Check Verification
}

// unusableDraftFile returns the file name of cfg's draft file when it is
// an installed model that cannot serve cfg's draft method, or "" when the
// file is fine or is not one the registry knows. For a head-based method
// the file must be a head of that kind; for a draft model, the registry's
// draft candidates for this model.
//
// A path the registry does not know is left alone: it may be a head
// placed by hand, and nothing here can say it is wrong.
func unusableDraftFile(reg *models.Registry, id string, cfg *models.ModelConfig) string {
	if !models.UsesDraftFile(cfg.SpecType) || cfg.DraftModelPath == "" {
		return ""
	}
	var file *models.Model
	for _, x := range reg.List() {
		if x.FilePath == cfg.DraftModelPath {
			file = x
			break
		}
	}
	if file == nil {
		return ""
	}
	cands := reg.FindDraftCandidates(id, cfg.SpecType)
	if models.IsHeadBasedDraftMode(cfg.SpecType) {
		cands = models.HeadsForMode(cfg.SpecType, cands)
	}
	for _, c := range cands {
		if c.FilePath == cfg.DraftModelPath {
			return ""
		}
	}
	return file.Filename
}

// Run proposes a starting profile for the model with registry ID id:
// the hardware fit first, then the model card's advice, then the
// validation a config save runs. The helper model is optional: without
// one, or when it fails, the proposal rests on the hardware fit alone and
// a note says so.
func Run(ctx context.Context, d Deps, id string, class models.ContextClass) (*Result, error) {
	progress := d.Progress
	if progress == nil {
		progress = func(string) {}
	}
	m, err := d.Registry.Get(id)
	if err != nil {
		return nil, err
	}
	cfgp, err := d.Registry.GetConfig(id)
	if err != nil {
		return nil, err
	}
	base := *cfgp

	// A draft file that cannot serve its method is turned off before
	// anything is planned: the estimate counts the file as GPU memory, so
	// a 22 GiB chat model set as a DSpark head would otherwise shrink the
	// plan to make room for a drafter that never loads.
	start := base
	badDraft := unusableDraftFile(d.Registry, id, &base)
	if badDraft != "" {
		start.SpecType, start.DraftModelPath = "", ""
	}

	// A model with its own draft layers has them turned on below, and
	// the draft context they need is memory like any other. The fit has
	// to be planned with it, or it plans a context that only fits while
	// drafting is off. The proposal itself starts from the user's own
	// setting, so the notes below still describe what changed.
	//
	// It is planned with MTP's own defaults, which the proposal gets when
	// MTP is turned on below. Without them the draft depth is unset, and
	// the estimate counts llama.cpp's default of 16 drafted tokens: on a
	// hybrid model that is the recurrent state 17 times over rather than
	// 7, and a 9B model that fits whole at 64K was planned with 5 of its
	// 33 layers on the GPU.
	fitBase := start
	if m.NextNLayers > 0 && fitBase.SpecType == "" {
		fitBase.SpecType = "draft-mtp"
		fitBase.ApplyDraftDefaults()
	}

	progress("Checking what fits on your GPU")
	fit := models.PlanFit(m, fitBase, d.Hardware, class)
	fit.Config.SpecType = start.SpecType
	res := &Result{ModelID: id, Class: class, Base: base, Proposed: fit.Config, Fit: fit}
	notes := map[string][]models.ProfileNote{} // by field; "" for general notes
	var order []string
	addNote := func(n models.ProfileNote) {
		if _, seen := notes[n.Field]; !seen {
			order = append(order, n.Field)
		}
		notes[n.Field] = append(notes[n.Field], n)
	}
	for _, n := range fit.Notes {
		addNote(n)
	}
	if badDraft != "" {
		addNote(models.ProfileNote{Field: "spec_type", Origin: "draft file", Reason: fmt.Sprintf(
			"Your %s setting loads %s, which is not a %s drafter for this model, so llama.cpp cannot use it. It is turned off, and its file no longer counts against GPU memory.",
			base.SpecType, badDraft, models.SpecModeLabel(base.SpecType))})
	}

	// A model with its own draft layers gets them turned on whatever the
	// card says: speculative decoding does not change the answers. When
	// they are already on, say so rather than saying nothing — an unspoken
	// setting reads as an overlooked one.
	const mtpWhy = "This model includes its own draft layers (multi-token prediction, MTP), which usually speed up generation with no change to the answers."
	mtpTurnedOn := false
	if m.NextNLayers > 0 {
		switch res.Proposed.SpecType {
		case "":
			p := Proposal{Field: "spec_type", Value: "draft-mtp", Origin: "model file", Reason: mtpWhy}
			ApplyProposal(&res.Proposed, p)
			// Turning MTP on applies its defaults; a draft depth the fit
			// lowered to make the plan fit is what the plan holds for.
			if fit.Config.DraftMax != fitBase.DraftMax {
				res.Proposed.DraftMax = fit.Config.DraftMax
			}
			addNote(models.ProfileNote{Field: p.Field, Reason: p.Reason, Origin: p.Origin})
			mtpTurnedOn = true
		case "draft-mtp":
			addNote(models.ProfileNote{Field: "spec_type", Origin: "model file",
				Reason: "Already on and kept. " + mtpWhy})
		default:
			addNote(models.ProfileNote{Field: "spec_type", Origin: "model file",
				Reason: fmt.Sprintf("Kept your %s setting. This model also has its own draft layers (MTP), which Autotune can compare against it.", res.Proposed.SpecType)})
		}
	}

	var checked Checked
	switch {
	case d.HelperID == "" || d.LLM == nil:
		addNote(models.ProfileNote{Origin: "default", Reason: "No helper model is installed, so the model card was not read. These settings come from the hardware fit only. Install a helper model in Settings to include the publisher's advice."})
	default:
		progress("Reading the model card")
		card := FetchCard(ctx, d.Fetcher, d.HFBase, m, CardCharsForContext(d.HelperContext))
		res.CardSources = card.Sources
		if card.Text == "" {
			addNote(models.ProfileNote{Origin: "default", Reason: "No model card with advice about running this model was found. These settings come from the hardware fit only."})
			break
		}
		progress("Asking the helper model to read it (this can take a minute)")
		adv, err := Ask(ctx, d.LLM, d.HelperID, m, card)
		if err != nil {
			if ctx.Err() != nil {
				return nil, ctx.Err()
			}
			addNote(models.ProfileNote{Origin: "default", Reason: "The helper model could not read the model card (" + err.Error() + "). These settings come from the hardware fit only."})
			break
		}
		progress("Checking the answer")
		checked = Validate(adv, Inputs{
			Model: m, Base: base, Class: class, Fit: fit,
			DraftCandidates: func(mode string) []models.DraftCandidate { return d.Registry.FindDraftCandidates(id, mode) },
		})
	}

	// Advice never changes what the fit decided about memory: the context
	// size, the KV cache type and where the layers run are the planner's,
	// because only it knows this machine. A card that says otherwise is
	// recorded as a note.
	for _, p := range checked.Proposals {
		if p.Field == "spec_type" && res.Proposed.SpecType == p.Value {
			continue // already on, from the model file
		}
		ApplyProposal(&res.Proposed, p)
		addNote(models.ProfileNote{Field: p.Field, Reason: p.Reason, Origin: p.Origin})
	}
	for _, n := range checked.Notes {
		addNote(n)
	}
	// MTP was turned on above, and the card's own method then replaced
	// it. The note that said why MTP is on now describes a setting that
	// is not proposed; say what it is instead.
	if mtpTurnedOn && res.Proposed.SpecType != "draft-mtp" {
		for i, n := range notes["spec_type"] {
			if n.Origin == "model file" && n.Reason == mtpWhy {
				notes["spec_type"][i].Reason = fmt.Sprintf(
					"This model also has its own draft layers (multi-token prediction, MTP). The model card's %s is proposed instead; Autotune can compare the two.",
					models.SpecModeLabel(res.Proposed.SpecType))
			}
		}
	}
	// The n-gram assist is measured, never assumed: it drafts from text
	// already in the context, so it wins on answers that repeat the
	// prompt and costs a little on those that do not. Which one a model
	// is used for is not something a model card knows.
	if res.Proposed.SpecAssist == "" {
		reason := "Autotune can test an n-gram assist. It speeds up answers that repeat text from the prompt, such as editing code or summarising a document, and Autotune measures whether it helps here."
		if res.Proposed.SpecType != "" {
			reason = "Autotune can test running an n-gram assist alongside " + res.Proposed.SpecType +
				". The two together are often faster than either alone on answers that repeat text from the prompt, such as editing code or summarising a document, and Autotune measures whether that holds on this machine."
		}
		addNote(models.ProfileNote{Origin: "default", Reason: reason})
	}

	models.NormalizeSpec(&res.Proposed)
	if err := validateProposal(&res.Proposed); err != nil {
		return nil, err
	}

	// The proposal is complete; find out whether it runs. An estimate
	// that says it does not fit is not loaded: the review already says
	// so, and the load would only take minutes to agree.
	if d.Verify != nil && fit.Fits {
		cfg, changes, check := verifyProposal(ctx, d, m, res.Proposed, progress)
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		res.Check = check
		if check.Status == VerifyPassed && len(changes) > 0 {
			if err := validateProposal(&cfg); err != nil {
				return nil, err
			}
			res.Proposed = cfg
			res.Fit.Config = cfg
			res.Fit.EstimateGiB = models.VRAMEstimateForConfigOn(m, &cfg, models.DeviceCountForConfig(&cfg, len(d.Hardware.GPUs)))
			for _, n := range changes {
				// What the fit said about a size the test load then
				// reduced is no longer true, and reads as a contradiction
				// beside the note that replaces it.
				if existing, have := notes[n.Field]; have && (n.Field == "context_size" || n.Field == "ubatch_size") {
					kept := existing[:0]
					for _, old := range existing {
						if old.Origin != "hardware fit" {
							kept = append(kept, old)
						}
					}
					notes[n.Field] = kept
				}
				addNote(n)
			}
		}
	}

	if checked.WantDraft != "" && d.Hub != nil {
		progress("Looking for the file that speculative decoding would need")
		installed := func(repo, file string) bool {
			for _, x := range d.Registry.List() {
				if x.ModelID == repo && x.Filename == file {
					return true
				}
			}
			return false
		}
		res.Suggestions = FindDraftSuggestions(ctx, d.Hub, m, checked, installed)
	}
	// Finish the speculative-decoding note now that it is known whether a
	// file for it can be downloaded. Without this it would end by pointing
	// at a list of suggested downloads that may be empty.
	if checked.WantDraft != "" {
		tail := " No file for it was found to download, so speculative decoding stays off."
		if res.Proposed.SpecType != "" {
			// Something else drafts — the model's own MTP layers, or the
			// user's own method — and saying "off" beside it contradicts
			// the row.
			tail = fmt.Sprintf(" No file for it was found to download, so it is not used; %s stays on.", models.SpecModeLabel(res.Proposed.SpecType))
		}
		if len(res.Suggestions) > 0 {
			tail = " The suggested downloads below can provide it."
		}
		for i, n := range notes["spec_type"] {
			if n.Origin == "model card" {
				notes["spec_type"][i].Reason += tail
			}
		}
	}

	for _, f := range order {
		res.Notes = append(res.Notes, notes[f]...)
	}
	return res, nil
}

// validateProposal runs the checks a config save runs.
func validateProposal(cfg *models.ModelConfig) error {
	if err := cfg.ValidateBatchSizes(); err != nil {
		return fmt.Errorf("the proposed settings are not valid: %w", err)
	}
	if err := cfg.ValidateFlashAttention(); err != nil {
		return fmt.Errorf("the proposed settings are not valid: %w", err)
	}
	if err := cfg.ValidateSpec(); err != nil {
		return fmt.Errorf("the proposed settings are not valid: %w", err)
	}
	return nil
}

// Profile builds the profile to save from a result.
func (r *Result) Profile(buildID string) models.ConfigProfile {
	return models.ConfigProfile{
		Config:  r.Proposed,
		Source:  models.ProfileSourceAutoconfig,
		Notes:   r.Notes,
		BuildID: buildID,
	}
}
