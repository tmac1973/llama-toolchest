package models

// SpecModeParam is one tunable parameter of a speculative decoding mode,
// with the recommended default the model config form applies when the
// mode is selected. The benchmark job form shows the same parameters
// with the same defaults under each mode, so the two surfaces cannot
// disagree about what a mode's settings are.
type SpecModeParam struct {
	Key     string // form key: draft_max, draft_min, draft_p_min, assist_n_max, …
	Label   string
	Default string // recommended default as entered in a form field; "" = leave llama.cpp's default
}

// SpecDraftParams returns the tunable parameters of a draft method, in
// display order. An unknown or empty mode has none.
//
// The keys are all draft_*, and the assist keys are all assist_*, so a
// caller handed one list can never mistake a key for a field it doesn't
// have. That separation is the whole reason these are two functions.
func SpecDraftParams(mode string) []SpecModeParam {
	draftMax := func(d string) SpecModeParam { return SpecModeParam{"draft_max", "Draft tokens max", d} }
	draftMin := func(d string) SpecModeParam { return SpecModeParam{"draft_min", "Draft tokens min", d} }
	pMin := func(d string) SpecModeParam { return SpecModeParam{"draft_p_min", "Draft probability min", d} }

	switch mode {
	case "draft":
		return []SpecModeParam{draftMax("16"), draftMin("0"), pMin("0.75")}
	case "draft-mtp":
		// unsloth's MTP guidance uses --spec-draft-n-max 6; p-min is left
		// at llama.cpp's default — MTP heads have their own acceptance
		// logic.
		return []SpecModeParam{draftMax("6"), draftMin("0"), pMin("")}
	case "draft-eagle3", "draft-dflash", "draft-dspark":
		// Deliberately blank: llama.cpp's own defaults (n-max 3, n-min 0)
		// apply, and there is no measured guidance for these three to
		// present as a recommendation. The benchmark sweep is where good
		// values come from, not a guess in this table.
		return []SpecModeParam{draftMax(""), draftMin(""), pMin("")}
	default:
		return nil
	}
}

// SpecAssistParams returns the tunable parameters of a draftless n-gram
// method, in display order. An unknown or empty mode has none, and so
// does ngram-cache, which takes no settings.
func SpecAssistParams(mode string) []SpecModeParam {
	nMax := func(d string) SpecModeParam { return SpecModeParam{"assist_n_max", "Draft tokens max", d} }
	nMin := func(d string) SpecModeParam { return SpecModeParam{"assist_n_min", "Draft tokens min", d} }
	nMatch := func(d string) SpecModeParam { return SpecModeParam{"assist_n_match", "Match length", d} }
	sizeN := func(d string) SpecModeParam { return SpecModeParam{"assist_size_n", "Lookup size", d} }
	sizeM := func(d string) SpecModeParam { return SpecModeParam{"assist_size_m", "Draft size", d} }
	minHits := func(d string) SpecModeParam { return SpecModeParam{"assist_min_hits", "Minimum hits", d} }

	switch mode {
	case "ngram-mod":
		// ngram-mod has no m-gram size — the old table offered one, and
		// it meant nothing for this mode.
		return []SpecModeParam{nMax("64"), nMin("48"), nMatch("24")}
	case "ngram-simple", "ngram-map-k", "ngram-map-k4v":
		return []SpecModeParam{sizeN("12"), sizeM("48"), minHits("1")}
	default:
		// ngram-cache takes no settings.
		return nil
	}
}
