package models

// SpecModeParam is one tunable parameter of a speculative decoding mode,
// with the recommended default the model config form applies when the
// mode is selected. The benchmark job form shows the same parameters
// with the same defaults under each mode, so the two surfaces cannot
// disagree about what a mode's settings are.
type SpecModeParam struct {
	Key     string // form key: draft_max, draft_min, draft_p_min, ngram_size_n, ngram_size_m
	Label   string
	Default string // recommended default as entered in a form field; "" = leave llama.cpp's default
}

// SpecModeParams returns the tunable parameters for a speculative
// decoding mode, in display order. An unknown or empty mode has none.
func SpecModeParams(mode string) []SpecModeParam {
	draftMax := func(d string) SpecModeParam { return SpecModeParam{"draft_max", "Draft tokens max", d} }
	draftMin := func(d string) SpecModeParam { return SpecModeParam{"draft_min", "Draft tokens min", d} }
	pMin := func(d string) SpecModeParam { return SpecModeParam{"draft_p_min", "Draft probability min", d} }
	ngramN := func(d string) SpecModeParam { return SpecModeParam{"ngram_size_n", "N-gram size N", d} }
	ngramM := func(d string) SpecModeParam { return SpecModeParam{"ngram_size_m", "N-gram size M", d} }

	switch mode {
	case "draft":
		return []SpecModeParam{draftMax("16"), draftMin("0"), pMin("0.75")}
	case "draft-mtp":
		// unsloth's MTP guidance uses --spec-draft-n-max 6; p-min is left
		// at llama.cpp's default — MTP heads have their own acceptance
		// logic.
		return []SpecModeParam{draftMax("6"), draftMin("0"), pMin("")}
	case "ngram-simple", "ngram-cache", "ngram-map-k", "ngram-map-k4v":
		return []SpecModeParam{draftMax("16"), ngramN("12"), ngramM("48")}
	case "ngram-mod":
		return []SpecModeParam{draftMax("64"), draftMin("48"), ngramN("24"), ngramM("48")}
	default:
		return nil
	}
}
