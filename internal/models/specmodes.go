package models

// Speculative decoding has two independent slots. llama.cpp accepts a
// comma-separated --spec-type list and its docs sanction exactly one
// shape of combination:
//
//	An implementation with draft model can be mixed with an
//	implementation without draft model.
//
// So SpecType holds the draft method and SpecAssist the draftless one,
// each with its own draft-length flags. Two draft methods in one list
// crash llama-server at startup (upstream issue #27897), which the two
// separate lists below make unreachable rather than something to
// validate against.
//
// Ordering inside the list is not exposed because it does not exist:
// common_speculative_init turns the names into a bitmask and walks a
// fixed priority order, so "draft-mtp,ngram-mod" and "ngram-mod,draft-mtp"
// behave identically.

// SpecMode is one speculative decoding mode and the label every surface
// shows for it. Sharing the labels is what stops the model config form,
// the benchmark job form and the sweep choices from drifting apart.
//
// There is deliberately no entry for "no mode selected": it reads
// differently in each place ("Disabled" in the draft picker, "None" in
// the assist picker, "Off (no speculative decoding)" in the sweep
// choices), so each surface writes its own.
type SpecMode struct {
	Name  string
	Label string
}

// draftModes are the modes that speculate with a drafter — a smaller
// model of the same family, or a trained head loaded through
// --model-draft, or one baked into the main GGUF. At most one can run.
//
// "draft" is the internal name for llama.cpp's "draft-simple"; the
// rename happened after configs were already saved with the old value,
// so the config keeps it and specDecodingParams translates on the way
// out.
var draftModes = []SpecMode{
	{"draft", "Draft Model"},
	{"draft-mtp", "MTP (self-speculation)"},
	{"draft-eagle3", "EAGLE-3"},
	{"draft-dflash", "DFlash"},
	{"draft-dspark", "DSpark"},
}

// assistModes are the draftless n-gram methods. They propose tokens by
// matching text already in the prompt and the generation so far, so they
// load no extra file and cost no extra memory.
var assistModes = []SpecMode{
	{"ngram-mod", "N-gram Mod"},
	{"ngram-simple", "N-gram Simple"},
	{"ngram-cache", "N-gram Cache"},
	{"ngram-map-k", "N-gram Map-K"},
	{"ngram-map-k4v", "N-gram Map-K4V"},
}

// headBasedDraftModes are the draft methods whose --model-draft target is
// a converted head rather than a smaller model of the same family. It
// matters for the draft-model picker: a head matches neither the
// architecture nor the size filter that makes the picker safe for
// "draft".
var headBasedDraftModes = map[string]bool{
	"draft-eagle3": true,
	"draft-dflash": true,
	"draft-dspark": true,
}

// DraftModes returns the draft methods in display order.
func DraftModes() []SpecMode { return draftModes }

// AssistModes returns the draftless n-gram methods in display order.
func AssistModes() []SpecMode { return assistModes }

// IsDraftMode reports whether name is a draft method, so belongs in
// ModelConfig.SpecType.
func IsDraftMode(name string) bool { return specModeIn(draftModes, name) }

// IsAssistMode reports whether name is a draftless n-gram method, so
// belongs in ModelConfig.SpecAssist.
func IsAssistMode(name string) bool { return specModeIn(assistModes, name) }

// IsHeadBasedDraftMode reports whether the mode loads a converted head
// through --model-draft.
func IsHeadBasedDraftMode(name string) bool { return headBasedDraftModes[name] }

// SpecModeLabel returns the display label for a mode, or the raw name for
// one this build doesn't know — a config written by a newer build should
// render as something rather than as an empty cell.
func SpecModeLabel(name string) string {
	for _, m := range draftModes {
		if m.Name == name {
			return m.Label
		}
	}
	for _, m := range assistModes {
		if m.Name == name {
			return m.Label
		}
	}
	return name
}

func specModeIn(modes []SpecMode, name string) bool {
	for _, m := range modes {
		if m.Name == name {
			return true
		}
	}
	return false
}

// NormalizeSpec moves a draftless mode found in SpecType into SpecAssist,
// carrying its parameters onto the assist fields.
//
// Before the two slots existed, SpecType held whichever single mode was
// selected and DraftMax/DraftMin/NgramSizeN/NgramSizeM changed meaning
// depending on which one it was. Saved model configs, stored benchmark
// jobs and completed run records all still carry that shape, so every
// read path runs this: the launch path (specDecodingParams), the
// benchmark override merge, and the one-shot registry backfill.
//
// It is idempotent, and does nothing when SpecType is empty or already a
// draft method.
func NormalizeSpec(c *ModelConfig) {
	if c == nil || !IsAssistMode(c.SpecType) {
		return
	}
	mode := c.SpecType
	c.SpecType = ""
	c.SpecAssist = mode

	// ngram-mod's lengths are n-max/n-min, and the field the form
	// labelled "N-gram size N" was always its match length — the value
	// specDecodingParams never emitted. The other four use size-n/size-m
	// and have no match length.
	c.AssistNMax, c.AssistNMin = c.DraftMax, c.DraftMin
	if mode == "ngram-mod" {
		// ngram-mod has no m-gram size, so NgramSizeM is dropped rather
		// than migrated: the form offered it, but the mode has no such
		// parameter.
		c.AssistNMatch = c.NgramSizeN
	} else {
		c.AssistSizeN, c.AssistSizeM = c.NgramSizeN, c.NgramSizeM
	}

	c.DraftMax, c.DraftMin = 0, 0
	c.NgramSizeN, c.NgramSizeM = 0, 0
	// DraftPMin is left alone: it is a draft-method knob, and a draftless
	// config that carries one was never emitting it anyway.
	c.DraftPMin = ""
}
