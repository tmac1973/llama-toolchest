package api

import (
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/models"
)

// The two default functions are split per slot precisely so that changing
// one picker cannot wipe the other's tuned values. A single function that
// reset everything — which is what existed before there were two slots —
// would silently discard a tuned draft depth the moment the user picked
// an n-gram assist.
func TestApplyDefaultsTouchOnlyTheirOwnSlot(t *testing.T) {
	cfg := &models.ModelConfig{
		SpecType: "draft-mtp", DraftMax: 3, DraftMin: 1, DraftPMin: "0.5",
		SpecAssist: "ngram-mod", AssistNMax: 99, AssistNMin: 7, AssistNMatch: 5,
	}

	applyAssistDefaults(cfg)
	if cfg.DraftMax != 3 || cfg.DraftMin != 1 || cfg.DraftPMin != "0.5" {
		t.Errorf("assist defaults clobbered the draft slot: %+v", cfg)
	}
	if cfg.AssistNMax != 64 || cfg.AssistNMin != 48 || cfg.AssistNMatch != 24 {
		t.Errorf("assist defaults not applied: %+v", cfg)
	}

	cfg.DraftMax, cfg.DraftMin, cfg.DraftPMin = 3, 1, "0.5"
	applyDraftDefaults(cfg)
	if cfg.AssistNMax != 64 || cfg.AssistNMin != 48 || cfg.AssistNMatch != 24 {
		t.Errorf("draft defaults clobbered the assist slot: %+v", cfg)
	}
	if cfg.DraftMax != 6 || cfg.DraftMin != 0 || cfg.DraftPMin != "" {
		t.Errorf("draft defaults not applied: %+v", cfg)
	}
}

// The three new draft methods carry no recommended values, so selecting
// one must leave the fields empty rather than inventing a number.
func TestApplyDraftDefaultsLeavesHeadModesBlank(t *testing.T) {
	for _, mode := range []string{"draft-eagle3", "draft-dflash", "draft-dspark"} {
		cfg := &models.ModelConfig{SpecType: mode, DraftMax: 16, DraftMin: 2, DraftPMin: "0.9"}
		applyDraftDefaults(cfg)
		if cfg.DraftMax != 0 || cfg.DraftMin != 0 || cfg.DraftPMin != "" {
			t.Errorf("%s should apply no defaults, got %+v", mode, cfg)
		}
	}
}

// ngram-cache takes no settings, so switching to it must clear whatever
// the previous assist mode left behind.
func TestApplyAssistDefaultsClearsForSettinglessMode(t *testing.T) {
	cfg := &models.ModelConfig{SpecAssist: "ngram-cache", AssistNMax: 64, AssistSizeN: 12}
	applyAssistDefaults(cfg)
	if cfg.AssistNMax != 0 || cfg.AssistSizeN != 0 {
		t.Errorf("ngram-cache should carry no settings, got %+v", cfg)
	}
}

func TestConfigFormShowsBothPickers(t *testing.T) {
	out := renderModelConfig(t, &models.ModelConfig{
		SpecType: "draft-mtp", DraftMax: 3,
		SpecAssist: "ngram-mod", AssistNMax: 64, AssistNMin: 48, AssistNMatch: 24,
	}, false, "")

	for _, want := range []string{
		`name="spec_type"`, `name="spec_assist"`,
		"Draft method", "N-gram assist",
		`name="draft_max"`, `name="assist_n_max"`, `name="assist_n_match"`,
		// Every draft method reachable, including the three that were
		// merged upstream but offered nowhere here.
		`value="draft-eagle3"`, `value="draft-dflash"`, `value="draft-dspark"`,
	} {
		if !strings.Contains(out, want) {
			t.Errorf("config form missing %q", want)
		}
	}
	// ngram-mod has no m-gram size; the old form offered one anyway.
	if strings.Contains(out, `name="assist_size_m"`) || strings.Contains(out, `name="ngram_size_m"`) {
		t.Errorf("ngram-mod must not be offered an m-gram size:\n%s", out)
	}
	// Two draft methods cannot be selected, because they share one picker.
	if n := strings.Count(out, `name="spec_type"`); n != 1 {
		t.Errorf("want exactly one draft-method picker, got %d", n)
	}
}

func TestConfigFormShowsEffectiveSpecType(t *testing.T) {
	combined := renderModelConfig(t, &models.ModelConfig{
		SpecType: "draft-mtp", SpecAssist: "ngram-mod",
	}, false, "")
	if !strings.Contains(combined, "draft-mtp,ngram-mod") {
		t.Errorf("effective --spec-type not shown:\n%s", combined)
	}

	off := renderModelConfig(t, &models.ModelConfig{}, false, "")
	if !strings.Contains(off, "speculative decoding is off") {
		t.Errorf("off state should say so, not print a value:\n%s", off)
	}
	// "none" is a real llama.cpp value that discards every other mode in a
	// list. The launch never emits it, so the form must never show it.
	if strings.Contains(off, "<code>none</code>") {
		t.Errorf("form must not display none as an effective value")
	}
}

func TestConfigFormOffersHeadPickerForHeadModes(t *testing.T) {
	for _, mode := range []string{"draft-eagle3", "draft-dflash", "draft-dspark"} {
		out := renderModelConfig(t, &models.ModelConfig{SpecType: mode}, false, "")
		if !strings.Contains(out, "Drafter Head") {
			t.Errorf("%s should offer a head picker", mode)
		}
		if !strings.Contains(out, `name="draft_model_path"`) {
			t.Errorf("%s head picker should post draft_model_path", mode)
		}
	}
}
