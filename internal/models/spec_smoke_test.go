package models

import (
	"reflect"
	"strings"
	"testing"
)

func TestSpecMTPEffectiveFlags(t *testing.T) {
	cfg := &ModelConfig{
		GPULayers:   99,
		ContextSize: 8192,
		Threads:     8,
		SpecType:    "draft-mtp",
		DraftMax:    6,
	}
	got := cfg.EffectiveFlags()
	for _, want := range []string{"--spec-type draft-mtp", "--spec-draft-n-max 6"} {
		if !strings.Contains(got, want) {
			t.Errorf("MTP flags missing %q in: %s", want, got)
		}
	}
	if strings.Contains(got, "--model-draft") {
		t.Errorf("MTP should not emit --model-draft, got: %s", got)
	}
}

func TestSpecMTPSeparateDrafterFlags(t *testing.T) {
	// gemma-4 style: MTP head ships as its own GGUF, loaded via --model-draft
	// under spec-type draft-mtp, including draft-resource overrides.
	cfg := &ModelConfig{
		GPULayers:   99,
		ContextSize: 8192,
		Threads:     8,
		SpecType:    "draft-mtp",
		MtpPath:     "/models/gemma-4-12B-it-MTP-Q8_0.gguf",
		DraftMax:    4,
		DraftDevice: "CUDA0",
	}
	got := cfg.EffectiveFlags()
	for _, want := range []string{
		"--spec-type draft-mtp",
		"--model-draft /models/gemma-4-12B-it-MTP-Q8_0.gguf",
		"--spec-draft-n-max 4",
		"--device-draft CUDA0",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("gemma MTP flags missing %q in: %s", want, got)
		}
	}
}

func TestSpecMTPSeparateDrafterDisabled(t *testing.T) {
	// MtpDisabled suppresses --model-draft while keeping self-speculation flags.
	cfg := &ModelConfig{
		GPULayers:   99,
		ContextSize: 8192,
		Threads:     8,
		SpecType:    "draft-mtp",
		MtpPath:     "/models/gemma-4-12B-it-MTP-Q8_0.gguf",
		MtpDisabled: true,
		DraftMax:    4,
	}
	got := cfg.EffectiveFlags()
	if strings.Contains(got, "--model-draft") {
		t.Errorf("disabled MTP head should not emit --model-draft, got: %s", got)
	}
	if !strings.Contains(got, "--spec-type draft-mtp") {
		t.Errorf("MTP flags missing --spec-type draft-mtp in: %s", got)
	}
}

func TestIsMTPHeadArch(t *testing.T) {
	for _, arch := range []string{"gemma4-assistant", "gemma4_assistant"} {
		if !IsMTPHeadArch(arch) {
			t.Errorf("IsMTPHeadArch(%q) = false, want true", arch)
		}
	}
	// Qwen self-speculation MTP models use a normal runnable arch — must NOT
	// be classified as a standalone drafter head.
	for _, arch := range []string{"qwen3", "qwen3moe", "gemma3", "llama"} {
		if IsMTPHeadArch(arch) {
			t.Errorf("IsMTPHeadArch(%q) = true, want false", arch)
		}
	}
}

func TestSpecDraftResourceFlags(t *testing.T) {
	cfg := &ModelConfig{
		GPULayers:         99,
		ContextSize:       16384,
		Threads:           8,
		SpecType:          "draft",
		DraftModelPath:    "/models/qwen-0.5b.gguf",
		DraftMax:          16,
		DraftPMin:         "0.75",
		DraftCtxSize:      4096,
		DraftGPULayers:    99,
		DraftDevice:       "CUDA1",
		DraftCPUMoE:       2,
		DraftKVCacheQuant: "q8_0",
	}
	got := cfg.EffectiveFlags()
	for _, want := range []string{
		"--spec-type draft-simple",
		"--model-draft /models/qwen-0.5b.gguf",
		"--spec-draft-n-max 16",
		"--spec-draft-p-min 0.75",
		"--ctx-size-draft 4096",
		"--gpu-layers-draft 99",
		"--device-draft CUDA1",
		"--n-cpu-moe-draft 2",
		"--cache-type-k-draft q8_0",
		"--cache-type-v-draft q8_0",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("Draft flags missing %q in: %s", want, got)
		}
	}
}

func TestSpecMTPPresetINI(t *testing.T) {
	cfg := &ModelConfig{
		Enabled:     true,
		GPULayers:   99,
		ContextSize: 8192,
		Threads:     8,
		SpecType:    "draft-mtp",
		DraftMax:    6,
	}
	var b strings.Builder
	writeConfigParams(&b, cfg, false, "")
	out := b.String()
	for _, want := range []string{"spec-type = draft-mtp", "spec-draft-n-max = 6"} {
		if !strings.Contains(out, want) {
			t.Errorf("MTP preset missing %q in:\n%s", want, out)
		}
	}
	if strings.Contains(out, "model-draft") {
		t.Errorf("MTP preset should not emit model-draft, got:\n%s", out)
	}
}

func TestSpecCombinedFlags(t *testing.T) {
	// The configuration the whole two-slot design exists for: MTP drafting
	// 3 tokens while ngram-mod drafts up to 64. Before the split this was
	// not merely unsupported, it was inexpressible — DraftMax had to be
	// either --spec-draft-n-max or --spec-ngram-mod-n-max, not both.
	cfg := &ModelConfig{
		GPULayers:    99,
		ContextSize:  8192,
		Threads:      8,
		SpecType:     "draft-mtp",
		DraftMax:     3,
		SpecAssist:   "ngram-mod",
		AssistNMax:   64,
		AssistNMin:   48,
		AssistNMatch: 24,
	}
	got := cfg.EffectiveFlags()
	for _, want := range []string{
		"--spec-type draft-mtp,ngram-mod",
		"--spec-draft-n-max 3",
		"--spec-ngram-mod-n-max 64",
		"--spec-ngram-mod-n-min 48",
		"--spec-ngram-mod-n-match 24",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("combined flags missing %q in: %s", want, got)
		}
	}
	// "none" anywhere in the list makes llama.cpp discard every other mode.
	if strings.Contains(got, "none") {
		t.Errorf("combined flags must never contain %q, got: %s", "none", got)
	}
	if n := strings.Count(got, "--spec-type"); n != 1 {
		t.Errorf("want exactly one --spec-type, got %d in: %s", n, got)
	}
}

func TestSpecCombinedPresetINISingleSpecTypeLine(t *testing.T) {
	// common/preset.cpp parses an INI section into a map, so a repeated
	// "spec-type =" line silently keeps only the last value. The modes
	// must arrive comma-joined on one line.
	cfg := &ModelConfig{
		Enabled:     true,
		GPULayers:   99,
		ContextSize: 8192,
		Threads:     8,
		SpecType:    "draft-mtp",
		DraftMax:    3,
		SpecAssist:  "ngram-mod",
		AssistNMax:  64,
	}
	var b strings.Builder
	writeConfigParams(&b, cfg, false, "")

	var specLines []string
	for _, line := range strings.Split(b.String(), "\n") {
		if strings.HasPrefix(strings.TrimSpace(line), "spec-type") {
			specLines = append(specLines, strings.TrimSpace(line))
		}
	}
	if len(specLines) != 1 {
		t.Fatalf("want exactly one spec-type line, got %d: %v", len(specLines), specLines)
	}
	if want := "spec-type = draft-mtp,ngram-mod"; specLines[0] != want {
		t.Errorf("spec-type line = %q, want %q", specLines[0], want)
	}
}

func TestSpecNgramModMatchEmitted(t *testing.T) {
	// The form stored this value and specDecodingParams never emitted it,
	// so tuning it did nothing.
	cfg := &ModelConfig{
		GPULayers:    99,
		ContextSize:  8192,
		Threads:      8,
		SpecAssist:   "ngram-mod",
		AssistNMatch: 32,
	}
	if got := cfg.EffectiveFlags(); !strings.Contains(got, "--spec-ngram-mod-n-match 32") {
		t.Errorf("ngram-mod match length not emitted in: %s", got)
	}
}

func TestSpecMinHitsEmitted(t *testing.T) {
	// Offered as a default in no table and emitted by no branch before the
	// assist slot existed.
	for _, mode := range []string{"ngram-simple", "ngram-map-k", "ngram-map-k4v"} {
		cfg := &ModelConfig{
			GPULayers:     99,
			ContextSize:   8192,
			Threads:       8,
			SpecAssist:    mode,
			AssistSizeN:   12,
			AssistSizeM:   48,
			AssistMinHits: 2,
		}
		got := cfg.EffectiveFlags()
		for _, want := range []string{
			"--spec-type " + mode,
			"--spec-" + mode + "-size-n 12",
			"--spec-" + mode + "-size-m 48",
			"--spec-" + mode + "-min-hits 2",
		} {
			if !strings.Contains(got, want) {
				t.Errorf("%s flags missing %q in: %s", mode, want, got)
			}
		}
	}
}

func TestSpecNewDraftModes(t *testing.T) {
	// Merged upstream, documented, in the build we ship, and reachable
	// from nowhere in the toolchest until now.
	for _, mode := range []string{"draft-eagle3", "draft-dflash", "draft-dspark"} {
		cfg := &ModelConfig{
			GPULayers:      99,
			ContextSize:    8192,
			Threads:        8,
			SpecType:       mode,
			DraftModelPath: "/models/head.gguf",
		}
		got := cfg.EffectiveFlags()
		for _, want := range []string{
			"--spec-type " + mode,
			"--model-draft /models/head.gguf",
		} {
			if !strings.Contains(got, want) {
				t.Errorf("%s flags missing %q in: %s", mode, want, got)
			}
		}
		// Blank defaults: nothing is emitted unless the user sets it.
		if strings.Contains(got, "--spec-draft-n-max") {
			t.Errorf("%s should emit no draft length by default, got: %s", mode, got)
		}
	}
}

func TestSpecLegacyDraftlessConfig(t *testing.T) {
	// The shape a registry written before the split holds. It must launch
	// as it always did, with the one deliberate addition of the n-match
	// flag the form was already storing.
	legacy := &ModelConfig{
		GPULayers:   99,
		ContextSize: 8192,
		Threads:     8,
		SpecType:    "ngram-mod",
		DraftMax:    64,
		DraftMin:    48,
		NgramSizeN:  24,
		NgramSizeM:  48,
	}
	got := legacy.EffectiveFlags()
	for _, want := range []string{
		"--spec-type ngram-mod",
		"--spec-ngram-mod-n-max 64",
		"--spec-ngram-mod-n-min 48",
		"--spec-ngram-mod-n-match 24",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("legacy ngram-mod flags missing %q in: %s", want, got)
		}
	}
	// ngram-mod has no m-gram size; the form offered one and it meant
	// nothing, so it must not reach the command line under any name.
	if strings.Contains(got, "size-m") {
		t.Errorf("ngram-mod must not emit an m-gram size, got: %s", got)
	}
	// Reading must not rewrite the caller's config.
	if legacy.SpecType != "ngram-mod" || legacy.DraftMax != 64 {
		t.Errorf("EffectiveFlags mutated the config: %+v", legacy)
	}

	// The migrated shape emits exactly the same thing.
	migrated := *legacy
	NormalizeSpec(&migrated)
	if migrated.SpecType != "" || migrated.SpecAssist != "ngram-mod" {
		t.Fatalf("NormalizeSpec left slots wrong: type=%q assist=%q", migrated.SpecType, migrated.SpecAssist)
	}
	if migrated.AssistNMax != 64 || migrated.AssistNMin != 48 || migrated.AssistNMatch != 24 {
		t.Errorf("NormalizeSpec lost parameters: %+v", migrated)
	}
	if migrated.NgramSizeM != 0 || migrated.AssistSizeM != 0 {
		t.Errorf("ngram-mod m-gram size should be dropped, got %+v", migrated)
	}
	if migrated.EffectiveFlags() != got {
		t.Errorf("migrated config emits different flags:\n legacy:   %s\n migrated: %s", got, migrated.EffectiveFlags())
	}

	// Idempotent: the startup backfill runs on every boot.
	twice := migrated
	NormalizeSpec(&twice)
	if !reflect.DeepEqual(twice, migrated) {
		t.Errorf("NormalizeSpec is not idempotent: %+v vs %+v", twice, migrated)
	}
}

func TestSpecLegacyDraftlessSizeModes(t *testing.T) {
	// The size-n/size-m family migrates onto different fields than
	// ngram-mod does, since "N-gram size N" meant lookup size there.
	legacy := &ModelConfig{
		GPULayers:   99,
		ContextSize: 8192,
		Threads:     8,
		SpecType:    "ngram-simple",
		NgramSizeN:  12,
		NgramSizeM:  48,
	}
	got := legacy.EffectiveFlags()
	for _, want := range []string{
		"--spec-type ngram-simple",
		"--spec-ngram-simple-size-n 12",
		"--spec-ngram-simple-size-m 48",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("legacy ngram-simple flags missing %q in: %s", want, got)
		}
	}
	// No legacy field feeds min-hits, so a migrated config emits none —
	// llama.cpp's own default is the 1 the form now offers, so nothing
	// changes for it.
	if strings.Contains(got, "min-hits") {
		t.Errorf("a migrated config should emit no min-hits, got: %s", got)
	}
}

func TestSpecOffEmitsNothing(t *testing.T) {
	cfg := &ModelConfig{GPULayers: 99, ContextSize: 8192, Threads: 8}
	got := cfg.EffectiveFlags()
	if strings.Contains(got, "--spec-type") {
		t.Errorf("speculative decoding off should emit no --spec-type, got: %s", got)
	}
}

func TestValidateSpecRejectsWrongSlot(t *testing.T) {
	// The two pickers cannot produce these, but a hand-edited registry or
	// a crafted POST can, and llama-server treats an unknown --spec-type
	// name as a fatal startup error.
	cases := []struct {
		name    string
		cfg     ModelConfig
		wantErr bool
	}{
		{"empty is fine", ModelConfig{}, false},
		{"draft in draft slot", ModelConfig{SpecType: "draft-mtp"}, false},
		{"assist in assist slot", ModelConfig{SpecAssist: "ngram-mod"}, false},
		{"both slots filled", ModelConfig{SpecType: "draft-mtp", SpecAssist: "ngram-mod"}, false},
		{"assist in draft slot", ModelConfig{SpecType: "ngram-mod"}, true},
		{"draft in assist slot", ModelConfig{SpecAssist: "draft-mtp"}, true},
		{"unknown draft name", ModelConfig{SpecType: "draft-mtp-adaptive"}, true},
		{"unknown assist name", ModelConfig{SpecAssist: "ngram-whatever"}, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.cfg.ValidateSpec()
			if (err != nil) != tc.wantErr {
				t.Errorf("ValidateSpec() error = %v, wantErr %v", err, tc.wantErr)
			}
		})
	}
}

func TestEffectiveSpecType(t *testing.T) {
	cases := []struct {
		cfg  ModelConfig
		want string
	}{
		{ModelConfig{}, ""},
		{ModelConfig{SpecType: "draft-mtp"}, "draft-mtp"},
		{ModelConfig{SpecType: "draft"}, "draft-simple"},
		{ModelConfig{SpecAssist: "ngram-mod"}, "ngram-mod"},
		{ModelConfig{SpecType: "draft-mtp", SpecAssist: "ngram-mod"}, "draft-mtp,ngram-mod"},
		// A config still in the pre-split shape reads the same as its
		// migrated equivalent.
		{ModelConfig{SpecType: "ngram-mod"}, "ngram-mod"},
	}
	for _, tc := range cases {
		if got := tc.cfg.EffectiveSpecType(); got != tc.want {
			t.Errorf("EffectiveSpecType(%+v) = %q, want %q", tc.cfg, got, tc.want)
		}
	}
}
