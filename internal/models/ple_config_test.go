package models

import (
	"strings"
	"testing"
)

// The three UI modes map onto llama.cpp's on-demand reading option, with
// auto emitting nothing so llama.cpp's own default applies. The option is
// spelled as the build spells it: --lazy-mode from b10700, before that
// --tensor-read-lazy from b10653, and before that it does not exist. An
// option a build does not know stops the router loading any model, so a
// build without it gets nothing rather than a guess.
func TestPLEModeEmitsTheBuildsLazyReadOption(t *testing.T) {
	tests := []struct {
		name    string
		version int
		mode    string
		want    string // "" means neither key may appear
	}{
		{"auto", 0, "", ""},
		{"on, current llama.cpp", 11064, "on", "lazy-mode = on"},
		{"off, current llama.cpp", 11064, "off", "lazy-mode = off"},
		{"on, build of unknown version", 0, "on", "lazy-mode = on"},
		{"on, first build with the new name", 10700, "on", "lazy-mode = on"},
		{"on, build before the rename", 10699, "on", "tensor-read-lazy = on"},
		{"on, first build with the option", 10653, "on", "tensor-read-lazy = on"},
		{"on, build before the option existed", 10448, "on", ""},
		{"nonsense", 11064, "nonsense", ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := &Model{ID: "m", Filename: "m.gguf", FilePath: "/models/m.gguf"}
			cfg := &ModelConfig{Enabled: true, PLEMode: tt.mode}
			target := Target{Backend: "rocm", Version: tt.version}
			ini := GeneratePresetINI("/models", []*Model{m}, map[string]*ModelConfig{"m": cfg}, target)
			flags := cfg.EffectiveFlagsFor(false, target)
			if tt.want == "" {
				for _, key := range []string{"lazy-mode", "tensor-read-lazy"} {
					if strings.Contains(ini, key) || strings.Contains(flags, key) {
						t.Errorf("emitted %s; preset:\n%s\nflags: %s", key, ini, flags)
					}
				}
				return
			}
			if !strings.Contains(ini, tt.want) {
				t.Errorf("want %q in preset:\n%s", tt.want, ini)
			}
			flag := "--" + strings.Replace(tt.want, " = ", " ", 1)
			if !strings.Contains(flags, flag) {
				t.Errorf("want %q in flags: %s", flag, flags)
			}
			other := "tensor-read-lazy"
			if strings.HasPrefix(tt.want, "tensor-read-lazy") {
				other = "lazy-mode"
			}
			if strings.Contains(ini, other) {
				t.Errorf("preset has both spellings:\n%s", ini)
			}
		})
	}
}

const testPLEBytes = 30 * 1024 * 1024 * 1024 // 30 GiB, comfortably over the auto threshold

// The table is never on a device, whatever the mode. Measured on four
// architectures: llama.cpp reports it under CPU_Mapped, and total VRAM is
// identical with the mode on, off and auto — see plan/ple-vram-findings.md.
//
// This replaces a pair of tests asserting the opposite. They encoded the
// belief the feature was built on: that "resident" meant resident in VRAM
// and streaming saved GPU memory. Hardware says streaming saves host
// memory and changes VRAM by nothing.
func TestPLEModeDoesNotChangeVRAM(t *testing.T) {
	m := &Model{
		SizeBytes: 40 * 1024 * 1024 * 1024,
		PLEBytes:  testPLEBytes,
		NLayers:   48, AttnLayers: 48, NEmbd: 4096, NHead: 32, NKVHead: 8,
		KVFullPerTok: 48 * 8 * 256, ContextLength: 4096,
	}
	est := func(mode string, directIO bool) float64 {
		return VRAMEstimateForConfig(m, &ModelConfig{ContextSize: 4096, PLEMode: mode, DirectIO: directIO})
	}
	want := est("", false)
	for _, mode := range []string{"on", "off"} {
		if got := est(mode, false); got != want {
			t.Errorf("mode %q gave %.4f, want %.4f — the table is host-mapped in every mode", mode, got, want)
		}
	}
	// Direct I/O changes how the file is read, not where the table lives.
	if got := est("", true); got != want {
		t.Errorf("direct I/O gave %.4f, want %.4f", got, want)
	}
}

// And it is excluded from the estimate at every size. The old code only
// excluded tables over 4 GiB, mirroring llama.cpp's streaming threshold —
// but that threshold governs host residency, which was never the question.
func TestSmallPLETableAlsoExcluded(t *testing.T) {
	base := Model{
		SizeBytes: 8 * 1024 * 1024 * 1024,
		NLayers:   32, AttnLayers: 32, NEmbd: 2048, NHead: 16, NKVHead: 4,
		KVFullPerTok: 32 * 4 * 128, ContextLength: 4096,
	}
	withTable := base
	withTable.PLEBytes = 2 * 1024 * 1024 * 1024 // well under the 4 GiB threshold
	cfg := &ModelConfig{ContextSize: 4096}

	diff := VRAMEstimateForConfig(&base, cfg) - VRAMEstimateForConfig(&withTable, cfg)
	if diff < 1.9 || diff > 2.1 {
		t.Errorf("a 2 GiB table changed the estimate by %.2f GiB, want the whole 2", diff)
	}
}

// Models without such a table must be estimated exactly as before.
func TestVRAMUnchangedWithoutPLETable(t *testing.T) {
	m := &Model{
		SizeBytes: 8 * 1024 * 1024 * 1024,
		NLayers:   32, NEmbd: 2048, NHead: 16, NKVHead: 4, ContextLength: 4096,
	}
	want := VRAMEstimateForConfig(m, &ModelConfig{ContextSize: 4096})
	for _, mode := range []string{"", "on", "off"} {
		got := VRAMEstimateForConfig(m, &ModelConfig{ContextSize: 4096, PLEMode: mode})
		if got != want {
			t.Errorf("mode %q changed the estimate for a model with no PLE table: %.4f vs %.4f", mode, got, want)
		}
	}
}
