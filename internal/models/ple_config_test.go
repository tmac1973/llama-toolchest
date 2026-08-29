package models

import (
	"strings"
	"testing"
)

// The three UI modes must map onto llama.cpp's --tensor-read-lazy, with
// auto emitting nothing so llama.cpp's own default applies.
func TestPLEModeEmitsTensorReadLazy(t *testing.T) {
	tests := []struct {
		mode string
		want string // "" means the key must be absent
	}{
		{"", ""},
		{"on", "tensor-read-lazy = on"},
		{"off", "tensor-read-lazy = off"},
		{"nonsense", ""},
	}
	for _, tt := range tests {
		t.Run("mode="+tt.mode, func(t *testing.T) {
			m := &Model{ID: "m", Filename: "m.gguf", FilePath: "/models/m.gguf"}
			cfg := &ModelConfig{Enabled: true, PLEMode: tt.mode}
			ini := GeneratePresetINI("/models", []*Model{m}, map[string]*ModelConfig{"m": cfg}, "rocm")
			has := strings.Contains(ini, "tensor-read-lazy")
			if tt.want == "" && has {
				t.Errorf("mode %q emitted a tensor-read-lazy key; preset:\n%s", tt.mode, ini)
			}
			if tt.want != "" && !strings.Contains(ini, tt.want) {
				t.Errorf("mode %q: want %q in preset:\n%s", tt.mode, tt.want, ini)
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
