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

// A streamed table is never resident, so it must not be counted toward the
// VRAM estimate — the bug this fixes had a 30 GiB table inflating the
// estimate past every consumer GPU.
func TestVRAMExcludesStreamedPLETable(t *testing.T) {
	// 40 GiB file, 30 GiB of which is the PLE table.
	m := &Model{
		SizeBytes: 40 * 1024 * 1024 * 1024,
		PLEBytes:  testPLEBytes,
		NLayers:   48, NEmbd: 4096, NHead: 32, NKVHead: 8, ContextLength: 4096,
	}
	base := func(mode string, directIO bool) float64 {
		return VRAMEstimateForConfig(m, &ModelConfig{ContextSize: 4096, PLEMode: mode, DirectIO: directIO})
	}

	auto, on, off := base("", false), base("on", false), base("off", false)
	if auto != on {
		t.Errorf("auto (%.2f) should match on (%.2f) for a table over 4 GiB", auto, on)
	}
	if got := off - on; got < 29 || got > 31 {
		t.Errorf("off minus on = %.2f GiB, want ~30 (the table)", got)
	}
	// Streaming needs mmap, so direct I/O has to count the table again.
	if dio := base("", true); dio != off {
		t.Errorf("direct I/O estimate %.2f should match resident %.2f", dio, off)
	}
}

// Under auto, llama.cpp keeps tables below 4 GiB resident, so a small one
// must still be counted.
func TestVRAMCountsSmallPLETableUnderAuto(t *testing.T) {
	m := &Model{
		SizeBytes: 8 * 1024 * 1024 * 1024,
		PLEBytes:  2 * 1024 * 1024 * 1024, // under the 4 GiB threshold
		NLayers:   32, NEmbd: 2048, NHead: 16, NKVHead: 4, ContextLength: 4096,
	}
	auto := VRAMEstimateForConfig(m, &ModelConfig{ContextSize: 4096})
	off := VRAMEstimateForConfig(m, &ModelConfig{ContextSize: 4096, PLEMode: "off"})
	if auto != off {
		t.Errorf("auto (%.2f) should match off (%.2f) below the 4 GiB threshold", auto, off)
	}
	// ...but an explicit "on" streams it regardless of size.
	if on := VRAMEstimateForConfig(m, &ModelConfig{ContextSize: 4096, PLEMode: "on"}); on >= auto {
		t.Errorf("explicit on (%.2f) should be below auto (%.2f)", on, auto)
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
