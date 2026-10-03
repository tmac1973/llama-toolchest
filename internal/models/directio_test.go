package models

import (
	"strings"
	"testing"
)

// Direct I/O is written as the build spells it. llama.cpp folded
// --direct-io into --load-mode at b10105 and removed the old flag at
// b10875; the router rejects a preset with an option it does not know
// ("option 'direct-io' not recognized in preset ...") and then serves no
// model at all, so a current build must never see "direct-io".
func TestDirectIOEmitsTheBuildsLoadOption(t *testing.T) {
	tests := []struct {
		name    string
		version int
		ini     string
		flags   string
	}{
		{"current llama.cpp", 11364, "load-mode = dio", "--load-mode dio"},
		{"build of unknown version", 0, "load-mode = dio", "--load-mode dio"},
		{"first build without --direct-io", 10875, "load-mode = dio", "--load-mode dio"},
		{"build with both spellings", 10500, "load-mode = dio", "--load-mode dio"},
		{"first build with --load-mode", 10105, "load-mode = dio", "--load-mode dio"},
		{"build before --load-mode", 10104, "direct-io = true", "--direct-io"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := &Model{ID: "m", Filename: "m.gguf", FilePath: "/models/m.gguf"}
			cfg := &ModelConfig{Enabled: true, DirectIO: true}
			target := Target{Backend: "rocm", Version: tt.version}
			ini := GeneratePresetINI("/models", []*Model{m}, map[string]*ModelConfig{"m": cfg}, target)
			if !strings.Contains(ini, tt.ini) {
				t.Errorf("want %q in preset:\n%s", tt.ini, ini)
			}
			if tt.ini != "direct-io = true" && strings.Contains(ini, "direct-io") {
				t.Errorf("preset still carries direct-io:\n%s", ini)
			}
			flags := cfg.EffectiveFlagsFor(false, target)
			if !strings.Contains(flags, tt.flags) {
				t.Errorf("want %q in flags: %s", tt.flags, flags)
			}
			if got := strings.Join(target.DirectIOFlags(), " "); got != tt.flags {
				t.Errorf("DirectIOFlags = %q, want %q", got, tt.flags)
			}
		})
	}
}

// Off writes nothing: llama.cpp's own default (auto — mmap unless a
// device cannot use it) applies.
func TestDirectIOOffWritesNoLoadOption(t *testing.T) {
	m := &Model{ID: "m", Filename: "m.gguf", FilePath: "/models/m.gguf"}
	cfg := &ModelConfig{Enabled: true}
	ini := GeneratePresetINI("/models", []*Model{m}, map[string]*ModelConfig{"m": cfg}, Target{})
	for _, key := range []string{"load-mode", "direct-io"} {
		if strings.Contains(ini, key) {
			t.Errorf("emitted %s with Direct I/O off:\n%s", key, ini)
		}
	}
}
