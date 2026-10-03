package models

import (
	"strings"
	"testing"
)

// --reasoning-preserve arrived at b9837; an older build must never see
// it, or the router stops. Default writes nothing, so llama.cpp's own
// default (keep, from b10763) applies.
func TestReasoningPreserveIsWrittenForBuildsThatHaveIt(t *testing.T) {
	tests := []struct {
		name    string
		version int
		mode    string
		ini     string // "" = no reasoning-preserve key
		flag    string // "" = no flag
	}{
		{"keep, current", 11364, "on", "reasoning-preserve = true", "--reasoning-preserve"},
		{"drop, current", 11364, "off", "reasoning-preserve = false", "--no-reasoning-preserve"},
		{"drop, unknown version", 0, "off", "reasoning-preserve = false", "--no-reasoning-preserve"},
		{"drop, first build with it", 9837, "off", "reasoning-preserve = false", "--no-reasoning-preserve"},
		{"default writes nothing", 11364, "", "", ""},
		{"build before the option", 9836, "on", "", ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := &Model{ID: "m", Filename: "m.gguf", FilePath: "/models/m.gguf"}
			cfg := &ModelConfig{Enabled: true, ReasoningPreserve: tt.mode}
			target := Target{Backend: "rocm", Version: tt.version}
			ini := GeneratePresetINI("/models", []*Model{m}, map[string]*ModelConfig{"m": cfg}, target)
			flags := cfg.EffectiveFlagsFor(false, target)
			if tt.ini == "" {
				if strings.Contains(ini, "reasoning-preserve") || strings.Contains(flags, "reasoning-preserve") {
					t.Errorf("want no reasoning-preserve:\n%s\n%s", ini, flags)
				}
				return
			}
			if !strings.Contains(ini, tt.ini) {
				t.Errorf("want %q in preset:\n%s", tt.ini, ini)
			}
			if !strings.Contains(flags, tt.flag) {
				t.Errorf("want %q in flags: %s", tt.flag, flags)
			}
		})
	}
}

func TestReasoningPreservedByDefaultFromB10763(t *testing.T) {
	for v, want := range map[int]bool{0: true, 10763: true, 10762: false, 10453: false} {
		if got := (Target{Version: v}).ReasoningPreservedByDefault(); got != want {
			t.Errorf("version %d: got %v, want %v", v, got, want)
		}
	}
}
