package benchmark

import "testing"

// The two axes added for the VRAM-estimate investigation (plan/ple-vram-findings.md).
// Both are load-time settings, so the only thing that makes them useful is
// reaching the preset INI — a sweep that parsed but never changed the
// launched config would report several labels for one measurement.
func TestPLEModeSweepReachesSnapshot(t *testing.T) {
	for _, tt := range []struct{ raw, want string }{
		{"auto", ""}, // llama.cpp's default: emit no flag at all
		{"on", "on"},
		{"off", "off"},
	} {
		var o ConfigOverrides
		if err := sweepFields["ple_mode"].set(&o, tt.raw); err != nil {
			t.Fatalf("%s: %v", tt.raw, err)
		}
		if got := applyOverrides(ConfigSnapshot{}, &o).PLEMode; got != tt.want {
			t.Errorf("ple_mode %q reached the snapshot as %q, want %q", tt.raw, got, tt.want)
		}
	}
}

func TestPLEModeSweepRejectsNonsense(t *testing.T) {
	var o ConfigOverrides
	if err := sweepFields["ple_mode"].set(&o, "resident"); err == nil {
		t.Error("accepted a mode llama.cpp does not have; a typo would silently run the model's saved setting under a wrong label")
	}
}

// Flag values contain commas of their own, so the axis splits on "|".
// Splitting on "," would shred one --override-tensor pattern list into
// several bogus sweep points.
func TestExtraFlagsSweepKeepsCommasIntact(t *testing.T) {
	f := sweepFields["extra_flags"]
	if f.Separator != "|" {
		t.Fatalf("separator = %q, want |", f.Separator)
	}
	const raw = "--override-tensor per_layer_token_embd=CPU,blk.1=CPU --load-mode none"
	var o ConfigOverrides
	if err := f.set(&o, raw); err != nil {
		t.Fatal(err)
	}
	if got := applyOverrides(ConfigSnapshot{}, &o).ExtraFlags; got != raw {
		t.Errorf("extra_flags reached the snapshot as %q, want it verbatim", got)
	}
}
