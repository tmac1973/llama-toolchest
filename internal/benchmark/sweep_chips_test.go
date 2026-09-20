package benchmark

import (
	"reflect"
	"strings"
	"testing"
)

func TestSweepChipsSplitsTheSpeculativeValue(t *testing.T) {
	got := SweepChips(map[string]string{
		"batch_size":      "2048",
		"flash_attention": "true",
		"split_mode":      "layer",
		"ubatch_size":     "256",
		"spec_type":       "draft-mtp+ngram-map-k4v:assist_min_hits=1,assist_size_m=48,assist_size_n=6,draft_max=6,draft_min=0",
	})
	want := []string{
		"batch_size=2048",
		"flash_attention=true",
		"spec_type=draft-mtp+ngram-map-k4v",
		"assist_min_hits=1",
		"assist_size_m=48",
		"assist_size_n=6",
		"draft_max=6",
		"draft_min=0",
		"split_mode=layer",
		"ubatch_size=256",
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("chips =\n%v\nwant\n%v", got, want)
	}
	// The reason for splitting: no single label is long enough to stretch
	// the column past the pane on its own.
	for _, c := range got {
		if len(c) > 40 {
			t.Errorf("label %q is %d characters; it will widen the column", c, len(c))
		}
	}
}

func TestSweepChipsLeavesOtherValuesWhole(t *testing.T) {
	cases := []struct {
		name string
		in   map[string]string
		want []string
	}{
		{"nothing swept", nil, nil},
		{"no settings on the mode", map[string]string{"spec_type": "ngram-cache"},
			[]string{"spec_type=ngram-cache"}},
		{"speculative decoding off", map[string]string{"spec_type": "none"},
			[]string{"spec_type=none"}},
		{"an empty settings list", map[string]string{"spec_type": "draft-mtp:"},
			[]string{"spec_type=draft-mtp"}},
		{"a value that is not a spec type", map[string]string{"gpu_assign": "0-1"},
			[]string{"gpu_assign=0-1"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := SweepChips(tc.in); !reflect.DeepEqual(got, tc.want) {
				t.Errorf("chips = %v, want %v", got, tc.want)
			}
		})
	}
}

// Rendering must never fail on a value this build cannot parse — one
// written by a newer build, say. A label nobody can read still beats a
// table cell that does not render.
func TestSweepChipsRendersAnUnknownValue(t *testing.T) {
	got := SweepChips(map[string]string{"spec_type": "draft-whatever+ngram-new:some_key=3"})
	want := []string{"spec_type=draft-whatever+ngram-new", "some_key=3"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("chips = %v, want %v", got, want)
	}
}

// The order is stable, so two rows of a table can be compared by eye.
func TestSweepChipsOrderIsStable(t *testing.T) {
	in := map[string]string{"ubatch_size": "256", "batch_size": "2048", "spec_type": "draft-mtp:draft_max=6"}
	first := strings.Join(SweepChips(in), "|")
	for i := 0; i < 20; i++ {
		if got := strings.Join(SweepChips(in), "|"); got != first {
			t.Fatalf("order changed between renders: %q then %q", first, got)
		}
	}
}
