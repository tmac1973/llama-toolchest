package api

import (
	"bytes"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/generation"
)

func TestMeasuredStale(t *testing.T) {
	catalog := ti2vSet()
	custom := generation.ModelSet{ID: "mine", Kind: generation.KindSD, LaunchArgs: []string{"--steps", "20"}}
	ace := generation.ModelSet{ID: "ace", Kind: generation.KindAce, CatalogID: "ace-step-turbo"}
	now := func(set generation.ModelSet, inst generation.Instance) []string { return runArgs(set, inst) }
	for _, tc := range []struct {
		name string
		inst generation.Instance
		set  generation.ModelSet
		want bool
	}{
		{"not measured", generation.Instance{}, catalog, false},
		{"measured with today's flags", generation.Instance{MeasuredPeakMiB: 1, MeasuredArgs: now(catalog, generation.Instance{})}, catalog, false},
		{"measured with older flags", generation.Instance{MeasuredPeakMiB: 1, MeasuredArgs: []string{"--steps", "20"}}, catalog, true},
		{"extra args added since", generation.Instance{MeasuredPeakMiB: 1, MeasuredArgs: now(catalog, generation.Instance{}), ExtraArgs: []string{"-t", "8"}}, catalog, true},
		{"catalog set measured before flags were kept", generation.Instance{MeasuredPeakMiB: 1}, catalog, true},
		{"custom set measured before flags were kept", generation.Instance{MeasuredPeakMiB: 1}, custom, false},
		{"music set measured before flags were kept", generation.Instance{MeasuredPeakMiB: 1}, ace, false},
	} {
		if got := measuredStale(tc.inst, tc.set); got != tc.want {
			t.Errorf("%s: measuredStale = %v, want %v", tc.name, got, tc.want)
		}
	}
}

// The row says when the measurement is out of date, and how much system
// RAM the backend used.
func TestInstanceRowStaleAndRAM(t *testing.T) {
	s := instanceServer(t)
	if err := s.gen.AddModelSet(ti2vSet()); err != nil {
		t.Fatal(err)
	}
	addInstance(t, s, generation.Instance{ID: "ti2v", ModelSetID: "wan22-ti2v-5b", Port: 8090, Enabled: true, ReservedMiB: 26624,
		MeasuredPeakMiB: 24000, MeasuredSuggestedMiB: 26624, MeasuredFrames: 33, MeasuredWidth: 832, MeasuredHeight: 480,
		MeasuredArgs: []string{"--steps", "20"}, MeasuredRAMMiB: 7680})
	var buf bytes.Buffer
	if err := s.pages["server.html"].ExecuteTemplate(&buf, "gen_instances_table", genInstancesData{Rows: s.genInstanceRows(s.gen)}); err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"Measured with older settings; measure again, it may need less memory now", "uses 7.5 GiB of system RAM"} {
		if !strings.Contains(buf.String(), want) {
			t.Errorf("row lacks %q", want)
		}
	}
}
