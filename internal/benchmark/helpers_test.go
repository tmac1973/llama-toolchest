package benchmark

import (
	"fmt"
	"reflect"
	"slices"
	"testing"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/monitor"
)

// covProfile is a job's copy of a saved profile with the given context
// size, copied at the given time.
func covProfile(model, name string, ctx int, copied time.Time) JobProfile {
	return JobProfile{
		ModelID: model, Name: name,
		Config:   models.ModelConfig{ContextSize: ctx, GPULayers: 999},
		SavedAt:  copied.Add(-time.Hour),
		CopiedAt: copied,
	}
}

// After a job edit, a profile copy whose settings did not change keeps
// the old copy, so its copy time still says when those settings were
// taken. A copy with changed settings, or one for another model, is the
// new copy. The result follows the new list: profiles left out of the
// edit are gone.
func TestKeepUnchangedCopies(t *testing.T) {
	t0 := time.Date(2026, 10, 1, 0, 0, 0, 0, time.UTC)
	t1 := t0.Add(24 * time.Hour)
	prev := []JobProfile{
		covProfile("m", "fast", 8192, t0),
		covProfile("m", "long", 32768, t0),
		covProfile("m", "dropped", 4096, t0),
	}
	next := []JobProfile{
		covProfile("m2", "fast", 8192, t1), // same name and settings, other model
		covProfile("m", "long", 65536, t1), // settings changed
		covProfile("m", "fast", 8192, t1),  // unchanged
	}
	// The new copy's SavedAt differs; only the settings are compared.
	next[2].SavedAt = t1

	got := keepUnchangedCopies(prev, next)
	if len(got) != 3 {
		t.Fatalf("got %d copies, want 3: %+v", len(got), got)
	}
	if got[0].ModelID != "m2" || !got[0].CopiedAt.Equal(t1) {
		t.Errorf("other model's copy = %+v, want the new copy", got[0])
	}
	if got[1].Config.ContextSize != 65536 || !got[1].CopiedAt.Equal(t1) {
		t.Errorf("changed copy = %+v, want the new settings and copy time", got[1])
	}
	if !reflect.DeepEqual(got[2], prev[0]) {
		t.Errorf("unchanged copy = %+v, want the previous copy %+v", got[2], prev[0])
	}

	if out := keepUnchangedCopies(prev, nil); len(out) != 0 {
		t.Errorf("no new profiles = %+v, want none", out)
	}
	if out := keepUnchangedCopies(nil, next); !reflect.DeepEqual(out, next) {
		t.Errorf("no previous profiles = %+v, want the new list unchanged", out)
	}
}

// keepUnchangedCopies must not change the caller's slice: the job
// definition passed to an edit is also what the caller keeps.
func TestKeepUnchangedCopiesDoesNotChangeItsInput(t *testing.T) {
	t0 := time.Date(2026, 10, 1, 0, 0, 0, 0, time.UTC)
	prev := []JobProfile{covProfile("m", "fast", 8192, t0)}
	next := []JobProfile{covProfile("m", "fast", 8192, t0.Add(time.Hour))}
	want := slices.Clone(next)
	keepUnchangedCopies(prev, next)
	if !reflect.DeepEqual(next, want) {
		t.Errorf("input changed to %+v", next)
	}
}

// profileConfigKey compares settings only: two copies with the same
// settings and different names or times have the same key, and any
// change of a setting changes the key.
func TestProfileConfigKey(t *testing.T) {
	t0 := time.Date(2026, 10, 1, 0, 0, 0, 0, time.UTC)
	a := covProfile("m", "fast", 8192, t0)
	b := covProfile("other", "slow", 8192, t0.Add(time.Hour))
	if profileConfigKey(&a) != profileConfigKey(&b) {
		t.Errorf("same settings, different keys: %q / %q", profileConfigKey(&a), profileConfigKey(&b))
	}
	c := a
	c.Config.TensorSplit = "1,1"
	if profileConfigKey(&a) == profileConfigKey(&c) {
		t.Error("a changed tensor split kept the same key")
	}
	if profileConfigKey(&a) == "" {
		t.Error("key is empty")
	}
}

// identifyWithProfiles adds the settings of the cell's profile copy to
// the cell identity. A completed cell must keep its result after an edit
// that leaves the profile's settings alone (even with a new copy time),
// and must run again when the settings change. Cells without a profile,
// or naming a profile the job has no copy of, get no profile settings.
func TestIdentifyWithProfiles(t *testing.T) {
	t0 := time.Date(2026, 10, 1, 0, 0, 0, 0, time.UTC)
	cell := JobCell{ModelID: "m", BuildID: "b1", Preset: "internal-quick", Profile: "fast",
		SweepValues: map[string]string{"batch_size": "512"}}
	before := []JobProfile{covProfile("m", "fast", 8192, t0)}
	recopied := []JobProfile{covProfile("m", "fast", 8192, t0.Add(time.Hour))}
	changed := []JobProfile{covProfile("m", "fast", 16384, t0.Add(time.Hour))}

	id := identifyWithProfiles(cell, nil, before)
	if id.ProfileConfig == "" || id.ProfileConfig != profileConfigKey(&before[0]) {
		t.Errorf("ProfileConfig = %q, want the copy's settings", id.ProfileConfig)
	}
	if id != identifyWithProfiles(cell, nil, recopied) {
		t.Error("a new copy with the same settings changed the identity")
	}
	if id == identifyWithProfiles(cell, nil, changed) {
		t.Error("changed profile settings kept the same identity")
	}
	// An empty overrides struct means the same as none.
	if id != identifyWithProfiles(cell, &ConfigOverrides{}, before) {
		t.Error("empty overrides changed the identity")
	}
	ctx := 4096
	if id == identifyWithProfiles(cell, &ConfigOverrides{ContextSize: &ctx}, before) {
		t.Error("a fixed override did not change the identity")
	}

	// A copy for another model with the same profile name does not count.
	other := []JobProfile{covProfile("m2", "fast", 8192, t0)}
	if got := identifyWithProfiles(cell, nil, other).ProfileConfig; got != "" {
		t.Errorf("ProfileConfig from another model's copy = %q", got)
	}

	current := cell
	current.Profile = ""
	if got := identifyWithProfiles(current, nil, before); got.ProfileConfig != "" || got.Profile != "" {
		t.Errorf("cell without a profile = %+v, want no profile settings", got)
	}
}

// RecordTiming keeps at most maxTimingSamples samples per model, dropping
// the oldest first, and one model's samples never push out another's.
func TestRecordTimingCapsEachModel(t *testing.T) {
	s := &Store{timings: map[string][]TimingSample{}}
	start := time.Date(2026, 10, 1, 0, 0, 0, 0, time.UTC)
	extra := 5
	for i := range maxTimingSamples + extra {
		s.RecordTiming(TimingSample{ModelID: "busy", Timestamp: start.Add(time.Duration(i) * time.Second), GenTokens: i})
	}
	s.RecordTiming(TimingSample{ModelID: "quiet", Timestamp: start, GenTokens: 1})

	busy := s.Timings("busy")
	if len(busy) != maxTimingSamples {
		t.Fatalf("busy model keeps %d samples, want %d", len(busy), maxTimingSamples)
	}
	if busy[0].GenTokens != extra || busy[len(busy)-1].GenTokens != maxTimingSamples+extra-1 {
		t.Errorf("kept samples %d..%d, want %d..%d (oldest dropped first)",
			busy[0].GenTokens, busy[len(busy)-1].GenTokens, extra, maxTimingSamples+extra-1)
	}
	if quiet := s.Timings("quiet"); len(quiet) != 1 || quiet[0].GenTokens != 1 {
		t.Errorf("quiet model = %+v", quiet)
	}
}

// Timings for one model keeps the order recorded and returns a copy;
// Timings for all models merges every model newest first. An unknown
// model has no samples.
func TestTimingsOrderAndCopy(t *testing.T) {
	s := &Store{timings: map[string][]TimingSample{}}
	at := func(min int) time.Time { return time.Date(2026, 10, 1, 12, min, 0, 0, time.UTC) }
	for _, ts := range []struct {
		model string
		min   int
	}{{"a", 1}, {"b", 2}, {"a", 5}, {"c", 3}, {"b", 4}} {
		s.RecordTiming(TimingSample{ModelID: ts.model, Timestamp: at(ts.min)})
	}

	a := s.Timings("a")
	if len(a) != 2 || !a[0].Timestamp.Equal(at(1)) || !a[1].Timestamp.Equal(at(5)) {
		t.Errorf("Timings(a) = %+v, want minutes 1 then 5", a)
	}
	a[0].ModelID = "changed"
	if s.Timings("a")[0].ModelID != "a" {
		t.Error("changing the returned slice changed the store")
	}

	all := s.Timings("")
	var got []string
	for _, x := range all {
		got = append(got, fmt.Sprintf("%s%d", x.ModelID, x.Timestamp.Minute()))
	}
	if want := []string{"a5", "b4", "c3", "b2", "a1"}; !slices.Equal(got, want) {
		t.Errorf("Timings(\"\") = %v, want %v (newest first across models)", got, want)
	}

	if none := s.Timings("missing"); len(none) != 0 {
		t.Errorf("unknown model = %+v", none)
	}
}

// LeastFreeCard is the card with the least memory left, measured as
// free memory and not as used memory: a large card with more in use can
// still have more left. A tie keeps the first card, and a run without
// card readings gives nil so the template shows nothing.
func TestLeastFreeCard(t *testing.T) {
	if c := (BenchmarkRun{}).LeastFreeCard(); c != nil {
		t.Errorf("no cards = %+v, want nil", c)
	}
	run := BenchmarkRun{Cards: []CardMemory{
		{Index: 0, UsedMiB: 20000, TotalMiB: 24576}, // 4576 free
		{Index: 1, UsedMiB: 14000, TotalMiB: 16384}, // 2384 free
		{Index: 2, UsedMiB: 6000, TotalMiB: 8384},   // 2384 free, a tie
	}}
	c := run.LeastFreeCard()
	if c == nil || c.Index != 1 || c.FreeMiB() != 2384 {
		t.Errorf("LeastFreeCard = %+v, want GPU 1 with 2384 MiB free", c)
	}
	// The result is a copy; changing it leaves the run alone.
	c.UsedMiB = 0
	if run.Cards[1].UsedMiB != 14000 {
		t.Error("changing the result changed the run")
	}
}

// TooFull marks a card with less than MinCardFreeMiB left. Exactly the
// limit is enough, and a reading with more used than total (as some
// drivers report) is too full.
func TestTooFull(t *testing.T) {
	cases := []struct {
		used, total int
		want        bool
	}{
		{16384 - MinCardFreeMiB + 1, 16384, true},
		{16384 - MinCardFreeMiB, 16384, false},
		{0, 16384, false},
		{17000, 16384, true},
		{0, 0, true},
	}
	for _, c := range cases {
		card := CardMemory{UsedMiB: c.used, TotalMiB: c.total}
		if got := card.TooFull(); got != c.want {
			t.Errorf("used %d of %d: TooFull = %v, want %v", c.used, c.total, got, c.want)
		}
	}
}

// GPUSnapshotsFromMetrics copies each GPU's index, name and total memory
// in the monitor's order. The index is the GPU's own, not its position,
// and the memory number keeps its value (the monitor's "MB" is MiB).
func TestGPUSnapshotsFromMetrics(t *testing.T) {
	if got := GPUSnapshotsFromMetrics(monitor.Metrics{}); got == nil || len(got) != 0 {
		t.Errorf("no GPUs = %#v, want an empty list", got)
	}
	m := monitor.Metrics{GPU: []monitor.GPUInfo{
		{Index: 2, Name: "AMD Radeon AI PRO R9700", VRAMTotalMB: 32624, VRAMUsedMB: 100},
		{Index: 0, Name: "AMD Radeon Graphics", VRAMTotalMB: 512, IsIGPU: true},
	}}
	got := GPUSnapshotsFromMetrics(m)
	want := []GPUSnapshot{
		{Index: 2, Name: "AMD Radeon AI PRO R9700", VRAMTotalMiB: 32624},
		{Index: 0, Name: "AMD Radeon Graphics", VRAMTotalMiB: 512},
	}
	if !slices.Equal(got, want) {
		t.Errorf("GPUSnapshotsFromMetrics = %+v, want %+v", got, want)
	}
}
