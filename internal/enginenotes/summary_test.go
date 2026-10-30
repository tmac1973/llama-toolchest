package enginenotes

import (
	"testing"
	"time"
)

func names(id string) (string, string) { return id, "Name of " + id }

func TestWindows(t *testing.T) {
	s, c, _, _ := testStore(t)
	// Counting begins 60 days back.
	c.set(base.Add(-60 * 24 * time.Hour))
	s.DeleteAll()
	for _, ago := range []time.Duration{2 * time.Hour, 3 * 24 * time.Hour, 40 * 24 * time.Hour} {
		c.set(base.Add(-ago))
		s.Record(Event{Kind: KindCrash, ModelID: "m"})
	}
	c.set(base)
	for _, tc := range []struct {
		window time.Duration
		want   int
	}{
		{24 * time.Hour, 1},
		{7 * 24 * time.Hour, 2},
		{30 * 24 * time.Hour, 2},
		{0, 3},
	} {
		sum := s.Summary(tc.window, nil, names)
		if sum.Totals.Crashes != tc.want {
			t.Errorf("window %v: %d crashes, want %d", tc.window, sum.Totals.Crashes, tc.want)
		}
	}
	if from := s.Summary(24*time.Hour, nil, nil).From; !from.Equal(base.Add(-24 * time.Hour)) {
		t.Errorf("24 h from = %v", from)
	}
	if from := s.Summary(0, nil, nil).From; !from.Equal(base.Add(-60 * 24 * time.Hour)) {
		t.Errorf("all from = %v, want counting since", from)
	}
}

// A period longer than the record starts at the record's start.
func TestWindowStartsNoEarlierThanCounting(t *testing.T) {
	s, c, _, _ := testStore(t)
	c.set(base.Add(2 * time.Hour))
	if from := s.Summary(30*24*time.Hour, nil, nil).From; !from.Equal(base) {
		t.Errorf("from = %v, want counting since %v", from, base)
	}
}

func TestModelRows(t *testing.T) {
	s, c, _, _ := testStore(t)
	c.set(base.Add(-30 * time.Hour))
	s.DeleteAll()
	// Outside 24 h: counted only in "all".
	s.Record(Event{Kind: KindLoad, ModelID: "a", ModelName: "Alpha", LoadSeconds: 100})
	s.AddUsage("a", 1000, 100, false)

	c.set(base.Add(-3 * time.Hour))
	s.Record(Event{Kind: KindLoad, ModelID: "a", ModelName: "Alpha", LoadSeconds: 10})
	c.set(base.Add(-2 * time.Hour))
	s.Record(Event{Kind: KindLoad, ModelID: "a", ModelName: "Alpha", LoadSeconds: 20})
	s.AddUsage("a", 50, 5, false)
	s.AddUsage("a", 0, 0, true)
	c.set(base.Add(-time.Hour))
	s.Record(Event{Kind: KindCrash, ModelID: "a", ModelName: "Alpha", LoadedSeconds: 600, JobID: "j"})
	s.Record(Event{Kind: KindLoadFailed, ModelID: "b", ModelName: "Beta"})
	s.Record(Event{Kind: KindUnload, ModelID: "a", ModelName: "Alpha", LoadedSeconds: 60})
	s.Record(Event{Kind: KindRouterStart, StartReason: "job"})
	s.Record(Event{Kind: KindRouterStart, StartReason: "user"})
	s.Record(Event{Kind: KindRouterStop})
	s.Record(Event{Kind: KindRouterExit, RouterError: "signal: killed"})
	c.set(base)

	// "c" is loaded now, since 30 minutes ago, and has no events.
	open := []OpenInstance{{RouterName: "c", Port: "1", ReadySince: base.Add(-30 * time.Minute)}}
	sum := s.Summary(24*time.Hour, open, names)

	want := Totals{RouterStartsUser: 1, RouterStartsJob: 1, RouterStops: 1, RouterExits: 1,
		Loads: 2, FailedLoads: 1, Crashes: 1, Unloads: 1, FailuresInJobs: 1}
	if sum.Totals != want {
		t.Errorf("totals = %+v\nwant %+v", sum.Totals, want)
	}
	if len(sum.Models) != 3 {
		t.Fatalf("models = %+v, want Alpha, Beta and c", sum.Models)
	}
	a, b, cc := sum.Models[0], sum.Models[1], sum.Models[2]
	if a.Name != "Alpha" || b.Name != "Beta" || cc.Name != "Name of c" {
		t.Fatalf("model order = %s, %s, %s", a.Name, b.Name, cc.Name)
	}
	if a.Loads != 2 || a.Crashes != 1 || a.Unloads != 1 || a.AvgLoadSeconds != 15 || a.LoadedSeconds != 660 {
		t.Errorf("Alpha = %+v", a)
	}
	if a.Requests != 2 || a.FailedRequests != 1 || a.PromptTokens != 50 || a.GenTokens != 5 {
		t.Errorf("Alpha usage = %+v", a)
	}
	// Latest load at -2 h; usage in the hour starting at -2 h too.
	if !a.LastUsed.Equal(base.Add(-2 * time.Hour)) {
		t.Errorf("Alpha last used = %v", a.LastUsed)
	}
	if b.FailedLoads != 1 || !b.LastUsed.IsZero() {
		t.Errorf("Beta = %+v, want one failed load and never used", b)
	}
	if cc.LoadedSeconds != 1800 {
		t.Errorf("c loaded = %v s, want 1800", cc.LoadedSeconds)
	}
	if len(sum.Failures) != 2 || sum.Failures[0].Kind != KindCrash && sum.Failures[0].Kind != KindLoadFailed {
		t.Errorf("failures = %+v", sum.Failures)
	}

	all := s.Summary(0, open, names)
	if all.Models[0].Loads != 3 || all.Models[0].PromptTokens != 1050 || all.Models[0].AvgLoadSeconds != 130.0/3 {
		t.Errorf("Alpha over all = %+v", all.Models[0])
	}
}

// A model loaded before the period counts only its time inside it.
func TestOpenInstanceClippedToThePeriod(t *testing.T) {
	s, c, _, _ := testStore(t)
	c.set(base.Add(-48 * time.Hour))
	s.DeleteAll()
	c.set(base)
	open := []OpenInstance{{RouterName: "m", ReadySince: base.Add(-36 * time.Hour)}}
	sum := s.Summary(24*time.Hour, open, nil)
	if len(sum.Models) != 1 || sum.Models[0].LoadedSeconds != 24*3600 {
		t.Errorf("models = %+v, want 24 h loaded", sum.Models)
	}
}

func TestFailuresNewestFirst(t *testing.T) {
	s, c, _, _ := testStore(t)
	for i := 0; i < 3; i++ {
		c.set(base.Add(time.Duration(i) * time.Minute))
		s.Record(Event{Kind: KindCrash, ModelID: "m", ExitStatus: i})
	}
	sum := s.Summary(0, nil, nil)
	for i, f := range sum.Failures {
		if f.ExitStatus != 2-i {
			t.Fatalf("failure %d has status %d, want newest first", i, f.ExitStatus)
		}
	}
}

func TestUsageHourOutsideTheWindow(t *testing.T) {
	s, c, _, _ := testStore(t)
	c.set(base.Add(-48 * time.Hour))
	s.DeleteAll()
	c.set(base.Add(-25 * time.Hour))
	s.AddUsage("m", 5, 5, false)
	c.set(base.Add(-time.Minute))
	s.AddUsage("m", 7, 7, false)
	s.AddUsage("m", 1, 1, false)
	c.set(base)
	sum := s.Summary(24*time.Hour, nil, nil)
	if len(sum.Models) != 1 || sum.Models[0].Requests != 2 || sum.Models[0].PromptTokens != 8 {
		t.Errorf("models = %+v, want the two recent requests summed", sum.Models)
	}
}
