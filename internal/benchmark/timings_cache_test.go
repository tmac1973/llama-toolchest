package benchmark

import (
	"math"
	"testing"
)

func cached(s TimingSample, n int) TimingSample {
	s.CacheTokens = &n
	return s
}

// The cache share is token-weighted over the requests that report
// cache_n; the hit count says how many reused the cache at all.
func TestTimingSummaryCache(t *testing.T) {
	s := &Store{timings: map[string][]TimingSample{
		"m": {
			cached(sample("m", 1000, 1, 10, 1), 0),   // a first turn: all processed
			cached(sample("m", 100, 1, 10, 1), 1000), // the next turn: mostly cached
			cached(sample("m", 50, 1, 10, 1), 1100),
			sample("m", 500, 1, 10, 1), // an engine that does not say: left out
		},
		"old": {sample("old", 100, 1, 10, 1)},
	}}
	got := s.TimingSummary()
	m, old := got[0], got[1]
	if m.CacheCounted != 3 || m.CacheHits != 2 {
		t.Errorf("hits = %d/%d, want 2/3", m.CacheHits, m.CacheCounted)
	}
	want := 2100.0 / (2100 + 1150)
	if math.Abs(m.CacheShare-want) > 1e-9 || math.Abs(m.CachePercent()-want*100) > 1e-9 {
		t.Errorf("share = %v (%v%%), want %v", m.CacheShare, m.CachePercent(), want)
	}
	if old.CacheCounted != 0 || old.CacheShare != 0 {
		t.Errorf("a model with no cache_n = %+v, want nothing counted", old)
	}
}

func TestClearTimingsClearsOneModel(t *testing.T) {
	s := &Store{timings: map[string][]TimingSample{}}
	s.RecordTiming(sample("a", 100, 1, 10, 1))
	s.RecordTiming(sample("b", 100, 1, 10, 1))
	if !s.ClearTimings("a") || s.ClearTimings("a") {
		t.Error("ClearTimings should report true once, then false")
	}
	got := s.TimingSummary()
	if len(got) != 1 || got[0].ModelID != "b" {
		t.Errorf("after reset = %+v, want only b", got)
	}
	// New requests fill the row again.
	s.RecordTiming(sample("a", 100, 1, 10, 1))
	if len(s.TimingSummary()) != 2 {
		t.Error("a new sample did not bring the row back")
	}
}
