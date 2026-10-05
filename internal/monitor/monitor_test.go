package monitor

import (
	"testing"
	"time"
)

// A new subscriber starts with the latest snapshot instead of waiting a
// whole poll interval, and Current reads the same snapshot.
func TestSubscriberGetsTheLatestSnapshot(t *testing.T) {
	m := New(time.Hour)
	if !m.Current().Timestamp.IsZero() {
		t.Fatal("Current before the first poll should be empty")
	}
	snap := Metrics{Timestamp: time.Unix(100, 0), Memory: MemoryInfo{UsedMB: 1, TotalMB: 2}}
	m.metrics.Broadcast(snap)

	if got := m.Current(); !got.Timestamp.Equal(snap.Timestamp) {
		t.Errorf("Current = %v, want the broadcast snapshot", got.Timestamp)
	}
	ch := m.Subscribe()
	defer m.Unsubscribe(ch)
	select {
	case got := <-ch:
		if !got.Timestamp.Equal(snap.Timestamp) {
			t.Errorf("first value = %v, want the latest snapshot", got.Timestamp)
		}
	default:
		t.Error("a new subscriber got nothing until the next poll")
	}
}
