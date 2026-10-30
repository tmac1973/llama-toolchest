package enginenotes

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"
)

// clock is a settable time for tests.
type clock struct {
	mu sync.Mutex
	t  time.Time
}

func (c *clock) now() time.Time  { c.mu.Lock(); defer c.mu.Unlock(); return c.t }
func (c *clock) set(t time.Time) { c.mu.Lock(); c.t = t; c.mu.Unlock() }

// writes counts the store's writes and passes them to the real writer.
type writes struct {
	mu sync.Mutex
	n  int
}

func (w *writes) write(path string, data []byte) error {
	w.mu.Lock()
	w.n++
	w.mu.Unlock()
	return os.WriteFile(path, data, 0o644)
}

func (w *writes) count() int { w.mu.Lock(); defer w.mu.Unlock(); return w.n }

// testStore opens a store in a temporary folder that saves only when
// told to (the ticker is an hour), at a clock set to base.
func testStore(t *testing.T) (*Store, *clock, *writes, string) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "engine-notes.json")
	c := &clock{t: base}
	w := &writes{}
	s, err := open(path, c.now, time.Hour, w.write)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.Close() })
	return s, c, w, path
}

func TestRoundTrip(t *testing.T) {
	s, c, w, path := testStore(t)
	s.Record(Event{Kind: KindRouterStart, StartReason: "user", Build: "v0.8.0"})
	s.Record(Event{Kind: KindLoad, ModelID: "qwen", ModelName: "Qwen", LoadSeconds: 12})
	s.Record(Event{Kind: KindCrash, ModelID: "qwen", ModelName: "Qwen", ExitStatus: 1,
		ErrorLines: []string{"CUDA error"}, Tail: []string{"a", "b"}, JobID: "job-1", JobName: "night run"})
	s.AddUsage("qwen", 100, 20, false)
	s.AddUsage("qwen", 0, 0, true)
	c.set(base.Add(time.Minute))
	before := s.Summary(0, nil, nil)
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	if w.count() != 1 {
		t.Errorf("writes = %d, want 1 on close", w.count())
	}

	again, err := open(path, c.now, time.Hour, w.write)
	if err != nil {
		t.Fatal(err)
	}
	defer again.Close()
	if after := again.Summary(0, nil, nil); !reflect.DeepEqual(before, after) {
		t.Errorf("after reopening:\n%+v\nwant\n%+v", after, before)
	}
}

func TestUnreadableFileIsMovedAside(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "engine-notes.json")
	if err := os.WriteFile(path, []byte("{not json"), 0o644); err != nil {
		t.Fatal(err)
	}
	c := &clock{t: base}
	s, err := open(path, c.now, time.Hour, (&writes{}).write)
	if err != nil {
		t.Fatalf("open with a bad file: %v", err)
	}
	defer s.Close()
	if sum := s.Summary(0, nil, nil); len(sum.Models) != 0 || !sum.CountingSince.Equal(base) {
		t.Errorf("summary = %+v, want an empty store counting from now", sum)
	}
	bad, _ := filepath.Glob(path + ".bad-*")
	if len(bad) != 1 {
		t.Fatalf("moved-aside files = %v, want one", bad)
	}
	if raw, _ := os.ReadFile(bad[0]); string(raw) != "{not json" {
		t.Errorf("moved-aside file holds %q", raw)
	}
}

func TestUnknownVersionIsMovedAside(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "engine-notes.json")
	os.WriteFile(path, []byte(`{"version": 99, "events": []}`), 0o644)
	s, err := open(path, (&clock{t: base}).now, time.Hour, (&writes{}).write)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if bad, _ := filepath.Glob(path + ".bad-*"); len(bad) != 1 {
		t.Errorf("moved-aside files = %v, want one", bad)
	}
}

func TestFailuresHaveTheirOwnCap(t *testing.T) {
	s, c, _, _ := testStore(t)
	for i := 0; i < maxCrashEvents+20; i++ {
		c.set(base.Add(time.Duration(i) * time.Second))
		s.Record(Event{Kind: KindCrash, ModelID: "m", ExitStatus: i})
	}
	sum := s.Summary(0, nil, nil)
	if len(sum.Failures) != maxCrashEvents {
		t.Fatalf("kept %d crashes, want %d", len(sum.Failures), maxCrashEvents)
	}
	// Newest first, and the oldest 20 are the ones gone.
	if sum.Failures[0].ExitStatus != maxCrashEvents+19 || sum.Failures[len(sum.Failures)-1].ExitStatus != 20 {
		t.Errorf("kept crashes %d..%d, want %d..20", sum.Failures[0].ExitStatus,
			sum.Failures[len(sum.Failures)-1].ExitStatus, maxCrashEvents+19)
	}
}

// Over the total cap, other events go before any failure does.
func TestTotalCapKeepsFailures(t *testing.T) {
	s, c, _, _ := testStore(t)
	for i := 0; i < 50; i++ {
		c.set(base.Add(time.Duration(i) * time.Second))
		s.Record(Event{Kind: KindCrash, ModelID: "m"})
	}
	for i := 0; i < maxEvents+100; i++ {
		c.set(base.Add(time.Hour + time.Duration(i)*time.Second))
		s.Record(Event{Kind: KindLoad, ModelID: "m"})
	}
	sum := s.Summary(0, nil, nil)
	if sum.Totals.Crashes != 50 {
		t.Errorf("crashes = %d, want all 50 kept", sum.Totals.Crashes)
	}
	if got := sum.Totals.Crashes + sum.Totals.Loads; got != maxEvents {
		t.Errorf("events kept = %d, want %d", got, maxEvents)
	}
}

func TestRecordKeepsLinesOnlyOnFailures(t *testing.T) {
	s, _, _, _ := testStore(t)
	long := make([]string, 50)
	for i := range long {
		long[i] = strings.Repeat("x", i+1)
	}
	s.Record(Event{Kind: KindUnload, ModelID: "m", ErrorLines: []string{"e"}, Tail: []string{"t"}})
	s.Record(Event{Kind: KindCrash, ModelID: "m", ErrorLines: long, Tail: long})
	s.mu.Lock()
	unload, crash := s.data.Events[0], s.data.Events[1]
	s.mu.Unlock()
	if unload.Tail != nil || unload.ErrorLines != nil {
		t.Errorf("an unload kept lines: %+v", unload)
	}
	if len(crash.Tail) != tailLines || crash.Tail[len(crash.Tail)-1] != long[49] {
		t.Errorf("crash tail = %d lines ending %q, want the last %d", len(crash.Tail), crash.Tail[len(crash.Tail)-1], tailLines)
	}
	if len(crash.ErrorLines) != 12 || crash.ErrorLines[0] != long[0] {
		t.Errorf("crash error lines = %d starting %q, want the first 12", len(crash.ErrorLines), crash.ErrorLines[0])
	}
	if crash.Source != SourceLlamaServer {
		t.Errorf("source = %q, want %q", crash.Source, SourceLlamaServer)
	}
}

func TestDeleteOlderThan(t *testing.T) {
	s, c, w, _ := testStore(t)
	c.set(base.Add(-60 * 24 * time.Hour))
	s.DeleteAll()
	c.set(base.Add(-40 * 24 * time.Hour))
	s.Record(Event{Kind: KindLoad, ModelID: "old"})
	s.AddUsage("old", 10, 1, false)
	c.set(base.Add(-2 * time.Hour))
	s.Record(Event{Kind: KindLoad, ModelID: "new"})
	s.AddUsage("new", 10, 1, false)
	c.set(base)
	start := s.Summary(0, nil, nil).CountingSince

	events, hours, err := s.DeleteOlderThan(base.Add(-30 * 24 * time.Hour))
	if err != nil {
		t.Fatal(err)
	}
	if events != 1 || hours != 1 {
		t.Errorf("removed %d events and %d hours, want 1 and 1", events, hours)
	}
	if w.count() != 2 {
		t.Errorf("writes = %d, want a save at once (after the one for DeleteAll)", w.count())
	}
	sum := s.Summary(0, nil, nil)
	if len(sum.Models) != 1 || sum.Models[0].ModelID != "new" {
		t.Errorf("models after delete = %+v, want only new", sum.Models)
	}
	if !sum.CountingSince.Equal(base.Add(-30*24*time.Hour)) || sum.CountingSince.Equal(start) {
		t.Errorf("counting since = %v, want the cutoff", sum.CountingSince)
	}
}

func TestDeleteAll(t *testing.T) {
	s, c, _, _ := testStore(t)
	s.Record(Event{Kind: KindLoad, ModelID: "m"})
	s.AddUsage("m", 1, 1, false)
	c.set(base.Add(time.Hour))
	events, hours, err := s.DeleteAll()
	if err != nil || events != 1 || hours != 1 {
		t.Fatalf("DeleteAll = %d, %d, %v; want 1, 1, nil", events, hours, err)
	}
	sum := s.Summary(0, nil, nil)
	if len(sum.Models) != 0 || !sum.CountingSince.Equal(base.Add(time.Hour)) {
		t.Errorf("after delete all: %+v", sum)
	}
}

// Many changes between ticks make one write.
func TestSavesAreBatched(t *testing.T) {
	path := filepath.Join(t.TempDir(), "engine-notes.json")
	w := &writes{}
	s, err := open(path, (&clock{t: base}).now, 50*time.Millisecond, w.write)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 10; i++ {
		s.AddUsage("m", 1, 1, false)
	}
	deadline := time.Now().Add(2 * time.Second)
	for w.count() == 0 && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	time.Sleep(150 * time.Millisecond) // a few more ticks with nothing new
	if w.count() != 1 {
		t.Errorf("writes = %d, want 1 for ten changes and idle ticks", w.count())
	}
	s.Close()
	if w.count() != 1 {
		t.Errorf("writes after close = %d, want no write when nothing changed", w.count())
	}
}
