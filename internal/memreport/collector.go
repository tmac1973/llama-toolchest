package memreport

import (
	"regexp"
	"sync"
	"time"
)

// spawnRE matches the router's announcement that it has started a child
// process for a model, which is the only line tying an instance back to
// the model it serves:
//
//	0.05.123.456 I srv          load: spawning server instance with name=Qwen3.6-27B on port 46231
//
// The router logs this itself rather than through a model instance, at
// LOG_INF — verbosity 3, llama.cpp's default. The buffer report the
// instance then prints needs verbosity 4, so a stream can carry the
// mapping with nothing to map, but never the reverse.
//
// The name is matched greedily up to the port so a model whose id
// contains " on port " still resolves.
var spawnRE = regexp.MustCompile(`spawning server instance with name=(.+) on port (\d+)\s*$`)

// readyRE matches the line a child writes when it has finished loading
// and is ready to serve:
//
//	[46231] cmd_child_to_router:state:{"state":"ready","payload":{...}}
//
// The router forwards every line a child prints before checking whether
// it is one of these control messages, so it arrives in the log stream
// like any other. It is the end of the load — and so the moment at which
// the card counters are worth reading.
//
// A model waking from sleep reports ready again without a spawn. That
// leaves the first ready standing rather than replacing it, which is the
// honest reading: the load being described is still the one measured.
var readyRE = regexp.MustCompile(`^\[\s*(\d+)\] cmd_child_to_router:state:.*"state"\s*:\s*"ready"`)

// Measurement is what one model load allocated, as llama.cpp reported it
// while loading.
type Measurement struct {
	// Model is the router's name for the model, which is the registry id:
	// preset sections are written under it (models.GeneratePresetINI).
	Model string
	// Port is the child instance's port, the number the router prefixes
	// onto every line that instance logs. Instances are what the log
	// separates; the port is how.
	Port string
	// At is when the spawn line was seen; Ready is when the instance
	// reported itself loaded, zero while it is still loading. Between the
	// two is the window in which this model, and ideally nothing else,
	// was claiming memory.
	At    time.Time
	Ready time.Time
	// Note is whatever the caller attached at spawn time — the
	// configuration in force for this load, in practice. Left to the
	// caller because what makes two loads comparable is a question about
	// launch configuration, which this package deliberately knows nothing
	// about.
	Note any
	// Report holds the buffer lines this instance logged, unscaled: a
	// tensor-parallel load reports per card, and turning that into totals
	// needs a card count only the caller has (Report.ScaleAggregates).
	Report Report
}

// Collector turns the router's combined log stream into one Measurement
// per model, which is the only way to see what a load really cost rather
// than what it was predicted to cost.
//
// Feed it every line: it picks out the two kinds it needs and ignores the
// rest. Safe for concurrent use — the reader goroutine writes while
// request handlers read.
type Collector struct {
	mu sync.Mutex
	// byPort is the in-flight mapping: buffer lines carry a port and
	// nothing else, so this is what attributes them. Reset by a router
	// restart, which reallocates ports.
	byPort map[string]*Measurement
	// byModel keeps the most recent load per model, including loads whose
	// instance has since exited. A measurement outlives its instance
	// because it is still the last thing known about that model.
	byModel map[string]*Measurement
	now     func() time.Time
}

// A nil *Collector is a working no-op: every method below tolerates one.
// Collection is observation — a server assembled without a collector
// should still start its router and render its pages, rather than fault
// on a code path that has nothing to do with memory.

// NewCollector returns an empty collector.
func NewCollector() *Collector {
	return &Collector{
		byPort:  map[string]*Measurement{},
		byModel: map[string]*Measurement{},
		now:     time.Now,
	}
}

// EventKind is what a log line turned out to mean.
type EventKind int

const (
	// EventNone is every line that is not a boundary of a load, buffer
	// reports included: those are collected, not announced.
	EventNone EventKind = iota
	// EventSpawned is a load starting.
	EventSpawned
	// EventReady is a load finishing.
	EventReady
)

// Event reports a boundary of a model load. Both boundaries matter to a
// caller measuring from outside — the card counters have to be read
// before the first allocation and after the last.
type Event struct {
	Kind  EventKind
	Model string
	Port  string
}

// Add feeds one log line and reports whether it was the start or the end
// of a load. Buffer lines in between are collected silently.
func (c *Collector) Add(line string) Event {
	if c == nil {
		return Event{}
	}
	if m := spawnRE.FindStringSubmatch(line); m != nil {
		name, port := m[1], m[2]
		c.mu.Lock()
		defer c.mu.Unlock()
		// A fresh record rather than a reset one: the previous
		// measurement for this model may still be the answer to a
		// request being served right now, and callers hold copies.
		ld := &Measurement{Model: name, Port: port, At: c.now()}
		c.byPort[port] = ld
		c.byModel[name] = ld
		return Event{Kind: EventSpawned, Model: name, Port: port}
	}

	if m := readyRE.FindStringSubmatch(line); m != nil {
		port := m[1]
		c.mu.Lock()
		defer c.mu.Unlock()
		ld := c.byPort[port]
		if ld == nil {
			return Event{}
		}
		if ld.Ready.IsZero() {
			ld.Ready = c.now()
		}
		return Event{Kind: EventReady, Model: ld.Model, Port: port}
	}

	e, ok := ParseLine(line)
	if !ok || e.Instance == "" {
		// An unprefixed buffer line came from the router process itself,
		// which loads no model. Nothing to attribute it to.
		return Event{}
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if ld := c.byPort[e.Instance]; ld != nil {
		ld.Report.Entries = append(ld.Report.Entries, e)
	}
	return Event{}
}

// Annotate attaches the caller's note to the load on a port. Keyed by
// port rather than by model so a note computed for one load cannot land
// on a newer load of the same model that started while it was being
// prepared. A no-op for a port with no load.
func (c *Collector) Annotate(port string, note any) {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if ld := c.byPort[port]; ld != nil {
		ld.Note = note
	}
}

// InFlight is how many loads have started and not yet reported ready.
// More than one means any measurement taken from the card counters
// covers several models at once.
func (c *Collector) InFlight() int {
	if c == nil {
		return 0
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	n := 0
	for _, ld := range c.byPort {
		if ld.Ready.IsZero() {
			n++
		}
	}
	return n
}

// Latest returns the most recent measurement for a model. The Report in
// the copy is a snapshot: entries collected after this call don't appear
// in it, so a load still in progress reads as whatever had been reported
// by the time it was asked.
func (c *Collector) Latest(model string) (Measurement, bool) {
	if c == nil {
		return Measurement{}, false
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	ld, ok := c.byModel[model]
	if !ok {
		return Measurement{}, false
	}
	return *ld, true
}

// Reset drops the port mapping. Call it when the router is about to
// start: ports are allocated per instance and reused freely across
// runs, so a stale entry would quietly collect another model's buffers.
//
// Past measurements are kept. They are no longer live — the instances
// that produced them are gone — but they remain the last thing measured
// for that model, and whether that still applies is a question about
// configuration the caller is better placed to answer.
func (c *Collector) Reset() {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.byPort = map[string]*Measurement{}
}
