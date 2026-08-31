package memreport

import (
	"strings"
	"testing"
)

// twoInstances is the shape the router really produces when a second
// model is loaded while the first is still allocating: one spawn line per
// instance from the router itself, and buffer reports from both children
// interleaved in a single stream, told apart only by the bracketed port.
const twoInstances = `0.00.100.000 I srv          load: spawning server instance with name=modelA on port 46231
0.00.200.000 I srv          load: spawning server instance with name=modelB on port 39114
[46231] 0.14.275.751 I load_tensors:   CPU_Mapped model buffer size =  1024.00 MiB
[39114] 0.14.300.000 I load_tensors:        ROCm0 model buffer size =  4096.00 MiB
[46231] 0.14.275.753 I load_tensors:        ROCm0 model buffer size =  8192.00 MiB
[39114] 0.16.100.000 I llama_kv_cache:      ROCm0 KV buffer size =   512.00 MiB
[46231] 0.16.162.940 I llama_kv_cache:      ROCm0 KV buffer size =  2048.00 MiB
[46231] 0.16.260.669 I sched_reserve:   ROCm_Host compute buffer size =   256.00 MiB
[46231] 0.16.260.662 I sched_reserve:       ROCm0 compute buffer size =  1024.00 MiB`

func feed(c *Collector, log string) {
	for _, line := range strings.Split(log, "\n") {
		c.Add(line)
	}
}

func TestCollectorAttributesLinesToTheModelThatLoggedThem(t *testing.T) {
	c := NewCollector()
	feed(c, twoInstances)

	a, ok := c.Latest("modelA")
	if !ok {
		t.Fatal("modelA has no measurement")
	}
	if a.Port != "46231" {
		t.Errorf("modelA port = %q; want 46231", a.Port)
	}
	// 8192 weights + 2048 KV + 1024 compute; the host-mapped weights and
	// the host compute buffer are not on a device.
	if got := a.Report.GPUMiB(true); got != 11264 {
		t.Errorf("modelA GPU = %v MiB; want 11264", got)
	}
	if got := a.Report.HostMiB(true); got != 1280 {
		t.Errorf("modelA host = %v MiB; want 1280", got)
	}

	b, ok := c.Latest("modelB")
	if !ok {
		t.Fatal("modelB has no measurement")
	}
	if got := b.Report.GPUMiB(true); got != 4608 {
		t.Errorf("modelB GPU = %v MiB; want 4608 — lines from the other instance leaked in", got)
	}
}

func TestCollectorReportsTheBoundariesOfALoad(t *testing.T) {
	c := NewCollector()

	ev := c.Add("0.00.100.000 I srv          load: spawning server instance with name=modelA on port 46231")
	if ev.Kind != EventSpawned || ev.Model != "modelA" || ev.Port != "46231" {
		t.Errorf("spawn line = %+v; want a spawn of modelA on 46231", ev)
	}
	if ev := c.Add("[46231] 0.14.275.753 I load_tensors:  ROCm0 model buffer size = 8192.00 MiB"); ev.Kind != EventNone {
		t.Errorf("buffer line = %+v; want no event — it is collected, not announced", ev)
	}
	if ev := c.Add("[46231] 0.14.000.000 I main: server is listening"); ev.Kind != EventNone {
		t.Errorf("unrelated line = %+v; want no event", ev)
	}

	ev = c.Add(`[46231] cmd_child_to_router:state:{"state":"ready","payload":{"n_ctx":8192}}`)
	if ev.Kind != EventReady || ev.Model != "modelA" {
		t.Errorf("ready line = %+v; want a ready for modelA", ev)
	}
	if m, _ := c.Latest("modelA"); m.Ready.IsZero() {
		t.Error("the measurement does not record when loading finished")
	}
}

// A load is in flight from its spawn until it reports ready. It matters
// because a card-counter reading taken while two models are loading
// describes both of them.
func TestCollectorTracksLoadsInFlight(t *testing.T) {
	c := NewCollector()
	if got := c.InFlight(); got != 0 {
		t.Errorf("in flight = %d; want 0", got)
	}
	feed(c, twoInstances)
	if got := c.InFlight(); got != 2 {
		t.Errorf("in flight = %d; want 2 — both spawned, neither ready", got)
	}
	c.Add(`[46231] cmd_child_to_router:state:{"state":"ready","payload":{}}`)
	if got := c.InFlight(); got != 1 {
		t.Errorf("in flight = %d; want 1", got)
	}
}

// A ready for a port with no load — a leftover from a previous router
// run, say — must not invent one.
func TestCollectorIgnoresReadyForAnUnknownPort(t *testing.T) {
	c := NewCollector()
	if ev := c.Add(`[59999] cmd_child_to_router:state:{"state":"ready","payload":{}}`); ev.Kind != EventNone {
		t.Errorf("ready for an unknown port = %+v; want no event", ev)
	}
}

// Waking from sleep reports ready again. The load being described is
// still the one that was measured, so the first time stands.
func TestCollectorKeepsTheFirstReady(t *testing.T) {
	c := NewCollector()
	c.Add("0.00.100.000 I srv          load: spawning server instance with name=modelA on port 46231")
	c.Add(`[46231] cmd_child_to_router:state:{"state":"ready","payload":{}}`)
	first, _ := c.Latest("modelA")
	c.Add(`[46231] cmd_child_to_router:state:{"state":"ready","payload":{}}`)
	again, _ := c.Latest("modelA")
	if !again.Ready.Equal(first.Ready) {
		t.Errorf("ready moved from %v to %v on a wakeup", first.Ready, again.Ready)
	}
}

// A model id may contain the words the spawn line uses to delimit it. The
// port is what ends the line, so the name is whatever precedes it.
func TestCollectorSpawnLineWithAwkwardName(t *testing.T) {
	c := NewCollector()
	const id = "user--Weird on port 1-GGUF--Q4_K_M"
	ev := c.Add("0.00.100.000 I srv          load: spawning server instance with name=" + id + " on port 46231")
	if ev.Kind != EventSpawned || ev.Model != id {
		t.Errorf("event = %+v; want a spawn of %q", ev, id)
	}
}

func TestCollectorReloadStartsAFreshReport(t *testing.T) {
	c := NewCollector()
	feed(c, `0.00.100.000 I srv          load: spawning server instance with name=modelA on port 46231
[46231] 0.14.275.753 I load_tensors:  ROCm0 model buffer size = 8192.00 MiB
0.20.100.000 I srv          load: spawning server instance with name=modelA on port 51002
[51002] 0.24.275.753 I load_tensors:  ROCm0 model buffer size = 4096.00 MiB`)

	m, ok := c.Latest("modelA")
	if !ok {
		t.Fatal("no measurement")
	}
	// The second load is a different configuration or a different model
	// file; adding the two would report a model twice its size.
	if got := m.Report.GPUMiB(true); got != 4096 {
		t.Errorf("GPU = %v MiB; want 4096 — a reload must replace, not accumulate", got)
	}
	if m.Port != "51002" {
		t.Errorf("port = %q; want 51002", m.Port)
	}
}

func TestCollectorResetDropsPortsButKeepsMeasurements(t *testing.T) {
	c := NewCollector()
	feed(c, twoInstances)
	c.Reset()

	// A new router run reuses the port for something else entirely.
	c.Add("[46231] 0.14.275.753 I load_tensors:  ROCm0 model buffer size = 99999.00 MiB")

	a, ok := c.Latest("modelA")
	if !ok {
		t.Fatal("reset must keep what was already measured")
	}
	if got := a.Report.GPUMiB(true); got != 11264 {
		t.Errorf("GPU = %v MiB; want 11264 — a stale port collected another run's buffers", got)
	}
}

func TestCollectorAnnotate(t *testing.T) {
	c := NewCollector()
	feed(c, twoInstances)

	c.Annotate("46231", "ctx=8192") // modelA's port
	c.Annotate("50000", "ctx=4096") // no load there: nothing to attach to

	a, _ := c.Latest("modelA")
	if a.Note != "ctx=8192" {
		t.Errorf("note = %v; want ctx=8192", a.Note)
	}
	if b, _ := c.Latest("modelB"); b.Note != nil {
		t.Errorf("modelB note = %v; want nil — annotating one model must not touch another", b.Note)
	}
	if _, ok := c.Latest("modelC"); ok {
		t.Error("annotating an unknown port must not create a measurement")
	}
}

// Buffer lines with no port prefix come from the router process, which
// loads no model of its own. Attributing them to whichever model spawned
// last would inflate that model.
func TestCollectorIgnoresUnprefixedBufferLines(t *testing.T) {
	c := NewCollector()
	c.Add("0.00.100.000 I srv          load: spawning server instance with name=modelA on port 46231")
	c.Add("0.14.275.753 I load_tensors:  ROCm0 model buffer size = 8192.00 MiB")

	a, _ := c.Latest("modelA")
	if got := len(a.Report.Entries); got != 0 {
		t.Errorf("entries = %d; want 0", got)
	}
}

// A tensor-parallel load reports per card against llama.cpp's aggregate
// device, so the collector's raw figures need scaling by the card count
// the caller knows. Guards the pairing of the two.
func TestCollectorReportScalesAggregates(t *testing.T) {
	c := NewCollector()
	feed(c, `0.00.100.000 I srv          load: spawning server instance with name=modelA on port 46231
[46231] 0.14.275.753 I load_tensors:      Meta() model buffer size = 7252.51 MiB
[46231] 0.16.162.940 I llama_kv_cache:    Meta() KV buffer size = 1024.00 MiB`)

	m, _ := c.Latest("modelA")
	one := m.Report.GPUMiB(true)
	four := m.Report.ScaleAggregates(4).GPUMiB(true)
	if four != one*4 {
		t.Errorf("scaled = %v; want %v (4x the per-card figures)", four, one*4)
	}
}

// The router writes the instance prefix with "[%5d] ", so a port below
// 10000 arrives padded while the spawn line states it unpadded. Both
// have to reach the same record, or a load on a low port collects
// nothing at all.
func TestCollectorMatchesAPaddedPortPrefix(t *testing.T) {
	c := NewCollector()
	feed(c, `0.00.100.000 I srv          load: spawning server instance with name=modelA on port 4623
[ 4623] 0.14.275.753 I load_tensors:  ROCm0 model buffer size = 8192.00 MiB`)

	m, ok := c.Latest("modelA")
	if !ok {
		t.Fatal("no measurement")
	}
	if got := m.Report.GPUMiB(true); got != 8192 {
		t.Errorf("GPU = %v MiB; want 8192", got)
	}
}
