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

// The router logs the ready message at debug level, under its own
// timestamp and level prefix. Matching only the bare form meant the end
// of a load was never seen on a real router.
func TestCollectorSeesThePrefixedReady(t *testing.T) {
	c := NewCollector()
	c.Add("0.00.100.000 I srv          load: spawning server instance with name=modelA on port 46231")
	ev := c.Add(`0.14.275.900 D [46231] cmd_child_to_router:state:{"state":"ready","payload":{"n_ctx":8192}}`)
	if ev.Kind != EventReady || ev.Model != "modelA" {
		t.Errorf("prefixed ready line = %+v; want a ready for modelA", ev)
	}
}

// At verbosity 4, which the buffer report needs, the router does not log
// its ready line: it is a debug line. The instance's own "model loaded"
// line is all there is to mark the end of a load. This is the shape of
// a compute2 router log at verbosity 4 (llama.cpp v0.8.0).
func TestCollectorSeesTheModelLoadedLine(t *testing.T) {
	c := NewCollector()
	feed(c, `0.07.990.000 I srv          load: spawning server instance with name=modelA on port 45221
[45221] 0.07.995.000 I load_tensors:        CUDA0 model buffer size =  8192.00 MiB
[45221] 0.07.999.369 I srv  server_run_i: model loaded`)
	if got := c.InFlight(); got != 0 {
		t.Errorf("in flight = %d; want 0 — the load reported itself loaded", got)
	}
	m, _ := c.Latest("modelA")
	if m.Ready.IsZero() {
		t.Error("the measurement does not record when loading finished")
	}
	// The line carries the instance prefix, so it must not be taken for
	// anything else that instance logs.
	if len(m.Report.Entries) != 1 {
		t.Errorf("entries = %d; want 1, the buffer line only", len(m.Report.Entries))
	}

	c = NewCollector()
	c.Add("0.07.990.000 I srv          load: spawning server instance with name=modelA on port 45221")
	ev := c.Add("[45221] 0.07.999.369 I srv  server_run_i: model loaded")
	if ev.Kind != EventReady || ev.Model != "modelA" || ev.Port != "45221" {
		t.Errorf("model loaded line = %+v; want a ready for modelA on 45221", ev)
	}
}

// At verbosity 5 both lines mark the end of one load, in either order.
// The first is the ready; the second must not announce it again, or the
// card counters would be read twice for one load.
func TestCollectorAnnouncesOneReadyWhenBothLinesArrive(t *testing.T) {
	for _, order := range [][2]string{
		{"[45221] 0.07.999.369 I srv  server_run_i: model loaded", `0.07.999.400 D [45221] cmd_child_to_router:state:{"state":"ready","payload":{}}`},
		{`0.07.999.300 D [45221] cmd_child_to_router:state:{"state":"ready","payload":{}}`, "[45221] 0.07.999.369 I srv  server_run_i: model loaded"},
	} {
		c := NewCollector()
		c.Add("0.07.990.000 I srv          load: spawning server instance with name=modelA on port 45221")
		if ev := c.Add(order[0]); ev.Kind != EventReady {
			t.Errorf("first line %q = %+v; want a ready", order[0], ev)
		}
		first, _ := c.Latest("modelA")
		if ev := c.Add(order[1]); ev.Kind != EventNone {
			t.Errorf("second line %q = %+v; want no event", order[1], ev)
		}
		again, _ := c.Latest("modelA")
		if !again.Ready.Equal(first.Ready) {
			t.Errorf("ready moved from %v to %v on the second line", first.Ready, again.Ready)
		}
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
	if ev := c.Add(`[46231] cmd_child_to_router:state:{"state":"ready","payload":{}}`); ev.Kind != EventNone {
		t.Errorf("ready on a wakeup = %+v; want no event — the load was already announced", ev)
	}
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

// Qwen3.6-35B-A3B with MTP on two cards, in the shape llama.cpp v0.9.0
// logs it: the model's context and the drafter's each reserve once while
// loading, and the model's reserves again when a request changes its
// graph (the MTP drafter's embeddings, a request's sampler). The later
// reservation replaces the first; summing it in reported 31.2 GiB for
// cards that held 27.8.
const mtpReserves = `0.00.100.000 I srv          load: spawning server instance with name=q35 on port 46231
[46231] 0.10.000.001 I load_tensors:        CUDA0 model buffer size =  8000.00 MiB
[46231] 0.10.000.002 I load_tensors:        CUDA1 model buffer size =  8000.00 MiB
[46231] 0.11.000.000 I llama_context: constructing llama_context
[46231] 0.11.000.100 I llama_kv_cache:      CUDA0 KV buffer size =  1000.00 MiB
[46231] 0.11.000.101 I llama_kv_cache:      CUDA1 KV buffer size =  1000.00 MiB
[46231] 0.11.100.000 I sched_reserve: reserving ...
[46231] 0.11.200.000 I sched_reserve:      CUDA0 compute buffer size =  1500.00 MiB
[46231] 0.11.200.001 I sched_reserve:      CUDA1 compute buffer size =  1600.00 MiB
[46231] 0.11.200.002 I sched_reserve:  CUDA_Host compute buffer size =   200.00 MiB
[46231] 0.12.000.000 I llama_context: constructing llama_context
[46231] 0.12.000.100 I llama_kv_cache:      CUDA1 KV buffer size =   256.00 MiB
[46231] 0.12.100.000 I sched_reserve: reserving ...
[46231] 0.12.200.000 I sched_reserve:      CUDA1 compute buffer size =   300.00 MiB
[46231] 0.13.000.000 I srv  load_model: model loaded
[46231] 0.20.000.000 I sched_reserve: reserving ...
[46231] 0.20.100.000 I sched_reserve:      CUDA0 compute buffer size =  1510.00 MiB
[46231] 0.20.100.001 I sched_reserve:      CUDA1 compute buffer size =  1610.00 MiB
[46231] 0.20.100.002 I sched_reserve:  CUDA_Host compute buffer size =   200.00 MiB`

func TestCollectorCountsEachContextsFirstReservation(t *testing.T) {
	c := NewCollector()
	feed(c, mtpReserves)
	m, ok := c.Latest("q35")
	if !ok {
		t.Fatal("no measurement")
	}
	gpu := func(e Entry) bool { return e.Kind == KindCompute && e.OnGPU() }
	// The model's 1500 + 1600 and the drafter's 300; not the 3120 the
	// model's second reservation would add.
	if got := m.Report.SumMiB(gpu); got != 3400 {
		t.Errorf("GPU compute = %v MiB; want 3400", got)
	}
	if got := m.Report.GPUMiB(true); got != 16000+2256+3400 {
		t.Errorf("GPU total = %v MiB; want %v", got, 16000+2256+3400)
	}
}

// A build that logs no reservation lines keeps every compute line, as
// before: there is nothing to tell a second reservation by.
func TestCollectorWithoutReserveLinesKeepsEveryComputeLine(t *testing.T) {
	c := NewCollector()
	feed(c, twoInstances)
	a, _ := c.Latest("modelA")
	if got := a.Report.SumMiB(func(e Entry) bool { return e.Kind == KindCompute }); got != 1280 {
		t.Errorf("compute = %v MiB; want 1280", got)
	}
}

func TestParseMarker(t *testing.T) {
	for line, want := range map[string]Marker{
		"[46231] 0.11.000.000 I llama_context: constructing llama_context":                MarkerContext,
		"data: [ 8080] 0.11.100.000 I sched_reserve: reserving ...":                       MarkerReserve,
		"[46231] 0.11.200.000 I sched_reserve: reserve took 12.00 ms":                     MarkerNone,
		"[46231] 0.11.200.000 I sched_reserve:  CUDA0 compute buffer size =  1500.00 MiB": MarkerNone,
	} {
		if _, got := ParseMarker(line); got != want {
			t.Errorf("%q: marker %v; want %v", line, got, want)
		}
	}
}

// A measurement handed out mid-load keeps the hidden buffer it was given,
// even when a later reallocation replaces it in the collector.
func TestCollectorHiddenReplacementLeavesSnapshotsAlone(t *testing.T) {
	c := NewCollector()
	feed(c, `0.00.100.000 I srv          load: spawning server instance with name=dflash on port 46231
[46231] 0.16.300.000 I layer_inp_dev_alloc:      CUDA0 layer input buffer size =    40.00 MiB (1 layers)`)
	before, _ := c.Latest("dflash")
	feed(c, `[46231] 0.16.300.001 I layer_inp_dev_alloc:      CUDA0 layer input buffer size =    80.00 MiB (2 layers)`)
	after, _ := c.Latest("dflash")
	if got := before.Report.SumMiB(nil); got != 40 {
		t.Errorf("snapshot = %v MiB; want the 40 it was taken with", got)
	}
	if got := after.Report.SumMiB(nil); got != 80 {
		t.Errorf("latest = %v MiB; want 80, the reallocation replacing the first", got)
	}
}
