package enginenotes

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

var base = time.Date(2026, 10, 30, 13, 0, 0, 0, time.UTC)

// feed reads a recorded log from testdata into tr, one line per second
// from base, and returns every outcome it produced.
func feed(t *testing.T, tr *Tracker, name string) []Outcome {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatal(err)
	}
	return feedLines(tr, strings.Split(strings.TrimRight(string(data), "\n"), "\n"), base)
}

func feedLines(tr *Tracker, lines []string, start time.Time) []Outcome {
	var out []Outcome
	for i, line := range lines {
		out = append(out, tr.Add(line, start.Add(time.Duration(i)*time.Second))...)
	}
	return out
}

func kinds(out []Outcome) []OutcomeKind {
	k := make([]OutcomeKind, len(out))
	for i, o := range out {
		k[i] = o.Kind
	}
	return k
}

func wantKinds(t *testing.T, out []Outcome, want ...OutcomeKind) {
	t.Helper()
	got := kinds(out)
	if len(got) != len(want) {
		t.Fatalf("outcomes = %v, want %v", got, want)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("outcomes = %v, want %v", got, want)
		}
	}
}

func TestLoadThenUnload(t *testing.T) {
	tr := NewTracker()
	out := feed(t, tr, "load_unload.log")
	wantKinds(t, out, KindLoad, KindUnload)

	load, unload := out[0], out[1]
	if load.RouterName != "qwen" || load.Port != "45221" {
		t.Errorf("load = %+v, want qwen on 45221", load)
	}
	// Spawned at line 0, loaded at line 2, exited at line 5.
	if load.LoadSeconds != 2 {
		t.Errorf("load seconds = %v, want 2", load.LoadSeconds)
	}
	if unload.LoadedSeconds != 3 || unload.ExitStatus != 0 {
		t.Errorf("unload = %+v, want 3 s loaded and status 0", unload)
	}
	if len(tr.Open()) != 0 {
		t.Errorf("open after unload = %v, want none", tr.Open())
	}
}

// The real router output from compute2 (llama.cpp v0.8.0, verbosity 4):
// SSE "data: " prefixes, and no router "ready" line, which needs
// verbosity 5. The instance's own "model loaded" line marks the load.
func TestRecordedComputeLogShowsTheLoad(t *testing.T) {
	tr := NewTracker()
	out := feed(t, tr, "compute2_load.log")
	wantKinds(t, out, KindLoad)
	const name = "unsloth--Qwen3.6-35B-A3B-MTP-GGUF--Qwen3.6-35B-A3B-UD-IQ4_NL"
	if out[0].RouterName != name || out[0].Port != "45221" {
		t.Errorf("load = %+v, want %s on 45221", out[0], name)
	}
	open := tr.Open()
	if len(open) != 1 || open[0].RouterName != name {
		t.Errorf("open = %+v, want the one loaded instance", open)
	}
}

func TestCrashAfterLoadKeepsTheReason(t *testing.T) {
	tr := NewTracker()
	out := feed(t, tr, "crash_cuda.log")
	wantKinds(t, out, KindLoad, KindCrash)

	c := out[1]
	if c.RouterName != "big" || c.Port != "50915" || c.ExitStatus != 1 {
		t.Errorf("crash = %+v, want big on 50915 with status 1", c)
	}
	if len(c.ErrorLines) == 0 || !strings.Contains(c.ErrorLines[0], "CUDA error") {
		t.Errorf("error lines = %q, want a CUDA error first", c.ErrorLines)
	}
	for _, l := range c.ErrorLines {
		if strings.Contains(l, "[0x") {
			t.Errorf("error lines keep a backtrace frame: %q", l)
		}
	}
	if len(c.Tail) != tailLines {
		t.Errorf("tail has %d lines, want %d", len(c.Tail), tailLines)
	}
	if last := c.Tail[len(c.Tail)-1]; !strings.Contains(last, "llama-server[0x55d0c1a2b3e5]") {
		t.Errorf("tail ends with %q, want the instance's last line", last)
	}
	if c.LoadedSeconds <= 0 {
		t.Errorf("loaded seconds = %v, want the time since the load", c.LoadedSeconds)
	}
}

// A stop that has to be forced reports status 1, like a crash. The stop
// request makes it an unload.
func TestForcedStopIsNotACrash(t *testing.T) {
	out := feed(t, NewTracker(), "stop_timeout.log")
	wantKinds(t, out, KindLoad, KindUnload)
	if out[1].ExitStatus != 1 {
		t.Errorf("status = %d, want the 1 the router reported", out[1].ExitStatus)
	}
	if out[1].ErrorLines != nil || out[1].Tail != nil {
		t.Errorf("an unload kept crash details: %+v", out[1])
	}
}

func TestExitBeforeLoadIsAFailedLoad(t *testing.T) {
	out := feed(t, NewTracker(), "load_failed.log")
	wantKinds(t, out, KindLoadFailed)
	f := out[0]
	if len(f.ErrorLines) == 0 || !strings.Contains(f.ErrorLines[0], "cudaMalloc failed: out of memory") {
		t.Errorf("error lines = %q, want the allocation that failed first", f.ErrorLines)
	}
	if f.LoadedSeconds != 0 {
		t.Errorf("loaded seconds = %v for a model that never loaded", f.LoadedSeconds)
	}
}

// Two instances write at the same time. A crash keeps only its own
// instance's lines.
func TestInterleavedInstancesKeepTheirOwnLines(t *testing.T) {
	tr := NewTracker()
	out := feed(t, tr, "interleaved.log")
	wantKinds(t, out, KindLoad, KindLoad, KindCrash)

	c := out[2]
	if c.RouterName != "modelB" || c.Port != "9114" {
		t.Fatalf("crash = %+v, want modelB on 9114", c)
	}
	for _, l := range c.Tail {
		if strings.Contains(l, "(A)") {
			t.Errorf("modelB's tail holds modelA's line %q", l)
		}
	}
	if len(c.ErrorLines) == 0 || !strings.Contains(strings.Join(c.ErrorLines, "\n"), "misaligned address") {
		t.Errorf("error lines = %q, want the misaligned address", c.ErrorLines)
	}
	open := tr.Open()
	if len(open) != 1 || open[0].RouterName != "modelA" {
		t.Errorf("open = %+v, want modelA still loaded", open)
	}
}

func TestRouterExitEndsItsInstances(t *testing.T) {
	tr := NewTracker()
	out := feed(t, tr, "router_exit.log")
	wantKinds(t, out, KindLoad, KindRouterExit, KindUnload)
	if out[1].RouterError != "signal: aborted (core dumped)" {
		t.Errorf("router error = %q", out[1].RouterError)
	}
	if out[2].RouterName != "qwen" || out[2].LoadedSeconds != 1 {
		t.Errorf("instance outcome = %+v, want qwen unloaded after 1 s", out[2])
	}
	if len(tr.Open()) != 0 {
		t.Errorf("open after the router ended = %v", tr.Open())
	}
}

func TestRouterStopAndNormalExit(t *testing.T) {
	tr := NewTracker()
	out := feedLines(tr, []string{
		"0.02.475.230 I srv          load: spawning server instance with name=qwen on port 45221",
		"==> Router stopped",
	}, base)
	// Not loaded yet: an unload with no time loaded.
	wantKinds(t, out, KindRouterStop, KindUnload)
	if out[1].LoadedSeconds != 0 {
		t.Errorf("loaded seconds = %v for an instance still loading", out[1].LoadedSeconds)
	}

	out = feedLines(tr, []string{"==> Router exited normally"}, base)
	wantKinds(t, out, KindRouterExit)
	if out[0].RouterError != "" {
		t.Errorf("router error = %q, want none for a clean exit", out[0].RouterError)
	}
}

// After a router stop the next router reuses ports. A spawn on the same
// port starts a fresh instance with none of the old lines.
func TestPortsReusedAfterARouterStop(t *testing.T) {
	tr := NewTracker()
	feedLines(tr, []string{
		"0.00.100.000 I srv          load: spawning server instance with name=old on port 45221",
		"[45221] 0.01.000.000 E old instance problem",
		"==> Router stopped",
		"0.00.100.000 I srv          load: spawning server instance with name=new on port 45221",
		"[45221] 0.01.000.000 E new instance problem",
	}, base)
	out := feedLines(tr, []string{"0.03.000.000 I srv    operator(): instance name=new exited with status 1"}, base.Add(time.Minute))
	wantKinds(t, out, KindLoadFailed)
	if got := strings.Join(out[0].Tail, "\n"); strings.Contains(got, "old instance") {
		t.Errorf("the new instance's tail holds the old one's line: %q", got)
	}
}

// A stop request belongs to one instance. A new instance of the same
// name that later dies is a crash.
func TestStopDoesNotCarryOverToANewInstance(t *testing.T) {
	out := feedLines(NewTracker(), []string{
		"0.00.100.000 I srv          load: spawning server instance with name=qwen on port 45221",
		"[45221] 0.01.000.000 I srv  server_run_i: model loaded",
		"0.02.000.000 I srv        unload: stopping model instance name=qwen",
		"0.02.500.000 I srv    operator(): instance name=qwen exited with status 0",
		"0.03.000.000 I srv          load: spawning server instance with name=qwen on port 51002",
		"[51002] 0.01.000.000 I srv  server_run_i: model loaded",
		"0.04.000.000 I srv    operator(): instance name=qwen exited with status 1",
	}, base)
	wantKinds(t, out, KindLoad, KindUnload, KindLoad, KindCrash)
	if out[3].Port != "51002" {
		t.Errorf("crash on port %q, want the new instance's 51002", out[3].Port)
	}
}

// The toolchest can start while an instance is already running, so the
// spawn was never seen.
func TestExitWithoutASpawnSeen(t *testing.T) {
	cases := []struct {
		lines []string
		want  OutcomeKind
	}{
		{[]string{"0.04.000.000 I srv    operator(): instance name=qwen exited with status 1"}, KindCrash},
		{[]string{
			"0.03.000.000 I srv        unload: stopping model instance name=qwen",
			"0.04.000.000 I srv    operator(): instance name=qwen exited with status 1",
		}, KindUnload},
		{[]string{"0.04.000.000 I srv    operator(): instance name=qwen exited with status 0"}, KindUnload},
	}
	for _, c := range cases {
		out := feedLines(NewTracker(), c.lines, base)
		wantKinds(t, out, c.want)
		if out[0].Port != "" || out[0].Tail != nil {
			t.Errorf("%v: outcome = %+v, want no port and no tail", c.want, out[0])
		}
	}
}

// A model waking from sleep reports ready again. That is not a new load.
func TestReadyAgainIsNotASecondLoad(t *testing.T) {
	out := feedLines(NewTracker(), []string{
		"0.00.100.000 I srv          load: spawning server instance with name=qwen on port 45221",
		"[45221] 0.01.000.000 I srv  server_run_i: model loaded",
		`0.14.275.900 D [45221] cmd_child_to_router:state:{"state":"ready","payload":{}}`,
		`0.20.275.900 D [45221] cmd_child_to_router:state:{"state":"ready","payload":{}}`,
	}, base)
	wantKinds(t, out, KindLoad)
}

// At verbosity 5 the router's ready line can arrive first. Either line
// marks the load.
func TestRouterReadyLineAloneMarksTheLoad(t *testing.T) {
	out := feedLines(NewTracker(), []string{
		"0.00.100.000 I srv          load: spawning server instance with name=qwen on port 45221",
		`0.14.275.900 D [45221] cmd_child_to_router:state:{"state":"ready","payload":{}}`,
	}, base)
	wantKinds(t, out, KindLoad)
}
