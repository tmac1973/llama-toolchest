package benchmark

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

// A streamed reply: progress while the prompt is read, a delta per token,
// and the timings in the last message.
func TestReadStreamTimings(t *testing.T) {
	body := strings.Join([]string{
		`data: {"choices":[],"prompt_progress":{"total":1531,"cache":0,"processed":256,"time_ms":90}}`,
		``,
		`data: {"choices":[{"delta":{"content":"def"}}]}`,
		``,
		`data: {"choices":[],"usage":{"completion_tokens":512},"timings":{"prompt_n":1531,"prompt_per_second":2500,"predicted_n":512,"predicted_per_second":150}}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")
	lines := 0
	timings, err := readStreamTimings(strings.NewReader(body), func() { lines++ })
	if err != nil {
		t.Fatal(err)
	}
	if timings == nil || timings.PredictedN != 512 || timings.PromptN != 1531 || timings.PredictedPerSec != 150 {
		t.Errorf("timings = %+v", timings)
	}
	if lines != 7 {
		t.Errorf("progress called %d times, want once per line (7)", lines)
	}
}

func TestReadStreamTimingsError(t *testing.T) {
	body := "data: {\"error\":{\"code\":500,\"message\":\"failed to decode\"}}\n\n"
	if _, err := readStreamTimings(strings.NewReader(body), func() {}); err == nil || !strings.Contains(err.Error(), "failed to decode") {
		t.Errorf("err = %v, want the server's message", err)
	}
}

// stallingRouter answers the warmup, then starts the benchmark request and
// goes quiet, the way a hung llama-server does.
type stallingRouter struct {
	*httptest.Server
	mu      sync.Mutex
	stalled int // benchmark requests that were left hanging
}

func newStallingRouter(t *testing.T) *stallingRouter {
	t.Helper()
	sr := &stallingRouter{}
	mux := http.NewServeMux()
	mux.HandleFunc("/models", func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, `{"data":[],"object":"list"}`)
	})
	mux.HandleFunc("/models/load", func(w http.ResponseWriter, r *http.Request) {})
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)
		if body["max_tokens"] == float64(16) { // the warmup
			w.Header().Set("Content-Type", "text/event-stream")
			io.WriteString(w, `data: {"choices":[],"timings":{"prompt_n":64,"predicted_n":16,"predicted_per_second":50}}`+"\n\ndata: [DONE]\n\n")
			return
		}
		sr.mu.Lock()
		sr.stalled++
		sr.mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, `data: {"choices":[],"prompt_progress":{"total":1531,"processed":0}}`+"\n\n")
		w.(http.Flusher).Flush()
		<-r.Context().Done()
	})
	sr.Server = httptest.NewServer(mux)
	t.Cleanup(sr.Close)
	return sr
}

// A request that stops making progress fails within benchStallTimeout,
// not the whole request timeout, and the cell gives up on the hung
// server rather than sending it the remaining repetitions.
func TestStalledRequestEndsTheCell(t *testing.T) {
	old := benchStallTimeout
	benchStallTimeout = 100 * time.Millisecond
	t.Cleanup(func() { benchStallTimeout = old })

	router := newStallingRouter(t)
	r := &Runner{}
	_, err := r.sendCompletionWithTimings(context.Background(), router.URL, "m", 1536, 512, 1, SamplingParams{}, "", promptOptions{})
	if !errors.Is(err, errBenchStalled) {
		t.Fatalf("err = %v, want a stall", err)
	}

	env := &fakeEnv{routerURL: router.URL, saved: ConfigSnapshot{GPULayers: 999}}
	start := time.Now()
	// Three repetitions at each of three prompt sizes: nine requests if
	// the cell kept going.
	j := oneCellJob(nil)
	j.Presets = []string{"internal-standard"}
	j.Cells = ExpandCells(j.ModelIDs, j.BuildIDs, j.Presets)
	job, _ := runJob(t, j, env)
	if job.Cells[0].Status != CellStatusFailed || !strings.Contains(job.Cells[0].Error, "stopped responding") {
		t.Errorf("cell = %s %q, want failed with the stall", job.Cells[0].Status, job.Cells[0].Error)
	}
	if took := time.Since(start); took > 5*time.Second {
		t.Errorf("job took %s after the stall", took)
	}
	router.mu.Lock()
	defer router.mu.Unlock()
	if router.stalled != 2 { // the direct call above, and one from the cell
		t.Errorf("hung server sent %d benchmark requests, want 2", router.stalled)
	}
}

// A server that dies mid-reply is named as a likely crash, not as a reply
// without timings.
func TestStreamCutOffReadsAsCrash(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, `data: {"choices":[{"delta":{"content":"def"}}]}`+"\n\n")
	}))
	t.Cleanup(srv.Close)
	r := &Runner{}
	_, err := r.sendCompletionWithTimings(context.Background(), srv.URL, "m", 64, 16, 1, SamplingParams{}, "", promptOptions{})
	if err == nil || !strings.Contains(err.Error(), "probably crashed") {
		t.Errorf("err = %v, want the crash message", err)
	}
}

// scriptedRouter answers the warmup, then gives each benchmark request
// to bench, numbered from 1.
type scriptedRouter struct {
	*httptest.Server
	mu    sync.Mutex
	bench int
}

func newScriptedRouter(t *testing.T, bench func(n int, w http.ResponseWriter, r *http.Request)) *scriptedRouter {
	t.Helper()
	sr := &scriptedRouter{}
	mux := http.NewServeMux()
	mux.HandleFunc("/models", func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, `{"data":[],"object":"list"}`)
	})
	mux.HandleFunc("/models/load", func(w http.ResponseWriter, r *http.Request) {})
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)
		if body["max_tokens"] == float64(16) { // the warmup
			writeTimings(w)
			return
		}
		sr.mu.Lock()
		sr.bench++
		n := sr.bench
		sr.mu.Unlock()
		bench(n, w, r)
	})
	sr.Server = httptest.NewServer(mux)
	t.Cleanup(sr.Close)
	return sr
}

func (sr *scriptedRouter) requests() int {
	sr.mu.Lock()
	defer sr.mu.Unlock()
	return sr.bench
}

func writeTimings(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "text/event-stream")
	io.WriteString(w, `data: {"choices":[],"timings":{"prompt_n":64,"predicted_n":16,"predicted_per_second":50}}`+"\n\ndata: [DONE]\n\n")
}

func shortTimeouts(t *testing.T, stall, request time.Duration) {
	t.Helper()
	oldStall, oldRequest := benchStallTimeout, benchRequestTimeout
	benchStallTimeout, benchRequestTimeout = stall, request
	t.Cleanup(func() { benchStallTimeout, benchRequestTimeout = oldStall, oldRequest })
}

// runStandardCell runs one cell of internal-standard (nine requests if it
// ran to the end) against the router.
func runStandardCell(t *testing.T, routerURL string) *BenchmarkJob {
	t.Helper()
	env := &fakeEnv{routerURL: routerURL, saved: ConfigSnapshot{GPULayers: 999}}
	j := oneCellJob(nil)
	j.Presets = []string{"internal-standard"}
	j.Cells = ExpandCells(j.ModelIDs, j.BuildIDs, j.Presets)
	job, _ := runJob(t, j, env)
	return job
}

// A server can hang before its reply starts (seen on compute2 with a
// tensor split, 2026-11-04). After the warmup the model is loaded, so
// that is a stall, not a ten-minute wait.
func TestHangBeforeTheReplyStartsIsAStall(t *testing.T) {
	shortTimeouts(t, 100*time.Millisecond, time.Minute)
	router := newScriptedRouter(t, func(_ int, _ http.ResponseWriter, r *http.Request) {
		<-r.Context().Done()
	})
	start := time.Now()
	job := runStandardCell(t, router.URL)
	if job.Cells[0].Status != CellStatusFailed || !strings.Contains(job.Cells[0].Error, "stopped responding") {
		t.Errorf("cell = %s %q, want failed with the stall", job.Cells[0].Status, job.Cells[0].Error)
	}
	if took := time.Since(start); took > 5*time.Second {
		t.Errorf("job took %s", took)
	}
	if n := router.requests(); n != 1 {
		t.Errorf("hung server got %d benchmark requests, want 1", n)
	}
}

// The warmup may wait for the model to load before its reply starts, and
// that is no stall.
func TestWarmupMayWaitForTheLoad(t *testing.T) {
	shortTimeouts(t, 100*time.Millisecond, time.Minute)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(300 * time.Millisecond)
		writeTimings(w)
	}))
	t.Cleanup(srv.Close)
	r := &Runner{}
	if _, err := r.sendCompletionWithTimings(context.Background(), srv.URL, "m", 64, 16, 0, SamplingParams{}, "", promptOptions{}); err != nil {
		t.Errorf("warmup: %v, want it to wait for the load", err)
	}
}

// A request that runs into the time limit means the server is stuck, as
// a stall does: the cell ends rather than sending the next request into
// the same wait.
func TestTimedOutRequestEndsTheCell(t *testing.T) {
	shortTimeouts(t, time.Minute, 300*time.Millisecond)
	router := newScriptedRouter(t, func(_ int, w http.ResponseWriter, r *http.Request) {
		// Progress that never ends: no stall, only the time limit.
		w.Header().Set("Content-Type", "text/event-stream")
		for {
			io.WriteString(w, `data: {"choices":[],"prompt_progress":{"total":1531,"processed":0}}`+"\n\n")
			w.(http.Flusher).Flush()
			select {
			case <-r.Context().Done():
				return
			case <-time.After(20 * time.Millisecond):
			}
		}
	})
	job := runStandardCell(t, router.URL)
	if job.Cells[0].Status != CellStatusFailed || !strings.Contains(job.Cells[0].Error, "time limit") {
		t.Errorf("cell = %s %q, want failed with the time limit", job.Cells[0].Status, job.Cells[0].Error)
	}
	if n := router.requests(); n != 1 {
		t.Errorf("stuck server got %d benchmark requests, want 1", n)
	}
}

// Tests that finished before a hang do not make the cell a result: one
// sample of nine is not the cell's numbers, and a setting that hangs the
// server is not one to rank.
func TestHangAfterSomeTestsFailsTheCell(t *testing.T) {
	shortTimeouts(t, 100*time.Millisecond, time.Minute)
	router := newScriptedRouter(t, func(n int, w http.ResponseWriter, r *http.Request) {
		if n == 1 {
			writeTimings(w)
			return
		}
		<-r.Context().Done()
	})
	job := runStandardCell(t, router.URL)
	if job.Cells[0].Status != CellStatusFailed || !strings.Contains(job.Cells[0].Error, "remaining tests were not run") {
		t.Errorf("cell = %s %q, want failed with the hang", job.Cells[0].Status, job.Cells[0].Error)
	}
	if n := router.requests(); n != 2 {
		t.Errorf("server got %d benchmark requests, want 2", n)
	}
}
