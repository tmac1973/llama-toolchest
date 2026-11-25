package api

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

// fakeDGL acts like sd-server's DGL job routes. Each job runs for polls
// status polls, then ends with fail (when set) or completes with result.
// When workDir is set, a completed job writes its script's output_path
// there, as sd-server would.
type fakeDGL struct {
	t        *testing.T
	srv      *httptest.Server
	mu       sync.Mutex
	sources  []string
	polls    int
	fail     string
	failCode string // the failed job's error code; "" = generation_failed
	submitSC int    // HTTP status for submits; 0 = 202
	result   string
	workDir  string
	requests int // requests after the submit
	seen     map[string]int
	// cancelable makes POST .../cancel work, as from master-1132; without
	// it the route answers 404, as before. cancelled counts those posts.
	cancelable bool
	cancelled  int
}

func newFakeDGL(t *testing.T) *fakeDGL {
	f := &fakeDGL{t: t, polls: 2, seen: map[string]int{},
		result: `{"type":"dict","outputs":[{"name":"frames","value":"17"},{"name":"height","value":"960"},{"name":"width","value":1664}],"media":[]}`}
	f.srv = httptest.NewServer(http.HandlerFunc(f.serve))
	t.Cleanup(f.srv.Close)
	return f
}

var outputPathRE = regexp.MustCompile(`output_path = "([^"]+)"`)

func (f *fakeDGL) serve(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	defer f.mu.Unlock()
	switch {
	case r.Method == "POST" && r.URL.Path == "/sdcpp/v1/dgl/jobs":
		if f.submitSC != 0 {
			w.WriteHeader(f.submitSC)
			_, _ = w.Write([]byte(`{"error":"no"}`))
			return
		}
		var body struct{ Source string }
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil || r.Header.Get("Content-Type") != "application/json" {
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		f.sources = append(f.sources, body.Source)
		w.WriteHeader(http.StatusAccepted)
		_, _ = w.Write([]byte(`{"id":"job1","kind":"dgl","status":"queued"}`))
	case r.Method == "POST" && f.cancelable && strings.HasSuffix(r.URL.Path, "/cancel"):
		f.cancelled++
		_, _ = w.Write([]byte(`{"id":"job1","status":"running","cancel_requested":true}`))
	case r.Method == "GET" && strings.HasPrefix(r.URL.Path, "/sdcpp/v1/dgl/jobs/"):
		f.requests++
		id := strings.TrimPrefix(r.URL.Path, "/sdcpp/v1/dgl/jobs/")
		f.seen[id]++
		n := f.seen[id]
		switch {
		case f.cancelled > 0:
			_, _ = w.Write([]byte(`{"id":"job1","status":"cancelled","error":null,"result":null}`))
		case n <= f.polls:
			_, _ = w.Write([]byte(`{"id":"job1","status":"running","progress":{"stage":"executing","detail":"dgl","completed":null,"total":null,"stage_elapsed_ms":` + strconv.Itoa(n) + `}}`))
		case f.fail != "":
			code := f.failCode
			if code == "" {
				code = "generation_failed"
			}
			b, _ := json.Marshal(map[string]any{"id": "job1", "status": "failed", "result": nil, "error": map[string]string{"code": code, "message": f.fail}})
			_, _ = w.Write(b)
		default:
			if f.workDir != "" && len(f.sources) > 0 {
				if m := outputPathRE.FindStringSubmatch(f.sources[len(f.sources)-1]); m != nil {
					p := filepath.Join(f.workDir, m[1])
					_ = os.MkdirAll(filepath.Dir(p), 0o755)
					_ = os.WriteFile(p, []byte("upscaled video"), 0o644)
				}
			}
			_, _ = w.Write([]byte(`{"id":"job1","status":"completed","error":null,"result":` + f.result + `}`))
		}
	case r.Method == "GET" && r.URL.Path == "/v1/models":
		_, _ = w.Write([]byte(`{"data":[]}`))
	default:
		f.requests++
		http.NotFound(w, r)
	}
}

func fastPoll(t *testing.T) {
	old := vidGenPoll
	vidGenPoll = 5 * time.Millisecond
	t.Cleanup(func() { vidGenPoll = old })
}

func TestRunDGLJob(t *testing.T) {
	fastPoll(t)
	f := newFakeDGL(t)
	var stages []string
	res, err := runDGLJob(context.Background(), f.srv.URL, "fn main() { return 1 }", func(p dglProgress) { stages = append(stages, p.Stage) })
	if err != nil {
		t.Fatal(err)
	}
	if res != (dglResult{Width: 1664, Height: 960, Frames: 17}) {
		t.Errorf("result = %+v", res)
	}
	if len(f.sources) != 1 || f.sources[0] != "fn main() { return 1 }" {
		t.Errorf("sources = %q", f.sources)
	}
	if len(stages) != 1 || stages[0] != "executing" {
		t.Errorf("progress = %v (want one change)", stages)
	}
}

func TestRunDGLJobErrors(t *testing.T) {
	fastPoll(t)
	for _, tt := range []struct {
		name   string
		setup  func(*fakeDGL)
		want   string
		target error
	}{
		{"failed", func(f *fakeDGL) { f.fail = "load_dit: file not found" }, "sd.cpp: load_dit: file not found", nil},
		{"too large", func(f *fakeDGL) { f.fail = "decoded video exceeds 256 MiB; set max_frames" }, "too large for sd.cpp's DGL limits", nil},
		{"no memory", func(f *fakeDGL) {
			f.fail = "line 47, column 16: line 36, column 14: diffusion output shape [] does not match input [208, 120, 2, 48, 1]"
		}, "ran out of GPU memory", errDGLMemory},
		{"oom code", func(f *fakeDGL) {
			f.fail, f.failCode = "line 47, column 16: CUDA0: need 2747.07 MB device, available 1269.56 MB device", "out_of_memory"
		}, "ran out of GPU memory", errDGLMemory},
		{"vae memory", func(f *fakeDGL) { f.fail = "line 64, column 16: line 40, column 15: VAE encode failed" }, "ran out of GPU memory", errDGLMemory},
		{"no binding", func(f *fakeDGL) { f.fail = "line 37, column 12: unknown function 'relayvsr::decode'" }, "no RelayVSR support", nil},
		{"busy", func(f *fakeDGL) { f.submitSC = http.StatusTooManyRequests }, "busy", errDGLBusy},
		{"forbidden", func(f *fakeDGL) { f.submitSC = http.StatusForbidden }, "local-only check", nil},
		{"bad result", func(f *fakeDGL) { f.result = `{"outputs":[{"name":"frames","value":"3"}]}` }, "lacks the width", nil},
	} {
		f := newFakeDGL(t)
		tt.setup(f)
		_, err := runDGLJob(context.Background(), f.srv.URL, "x", nil)
		if err == nil || !strings.Contains(err.Error(), tt.want) || (tt.target != nil && !errors.Is(err, tt.target)) {
			t.Errorf("%s: err = %v, want %q", tt.name, err, tt.want)
		}
	}
}

func TestRunDGLJobStopsWaiting(t *testing.T) {
	fastPoll(t)
	f := newFakeDGL(t)
	f.polls = 1 << 30
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		_, err := runDGLJob(ctx, f.srv.URL, "x", nil)
		done <- err
	}()
	time.Sleep(30 * time.Millisecond)
	cancel()
	select {
	case err := <-done:
		// The fake has no cancel route, like sd.cpp before master-1132.
		if !errors.Is(err, context.Canceled) || !errors.Is(err, errDGLStillRunning) {
			t.Errorf("err = %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("runDGLJob did not return after the context ended")
	}
	f.mu.Lock()
	n := f.requests
	f.mu.Unlock()
	time.Sleep(30 * time.Millisecond)
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.requests != n {
		t.Errorf("requests went on after cancel: %d then %d", n, f.requests)
	}
}

// From master-1132 runDGLJob cancels the job in sd-server and waits until
// it reports cancelled.
func TestRunDGLJobCancels(t *testing.T) {
	fastPoll(t)
	f := newFakeDGL(t)
	f.polls, f.cancelable = 1<<30, true
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		_, err := runDGLJob(ctx, f.srv.URL, "x", nil)
		done <- err
	}()
	time.Sleep(30 * time.Millisecond)
	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) || errors.Is(err, errDGLStillRunning) {
			t.Errorf("err = %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("runDGLJob did not return after the context ended")
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.cancelled != 1 {
		t.Errorf("cancel posts = %d, want 1", f.cancelled)
	}
}

// A job that does not stop within dglCancelWait counts as still running.
func TestRunDGLJobCancelTimesOut(t *testing.T) {
	fastPoll(t)
	old := dglCancelWait
	dglCancelWait = 50 * time.Millisecond
	t.Cleanup(func() { dglCancelWait = old })
	f := newFakeDGL(t)
	f.polls = 1 << 30
	f.srv.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == "POST" && strings.HasSuffix(r.URL.Path, "/cancel") {
			_, _ = w.Write([]byte(`{"id":"job1","status":"running","cancel_requested":true}`))
			return
		}
		f.serve(w, r)
	})
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		_, err := runDGLJob(ctx, f.srv.URL, "x", nil)
		done <- err
	}()
	time.Sleep(30 * time.Millisecond)
	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, errDGLStillRunning) {
			t.Errorf("err = %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("runDGLJob did not give up waiting for the cancel")
	}
}
