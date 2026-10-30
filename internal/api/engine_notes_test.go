package api

import (
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/benchmark"
	"github.com/tmac1973/llama-toolchest/internal/enginenotes"
	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/process"
	"github.com/tmac1973/llama-toolchest/internal/testutil"
)

// openTestNotes gives s an engine notes store in a temporary folder.
func openTestNotes(t *testing.T, s *Server) {
	t.Helper()
	st, err := enginenotes.Open(filepath.Join(t.TempDir(), "engine-notes.json"), time.Now)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { st.Close() })
	s.notes = st
}

func allNotes(s *Server) enginenotes.Summary {
	return s.notes.Summary(0, s.openInstances(), s.resolveRouterName)
}

// The whole path a crash takes: a router writing to stdout, the process
// manager broadcasting it, and watchEngineNotes recording it with the
// model, build and GPUs filled in.
func TestWatchEngineNotesRecordsACrash(t *testing.T) {
	fixture, err := filepath.Abs(filepath.Join("..", "enginenotes", "testdata", "crash_cuda.log"))
	if err != nil {
		t.Fatal(err)
	}
	binary := filepath.Join(fakeLlamaServerDir(t), "crashing-llama-server")
	script := "#!/bin/sh\ncat '" + fixture + "'\nwhile kill -0 $PPID 2>/dev/null; do sleep 1; done\n"
	if err := os.WriteFile(binary, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}

	s := newTestServer(t)
	openTestNotes(t, s)
	if err := s.registry.Add(&models.Model{ID: "big", ModelID: "org/Big-GGUF", Filename: "big.gguf"}); err != nil {
		t.Fatal(err)
	}
	if err := s.registry.SetConfig("big", &models.ModelConfig{Enabled: true, GPUAssign: "1"}); err != nil {
		t.Fatal(err)
	}
	s.setRunningBuild("b1")
	t.Cleanup(func() { s.process.Stop() })
	go s.watchEngineNotes()

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := ln.Addr().(*net.TCPAddr).Port
	ln.Close()
	if err := s.process.Start(process.RouterConfig{BinaryPath: binary, Host: "127.0.0.1", Port: port}); err != nil {
		t.Fatalf("start: %v", err)
	}

	testutil.Eventually(t, func() bool { return allNotes(s).Totals.Crashes == 1 }, "no crash recorded from the log stream")
	sum := allNotes(s)
	if sum.Totals.Loads != 1 {
		t.Errorf("loads = %d, want the load before the crash", sum.Totals.Loads)
	}
	c := sum.Failures[0]
	if c.ModelID != "big" || c.Build != "b1" || c.GPUs != "gpu:1" || c.ExitStatus != 1 {
		t.Errorf("crash = %+v, want big on build b1, gpu:1, status 1", c)
	}
	if len(c.ErrorLines) == 0 || !strings.Contains(c.ErrorLines[0], "CUDA error") {
		t.Errorf("error lines = %q", c.ErrorLines)
	}
	if c.JobID != "" {
		t.Errorf("job = %q with no job running", c.JobID)
	}
}

// A crash while a job holds the router carries the job; once the job
// lets go, events carry none.
func TestCrashDuringAJobIsTagged(t *testing.T) {
	s := newTestServer(t)
	openTestNotes(t, s)
	s.env = &jobEnv{s: s, ownsRouter: true}
	s.notesJob.Store(&jobRef{ID: "job-1", Name: "night run"})
	crash := enginenotes.Outcome{Kind: enginenotes.KindCrash, RouterName: "gone", ExitStatus: 1, Time: time.Now()}
	s.recordOutcome(crash)

	sum := allNotes(s)
	if sum.Totals.FailuresInJobs != 1 {
		t.Errorf("failures in jobs = %d, want 1", sum.Totals.FailuresInJobs)
	}
	c := sum.Failures[0]
	if c.JobID != "job-1" || c.JobName != "night run" {
		t.Errorf("crash job = %q %q, want job-1 night run", c.JobID, c.JobName)
	}
	// A model the registry does not have keeps the router's name.
	if c.ModelID != "gone" || c.ModelName != "gone" {
		t.Errorf("crash model = %q %q, want the router name", c.ModelID, c.ModelName)
	}

	s.env.mu.Lock()
	s.env.ownsRouter = false
	s.env.mu.Unlock()
	s.recordOutcome(crash)
	if got := allNotes(s).Totals.FailuresInJobs; got != 1 {
		t.Errorf("failures in jobs = %d after the job ended, want still 1", got)
	}
}

func TestRouterStartReasons(t *testing.T) {
	s, proc := evalStopServer(t)
	openTestNotes(t, s)
	s.builder = testBuilder(t, testBuild("b1", fakeLlamaServerDir(t)))
	s.cfg.ActiveBuild = "b1"

	if err := s.startRouter(); err != nil {
		t.Fatalf("start: %v", err)
	}
	waitRouterRunning(t, proc)
	if err := proc.Stop(); err != nil {
		t.Fatal(err)
	}
	if err := s.startRouterWith(routerOptions{job: true}); err != nil {
		t.Fatalf("job start: %v", err)
	}
	waitRouterRunning(t, proc)

	tot := allNotes(s).Totals
	if tot.RouterStartsUser != 1 || tot.RouterStartsJob != 1 {
		t.Errorf("starts = %d user, %d job; want 1 and 1", tot.RouterStartsUser, tot.RouterStartsJob)
	}
}

// usageServer is a proxy in front of a fake router that has the test
// model loaded and answers /v1 requests with v1.
func usageServer(t *testing.T, v1 http.HandlerFunc) (*Server, *models.Model, http.Handler) {
	t.Helper()
	s, f, m := startFakeRouter(t)
	openTestNotes(t, s)
	s.bench = benchmark.NewStore(t.TempDir(), nil) // Live Performance samples
	f.reset(map[string]string{m.ID: "loaded"}, nil)
	f.mu.Lock()
	f.v1 = v1
	f.mu.Unlock()
	return s, m, s.newProxyHandler()
}

func post(t *testing.T, h http.Handler, path, model string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(`{"model":"`+model+`"}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec
}

func usageOf(t *testing.T, s *Server, id string) enginenotes.ModelRow {
	t.Helper()
	for _, r := range s.notes.Summary(0, nil, nil).Models {
		if r.ModelID == id {
			return r
		}
	}
	return enginenotes.ModelRow{}
}

// Each response counts once, whatever shape it has.
func TestProxyCountsUsage(t *testing.T) {
	cases := []struct {
		name, path string
		answer     func(w http.ResponseWriter)
		want       enginenotes.ModelRow
	}{
		{"chat JSON", "/v1/chat/completions", func(w http.ResponseWriter) {
			w.Header().Set("Content-Type", "application/json")
			io.WriteString(w, `{"model":"x","timings":{"prompt_n":120,"predicted_n":30}}`)
		}, enginenotes.ModelRow{Requests: 1, PromptTokens: 120, GenTokens: 30}},
		{"chat stream", "/v1/chat/completions", func(w http.ResponseWriter) {
			w.Header().Set("Content-Type", "text/event-stream")
			io.WriteString(w, "data: {\"model\":\"x\",\"choices\":[]}\n\n")
			io.WriteString(w, "data: {\"model\":\"x\",\"timings\":{\"prompt_n\":50,\"predicted_n\":7}}\n\n")
			io.WriteString(w, "data: [DONE]\n\n")
		}, enginenotes.ModelRow{Requests: 1, PromptTokens: 50, GenTokens: 7}},
		{"stream without timings", "/v1/completions", func(w http.ResponseWriter) {
			w.Header().Set("Content-Type", "text/event-stream")
			io.WriteString(w, "data: {\"model\":\"x\"}\n\ndata: [DONE]\n\n")
		}, enginenotes.ModelRow{Requests: 1}},
		{"embeddings", "/v1/embeddings", func(w http.ResponseWriter) {
			w.Header().Set("Content-Type", "application/json")
			io.WriteString(w, `{"data":[],"usage":{"prompt_tokens":64,"total_tokens":64}}`)
		}, enginenotes.ModelRow{Requests: 1, PromptTokens: 64}},
		{"rerank with total only", "/v1/rerank", func(w http.ResponseWriter) {
			w.Header().Set("Content-Type", "application/json")
			io.WriteString(w, `{"results":[],"usage":{"total_tokens":9}}`)
		}, enginenotes.ModelRow{Requests: 1, PromptTokens: 9}},
		{"no usage object", "/v1/embeddings", func(w http.ResponseWriter) {
			io.WriteString(w, `not json`)
		}, enginenotes.ModelRow{Requests: 1}},
		{"failed", "/v1/chat/completions", func(w http.ResponseWriter) {
			w.WriteHeader(http.StatusInternalServerError)
			io.WriteString(w, `{"error":"boom"}`)
		}, enginenotes.ModelRow{Requests: 1, FailedRequests: 1}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			s, m, h := usageServer(t, func(w http.ResponseWriter, r *http.Request) { c.answer(w) })
			rec := post(t, h, c.path, m.ID)
			io.ReadAll(rec.Result().Body)
			testutil.Eventually(t, func() bool { return usageOf(t, s, m.ID).Requests > 0 }, "no request counted")
			got := usageOf(t, s, m.ID)
			if got.Requests != c.want.Requests || got.FailedRequests != c.want.FailedRequests ||
				got.PromptTokens != c.want.PromptTokens || got.GenTokens != c.want.GenTokens {
				t.Errorf("usage = %d requests (%d failed), %d prompt, %d gen; want %+v",
					got.Requests, got.FailedRequests, got.PromptTokens, got.GenTokens, c.want)
			}
		})
	}
}

// Job traffic stays out of usage, and so do paths that are not model
// requests.
func TestProxyUsageLeavesOutJobsAndOtherPaths(t *testing.T) {
	s, m, h := usageServer(t, func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, `{"timings":{"prompt_n":1,"predicted_n":1}}`)
	})
	s.env = &jobEnv{s: s, ownsRouter: true}
	post(t, h, "/v1/chat/completions", m.ID)
	s.env.mu.Lock()
	s.env.ownsRouter = false
	s.env.mu.Unlock()
	post(t, h, "/v1/audio/speech", m.ID)
	time.Sleep(50 * time.Millisecond)
	if got := usageOf(t, s, m.ID); got.Requests != 0 {
		t.Errorf("usage = %+v, want nothing counted", got)
	}
}

func TestEngineNotesEndpoints(t *testing.T) {
	// A record that began 60 days ago, so both crashes are inside it.
	path := filepath.Join(t.TempDir(), "engine-notes.json")
	since := time.Now().Add(-60 * 24 * time.Hour).UTC().Format(time.RFC3339)
	if err := os.WriteFile(path, []byte(`{"version":1,"counting_since":"`+since+`","events":[]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	st, err := enginenotes.Open(path, time.Now)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { st.Close() })
	s := newTestServer(t)
	s.notes = st
	s.notes.Record(enginenotes.Event{Kind: enginenotes.KindCrash, ModelID: "m", Time: time.Now().Add(-40 * 24 * time.Hour)})
	s.notes.Record(enginenotes.Event{Kind: enginenotes.KindCrash, ModelID: "m", Time: time.Now().Add(-time.Hour)})

	get := func(q string) *httptest.ResponseRecorder {
		rec := httptest.NewRecorder()
		s.handleEngineNotes(rec, httptest.NewRequest(http.MethodGet, "/api/service/notes"+q, nil))
		return rec
	}
	if rec := get("?period=bogus"); rec.Code != http.StatusBadRequest {
		t.Errorf("bogus period: status %d, want 400", rec.Code)
	}
	if rec := get(""); rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), `"crashes":1`) {
		t.Errorf("default period: %d %s, want 24 h with one crash", rec.Code, rec.Body.String())
	}

	del := func(age string) *httptest.ResponseRecorder {
		req := httptest.NewRequest(http.MethodPost, "/api/service/notes/delete",
			strings.NewReader(url.Values{"older_than": {age}}.Encode()))
		req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
		rec := httptest.NewRecorder()
		s.handleEngineNotesDelete(rec, req)
		return rec
	}
	if rec := del("bogus"); rec.Code != http.StatusBadRequest {
		t.Errorf("bogus age: status %d, want 400", rec.Code)
	}
	rec := del("30d")
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), `"removed_events":1`) {
		t.Errorf("delete 30d: %d %s, want one event removed", rec.Code, rec.Body.String())
	}
	if got := allNotes(s).Totals.Crashes; got != 1 {
		t.Errorf("crashes left = %d, want the recent one", got)
	}
}
