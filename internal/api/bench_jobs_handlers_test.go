package api

import (
	"context"
	"encoding/csv"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/tmac1973/llama-toolchest/internal/benchmark"
	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/testutil"
)

// bjhEnv is a job environment for handler tests. The first thing a cell
// does is ask whether its build can run; this answer waits until the
// test releases it or the job is cancelled, and then fails the cell. So
// a submitted job stays "running" for as long as the test needs, and no
// router, GPU or network is touched. Every other JobEnv method is left
// unset and would panic if a cell reached it.
type bjhEnv struct {
	benchmark.JobEnv
	release     chan struct{}
	releaseOnce sync.Once
	mu          sync.Mutex
	checked     []string
}

func (e *bjhEnv) CheckBuildRunnable(ctx context.Context, buildID string) error {
	e.mu.Lock()
	e.checked = append(e.checked, buildID)
	e.mu.Unlock()
	select {
	case <-ctx.Done():
	case <-e.release:
	}
	return errors.New("the test build cannot run")
}

func (e *bjhEnv) ClearEphemeralConfig(context.Context) error { return nil }

func (e *bjhEnv) checkCount() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return len(e.checked)
}

func (e *bjhEnv) releaseAll() { e.releaseOnce.Do(func() { close(e.release) }) }

// bjhServer returns a test server with a real benchmark store and a job
// queue that runs on bjhEnv. The store holds the given jobs. At the end
// of the test any running job is released and the queue is waited on,
// so no job goroutine outlives the test's temporary directories.
func bjhServer(t *testing.T, jobs ...benchmark.BenchmarkJob) (*Server, *bjhEnv) {
	t.Helper()
	s := newTestServer(t)
	s.bench = benchmark.NewStore(t.TempDir(), nil)
	env := &bjhEnv{release: make(chan struct{})}
	s.jobs = benchmark.NewJobQueue(s.bench, env)
	t.Cleanup(func() {
		env.releaseAll()
		testutil.Eventually(t, func() bool { _, running := s.jobs.Status(); return !running },
			"the job queue did not become idle")
	})
	for _, j := range jobs {
		if j.Kind == "" {
			j.Kind = benchmark.JobKindBatch
		}
		if j.CreatedAt.IsZero() {
			j.CreatedAt = time.Now()
		}
		s.bench.SaveJob(j)
	}
	return s, env
}

// bjhRequest builds a request with the chi "id" URL parameter set, as the
// router would for /api/benchmark-jobs/{id}/...
func bjhRequest(method, target, body, id string) *http.Request {
	r := httptest.NewRequest(method, target, strings.NewReader(body))
	if id != "" {
		rc := chi.NewRouteContext()
		rc.URLParams.Add("id", id)
		r = r.WithContext(context.WithValue(r.Context(), chi.RouteCtxKey, rc))
	}
	return r
}

// bjhWaitStatus waits until the stored job has the given status.
func bjhWaitStatus(t *testing.T, s *Server, id, status string) {
	t.Helper()
	testutil.EventuallyMsg(t, func() bool {
		j, err := s.bench.GetJob(id)
		return err == nil && j.Status == status
	}, func() string {
		j, err := s.bench.GetJob(id)
		if err != nil {
			return "job " + id + ": " + err.Error()
		}
		return "job " + id + " status is " + j.Status + ", want " + status
	})
}

// bjhStartBusyJob submits a one-cell job straight to the queue and waits
// until it runs, so a test has "another job is running" without going
// through the create handler.
func bjhStartBusyJob(t *testing.T, s *Server) string {
	t.Helper()
	const id = "job-busy"
	job := benchmark.BenchmarkJob{
		ID: id, Name: "busy", Kind: benchmark.JobKindBatch, CreatedAt: time.Now(),
		ModelIDs: []string{"m9"}, BuildIDs: []string{"b1"}, Presets: []string{"internal-quick"},
		Cells: benchmark.ExpandCellsWithStarts([]string{"m9"}, nil, []string{"b1"}, []string{"internal-quick"}, nil),
	}
	if err := s.jobs.Submit(job); err != nil {
		t.Fatalf("submit the busy job: %v", err)
	}
	bjhWaitStatus(t, s, id, benchmark.JobStatusRunning)
	return id
}

// bjhReadOnlyStore returns a benchmark store over a benchmarks.json that
// cannot be parsed, so the store refuses every write.
func bjhReadOnlyStore(t *testing.T) *benchmark.Store {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "config", "benchmarks.json")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("not json"), 0o644); err != nil {
		t.Fatal(err)
	}
	st := benchmark.NewStore(dir, nil)
	if st.Writable() == nil {
		t.Fatal("the store over an unreadable file is writable; the test setup is wrong")
	}
	return st
}

// bjhAddModel registers a model with a saved config in the server's
// registry.
func bjhAddModel(t *testing.T, s *Server, id string) {
	t.Helper()
	if err := s.registry.Add(&models.Model{
		ID: id, ModelID: "org/" + id + "-GGUF", Quant: "Q4_K_M", Filename: id + ".gguf",
		FilePath: filepath.Join(t.TempDir(), id+".gguf"),
	}); err != nil {
		t.Fatal(err)
	}
	cfg := models.DefaultConfig()
	cfg.Enabled = true
	if err := s.registry.SetConfig(id, &cfg); err != nil {
		t.Fatal(err)
	}
}

// Creating a job expands the matrix into one cell per model × build ×
// preset, answers 201 with the stored job, and starts it on the queue.
//
// The handler writes the job to the response after it has given the job
// to the queue, while the queue's goroutine sets each cell's status. The
// queue keeps its own copy of the cells, so with -race this checks that
// the two do not share them.
func TestBJHCreateJobQueuesTheMatrix(t *testing.T) {
	s, env := bjhServer(t)
	body := `{"name":"two models","description":"d","model_ids":["m1","m2"],"build_ids":["b1"],"presets":["internal-quick"]}`
	rec := httptest.NewRecorder()
	s.handleCreateJob(rec, bjhRequest(http.MethodPost, "/api/benchmark-jobs", body, ""))
	if rec.Code != http.StatusCreated {
		t.Fatalf("status %d, want 201: %s", rec.Code, rec.Body.String())
	}
	var job benchmark.BenchmarkJob
	if err := json.Unmarshal(rec.Body.Bytes(), &job); err != nil {
		t.Fatalf("response is not a job: %v\n%s", err, rec.Body.String())
	}
	if job.ID == "" || job.Name != "two models" || job.Description != "d" || job.Kind != benchmark.JobKindBatch {
		t.Errorf("job fields = id %q name %q description %q kind %q", job.ID, job.Name, job.Description, job.Kind)
	}
	if len(job.Cells) != 2 {
		t.Fatalf("cells = %d, want 2 (2 models × 1 build × 1 preset)", len(job.Cells))
	}
	seen := map[string]bool{}
	for _, c := range job.Cells {
		if c.BuildID != "b1" || c.Preset != "internal-quick" {
			t.Errorf("cell %+v does not carry the job's build and preset", c)
		}
		seen[c.ModelID] = true
	}
	if !seen["m1"] || !seen["m2"] {
		t.Errorf("cells cover models %v, want m1 and m2", seen)
	}

	bjhWaitStatus(t, s, job.ID, benchmark.JobStatusRunning)
	testutil.Eventually(t, func() bool { return env.checkCount() > 0 }, "the first cell never started")
}

// Cancelling the running job answers 204 and the job ends as cancelled,
// which is how the user stops a long job from the page.
func TestBJHCancelJobStopsTheRunningJob(t *testing.T) {
	s, env := bjhServer(t)
	id := bjhStartBusyJob(t, s)
	testutil.Eventually(t, func() bool { return env.checkCount() > 0 }, "the cell never started")

	rec := httptest.NewRecorder()
	s.handleCancelJob(rec, bjhRequest(http.MethodPost, "/x", "", id))
	if rec.Code != http.StatusNoContent {
		t.Fatalf("cancel status %d, want 204: %s", rec.Code, rec.Body.String())
	}
	bjhWaitStatus(t, s, id, benchmark.JobStatusCanceled)
}

// Every check a job must pass is applied when it is created, so a job
// that cannot run is refused with 400 and a message saying why, and
// nothing is stored or queued. A refusal here is much cheaper than a
// cell failing hours into a run.
func TestBJHCreateJobRefusesInvalidRequests(t *testing.T) {
	s, env := bjhServer(t)
	bjhAddModel(t, s, "m1")
	cases := []struct {
		name, body, want string
	}{
		{"bad JSON", `{"name":`, "invalid JSON"},
		{"no name", `{"name":" ","model_ids":["m1"],"build_ids":["b1"],"presets":["internal-quick"]}`, "name is required"},
		{"no presets", `{"name":"j","model_ids":["m1"],"build_ids":["b1"]}`, "are all required"},
		{"unknown parameter", `{"name":"j","model_ids":["m1"],"build_ids":["b1"],"presets":["internal-quick"],"params":{"no_such_field":["1"]}}`, "unknown parameter"},
		{"micro-batch above every batch", `{"name":"j","model_ids":["m1"],"build_ids":["b1"],"presets":["internal-quick"],"overrides":{"batch_size":512,"ubatch_size":4096}}`, "micro-batch"},
		{"GPU assignment with tensor split", `{"name":"j","model_ids":["m1"],"build_ids":["b1"],"presets":["internal-quick"],"overrides":{"gpu_assign":"all","tensor_split":"1,1"}}`, "not both"},
		{"GPU assignment with no GPUs", `{"name":"j","model_ids":["m1"],"build_ids":["b1"],"presets":["internal-quick"],"overrides":{"gpu_assign":"all"}}`, "no GPUs detected"},
		{"KL reference not installed", `{"name":"j","model_ids":["m1"],"build_ids":["b1"],"presets":["kl-divergence-quick"],"kl_reference":"ghost"}`, "not an installed model"},
		{"starting point for another model", `{"name":"j","model_ids":["m1"],"build_ids":["b1"],"presets":["internal-quick"],"starts":{"m9":[""]}}`, "not one of the job's models"},
		{"saved profile that does not exist", `{"name":"j","model_ids":["m1"],"build_ids":["b1"],"presets":["internal-quick"],"starts":{"m1":["ghost"]}}`, "no saved profile named"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			rec := httptest.NewRecorder()
			s.handleCreateJob(rec, bjhRequest(http.MethodPost, "/api/benchmark-jobs", c.body, ""))
			if rec.Code != http.StatusBadRequest {
				t.Fatalf("status %d, want 400: %s", rec.Code, rec.Body.String())
			}
			if !strings.Contains(rec.Body.String(), c.want) {
				t.Errorf("message %q does not contain %q", rec.Body.String(), c.want)
			}
		})
	}
	if jobs := s.bench.ListJobs(); len(jobs) != 0 {
		t.Errorf("refused requests stored %d jobs", len(jobs))
	}
	if env.checkCount() != 0 {
		t.Error("a refused request started a cell")
	}
}

// Only one job runs at a time. A second job created while the first is
// running is refused with 409 Conflict, so the page can say "a job is
// already running" instead of a generic error.
func TestBJHCreateJobConflictsWithRunningJob(t *testing.T) {
	s, _ := bjhServer(t)
	bjhStartBusyJob(t, s)
	body := `{"name":"j","model_ids":["m1"],"build_ids":["b1"],"presets":["internal-quick"]}`
	rec := httptest.NewRecorder()
	s.handleCreateJob(rec, bjhRequest(http.MethodPost, "/api/benchmark-jobs", body, ""))
	if rec.Code != http.StatusConflict {
		t.Fatalf("second job: status %d, want 409: %s", rec.Code, rec.Body.String())
	}
	if n := len(s.bench.ListJobs()); n != 1 {
		t.Errorf("stored jobs = %d, want only the running one", n)
	}
}

// A job created while the benchmark history is read-only is refused: it
// would run, and its results would be lost at the next restart.
func TestBJHCreateJobRefusedOnReadOnlyStore(t *testing.T) {
	s, _ := bjhServer(t)
	s.bench = bjhReadOnlyStore(t)
	s.jobs = benchmark.NewJobQueue(s.bench, &bjhEnv{release: make(chan struct{})})
	body := `{"name":"j","model_ids":["m1"],"build_ids":["b1"],"presets":["internal-quick"]}`
	rec := httptest.NewRecorder()
	s.handleCreateJob(rec, bjhRequest(http.MethodPost, "/api/benchmark-jobs", body, ""))
	if rec.Code == http.StatusCreated || rec.Code < 400 || rec.Code >= 500 {
		t.Fatalf("status %d, want a 4xx refusal: %s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "will not be saved") {
		t.Errorf("message %q does not say the history cannot be saved", rec.Body.String())
	}
	if _, running := s.jobs.Status(); running {
		t.Error("a job started on a read-only store")
	}
}

// A job that measures a saved profile carries a copy of that profile, and
// the starting point is rewritten to the profile's stored spelling, so the
// cells and the copy match even when the request used other letter case.
// decodeJobRequest is shared by create and edit, so both get this.
func TestBJHDecodeJobRequestCopiesNamedProfiles(t *testing.T) {
	s := newTestServer(t)
	bjhAddModel(t, s, "m1")
	if _, err := s.registry.SaveProfile("m1", "Fast", "test", "b1"); err != nil {
		t.Fatal(err)
	}
	body := `{"name":"j","model_ids":["m1"],"build_ids":["b1"],"presets":["internal-quick"],"starts":{"m1":["","FAST"]}}`
	req, profiles, err := s.decodeJobRequest(httptest.NewRequest(http.MethodPost, "/x", strings.NewReader(body)))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(profiles) != 1 || profiles[0].Name != "Fast" || profiles[0].ModelID != "m1" || profiles[0].BuildID != "b1" {
		t.Fatalf("profiles = %+v, want one copy of m1's Fast", profiles)
	}
	if profiles[0].CopiedAt.IsZero() {
		t.Error("the copy has no CopiedAt time")
	}
	if got := req.Starts["m1"]; len(got) != 2 || got[0] != "" || got[1] != "Fast" {
		t.Errorf("starts = %q, want current settings and the stored spelling Fast", got)
	}
}

// A request that passes every check decodes to the job it describes,
// with params already split into overrides and sweeps.
func TestBJHDecodeJobRequestAcceptsValidJob(t *testing.T) {
	s := newTestServer(t)
	body := `{"name":"j","description":"d","model_ids":["m1"],"build_ids":["b1","b2"],"presets":["internal-quick"],"params":{"context_size":["4096"],"top_k":["10","20"]}}`
	req, profiles, err := s.decodeJobRequest(httptest.NewRequest(http.MethodPost, "/x", strings.NewReader(body)))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if req.Name != "j" || req.Description != "d" || len(req.BuildIDs) != 2 {
		t.Errorf("decoded request = %+v", req)
	}
	if req.Overrides == nil || req.Overrides.ContextSize == nil || *req.Overrides.ContextSize != 4096 {
		t.Errorf("context_size was not fixed at 4096: %+v", req.Overrides)
	}
	if len(req.Sweeps) != 1 || req.Sweeps[0].Field != "top_k" {
		t.Errorf("sweeps = %+v, want the top_k axis", req.Sweeps)
	}
	if len(profiles) != 0 {
		t.Errorf("profiles = %+v, want none for a job with no saved profiles", profiles)
	}
}

// Editing a job keeps the cells that already completed when the new
// matrix still contains them, adds the new cells, and runs the job again.
// Without this, widening a finished job would measure everything again.
//
// Like TestBJHCreateJobQueuesTheMatrix, with -race this also checks that
// the answer and the running job do not share the cells.
func TestBJHUpdateJobKeepsCompletedCellsAndResubmits(t *testing.T) {
	cells := benchmark.ExpandCellsWithStarts([]string{"m1"}, nil, []string{"b1"}, []string{"internal-quick"}, nil)
	cells[0].Status = benchmark.CellStatusCompleted
	cells[0].BenchmarkRunID = "run-1"
	s, env := bjhServer(t, benchmark.BenchmarkJob{
		ID: "job-1", Name: "old", Status: benchmark.JobStatusCompleted,
		ModelIDs: []string{"m1"}, BuildIDs: []string{"b1"}, Presets: []string{"internal-quick"},
		Cells: cells,
	})
	body := `{"name":"new","model_ids":["m1","m2"],"build_ids":["b1"],"presets":["internal-quick"]}`
	rec := httptest.NewRecorder()
	s.handleUpdateJob(rec, bjhRequest(http.MethodPut, "/api/benchmark-jobs/job-1", body, "job-1"))
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	var job benchmark.BenchmarkJob
	if err := json.Unmarshal(rec.Body.Bytes(), &job); err != nil {
		t.Fatal(err)
	}
	if job.ID != "job-1" || job.Name != "new" {
		t.Errorf("updated job = id %q name %q", job.ID, job.Name)
	}
	if len(job.Cells) != 2 {
		t.Fatalf("cells = %d, want 2", len(job.Cells))
	}
	for _, c := range job.Cells {
		switch c.ModelID {
		case "m1":
			if c.Status != benchmark.CellStatusCompleted || c.BenchmarkRunID != "run-1" {
				t.Errorf("the completed m1 cell was not kept: %+v", c)
			}
		case "m2":
			if c.Status == benchmark.CellStatusCompleted {
				t.Errorf("the new m2 cell is already completed: %+v", c)
			}
		default:
			t.Errorf("unexpected cell %+v", c)
		}
	}
	bjhWaitStatus(t, s, "job-1", benchmark.JobStatusRunning)
	// Only the new cell runs; the kept one is not measured again.
	testutil.Eventually(t, func() bool { return env.checkCount() == 1 }, "the new cell never started")
}

// The edit handler refuses what it cannot do with the status that says
// why: 400 for the Ad-Hoc job and for an invalid definition, 404 for a
// job that does not exist, 409 when the history is read-only.
func TestBJHUpdateJobErrors(t *testing.T) {
	valid := `{"name":"j","model_ids":["m1"],"build_ids":["b1"],"presets":["internal-quick"]}`
	s, _ := bjhServer(t, benchmark.BenchmarkJob{
		ID: "job-1", Name: "old", Status: benchmark.JobStatusCompleted,
		ModelIDs: []string{"m1"}, BuildIDs: []string{"b1"}, Presets: []string{"internal-quick"},
	})
	cases := []struct {
		name, id, body string
		want           int
		msg            string
	}{
		{"Ad-Hoc job", benchmark.AdhocJobID, valid, http.StatusBadRequest, "cannot be edited"},
		{"bad JSON", "job-1", `nope`, http.StatusBadRequest, "invalid JSON"},
		{"invalid definition", "job-1", `{"name":"","model_ids":["m1"],"build_ids":["b1"],"presets":["internal-quick"]}`, http.StatusBadRequest, "name is required"},
		{"unknown job", "job-missing", valid, http.StatusNotFound, "not found"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			rec := httptest.NewRecorder()
			s.handleUpdateJob(rec, bjhRequest(http.MethodPut, "/x", c.body, c.id))
			if rec.Code != c.want {
				t.Fatalf("status %d, want %d: %s", rec.Code, c.want, rec.Body.String())
			}
			if !strings.Contains(rec.Body.String(), c.msg) {
				t.Errorf("message %q does not contain %q", rec.Body.String(), c.msg)
			}
		})
	}
	if j, _ := s.bench.GetJob("job-1"); j.Name != "old" {
		t.Errorf("a refused edit changed the job's name to %q", j.Name)
	}

	t.Run("read-only history", func(t *testing.T) {
		ro, _ := bjhServer(t)
		ro.bench = bjhReadOnlyStore(t)
		ro.jobs = benchmark.NewJobQueue(ro.bench, &bjhEnv{release: make(chan struct{})})
		rec := httptest.NewRecorder()
		ro.handleUpdateJob(rec, bjhRequest(http.MethodPut, "/x", valid, "job-1"))
		if rec.Code != http.StatusConflict {
			t.Fatalf("status %d, want 409: %s", rec.Code, rec.Body.String())
		}
	})
}

// An edit that is refused with 409 because another job is running must
// leave the job as it was. Saving the new definition first and having the
// queue refuse it afterwards would show the user an error while the job
// had in fact been changed and reset to "pending", with nothing that will
// ever run it.
func TestBJHUpdateJobRefusedWhileAnotherRunsLeavesJobUnchanged(t *testing.T) {
	s, _ := bjhServer(t, benchmark.BenchmarkJob{
		ID: "job-old", Name: "old", Status: benchmark.JobStatusCompleted,
		ModelIDs: []string{"m1"}, BuildIDs: []string{"b1"}, Presets: []string{"internal-quick"},
		Cells: benchmark.ExpandCellsWithStarts([]string{"m1"}, nil, []string{"b1"}, []string{"internal-quick"}, nil),
	})
	bjhStartBusyJob(t, s)

	rec := httptest.NewRecorder()
	s.handleUpdateJob(rec, bjhRequest(http.MethodPut, "/x",
		`{"name":"renamed","model_ids":["m1","m2"],"build_ids":["b1"],"presets":["internal-quick"]}`, "job-old"))
	if rec.Code != http.StatusConflict {
		t.Fatalf("status %d, want 409: %s", rec.Code, rec.Body.String())
	}
	j, err := s.bench.GetJob("job-old")
	if err != nil {
		t.Fatal(err)
	}
	if j.Name != "old" || j.Status != benchmark.JobStatusCompleted || len(j.Cells) != 1 {
		t.Errorf("after a refused edit the job is name %q, status %q, %d cells; want it unchanged (old, completed, 1 cell)",
			j.Name, j.Status, len(j.Cells))
	}
}

// Cancel only acts on the job that is running. Cancelling a finished or
// unknown job is refused with 400, and does not stop the running one.
func TestBJHCancelJobRefusesJobThatIsNotRunning(t *testing.T) {
	s, _ := bjhServer(t, benchmark.BenchmarkJob{ID: "job-done", Name: "done", Status: benchmark.JobStatusCompleted})
	busy := bjhStartBusyJob(t, s)

	for _, id := range []string{"job-done", "job-missing"} {
		rec := httptest.NewRecorder()
		s.handleCancelJob(rec, bjhRequest(http.MethodPost, "/x", "", id))
		if rec.Code != http.StatusBadRequest {
			t.Errorf("%s: status %d, want 400", id, rec.Code)
		}
		if !strings.Contains(rec.Body.String(), "not running") {
			t.Errorf("%s: message %q does not say the job is not running", id, rec.Body.String())
		}
	}
	if cur, ok := s.jobs.Status(); !ok || cur.ID != busy {
		t.Error("cancelling another job stopped the running one")
	}
}

// Retry puts only the failed cells back to run; completed cells keep their
// results. It answers 202 because the job then runs in the background.
func TestBJHRetryFailedCellsRequeuesOnlyFailedCells(t *testing.T) {
	cells := benchmark.ExpandCellsWithStarts([]string{"m1", "m2"}, nil, []string{"b1"}, []string{"internal-quick"}, nil)
	cells[0].Status, cells[0].BenchmarkRunID = benchmark.CellStatusCompleted, "run-1"
	cells[1].Status, cells[1].Error = benchmark.CellStatusFailed, "out of memory"
	failedModel := cells[1].ModelID
	s, env := bjhServer(t, benchmark.BenchmarkJob{
		ID: "job-1", Name: "j", Status: benchmark.JobStatusCompleted,
		ModelIDs: []string{"m1", "m2"}, BuildIDs: []string{"b1"}, Presets: []string{"internal-quick"},
		Cells: cells,
	})
	rec := httptest.NewRecorder()
	s.handleRetryFailedCells(rec, bjhRequest(http.MethodPost, "/x", "", "job-1"))
	if rec.Code != http.StatusAccepted {
		t.Fatalf("status %d, want 202: %s", rec.Code, rec.Body.String())
	}
	bjhWaitStatus(t, s, "job-1", benchmark.JobStatusRunning)
	testutil.Eventually(t, func() bool { return env.checkCount() == 1 }, "the failed cell was not run again")
	j, _ := s.bench.GetJob("job-1")
	for _, c := range j.Cells {
		if c.ModelID == failedModel {
			if c.Status == benchmark.CellStatusFailed && c.Error == "out of memory" {
				t.Errorf("the failed cell was not reset: %+v", c)
			}
		} else if c.Status != benchmark.CellStatusCompleted || c.BenchmarkRunID != "run-1" {
			t.Errorf("the completed cell was changed: %+v", c)
		}
	}
}

// Retry refuses a job with nothing to retry or that does not exist (400),
// and a retry while another job runs (409), so the page can tell the user
// to wait instead of showing a generic failure.
func TestBJHRetryFailedCellsErrors(t *testing.T) {
	failed := benchmark.ExpandCellsWithStarts([]string{"m1"}, nil, []string{"b1"}, []string{"internal-quick"}, nil)
	failed[0].Status = benchmark.CellStatusFailed
	done := benchmark.ExpandCellsWithStarts([]string{"m1"}, nil, []string{"b1"}, []string{"internal-quick"}, nil)
	done[0].Status = benchmark.CellStatusCompleted
	s, _ := bjhServer(t,
		benchmark.BenchmarkJob{ID: "job-ok", Name: "ok", Status: benchmark.JobStatusCompleted, Cells: done},
		benchmark.BenchmarkJob{ID: "job-failed", Name: "failed", Status: benchmark.JobStatusFailed, Cells: failed},
	)
	retry := func(id string) *httptest.ResponseRecorder {
		rec := httptest.NewRecorder()
		s.handleRetryFailedCells(rec, bjhRequest(http.MethodPost, "/x", "", id))
		return rec
	}
	if rec := retry("job-ok"); rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), "no failed cells") {
		t.Errorf("job with no failed cells: status %d, body %q; want 400 naming the reason", rec.Code, rec.Body.String())
	}
	if rec := retry("job-missing"); rec.Code != http.StatusBadRequest {
		t.Errorf("unknown job: status %d, want 400", rec.Code)
	}

	bjhStartBusyJob(t, s)
	if rec := retry("job-failed"); rec.Code != http.StatusConflict {
		t.Errorf("retry while another job runs: status %d, want 409", rec.Code)
	}
}

// The per-job export holds the job and only its own runs. JSON is the
// full record; CSV honours the scope. The file name carries the job ID
// and the scope so several downloads do not overwrite each other.
func TestBJHExportJob(t *testing.T) {
	s, _ := bjhServer(t,
		benchmark.BenchmarkJob{ID: "job-1", Name: "one", Status: benchmark.JobStatusCompleted},
		benchmark.BenchmarkJob{ID: "job-2", Name: "two", Status: benchmark.JobStatusCompleted},
	)
	run := func(id, job string) benchmark.BenchmarkRun {
		return benchmark.BenchmarkRun{
			ID: id, JobID: job, CreatedAt: time.Unix(1, 0).UTC(), Status: benchmark.StatusCompleted,
			ModelID: "m1", ModelName: "M1", Preset: "internal-quick",
			Summary: &benchmark.BenchmarkSummary{AvgPromptTokPerSec: 100},
		}
	}
	s.bench.Save(run("run-a", "job-1"))
	s.bench.Save(run("run-b", "job-1"))
	s.bench.Save(run("run-other", "job-2"))

	export := func(id, query string) *httptest.ResponseRecorder {
		rec := httptest.NewRecorder()
		s.handleExportJob(rec, bjhRequest(http.MethodGet, "/api/benchmark-jobs/"+id+"/export"+query, "", id))
		return rec
	}

	t.Run("JSON", func(t *testing.T) {
		rec := export("job-1", "?format=json")
		if rec.Code != http.StatusOK {
			t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
		}
		if cd := rec.Header().Get("Content-Disposition"); !strings.Contains(cd, `"job-job-1.json"`) {
			t.Errorf("Content-Disposition = %q", cd)
		}
		var env ExportEnvelope
		if err := json.Unmarshal(rec.Body.Bytes(), &env); err != nil {
			t.Fatalf("not an export envelope: %v", err)
		}
		if env.Version != exportEnvelopeVersion || len(env.Jobs) != 1 || env.Jobs[0].ID != "job-1" {
			t.Errorf("envelope version %d, jobs %v", env.Version, env.Jobs)
		}
		ids := map[string]bool{}
		for _, r := range env.Runs {
			ids[r.ID] = true
		}
		if len(env.Runs) != 2 || !ids["run-a"] || !ids["run-b"] {
			t.Errorf("runs = %v, want run-a and run-b only", ids)
		}
	})

	t.Run("CSV summary", func(t *testing.T) {
		rec := export("job-1", "?scope=summary")
		if rec.Code != http.StatusOK {
			t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
		}
		if ct := rec.Header().Get("Content-Type"); !strings.HasPrefix(ct, "text/csv") {
			t.Errorf("Content-Type = %q, want text/csv (CSV is the default format)", ct)
		}
		if cd := rec.Header().Get("Content-Disposition"); !strings.Contains(cd, `"job-job-1-summary.csv"`) {
			t.Errorf("Content-Disposition = %q", cd)
		}
		rows, err := csv.NewReader(strings.NewReader(rec.Body.String())).ReadAll()
		if err != nil {
			t.Fatalf("not CSV: %v", err)
		}
		if len(rows) != 3 {
			t.Errorf("rows = %d, want a header and one row per run of the job (3)", len(rows))
		}
		if strings.Contains(rec.Body.String(), "run-other") {
			t.Error("the export includes a run of another job")
		}
	})

	for _, c := range []struct {
		name, id, query string
		want            int
	}{
		{"unknown job", "job-missing", "", http.StatusNotFound},
		{"bad format", "job-1", "?format=xml", http.StatusBadRequest},
		{"bad scope", "job-1", "?scope=everything", http.StatusBadRequest},
	} {
		t.Run(c.name, func(t *testing.T) {
			if rec := export(c.id, c.query); rec.Code != c.want {
				t.Errorf("status %d, want %d: %s", rec.Code, c.want, rec.Body.String())
			}
		})
	}
}

// The "runs" choice of a job delete: empty and "cascade" delete the runs
// with the job, "orphan" keeps them under Ad-Hoc, and anything else is an
// error rather than a silent cascade that would delete data.
func TestBJHParseRunsDisposition(t *testing.T) {
	for in, want := range map[string]benchmark.DeleteDisposition{
		"":        benchmark.DeleteCascade,
		"cascade": benchmark.DeleteCascade,
		"orphan":  benchmark.DeleteOrphan,
	} {
		got, err := parseRunsDisposition(in)
		if err != nil || got != want {
			t.Errorf("parseRunsDisposition(%q) = %q, %v; want %q", in, got, err, want)
		}
	}
	for _, in := range []string{"keep", "Cascade", "delete"} {
		if _, err := parseRunsDisposition(in); err == nil {
			t.Errorf("parseRunsDisposition(%q) accepted an unknown choice", in)
		}
	}
}

// Params, the shape the job form posts, decide every parameter they can
// express: one value fixes it, two or more sweep it, and a parameter left
// out of params is unset even when the older overrides field set it.
// Without params the older fields are used as they are.
func TestBJHResolveSweeps(t *testing.T) {
	t.Run("params replace overrides and sweeps", func(t *testing.T) {
		req := jobCreateRequest{
			Overrides: &benchmark.ConfigOverrides{ContextSize: ptr(4096)},
			Sweeps:    []benchmark.SweepAxis{{Field: "top_k", Values: []string{"10", "20"}}},
			Params: map[string][]string{
				"gpu_layers":  {"40"},
				"ubatch_size": {"512", "1024"},
			},
		}
		if err := resolveSweeps(&req); err != nil {
			t.Fatal(err)
		}
		if req.Overrides == nil || req.Overrides.GPULayers == nil || *req.Overrides.GPULayers != 40 {
			t.Errorf("gpu_layers was not fixed at 40: %+v", req.Overrides)
		}
		if req.Overrides != nil && req.Overrides.ContextSize != nil {
			t.Errorf("context_size survived although params left it out: %d", *req.Overrides.ContextSize)
		}
		if len(req.Sweeps) != 1 || req.Sweeps[0].Field != "ubatch_size" || len(req.Sweeps[0].Values) != 2 {
			t.Errorf("sweeps = %+v, want only the ubatch_size axis", req.Sweeps)
		}
	})

	t.Run("no params keeps the older fields", func(t *testing.T) {
		ov := &benchmark.ConfigOverrides{ContextSize: ptr(4096)}
		sw := []benchmark.SweepAxis{{Field: "top_k", Values: []string{"10", "20"}}}
		req := jobCreateRequest{Overrides: ov, Sweeps: sw}
		if err := resolveSweeps(&req); err != nil {
			t.Fatal(err)
		}
		if req.Overrides != ov || len(req.Sweeps) != 1 || req.Sweeps[0].Field != "top_k" {
			t.Errorf("fields changed without params: %+v %+v", req.Overrides, req.Sweeps)
		}
	})

	t.Run("bad parameter value", func(t *testing.T) {
		req := jobCreateRequest{Params: map[string][]string{"gpu_layers": {"many"}}}
		if err := resolveSweeps(&req); err == nil {
			t.Error("a non-number gpu_layers value was accepted")
		}
	})
}
