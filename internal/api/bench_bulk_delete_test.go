package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/autotune"
	"github.com/tmac1973/llama-toolchest/internal/benchmark"
)

// bulkDeleteServer has a real benchmark store holding the given jobs, an
// idle job queue and an empty autotune store.
func bulkDeleteServer(t *testing.T, jobs ...benchmark.BenchmarkJob) *Server {
	t.Helper()
	s := benchListServer(t)
	s.bench = benchmark.NewStore(t.TempDir(), nil)
	s.jobs = benchmark.NewJobQueue(s.bench, nil)
	s.tuneStore = autotune.NewStore(t.TempDir())
	for _, j := range jobs {
		if j.Kind == "" {
			j.Kind = benchmark.JobKindBatch
		}
		if j.CreatedAt.IsZero() {
			j.CreatedAt = time.Now()
		}
		s.bench.SaveJob(j)
	}
	return s
}

func postBulkDelete(t *testing.T, s *Server, body string) (*httptest.ResponseRecorder, jobBulkDeleteResponse) {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/api/benchmark-jobs/delete", strings.NewReader(body))
	rec := httptest.NewRecorder()
	s.handleBulkDeleteJobs(rec, req)
	var out jobBulkDeleteResponse
	if rec.Code == http.StatusOK {
		if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
			t.Fatalf("response is not JSON: %v\n%s", err, rec.Body.String())
		}
	}
	return rec, out
}

func TestBulkDeleteJobs(t *testing.T) {
	s := bulkDeleteServer(t,
		benchmark.BenchmarkJob{ID: "job-a", Name: "a", Status: benchmark.JobStatusCompleted},
		benchmark.BenchmarkJob{ID: "job-b", Name: "b", Status: benchmark.JobStatusFailed},
	)
	rec, out := postBulkDelete(t, s, `{"ids":["job-a","job-b","adhoc"],"runs":"orphan"}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	if len(out.Deleted) != 2 {
		t.Errorf("deleted = %v, want job-a and job-b", out.Deleted)
	}
	if _, ok := out.Skipped["adhoc"]; !ok {
		t.Errorf("the Ad-Hoc job should be reported as skipped: %v", out.Skipped)
	}
}

func TestBulkDeleteJobsRejectsBadRequests(t *testing.T) {
	s := bulkDeleteServer(t)
	for _, body := range []string{`{"ids":[]}`, `{"ids":["x"],"runs":"shred"}`, `not json`} {
		rec, _ := postBulkDelete(t, s, body)
		if rec.Code != http.StatusBadRequest {
			t.Errorf("%s: status %d, want 400", body, rec.Code)
		}
	}
}

// A stage of the autotune run that is going now is looked up again as
// the run moves on, so it is skipped. A stage of a finished run is not.
func TestBulkDeleteJobsSkipsActiveAutotuneStages(t *testing.T) {
	s := bulkDeleteServer(t,
		benchmark.BenchmarkJob{ID: "stage-live", Name: "live", AutotuneID: "tune-live", Status: benchmark.JobStatusCompleted},
		benchmark.BenchmarkJob{ID: "stage-old", Name: "old", AutotuneID: "tune-old", Status: benchmark.JobStatusCompleted},
	)
	if err := s.tuneStore.Save(&autotune.Autotune{ID: "tune-live", Status: autotune.StatusRunning, CreatedAt: time.Now()}); err != nil {
		t.Fatal(err)
	}
	if err := s.tuneStore.Save(&autotune.Autotune{ID: "tune-old", Status: autotune.StatusDone, CreatedAt: time.Now().Add(-time.Hour)}); err != nil {
		t.Fatal(err)
	}
	_, out := postBulkDelete(t, s, `{"ids":["stage-live","stage-old"]}`)
	if len(out.Deleted) != 1 || out.Deleted[0] != "stage-old" {
		t.Errorf("deleted = %v, want [stage-old]", out.Deleted)
	}
	if _, ok := out.Skipped["stage-live"]; !ok {
		t.Errorf("the active autotune stage should be skipped: %v", out.Skipped)
	}
}

// The Ad-Hoc row and a running job get no selection checkbox, since the
// server refuses to delete either.
func TestJobListSelectionCheckboxes(t *testing.T) {
	s := bulkDeleteServer(t,
		benchmark.BenchmarkJob{ID: "job-done", Name: "done", Status: benchmark.JobStatusCompleted},
		benchmark.BenchmarkJob{ID: "job-live", Name: "live", Status: benchmark.JobStatusRunning},
	)
	s.bench.Save(benchmark.BenchmarkRun{ID: "r1", Status: benchmark.StatusCompleted, CreatedAt: time.Now()})
	rec := httptest.NewRecorder()
	s.renderJobList(rec, s.bench.ListJobs())
	out := rec.Body.String()

	if !strings.Contains(out, `class="job-select" value="job-done"`) {
		t.Errorf("the finished job has no checkbox\n%s", out)
	}
	for _, id := range []string{"job-live", "adhoc"} {
		if strings.Contains(out, `class="job-select" value="`+id+`"`) {
			t.Errorf("%s should have no checkbox", id)
		}
	}
}
