package api

import (
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/benchmark"
)

// The version changes with anything a collapsed row shows that can
// change on its own, and the list carries the version it was rendered
// at, so the page can tell when it is out of date.
func TestJobListVersion(t *testing.T) {
	s := bulkDeleteServer(t, benchmark.BenchmarkJob{
		ID: "job-a", Name: "a", Status: benchmark.JobStatusRunning,
		Cells: []benchmark.JobCell{{Status: benchmark.CellStatusRunning}, {Status: benchmark.CellStatusPending}},
	})
	version := func() string {
		rec := httptest.NewRecorder()
		s.handleJobListVersion(rec, httptest.NewRequest("GET", "/api/benchmark-jobs/version", nil))
		return rec.Body.String()
	}
	v1 := version()
	if v1 == "" || version() != v1 {
		t.Fatalf("version = %q, want a stable non-empty value", v1)
	}

	rec := httptest.NewRecorder()
	s.renderJobList(rec, s.bench.ListJobs())
	if !strings.Contains(rec.Body.String(), `id="job-list-version" data-version="`+v1+`"`) {
		t.Errorf("the list does not carry the version it was rendered at\n%s", rec.Body.String())
	}

	job, _ := s.bench.GetJob("job-a")
	job.Cells[0].Status = benchmark.CellStatusCompleted
	s.bench.SaveJob(*job)
	v2 := version()
	if v2 == v1 {
		t.Error("a finished cell did not change the version")
	}

	job.Status = benchmark.JobStatusCompleted
	s.bench.SaveJob(*job)
	if version() == v2 {
		t.Error("a finished job did not change the version")
	}

	v3 := version()
	s.bench.SaveJob(benchmark.BenchmarkJob{ID: "job-b", Name: "b", Kind: benchmark.JobKindBatch, CreatedAt: time.Now()})
	if version() == v3 {
		t.Error("a new job did not change the version")
	}
}
