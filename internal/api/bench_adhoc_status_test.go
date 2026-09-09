package api

import (
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/benchmark"
)

// adhocServer is a Server with a real store holding the given adhoc runs.
func adhocServer(t *testing.T, statuses ...string) *Server {
	t.Helper()
	s := benchListServer(t)
	s.bench = benchmark.NewStore(t.TempDir(), nil)
	for i, st := range statuses {
		// JobID left empty, which is how an ad-hoc run actually arrives:
		// Save assigns it and creates the synthetic job on first use.
		s.bench.Save(benchmark.BenchmarkRun{
			ID:        "r" + string(rune('1'+i)),
			Status:    st,
			CreatedAt: time.Unix(0, 0).UTC(),
			ModelID:   "m", ModelName: "M", Quant: "Q4_K_M", Preset: "quick",
		})
	}
	return s
}

// The synthetic adhoc job is built with a fixed "completed" status, so the
// collapsed row claimed completed while a run inside it was still going —
// and expanding it showed the running run directly under the badge
// contradicting it.
func TestAdhocRowStatusFollowsItsRuns(t *testing.T) {
	cases := []struct {
		name     string
		statuses []string
		want     string
	}{
		{"nothing running", []string{benchmark.StatusCompleted}, benchmark.JobStatusCompleted},
		{"one running", []string{benchmark.StatusCompleted, benchmark.StatusRunning}, benchmark.JobStatusRunning},
		{"several, one running", []string{
			benchmark.StatusCompleted, benchmark.StatusFailed, benchmark.StatusRunning,
		}, benchmark.JobStatusRunning},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s := adhocServer(t, tc.statuses...)
			rec := httptest.NewRecorder()
			s.renderJobList(rec, s.bench.ListJobs())
			out := rec.Body.String()

			want := `id="job-status-adhoc" class="job-status status-` + tc.want
			if !strings.Contains(out, want) {
				t.Errorf("adhoc row should read %q\n%s", tc.want, out)
			}
		})
	}
}

// There is no ad-hoc row until an ad-hoc run exists — the job is synthesized
// on first use, not seeded — so an empty store shows no row rather than an
// idle one.
func TestAdhocRowAbsentUntilARunExists(t *testing.T) {
	s := adhocServer(t)
	rec := httptest.NewRecorder()
	s.renderJobList(rec, s.bench.ListJobs())
	if strings.Contains(rec.Body.String(), "job-status-adhoc") {
		t.Errorf("no runs should mean no ad-hoc row\n%s", rec.Body.String())
	}
}

// The row is drawn by the job list, which has no idea a run started. Without
// these the badge stays stale until a full refresh.
func TestAdhocDetailUpdatesTheRowOutOfBand(t *testing.T) {
	s := adhocServer(t, benchmark.StatusRunning)
	rec := httptest.NewRecorder()
	s.renderAdhocDetail(rec)
	out := rec.Body.String()

	if !strings.Contains(out, `id="job-status-adhoc" hx-swap-oob="true"`) {
		t.Errorf("no out-of-band status update\n%s", out)
	}
	if !strings.Contains(out, "status-running") {
		t.Errorf("the out-of-band badge should read running\n%s", out)
	}
	if !strings.Contains(out, `id="job-progress-adhoc" hx-swap-oob="true"`) {
		t.Errorf("no out-of-band progress update\n%s", out)
	}
	if !strings.Contains(out, ">1 run<") {
		t.Errorf("progress should read the run count, singular\n%s", out)
	}
}

// Polling only while something is running: an idle history of hundreds of
// runs should not re-render itself every two seconds forever.
func TestAdhocDetailPollsOnlyWhileRunning(t *testing.T) {
	running := httptest.NewRecorder()
	adhocServer(t, benchmark.StatusRunning).renderAdhocDetail(running)
	if !strings.Contains(running.Body.String(), `hx-trigger="every 2s"`) {
		t.Error("a running list should poll so the rows and badge stay current")
	}

	idle := httptest.NewRecorder()
	adhocServer(t, benchmark.StatusCompleted).renderAdhocDetail(idle)
	if strings.Contains(idle.Body.String(), `hx-trigger="every 2s"`) {
		t.Error("an idle list should not poll")
	}
}

// The run list itself still renders — the wrapper must not replace it.
func TestAdhocDetailStillRendersTheRuns(t *testing.T) {
	s := adhocServer(t, benchmark.StatusCompleted)
	rec := httptest.NewRecorder()
	s.renderAdhocDetail(rec)
	if !strings.Contains(rec.Body.String(), "bench-runs-container") {
		t.Errorf("the run list is missing\n%s", rec.Body.String())
	}
}

// The shared renderer serves GET /api/benchmarks/ as well, where an adhoc
// badge update would target a row that view knows nothing about.
func TestBenchmarkListItselfEmitsNoAdhocUpdates(t *testing.T) {
	s := adhocServer(t, benchmark.StatusRunning)
	rec := httptest.NewRecorder()
	s.renderBenchmarkList(rec, s.bench.RunsForJob(benchmark.AdhocJobID))
	if strings.Contains(rec.Body.String(), "job-status-adhoc") {
		t.Error("the shared list renderer should not emit adhoc row updates")
	}
}
