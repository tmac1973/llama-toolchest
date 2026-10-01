package benchmark

import (
	"testing"
	"time"
)

// bulkStore holds three batch jobs, each with one run, plus one ad-hoc run.
func bulkStore(t *testing.T) *Store {
	t.Helper()
	s := NewStore(t.TempDir(), nil)
	for _, id := range []string{"job-a", "job-b", "job-c"} {
		s.SaveJob(BenchmarkJob{ID: id, Name: id, Kind: JobKindBatch, Status: JobStatusCompleted, CreatedAt: time.Now()})
		s.Save(BenchmarkRun{ID: "run-" + id, JobID: id, Status: StatusCompleted, CreatedAt: time.Now()})
	}
	s.Save(BenchmarkRun{ID: "run-adhoc", Status: StatusCompleted, CreatedAt: time.Now()})
	return s
}

func TestDeleteJobsCascadeRemovesJobsAndRuns(t *testing.T) {
	s := bulkStore(t)
	deleted, skipped, err := s.DeleteJobs([]string{"job-a", "job-b"}, DeleteCascade)
	if err != nil {
		t.Fatal(err)
	}
	if len(deleted) != 2 || len(skipped) != 0 {
		t.Fatalf("deleted %v, skipped %v; want both deleted", deleted, skipped)
	}
	for _, id := range []string{"job-a", "job-b"} {
		if _, err := s.GetJob(id); err == nil {
			t.Errorf("%s still exists", id)
		}
		if _, err := s.Get("run-" + id); err == nil {
			t.Errorf("run-%s still exists after a delete with runs", id)
		}
	}
	if _, err := s.Get("run-job-c"); err != nil {
		t.Errorf("an unselected job's run was deleted: %v", err)
	}
}

func TestDeleteJobsOrphanMovesRunsToAdhoc(t *testing.T) {
	s := bulkStore(t)
	if _, _, err := s.DeleteJobs([]string{"job-a", "job-c"}, DeleteOrphan); err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"run-job-a", "run-job-c"} {
		r, err := s.Get(id)
		if err != nil {
			t.Fatalf("%s was deleted, want it kept: %v", id, err)
		}
		if r.JobID != AdhocJobID {
			t.Errorf("%s belongs to %q, want %q", id, r.JobID, AdhocJobID)
		}
	}
}

// One job that cannot be deleted must not stop the rest.
func TestDeleteJobsSkipsAdhocAndMissing(t *testing.T) {
	s := bulkStore(t)
	deleted, skipped, err := s.DeleteJobs([]string{AdhocJobID, "job-gone", "job-b"}, DeleteCascade)
	if err != nil {
		t.Fatal(err)
	}
	if len(deleted) != 1 || deleted[0] != "job-b" {
		t.Errorf("deleted = %v, want [job-b]", deleted)
	}
	if _, ok := skipped[AdhocJobID]; !ok {
		t.Errorf("the Ad-Hoc job was not reported as skipped: %v", skipped)
	}
	if _, ok := skipped["job-gone"]; !ok {
		t.Errorf("the missing job was not reported as skipped: %v", skipped)
	}
	if _, err := s.GetJob(AdhocJobID); err != nil {
		t.Errorf("the Ad-Hoc job was deleted: %v", err)
	}
}

// The deletes are written to disk: a store opened on the same directory
// sees them.
func TestDeleteJobsPersists(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir, nil)
	s.SaveJob(BenchmarkJob{ID: "job-a", Name: "a", Kind: JobKindBatch, CreatedAt: time.Now()})
	s.SaveJob(BenchmarkJob{ID: "job-b", Name: "b", Kind: JobKindBatch, CreatedAt: time.Now()})
	if _, _, err := s.DeleteJobs([]string{"job-a"}, DeleteCascade); err != nil {
		t.Fatal(err)
	}
	reopened := NewStore(dir, nil)
	if _, err := reopened.GetJob("job-a"); err == nil {
		t.Error("job-a is back after reopening the store")
	}
	if _, err := reopened.GetJob("job-b"); err != nil {
		t.Errorf("job-b is gone after reopening the store: %v", err)
	}
}

func TestDeleteJobsRejectsUnknownDisposition(t *testing.T) {
	s := bulkStore(t)
	if _, _, err := s.DeleteJobs([]string{"job-a"}, "shred"); err == nil {
		t.Error("an unknown disposition was accepted")
	}
	if _, err := s.GetJob("job-a"); err != nil {
		t.Errorf("job-a was deleted by a refused request: %v", err)
	}
}
