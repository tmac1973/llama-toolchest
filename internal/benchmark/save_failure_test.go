package benchmark

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

// makeUnwritable stops benchmarks.json from being replaced: atomicfile
// writes a temporary file next to it, which a read-only folder refuses.
func makeUnwritable(t *testing.T, s *Store) {
	t.Helper()
	dir := filepath.Dir(s.benchmarkPath())
	if err := os.Chmod(dir, 0o500); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.Chmod(dir, 0o755) })
	if f, err := os.CreateTemp(dir, "probe"); err == nil {
		f.Close()
		os.Remove(f.Name())
		t.Skip("folder is still writable (running as root?)")
	}
}

// A delete whose save fails used to report success, and the deleted items
// came back after a restart. It now reports the failure and leaves the
// items in place, matching the file on disk.
func TestDeleteReportsFailedSaveAndKeepsItems(t *testing.T) {
	s := bulkStore(t)
	makeUnwritable(t, s)

	if err := s.Delete("run-adhoc"); !errors.Is(err, ErrSaveFailed) {
		t.Errorf("Delete: %v, want ErrSaveFailed", err)
	}
	if _, err := s.Get("run-adhoc"); err != nil {
		t.Errorf("run removed in memory although the save failed: %v", err)
	}

	if err := s.DeleteJob("job-a", DeleteCascade); !errors.Is(err, ErrSaveFailed) {
		t.Errorf("DeleteJob: %v, want ErrSaveFailed", err)
	}
	if _, _, err := s.DeleteJobs([]string{"job-b"}, DeleteCascade); !errors.Is(err, ErrSaveFailed) {
		t.Errorf("DeleteJobs: %v, want ErrSaveFailed", err)
	}
	for _, id := range []string{"job-a", "job-b"} {
		if _, err := s.GetJob(id); err != nil {
			t.Errorf("%s removed in memory although the save failed", id)
		}
		if _, err := s.Get("run-" + id); err != nil {
			t.Errorf("run-%s removed in memory although the save failed", id)
		}
	}
}
