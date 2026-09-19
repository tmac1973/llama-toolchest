package benchmark

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func writeBenchFile(t *testing.T, dir, contents string) string {
	t.Helper()
	path := filepath.Join(dir, "config", "benchmarks.json")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(contents), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

// assertStoreRefuses checks that user-started changes are refused, that a
// job cannot be submitted, and that the file is left byte-for-byte as found.
func assertStoreRefuses(t *testing.T, s *Store, path, before string) {
	t.Helper()
	if s.ReadOnlyReason() == "" {
		t.Fatal("ReadOnlyReason is empty; want the store read-only")
	}
	if err := s.Delete("r1"); !errors.Is(err, ErrStoreReadOnly) {
		t.Errorf("Delete: err = %v, want ErrStoreReadOnly", err)
	}
	if err := s.DeleteJob("job-1", DeleteCascade); !errors.Is(err, ErrStoreReadOnly) {
		t.Errorf("DeleteJob: err = %v, want ErrStoreReadOnly", err)
	}
	if _, err := s.UpdateJobDefinition("job-1", JobDefinition{Name: "x"}); !errors.Is(err, ErrStoreReadOnly) {
		t.Errorf("UpdateJobDefinition: err = %v, want ErrStoreReadOnly", err)
	}
	q := NewJobQueue(s, nil)
	if err := q.Submit(oneCellJob(nil)); !errors.Is(err, ErrStoreReadOnly) {
		t.Errorf("Submit: err = %v, want ErrStoreReadOnly", err)
	}
	// The backstop: a write that does get through is not persisted.
	s.Save(BenchmarkRun{ID: "r2", JobID: AdhocJobID})

	after, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(after) != before {
		t.Errorf("benchmarks.json changed on disk:\n%s", after)
	}
}

const benchJobAndRun = `"jobs": [{"id": "job-1", "name": "J"}, {"id": "adhoc", "name": "Ad-Hoc Runs"}],
  "runs": [{"id": "r1", "job_id": "job-1", "status": "completed"}]`

// A file in neither known layout used to leave the store empty, and the
// next save wrote that empty history over it.
func TestStoreCorruptFileIsReadOnly(t *testing.T) {
	dir := t.TempDir()
	corrupt := `{"version": 4, ` + benchJobAndRun // truncated: no closing brace
	path := writeBenchFile(t, dir, corrupt)

	s := NewStore(dir, nil)
	if !strings.Contains(s.ReadOnlyReason(), "could not be parsed") {
		t.Errorf("ReadOnlyReason = %q, want it to say the file could not be parsed", s.ReadOnlyReason())
	}
	assertStoreRefuses(t, s, path, corrupt)
}

// A newer file is shown but never rewritten, not even by the load-time
// migrations and fix-ups (which would otherwise mark running jobs failed
// and save).
func TestStoreNewerVersionIsReadOnly(t *testing.T) {
	dir := t.TempDir()
	newer := `{"version": 99, "future": true, ` + benchJobAndRun + `}`
	path := writeBenchFile(t, dir, newer)

	s := NewStore(dir, nil)
	if !strings.Contains(s.ReadOnlyReason(), "newer version") {
		t.Errorf("ReadOnlyReason = %q, want it to name a newer version", s.ReadOnlyReason())
	}
	if _, err := s.Get("r1"); err != nil {
		t.Errorf("run from the newer file is not listed: %v", err)
	}
	assertStoreRefuses(t, s, path, newer)
}

// The current version loads writable, and saving leaves no temporary file.
func TestStoreCurrentVersionIsWritable(t *testing.T) {
	dir := t.TempDir()
	path := writeBenchFile(t, dir, `{"version": 4, `+benchJobAndRun+`}`)

	s := NewStore(dir, nil)
	if reason := s.ReadOnlyReason(); reason != "" {
		t.Fatalf("current file made the store read-only: %s", reason)
	}
	if err := s.Delete("r1"); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(raw), `"r1"`) {
		t.Error("deleted run is still in benchmarks.json")
	}
	entries, err := os.ReadDir(filepath.Dir(path))
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range entries {
		if strings.HasSuffix(e.Name(), ".tmp") {
			t.Errorf("temporary file left behind: %s", e.Name())
		}
	}
}
