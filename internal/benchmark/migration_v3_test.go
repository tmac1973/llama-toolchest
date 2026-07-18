package benchmark

import "testing"

func TestMarkUnverifiedConfigsFlagsOverriddenJobsOnly(t *testing.T) {
	ngl := 50
	jobs := []BenchmarkJob{
		{ID: "with-overrides", Overrides: &ConfigOverrides{GPULayers: &ngl}},
		{ID: "clean"},
	}
	runs := []BenchmarkRun{
		{ID: "r1", JobID: "with-overrides"},
		{ID: "r2", JobID: "clean"},
		{ID: "r3", JobID: "with-overrides"},
	}

	if !markUnverifiedConfigs(jobs, runs) {
		t.Fatal("expected migration to report a change")
	}
	if !runs[0].ConfigUnverified || !runs[2].ConfigUnverified {
		t.Error("runs from a job with overrides must be flagged")
	}
	if runs[1].ConfigUnverified {
		t.Error("a run from a job without overrides must not be flagged")
	}
}

// No overrides anywhere means nothing to flag and no rewrite of the
// benchmarks file.
func TestMarkUnverifiedConfigsNoopWithoutOverrides(t *testing.T) {
	jobs := []BenchmarkJob{{ID: "clean"}}
	runs := []BenchmarkRun{{ID: "r1", JobID: "clean"}}

	if markUnverifiedConfigs(jobs, runs) {
		t.Error("expected no change")
	}
	if runs[0].ConfigUnverified {
		t.Error("run should not be flagged")
	}
}

// Re-running the migration must not report spurious changes, so loading
// an already-migrated file doesn't rewrite it on every startup.
func TestMarkUnverifiedConfigsIsIdempotent(t *testing.T) {
	ngl := 50
	jobs := []BenchmarkJob{{ID: "j", Overrides: &ConfigOverrides{GPULayers: &ngl}}}
	runs := []BenchmarkRun{{ID: "r", JobID: "j"}}

	if !markUnverifiedConfigs(jobs, runs) {
		t.Fatal("first pass should report a change")
	}
	if markUnverifiedConfigs(jobs, runs) {
		t.Error("second pass should report no change")
	}
}

// An orphaned run whose job is gone can't be attributed, so it stays
// unflagged rather than being guessed at.
func TestMarkUnverifiedConfigsIgnoresOrphanRuns(t *testing.T) {
	ngl := 50
	jobs := []BenchmarkJob{{ID: "j", Overrides: &ConfigOverrides{GPULayers: &ngl}}}
	runs := []BenchmarkRun{{ID: "orphan", JobID: "deleted-job"}}

	if markUnverifiedConfigs(jobs, runs) {
		t.Error("orphan run should not be flagged")
	}
}
