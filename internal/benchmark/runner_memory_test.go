package benchmark

import "testing"

// The measurement is taken inside Run, after the model is up: the figure
// does not exist when the RunConfig is built. These cover the wiring
// that carries it onto the run.

func TestRunConfigMemoryCallbackRecordsTheFootprint(t *testing.T) {
	var asked string
	cfg := RunConfig{
		Run: BenchmarkRun{ModelID: "m1"},
		Memory: func(modelID string) (MemorySnapshot, bool) {
			asked = modelID
			return MemorySnapshot{GPUGiB: 23, Cards: 4}, true
		},
	}

	run := cfg.Run
	if cfg.Memory != nil {
		if mem, ok := cfg.Memory(run.ModelID); ok {
			run.Memory = &mem
		}
	}

	if asked != "m1" {
		t.Errorf("callback asked about %q; want the run's own model", asked)
	}
	if run.Memory == nil || run.Memory.GPUGiB != 23 {
		t.Errorf("footprint not recorded: %+v", run.Memory)
	}
}

// Below log verbosity 4 llama.cpp itemises nothing, so a cell has
// timings and no footprint. That must leave the field nil rather than a
// zeroed struct, which every reader downstream treats as "measured, and
// it used nothing".
func TestRunKeepsMemoryNilWhenNothingWasMeasured(t *testing.T) {
	cfg := RunConfig{
		Run:    BenchmarkRun{ModelID: "m1"},
		Memory: func(string) (MemorySnapshot, bool) { return MemorySnapshot{}, false },
	}

	run := cfg.Run
	if mem, ok := cfg.Memory(run.ModelID); ok {
		run.Memory = &mem
	}
	if run.Memory != nil {
		t.Errorf("memory = %+v; want nil", run.Memory)
	}
}

func TestMemorySnapshotUnreported(t *testing.T) {
	m := MemorySnapshot{GPUGiB: 23.0, CardDeltaGiB: 24.5}
	if got := m.Unreported(); got < 1.49 || got > 1.51 {
		t.Errorf("unreported = %v; want 1.5", got)
	}
	// Without a card reading there is no remainder to state.
	none := MemorySnapshot{GPUGiB: 23.0}
	if got := none.Unreported(); got != 0 {
		t.Errorf("unreported = %v; want 0 when the counters were not captured", got)
	}
}

// End to end through a job cell: the cell already pays for the load, so
// what that load cost is recorded on the run beside its timings.
func TestJobCellRecordsWhatTheLoadCost(t *testing.T) {
	router := newFakeRouter(t)
	env := &fakeEnv{
		routerURL: router.URL,
		saved:     ConfigSnapshot{GPULayers: 999, ContextSize: 8192, Threads: 8},
		memReport: &MemorySnapshot{
			GPUGiB: 23, WeightsGiB: 20, KVGiB: 2, ComputeGiB: 1,
			HostGiB: 1.5, CardDeltaGiB: 24.5, Cards: 4,
		},
	}

	job, store := runJob(t, oneCellJob(nil), env)
	if job.Status != JobStatusCompleted {
		t.Fatalf("job status = %s, want completed", job.Status)
	}
	runs := store.RunsForJob(job.ID)
	if len(runs) != 1 {
		t.Fatalf("got %d runs, want 1", len(runs))
	}
	mem := runs[0].Memory
	if mem == nil {
		t.Fatal("the cell recorded no footprint")
	}
	if mem.GPUGiB != 23 || mem.WeightsGiB != 20 || mem.Cards != 4 {
		t.Errorf("footprint = %+v", mem)
	}
}

// A router below log verbosity 4 itemises nothing. The cell still runs
// and still reports its timings; the footprint is simply absent.
func TestJobCellSurvivesAnUnmeasuredLoad(t *testing.T) {
	router := newFakeRouter(t)
	env := &fakeEnv{
		routerURL: router.URL,
		saved:     ConfigSnapshot{GPULayers: 999, ContextSize: 8192, Threads: 8},
	}

	job, store := runJob(t, oneCellJob(nil), env)
	if job.Status != JobStatusCompleted {
		t.Fatalf("job status = %s, want completed", job.Status)
	}
	runs := store.RunsForJob(job.ID)
	if len(runs) != 1 {
		t.Fatalf("got %d runs, want 1", len(runs))
	}
	if runs[0].Memory != nil {
		t.Errorf("memory = %+v; want nil", runs[0].Memory)
	}
	if runs[0].Summary == nil {
		t.Error("the cell lost its timings")
	}
}
