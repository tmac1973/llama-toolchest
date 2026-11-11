package benchmark

import (
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/models"
)

// Every run records the whole config it ran under: the cell's swept
// values over the live config for a current-settings cell, and over the
// job's profile copy for a profile cell, with the settings a snapshot
// leaves out (sampling, the system prompt) kept from that base.
func TestRunsRecordTheirLaunchConfig(t *testing.T) {
	router := newFakeRouter(t)
	temp := 0.6
	liveCfg := models.ModelConfig{Enabled: true, GPULayers: 999, ContextSize: 8192, Threads: 8, UBatchSize: 512,
		SystemPrompt: "live prompt", Temperature: &temp}
	env := &fakeEnv{routerURL: router.URL, saved: SnapshotFromConfig(liveCfg, "", false), savedConfig: liveCfg}

	job := oneCellJob(nil)
	job.Starts = map[string][]string{"m": {"", "Fast"}}
	fast := profileCopy("m", "Fast", 1024)
	fast.Config.SystemPrompt = "profile prompt"
	job.Profiles = []JobProfile{fast}
	job.Sweeps = []SweepAxis{{Field: "threads", Values: []string{"4"}}}
	job.Cells = ExpandCellsWithStarts(job.ModelIDs, job.Starts, job.BuildIDs, job.Presets, job.Sweeps)

	done, store := runJob(t, job, env)
	if done.Status != JobStatusCompleted {
		t.Fatalf("status = %s, cells %+v", done.Status, done.Cells)
	}
	want := map[string]struct {
		ubatch int
		prompt string
	}{"": {512, "live prompt"}, "Fast": {1024, "profile prompt"}}
	for _, c := range done.Cells {
		run, err := store.Get(c.BenchmarkRunID)
		if err != nil {
			t.Fatalf("cell %q has no run: %v", c.Profile, err)
		}
		lc := run.LaunchConfig
		if lc == nil {
			t.Fatalf("cell %q recorded no launch config", c.Profile)
		}
		w := want[c.Profile]
		if lc.UBatchSize != w.ubatch || lc.SystemPrompt != w.prompt || lc.Threads != 4 {
			t.Errorf("cell %q launch config ubatch=%d prompt=%q threads=%d, want %d, %q, 4",
				c.Profile, lc.UBatchSize, lc.SystemPrompt, lc.Threads, w.ubatch, w.prompt)
		}
	}
}
