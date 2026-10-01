package benchmark

import (
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/evaluate"
	"github.com/tmac1973/llama-toolchest/internal/models"
)

func profileCopy(model, name string, ubatch int) JobProfile {
	return JobProfile{
		ModelID: model, Name: name, CopiedAt: time.Now(),
		Config: models.ModelConfig{Enabled: true, GPULayers: 999, ContextSize: 8192, Threads: 8, UBatchSize: ubatch},
	}
}

// Starting points multiply only their own model's cells, and a model's
// performance cells are grouped by starting point so the router reloads
// once per profile.
func TestExpandCellsWithStarts(t *testing.T) {
	starts := map[string][]string{"a": {"", "Fast", "Long"}}
	cells := ExpandCellsWithStarts([]string{"a", "b"}, starts, []string{"b1"},
		[]string{"internal-quick", "internal-standard"}, nil)

	// a: 3 starting points × 2 presets; b: current settings × 2 presets.
	if len(cells) != 8 {
		t.Fatalf("got %d cells, want 8: %+v", len(cells), cells)
	}
	var order []string
	for _, c := range cells {
		order = append(order, c.ModelID+"/"+c.Profile)
	}
	want := []string{"a/", "a/", "a/Fast", "a/Fast", "a/Long", "a/Long", "b/", "b/"}
	if !slices.Equal(order, want) {
		t.Errorf("cell order = %v, want %v", order, want)
	}
}

// A job saved before profile comparison has no Starts, and expands
// exactly as it did.
func TestExpandCellsWithoutStartsIsUnchanged(t *testing.T) {
	models, builds, presets := []string{"a", "b"}, []string{"b1", "b2"}, []string{"internal-quick"}
	axes := []SweepAxis{{Field: "ubatch_size", Values: []string{"256", "512"}}}
	old := ExpandCellsWithSweeps(models, builds, presets, axes)
	got := ExpandCellsWithStarts(models, nil, builds, presets, axes)
	if len(old) != len(got) {
		t.Fatalf("%d cells, want %d", len(got), len(old))
	}
	for i := range old {
		if identify(old[i]) != identify(got[i]) || got[i].Profile != "" {
			t.Errorf("cell %d = %+v, want %+v", i, got[i], old[i])
		}
	}
}

// Capability cells collapse per starting point, not across them: two
// profiles with different settings are two different evaluations.
func TestCapabilityCellsKeepOnePerStartingPoint(t *testing.T) {
	starts := map[string][]string{"a": {"", "Fast"}}
	cells := ExpandCellsWithStarts([]string{"a"}, starts, []string{"b1"},
		[]string{"perplexity-quick"}, []SweepAxis{{Field: "threads", Values: []string{"4", "8"}}})
	profiles := map[string]int{}
	for _, c := range cells {
		profiles[c.Profile]++
	}
	if profiles[""] == 0 || profiles["Fast"] == 0 || profiles[""] != profiles["Fast"] {
		t.Errorf("capability cells per starting point = %v, want the same non-zero count for each", profiles)
	}
}

// Each cell runs its own profile's copy, records the profile's name, and
// the current-settings cell runs the live config.
func TestProfileCellsRunTheirOwnCopy(t *testing.T) {
	router := newFakeRouter(t)
	live := ConfigSnapshot{GPULayers: 999, ContextSize: 8192, Threads: 8, UBatchSize: 512}
	env := &fakeEnv{routerURL: router.URL, saved: live}

	job := oneCellJob(nil)
	job.Starts = map[string][]string{"m": {"", "Fast", "Big"}}
	job.Profiles = []JobProfile{profileCopy("m", "Fast", 1024), profileCopy("m", "Big", 2048)}
	job.Cells = ExpandCellsWithStarts(job.ModelIDs, job.Starts, job.BuildIDs, job.Presets, nil)

	done, store := runJob(t, job, env)
	if done.Status != JobStatusCompleted {
		t.Fatalf("status = %s, cells %+v", done.Status, done.Cells)
	}
	want := map[string]struct {
		ubatch  int
		profile string
	}{"": {512, ""}, "Fast": {1024, "Fast"}, "Big": {2048, "Big"}}
	for _, c := range done.Cells {
		run, err := store.Get(c.BenchmarkRunID)
		if err != nil {
			t.Fatalf("cell %q has no run: %v", c.Profile, err)
		}
		w := want[c.Profile]
		if run.Config.UBatchSize != w.ubatch {
			t.Errorf("cell %q ran ubatch %d, want %d", c.Profile, run.Config.UBatchSize, w.ubatch)
		}
		if run.Config.ProfileName != w.profile || run.Config.ProfileEdited {
			t.Errorf("cell %q recorded profile %q edited=%v, want %q unedited",
				c.Profile, run.Config.ProfileName, run.Config.ProfileEdited, w.profile)
		}
	}
	// Two profiles, two applies. The current-settings cell needs none.
	applied, _ := env.snapshotCalls()
	if len(applied) != 2 {
		t.Errorf("applied %d configs, want one per profile (2)", len(applied))
	}
}

// A cell naming a profile the job holds no copy of fails; it must not
// measure the current settings under the profile's name.
func TestProfileCellWithoutCopyFails(t *testing.T) {
	router := newFakeRouter(t)
	env := &fakeEnv{routerURL: router.URL, saved: ConfigSnapshot{GPULayers: 999, ContextSize: 8192}}
	job := oneCellJob(nil)
	job.Starts = map[string][]string{"m": {"Gone"}}
	job.Cells = ExpandCellsWithStarts(job.ModelIDs, job.Starts, job.BuildIDs, job.Presets, nil)

	done, store := runJob(t, job, env)
	if done.Cells[0].Status != CellStatusFailed || !strings.Contains(done.Cells[0].Error, "Gone") {
		t.Errorf("cell = %+v, want failed naming the profile", done.Cells[0])
	}
	if n := len(store.RunsForJob(done.ID)); n != 0 {
		t.Errorf("%d runs recorded, want none", n)
	}
}

// A capability cell measuring a profile evaluates the profile's settings.
func TestCapabilityCellEvaluatesTheProfile(t *testing.T) {
	env := capabilityEnv(t, true)
	env.evalResult = evaluate.Result{Mode: "perplexity", Perplexity: 6.1}

	job := capJob("job-cap-profile", []string{"m4"}, []string{"perplexity-quick"})
	job.Starts = map[string][]string{"m4": {"Fast"}}
	p := profileCopy("m4", "Fast", 1024)
	p.Config.Threads = 3
	job.Profiles = []JobProfile{p}
	job.Cells = ExpandCellsWithStarts(job.ModelIDs, job.Starts, job.BuildIDs, job.Presets, nil)

	done, store := runJob(t, job, env)
	if done.Status != JobStatusCompleted {
		t.Fatalf("status = %s, cells %+v", done.Status, done.Cells)
	}
	run, err := store.Get(done.Cells[0].BenchmarkRunID)
	if err != nil {
		t.Fatal(err)
	}
	if run.Config.Threads != 3 || run.Config.ProfileName != "Fast" {
		t.Errorf("recorded threads %d profile %q, want the profile's 3 and \"Fast\"", run.Config.Threads, run.Config.ProfileName)
	}
	env.mu.Lock()
	calls := env.evalCalls
	env.mu.Unlock()
	if len(calls) != 1 || !slices.Contains(calls[0].Flags, "3") {
		t.Errorf("eval flags = %v, want the profile's thread count", calls)
	}
}

// Editing a job keeps the results of a profile whose settings did not
// change, and runs again the cells of one whose settings did.
func TestEditKeepsUnchangedProfileResults(t *testing.T) {
	s := NewStore(t.TempDir(), nil)
	starts := map[string][]string{"m": {"Fast", "Big"}}
	profiles := []JobProfile{profileCopy("m", "Fast", 1024), profileCopy("m", "Big", 2048)}
	job := BenchmarkJob{
		ID: "job-e", Name: "e", Kind: JobKindBatch, CreatedAt: time.Now(),
		ModelIDs: []string{"m"}, BuildIDs: []string{"b"}, Presets: []string{"internal-quick"},
		Starts: starts, Profiles: profiles,
		Cells: ExpandCellsWithStarts([]string{"m"}, starts, []string{"b"}, []string{"internal-quick"}, nil),
	}
	for i := range job.Cells {
		job.Cells[i].Status = CellStatusCompleted
		job.Cells[i].BenchmarkRunID = "run-" + job.Cells[i].Profile
		s.Save(BenchmarkRun{ID: job.Cells[i].BenchmarkRunID, JobID: job.ID, Status: StatusCompleted, CreatedAt: time.Now()})
	}
	s.SaveJob(job)

	changed := []JobProfile{profileCopy("m", "Fast", 1024), profileCopy("m", "Big", 4096)}
	got, err := s.UpdateJobDefinition(job.ID, JobDefinition{
		Name: "e", ModelIDs: job.ModelIDs, BuildIDs: job.BuildIDs, Presets: job.Presets,
		Starts: starts, Profiles: changed,
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range got.Cells {
		switch c.Profile {
		case "Fast":
			if c.Status != CellStatusCompleted || c.BenchmarkRunID != "run-Fast" {
				t.Errorf("unchanged profile lost its result: %+v", c)
			}
		case "Big":
			if c.Status != CellStatusPending {
				t.Errorf("changed profile kept a result measured with old settings: %+v", c)
			}
		}
	}
	if len(got.Profiles) != 2 || got.Profiles[1].Config.UBatchSize != 4096 {
		t.Errorf("job kept the old profile copies: %+v", got.Profiles)
	}
}
