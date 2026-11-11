package api

import (
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/go-chi/chi/v5"

	"github.com/tmac1973/llama-toolchest/internal/benchmark"
	"github.com/tmac1973/llama-toolchest/internal/models"
)

// newBenchUseServer is a profile server with a benchmark store holding one
// completed run of the test model, measured at ubatch 1024 and a prompt
// the live config does not have.
func newBenchUseServer(t *testing.T) (*Server, *benchmark.BenchmarkRun) {
	t.Helper()
	s := newProfileServer(t, "b100")
	s.bench = benchmark.NewStore(t.TempDir(), nil)
	setLiveContext(t, s, 8192)
	live, _ := s.registry.GetConfig(profTestID)
	launch := *live
	launch.UBatchSize = 1024
	launch.SystemPrompt = "be brief"
	job := benchmark.BenchmarkJob{ID: "job1", Name: "ubatch sweep", Status: benchmark.JobStatusCompleted, CreatedAt: time.Now()}
	s.bench.SaveJob(job)
	run := benchmark.BenchmarkRun{
		ID: "run1", JobID: job.ID, CreatedAt: time.Now(), Status: benchmark.StatusCompleted,
		ModelID: profTestID, ModelName: "r", Quant: "Q4_K_M", BuildID: "b100", Preset: "internal-quick",
		Config:        benchmark.SnapshotFromConfig(launch, "", false),
		LaunchConfig:  &launch,
		SweepValues:   map[string]string{"ubatch_size": "1024"},
		StartingPoint: benchmark.StartingPointCurrent,
		Summary:       &benchmark.BenchmarkSummary{AvgGenTokPerSec: 42.1, AvgPromptTokPerSec: 1830},
	}
	s.bench.Save(run)
	return s, &run
}

func (s *Server) doBenchUse(t *testing.T, method, runID string, form url.Values) string {
	t.Helper()
	r := chi.NewRouter()
	r.Get("/api/benchmarks/{id}/use", s.handleBenchUseForm)
	r.Post("/api/benchmarks/{id}/use", s.handleBenchUseSave)
	req := httptest.NewRequest(method, "/api/benchmarks/"+runID+"/use", strings.NewReader(form.Encode()))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("HX-Request", "true")
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)
	return rec.Body.String()
}

// The dialog suggests a name from the sweep point and lists what differs
// from the model's current settings.
func TestBenchUseFormSuggestsANameAndShowsChanges(t *testing.T) {
	s, _ := newBenchUseServer(t)
	body := s.doBenchUse(t, "GET", "run1", nil)
	for _, want := range []string{`value="ubatch_size=1024"`, "ubatch_size", "system_prompt", "be brief", "Save and apply"} {
		if !strings.Contains(body, want) {
			t.Errorf("dialog missing %q:\n%s", want, body)
		}
	}
}

// Save stores the run's whole config as a benchmark profile and leaves
// the live config alone.
func TestBenchUseSaveKeepsLiveConfig(t *testing.T) {
	s, run := newBenchUseServer(t)
	body := s.doBenchUse(t, "POST", "run1", url.Values{"profile_name": {"Fast prompts"}, "apply": {"0"}})
	if !strings.Contains(body, "Saved profile") {
		t.Fatalf("no confirmation:\n%s", body)
	}
	p, err := s.registry.GetProfile(profTestID, "Fast prompts")
	if err != nil {
		t.Fatal(err)
	}
	if p.Source != models.ProfileSourceBenchmark || p.BuildID != "b100" || !models.ProfileEqual(p.Config, *run.LaunchConfig) {
		t.Errorf("profile = %+v, want the run's config from a benchmark on b100", p)
	}
	if len(p.Notes) != 1 || !strings.Contains(p.Notes[0].Reason, `"ubatch sweep"`) {
		t.Errorf("notes = %+v, want one naming the job", p.Notes)
	}
	if cfg, _ := s.registry.GetConfig(profTestID); cfg.UBatchSize == 1024 || cfg.SystemPrompt != "" {
		t.Errorf("live config changed by a plain save: %+v", cfg)
	}
}

// Save and apply also makes the profile the model's settings.
func TestBenchUseSaveAndApply(t *testing.T) {
	s, _ := newBenchUseServer(t)
	body := s.doBenchUse(t, "POST", "run1", url.Values{"profile_name": {"Fast prompts"}, "apply": {"1"}})
	if !strings.Contains(body, `Saved and applied profile &#34;Fast prompts&#34; for r. The new settings take effect`) {
		t.Fatalf("no apply confirmation:\n%s", body)
	}
	cfg, _ := s.registry.GetConfig(profTestID)
	if cfg.UBatchSize != 1024 || cfg.SystemPrompt != "be brief" || cfg.ActiveProfile != "Fast prompts" {
		t.Errorf("live config = %+v, want the run's settings under the new profile", cfg)
	}
}

// A run recorded before launch configs were kept cannot be saved, and
// the dialog says why instead of saving a guess.
func TestBenchUseRefusesARunWithoutLaunchConfig(t *testing.T) {
	s, run := newBenchUseServer(t)
	run.LaunchConfig = nil
	s.bench.Save(*run)
	body := s.doBenchUse(t, "POST", "run1", url.Values{"profile_name": {"Old"}})
	if !strings.Contains(body, "before benchmarks kept their full settings") {
		t.Errorf("no reason given:\n%s", body)
	}
	if _, err := s.registry.GetProfile(profTestID, "Old"); err == nil {
		t.Error("a profile was saved from a run with no launch config")
	}
}

func TestSuggestBenchProfileName(t *testing.T) {
	s, run := newBenchUseServer(t)
	cases := []struct {
		name    string
		start   string
		profile string
		sweep   map[string]string
		want    string
	}{
		{"sweep", benchmark.StartingPointCurrent, "", map[string]string{"ubatch_size": "1024", "threads": "8"}, "threads=8, ubatch_size=1024"},
		{"profile and sweep", benchmark.StartingPointProfile, "Long", map[string]string{"threads": "8"}, "Long · threads=8"},
		{"profile only", benchmark.StartingPointProfile, "Long", nil, "Long (benchmarked)"},
		{"current settings, profile name ignored", benchmark.StartingPointCurrent, "Long", nil, "ubatch sweep"},
	}
	for _, c := range cases {
		r := *run
		r.StartingPoint, r.Config.ProfileName, r.SweepValues = c.start, c.profile, c.sweep
		if got := s.suggestBenchProfileName(&r); got != c.want {
			t.Errorf("%s: got %q, want %q", c.name, got, c.want)
		}
	}
}

// A completed cell offers Use; one whose run predates launch configs
// shows it turned off, with the reason.
func TestJobDetailOffersUse(t *testing.T) {
	s, run := newBenchUseServer(t)
	old := *run
	old.ID, old.LaunchConfig = "run-old", nil
	s.bench.Save(old)
	job, _ := s.bench.GetJob("job1")
	job.Cells = []benchmark.JobCell{
		{ModelID: profTestID, BuildID: "b100", Preset: "internal-quick", Status: benchmark.CellStatusCompleted, BenchmarkRunID: "run1", Attempt: 1},
		{ModelID: profTestID, BuildID: "b100", Preset: "internal-quick", Status: benchmark.CellStatusCompleted, BenchmarkRunID: "run-old", Attempt: 1},
		{ModelID: profTestID, BuildID: "b100", Preset: "internal-quick", Status: benchmark.CellStatusPending},
	}
	rec := httptest.NewRecorder()
	s.renderJobDetail(rec, job)
	out := rec.Body.String()
	if !strings.Contains(out, `openBenchUse('run1')`) {
		t.Errorf("the completed cell has no Use button\n%s", out)
	}
	if strings.Count(out, ">Use</button>") != 2 || !strings.Contains(out, "before benchmarks kept their full settings") {
		t.Errorf("want Use on both completed cells, the old one turned off with its reason\n%s", out)
	}
}
