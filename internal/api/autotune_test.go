package api

import (
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/autotune"
	"github.com/tmac1973/llama-toolchest/internal/benchmark"
	"github.com/tmac1973/llama-toolchest/internal/models"
)

// newTuneServer is a server with the autotune store and runner wired up,
// and one installed model.
func newTuneServer(t *testing.T) *Server {
	t.Helper()
	s := newHelperServer(t)
	dir := t.TempDir()
	s.tuneStore = autotune.NewStore(dir)
	s.bench = benchmark.NewStore(dir, nil)
	s.tuner = autotune.NewRunner(autotune.Deps{
		Store: s.tuneStore, Runs: s.bench, Registry: s.registry,
		ActiveBuild: func() string { return "b1" },
		Hardware:    func() (int, int) { return 1, 16 },
	})
	return s
}

func (s *Server) doTune(t *testing.T, method, path string, form url.Values) string {
	t.Helper()
	body := strings.NewReader("")
	if form != nil {
		body = strings.NewReader(form.Encode())
	}
	req := httptest.NewRequest(method, path, body)
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("HX-Request", "true")
	rec := httptest.NewRecorder()
	s.buildRouter().ServeHTTP(rec, req)
	return rec.Body.String()
}

// A model with no profile is offered one, because autotune measures from
// a profile rather than from a config that can change under it.
func TestAutotuneDialogWithoutAProfile(t *testing.T) {
	s := newTuneServer(t)
	out := s.doTune(t, "GET", "/api/models/"+profTestID+"/autotune", nil)
	if !strings.Contains(out, "measures from a saved profile") || !strings.Contains(out, "autotune/save-current") {
		t.Errorf("dialog does not offer to save a profile:\n%s", out)
	}

	out = s.doTune(t, "POST", "/api/models/"+profTestID+"/autotune/save-current", nil)
	if !strings.Contains(out, "Saved the current settings") {
		t.Errorf("saving the current settings failed:\n%s", out)
	}
	if _, err := s.registry.GetProfile(profTestID, currentSettingsProfile); err != nil {
		t.Errorf("the profile was not saved: %v", err)
	}
	if !strings.Contains(out, "What do you use this model for?") {
		t.Error("the dialog did not come back with the use-case question")
	}
}

func TestAutotuneDialogWithProfiles(t *testing.T) {
	s := newTuneServer(t)
	for _, name := range []string{"Mine", autoconfigProfileName} {
		if _, err := s.registry.SaveProfile(profTestID, name, models.ProfileSourceUser, "b1"); err != nil {
			t.Fatal(err)
		}
	}
	out := s.doTune(t, "GET", "/api/models/"+profTestID+"/autotune", nil)
	for _, want := range []string{"Start from profile", "General chat", "Coding and editing", "Mixed",
		"settings to measure", "cannot answer other requests"} {
		if !strings.Contains(out, want) {
			t.Errorf("dialog missing %q", want)
		}
	}
	// Autoconfig is offered first and selected.
	first := strings.Index(out, `<option value="Autoconfig"`)
	mine := strings.Index(out, `<option value="Mine"`)
	if first < 0 || mine < 0 || first > mine {
		t.Errorf("the Autoconfig profile is not offered first:\n%s", out)
	}
	if !strings.Contains(out, `<option value="Autoconfig" selected>`) {
		t.Error("the Autoconfig profile is not selected")
	}
}

// The estimate follows the choices: a mixed workload measures each
// setting twice, so it takes longer than one kind of work.
func TestAutotuneEstimateFollowsTheChoices(t *testing.T) {
	s := newTuneServer(t)
	if _, err := s.registry.SaveProfile(profTestID, autoconfigProfileName, models.ProfileSourceAutoconfig, "b1"); err != nil {
		t.Fatal(err)
	}
	minutes := func(uc string) int {
		out := s.doTune(t, "POST", "/api/models/"+profTestID+"/autotune/estimate",
			url.Values{"profile": {autoconfigProfileName}, "use_case": {uc}})
		var n int
		if _, err := fmtSscan(out, &n); err != nil {
			t.Fatalf("no estimate in %q", out)
		}
		return n
	}
	chat, mixed := minutes("chat"), minutes("mixed")
	if chat <= 0 || mixed <= chat {
		t.Errorf("chat = %d minutes, mixed = %d; mixed should take longer", chat, mixed)
	}
}

// fmtSscan pulls "roughly N minutes" out of the rendered estimate.
func fmtSscan(s string, n *int) (int, error) {
	i := strings.Index(s, "about ")
	if i < 0 {
		return 0, errNoEstimate
	}
	rest := s[i+len("about "):]
	j := strings.Index(rest, " ")
	if j < 0 {
		return 0, errNoEstimate
	}
	var v int
	for _, c := range rest[:j] {
		if c < '0' || c > '9' {
			return 0, errNoEstimate
		}
		v = v*10 + int(c-'0')
	}
	*n = v
	return 1, nil
}

var errNoEstimate = errStr("no estimate")

type errStr string

func (e errStr) Error() string { return string(e) }

// Starting is refused while the GPU is in use, and says so where the user
// is looking.
func TestAutotuneStartRefusedWhileBusy(t *testing.T) {
	s := newTuneServer(t)
	if _, err := s.registry.SaveProfile(profTestID, autoconfigProfileName, models.ProfileSourceAutoconfig, "b1"); err != nil {
		t.Fatal(err)
	}
	s.autoconf.run = &autoconfigRun{modelID: "other"} // autoconfigure is running
	out := s.doTune(t, "GET", "/api/models/"+profTestID+"/autotune", nil)
	if !strings.Contains(out, "Autoconfigure is running") || !strings.Contains(out, "disabled") {
		t.Errorf("dialog does not explain the wait:\n%s", out)
	}
}

// progress shows the four stages, what is being measured, and how to stop.
func TestAutotuneProgressScreen(t *testing.T) {
	s := newTuneServer(t)
	rec := &autotune.Autotune{ID: "at-1", ModelID: profTestID, BaseProfile: "Autoconfig",
		UseCase: autotune.UseCode, Status: autotune.StatusRunning, CreatedAt: time.Now()}
	rec.SetStage(autotune.StageRecord{Key: autotune.StageBatch, Status: autotune.StatusDone,
		Finalists: []autotune.Candidate{{Values: map[string]string{"ubatch_size": "1024"},
			Label: "prompt batch 1024", Goals: []autotune.Goal{autotune.GoalPrompt}}}})
	rec.SetStage(autotune.StageRecord{Key: autotune.StageSpec, Status: autotune.StatusRunning, JobID: "job-spec"})
	if err := s.tuneStore.Save(rec); err != nil {
		t.Fatal(err)
	}
	s.bench.SaveJob(benchmark.BenchmarkJob{ID: "job-spec", Cells: []benchmark.JobCell{
		{Status: benchmark.CellStatusCompleted},
		{Status: benchmark.CellStatusRunning, SweepValues: map[string]string{"spec_type": "draft-mtp+ngram-mod"}},
		{Status: benchmark.CellStatusPending},
	}})

	out := s.doTune(t, "GET", "/api/models/"+profTestID+"/autotune/status", nil)
	for _, want := range []string{"Batch sizes and attention", "Speculative decoding", "Confirming the finalists",
		"MTP draft layers with an ngram-mod assist", "1 of 3", "prompt batch 1024", "autotune/cancel"} {
		if !strings.Contains(out, want) {
			t.Errorf("progress screen missing %q", want)
		}
	}
	if !strings.Contains(out, `hx-trigger="every 2s"`) {
		t.Error("the progress screen does not refresh itself")
	}
}

// A stopped run offers to continue rather than starting over.
func TestAutotuneStoppedRunOffersToContinue(t *testing.T) {
	s := newTuneServer(t)
	rec := &autotune.Autotune{ID: "at-2", ModelID: profTestID, BaseProfile: "Autoconfig",
		UseCase: autotune.UseChat, Status: autotune.StatusInterrupted, CreatedAt: time.Now()}
	s.tuneStore.Save(rec)
	out := s.doTune(t, "GET", "/api/models/"+profTestID+"/autotune", nil)
	if !strings.Contains(out, "autotune/resume") || !strings.Contains(out, "Continue") {
		t.Errorf("a stopped run does not offer to continue:\n%s", out)
	}
}

// The results screen shows each goal, and restores a profile on request.
func TestAutotuneResultsScreen(t *testing.T) {
	s := newTuneServer(t)
	if _, err := s.registry.SaveProfile(profTestID, "Autoconfig", models.ProfileSourceAutoconfig, "b1"); err != nil {
		t.Fatal(err)
	}
	winner := autotune.Candidate{
		Values: map[string]string{"spec_type": "draft-mtp+ngram-mod", "ubatch_size": "1024"},
		Scores: map[autotune.Goal]autotune.Score{
			autotune.GoalGeneration: {Value: 61}, autotune.GoalPrompt: {Value: 1400},
			autotune.GoalResponse: {Value: -5.5},
		},
	}
	// The profile the results offer to restore.
	if _, err := s.registry.SaveProfileFrom(profTestID, "Autotune – fastest generation", models.ConfigProfile{
		Config: models.ModelConfig{Enabled: true, GPULayers: 999, ContextSize: 8192, UBatchSize: 1024},
		Source: models.ProfileSourceAutotune,
	}); err != nil {
		t.Fatal(err)
	}
	rec := &autotune.Autotune{ID: "at-3", ModelID: profTestID, BaseProfile: "Autoconfig",
		UseCase: autotune.UseCode, BuildID: "b1", Status: autotune.StatusDone, CreatedAt: time.Now(),
		Results: map[autotune.Goal]autotune.Outcome{
			autotune.GoalGeneration: {Goal: autotune.GoalGeneration, Saved: true,
				ProfileName: "Autotune – fastest generation", Winner: winner,
				Message: "fastest generation: 38.0 → 61.0 tokens per second (61% faster). Saved as \"Autotune – fastest generation\"."},
			autotune.GoalPrompt: {Goal: autotune.GoalPrompt,
				Message: "Your \"Autoconfig\" profile is already the fastest prompt."},
		},
	}
	rec.SetStage(autotune.StageRecord{Key: autotune.StageConfirm, Status: autotune.StatusDone, JobID: "job-confirm",
		Failed: []autotune.FailedCell{{Label: "prompt batch 4096", Error: "out of memory"}}})
	s.tuneStore.Save(rec)

	out := s.doTune(t, "GET", "/api/models/"+profTestID+"/autotune/status", nil)
	for _, want := range []string{"61.0 tokens per second", "already the fastest prompt", "Use these settings",
		"MTP draft layers with an ngram-mod assist", "prompt batch 1024", "out of memory", "Measured for Coding and editing work"} {
		if !strings.Contains(out, want) {
			t.Errorf("results screen missing %q", want)
		}
	}

	out = s.doTune(t, "POST", "/api/models/"+profTestID+"/autotune/restore",
		url.Values{"profile": {"Autotune – fastest generation"}})
	if !strings.Contains(out, "Restored profile") {
		t.Errorf("restore failed:\n%s", out)
	}
	cfg, _ := s.registry.GetConfig(profTestID)
	if cfg.UBatchSize != 1024 || cfg.ActiveProfile != "Autotune – fastest generation" {
		t.Errorf("live config after restore = ubatch %d, profile %q", cfg.UBatchSize, cfg.ActiveProfile)
	}
}

// The model card offers Autotune and a place to render it.
func TestModelCardOffersAutotune(t *testing.T) {
	s := newTuneServer(t)
	out := s.renderCard(t)
	if !strings.Contains(out, "Autotune\n") || !strings.Contains(out, `id="autotune-`) {
		t.Errorf("card lacks the button or the container:\n%s", out)
	}
}

// A run that was stopped must not trap the model: the screen offers to
// continue it, and to start a new one with different choices.
func TestAutotuneStoppedRunCanBeLeftBehind(t *testing.T) {
	s := newTuneServer(t)
	if _, err := s.registry.SaveProfile(profTestID, autoconfigProfileName, models.ProfileSourceAutoconfig, "b1"); err != nil {
		t.Fatal(err)
	}
	s.tuneStore.Save(&autotune.Autotune{ID: "at-old", ModelID: profTestID, BaseProfile: "Autoconfig",
		UseCase: autotune.UseChat, Status: autotune.StatusCancelled, CreatedAt: time.Now()})

	out := s.doTune(t, "GET", "/api/models/"+profTestID+"/autotune", nil)
	if !strings.Contains(out, "autotune/resume") || !strings.Contains(out, "autotune?new=1") {
		t.Errorf("a stopped run offers no way to continue or to start again:\n%s", out)
	}
	fresh := s.doTune(t, "GET", "/api/models/"+profTestID+"/autotune?new=1", nil)
	if !strings.Contains(fresh, "What do you use this model for?") {
		t.Errorf("the start dialog is unreachable after a stopped run:\n%s", fresh)
	}

	// The same for a failed run, which the status screen does not show at
	// all.
	s.tuneStore.Save(&autotune.Autotune{ID: "at-bad", ModelID: profTestID, BaseProfile: "Autoconfig",
		UseCase: autotune.UseChat, Status: autotune.StatusFailed, Error: "every setting failed",
		CreatedAt: time.Now().Add(time.Minute)})
	out = s.doTune(t, "GET", "/api/models/"+profTestID+"/autotune", nil)
	if !strings.Contains(out, "every setting failed") || !strings.Contains(out, "autotune?new=1") {
		t.Errorf("a failed run does not explain itself or offer a new run:\n%s", out)
	}
}

// Autotune does not do Autoconfigure's job, and says so: the dialog tells
// the user to set the model up first, and warns when nothing has.
func TestAutotuneDialogSendsYouToAutoconfigureFirst(t *testing.T) {
	s := newTuneServer(t)

	// No profile at all: Autoconfigure is the recommendation, and saving
	// the current settings is the alternative.
	out := s.doTune(t, "GET", "/api/models/"+profTestID+"/autotune", nil)
	if !strings.Contains(out, "Run <strong>Autoconfigure</strong> first") {
		t.Errorf("a model with no profile is not sent to Autoconfigure:\n%s", out)
	}

	// A profile, but not one Autoconfigure wrote: still worth a warning,
	// because nothing has checked that the settings fit this machine.
	if _, err := s.registry.SaveProfile(profTestID, "Mine", models.ProfileSourceUser, "b1"); err != nil {
		t.Fatal(err)
	}
	out = s.doTune(t, "GET", "/api/models/"+profTestID+"/autotune", nil)
	if !strings.Contains(out, "no Autoconfigure profile") {
		t.Errorf("no warning without an Autoconfigure profile:\n%s", out)
	}
	if !strings.Contains(out, "Set the model up first") {
		t.Errorf("the dialog does not say what Autotune leaves alone:\n%s", out)
	}

	// With an Autoconfigure profile the warning goes away.
	if _, err := s.registry.SaveProfile(profTestID, autoconfigProfileName, models.ProfileSourceAutoconfig, "b1"); err != nil {
		t.Fatal(err)
	}
	out = s.doTune(t, "GET", "/api/models/"+profTestID+"/autotune", nil)
	if strings.Contains(out, "no Autoconfigure profile") {
		t.Errorf("the warning is still shown with an Autoconfigure profile:\n%s", out)
	}
}
