package api

import (
	"errors"
	"fmt"
	"net/http"
	"strings"

	"github.com/go-chi/chi/v5"

	"github.com/tmac1973/llama-toolchest/internal/benchmark"
	"github.com/tmac1973/llama-toolchest/internal/models"
)

// benchUseData is what the bench_use partial renders: the dialog that
// saves a benchmark run's settings as a profile of its model.
type benchUseData struct {
	RunID     string
	ModelName string
	Quant     string
	// Name is the name in the field: the suggestion, or what was typed.
	Name string
	// Existing names the model's profiles, so the dialog can say before
	// saving that a name replaces one.
	Existing []string
	Measured string
	// Changes are the settings that differ from the model's current
	// ones: what "Save and apply" would change.
	Changes []models.ProfileChange
	// BuildNote warns that the run measured another build than the
	// active one.
	BuildNote string
	// Unavailable says why the run cannot be saved; the dialog then
	// shows only that.
	Unavailable string
	Banner      *panelBanner
	// Done hides the form once the profile is saved.
	Done bool
}

// benchUseRun resolves the run a request names to the run and its model's
// registry ID, or the reason it cannot be saved as a profile.
func (s *Server) benchUseRun(r *http.Request) (*benchmark.BenchmarkRun, string, string) {
	run, err := s.bench.Get(chi.URLParam(r, "id"))
	if err != nil {
		return nil, "", "This benchmark run no longer exists."
	}
	if why := benchUseUnavailable(run); why != "" {
		return run, "", why
	}
	id := s.registry.ResolveID(run.ModelID)
	if _, err := s.registry.Get(id); err != nil {
		return run, "", "The model this run measured is no longer installed."
	}
	return run, id, ""
}

// benchUseUnavailable is why a run's settings cannot be saved as a
// profile, or "" when they can.
func benchUseUnavailable(run *benchmark.BenchmarkRun) string {
	switch {
	case run.Status != benchmark.StatusCompleted:
		return "Only a completed run can be saved as a profile."
	case run.LaunchConfig == nil:
		return "This run was recorded before benchmarks kept their full settings, so it cannot be saved as a profile. Run the cell again to save it."
	}
	return ""
}

// benchUseOffer is the "Use" button of a job cell's completed run: whether
// it can save the run's settings as a profile, and its tooltip.
func (s *Server) benchUseOffer(run *benchmark.BenchmarkRun) (bool, string) {
	if why := benchUseUnavailable(run); why != "" {
		return false, why
	}
	if _, err := s.registry.Get(s.registry.ResolveID(run.ModelID)); err != nil {
		return false, "The model this run measured is no longer installed."
	}
	return true, "Save the settings this run measured as a profile of the model."
}

func (s *Server) benchUseData(run *benchmark.BenchmarkRun, modelID string) benchUseData {
	d := benchUseData{RunID: run.ID, ModelName: run.ModelName, Quant: run.Quant, Measured: benchRunMeasured(run)}
	if d.ModelName == "" {
		d.ModelName = models.ShortModelName(run.ModelID)
	}
	for _, p := range s.registry.Profiles(modelID) {
		d.Existing = append(d.Existing, p.Name)
	}
	if live, err := s.registry.GetConfig(modelID); err == nil {
		for _, c := range models.ProfileChanges(*live, *run.LaunchConfig) {
			c.From, c.To = models.TruncateText(c.From, 60), models.TruncateText(c.To, 60)
			d.Changes = append(d.Changes, c)
		}
	}
	if active := s.activeBuild(); run.BuildID != "" && active != "" && run.BuildID != active {
		d.BuildNote = fmt.Sprintf("This run measured build %s; the active build is %s. Speculative decoding methods and extra flags may behave differently.",
			run.BuildID, active)
	}
	return d
}

// benchRunMeasured is what a run measured, in one line.
func benchRunMeasured(run *benchmark.BenchmarkRun) string {
	if run.Eval != nil {
		if e := evalScoreText(run.Eval); e != "" {
			return e
		}
		return ""
	}
	if run.Summary != nil {
		return fmt.Sprintf("generation %.1f tokens per second, prompt %.0f tokens per second",
			run.Summary.AvgGenTokPerSec, run.Summary.AvgPromptTokPerSec)
	}
	return ""
}

// suggestBenchProfileName names a profile after what set the run apart:
// its sweep point, after the profile it started from when there was one,
// otherwise its job.
func (s *Server) suggestBenchProfileName(run *benchmark.BenchmarkRun) string {
	var base string
	if run.StartingPoint == benchmark.StartingPointProfile {
		base = run.Config.ProfileName
	}
	var name string
	switch chips := strings.Join(benchmark.SweepChips(run.SweepValues), ", "); {
	case chips != "" && base != "":
		name = base + " · " + chips
	case chips != "":
		name = chips
	case base != "":
		name = base + " (benchmarked)"
	}
	if name == "" && run.JobID != "" && run.JobID != benchmark.AdhocJobID {
		if job, err := s.bench.GetJob(run.JobID); err == nil {
			name = job.Name
		}
	}
	if name = models.NormalizeProfileName(name); name == "" {
		name = "Benchmark " + run.CreatedAt.Local().Format("Jan 2 15:04")
	}
	if r := []rune(name); len(r) > models.MaxProfileNameLen {
		name = strings.TrimSpace(string(r[:models.MaxProfileNameLen-1])) + "…"
	}
	return name
}

// handleBenchUseForm renders the dialog that saves a run's settings as a
// profile, with a suggested name.
func (s *Server) handleBenchUseForm(w http.ResponseWriter, r *http.Request) {
	run, id, why := s.benchUseRun(r)
	respondHTML(w)
	if why != "" {
		s.renderPartial(w, "bench_use", benchUseData{Unavailable: why})
		return
	}
	d := s.benchUseData(run, id)
	d.Name = s.suggestBenchProfileName(run)
	s.renderPartial(w, "bench_use", d)
}

// handleBenchUseSave saves a run's settings as a profile of its model
// (form: profile_name), and with apply=1 also puts the profile into use.
func (s *Server) handleBenchUseSave(w http.ResponseWriter, r *http.Request) {
	if !parseForm(w, r) {
		return
	}
	run, id, why := s.benchUseRun(r)
	if why != "" {
		respondHTML(w)
		s.renderPartial(w, "bench_use", benchUseData{Unavailable: why})
		return
	}
	d := s.benchUseData(run, id)
	d.Name = models.NormalizeProfileName(r.FormValue("profile_name"))
	apply := r.FormValue("apply") == "1"

	replaced, err := s.registry.SaveProfileFrom(id, d.Name, models.ConfigProfile{
		Config:  *run.LaunchConfig,
		Source:  models.ProfileSourceBenchmark,
		BuildID: run.BuildID,
		Notes:   []models.ProfileNote{{Origin: "benchmark", Reason: s.benchProfileNote(run)}},
	})
	if err != nil {
		msg := "Profile not saved: " + err.Error()
		if errors.Is(err, models.ErrProfileName) {
			msg = "Type a name for the profile first."
		}
		d.Banner = &panelBanner{"error", msg}
		respondHTML(w)
		s.renderPartial(w, "bench_use", d)
		return
	}

	verb := "Saved"
	if replaced {
		verb = "Replaced"
	}
	saved := fmt.Sprintf("%s profile %q for %s.", verb, d.Name, d.ModelName)
	banner := panelBanner{"ok", saved + " Apply it from the model's Configure panel when you want to use it."}
	if apply {
		p, err := s.registry.GetProfile(id, d.Name)
		if err != nil {
			banner = panelBanner{"error", saved + " It could not be read back to apply: " + err.Error()}
		} else {
			banner = s.restoreProfile(w, r, id, p)
			if restored := fmt.Sprintf("Restored profile %q.", p.Name); banner.Kind != "error" && strings.HasPrefix(banner.Text, restored) {
				// One sentence for both steps, rather than "Saved …
				// Restored …" about the profile that was just made.
				banner.Text = fmt.Sprintf("%s and applied profile %q for %s.", verb, d.Name, d.ModelName) + strings.TrimPrefix(banner.Text, restored)
			} else {
				banner.Text = saved + " " + banner.Text
			}
		}
	}
	// Saved either way, so the form goes: a second press would only save
	// the same settings again.
	d.Banner, d.Done = &banner, true
	respondHTML(w)
	s.renderPartial(w, "bench_use", d)
}

// benchProfileNote is the profile's note on where its settings came from.
func (s *Server) benchProfileNote(run *benchmark.BenchmarkRun) string {
	where := "a benchmark run"
	if run.JobID != "" && run.JobID != benchmark.AdhocJobID {
		if job, err := s.bench.GetJob(run.JobID); err == nil && job.Name != "" {
			where = fmt.Sprintf("benchmark job %q", job.Name)
		}
	}
	note := fmt.Sprintf("Saved from %s (%s preset, build %s, %s)", where, run.Preset, run.BuildID,
		run.CreatedAt.Local().Format("2006-01-02 15:04"))
	if m := benchRunMeasured(run); m != "" {
		note += ": " + m
	}
	return note + "."
}
