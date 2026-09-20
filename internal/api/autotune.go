package api

import (
	"fmt"
	"net/http"
	"strings"

	"github.com/go-chi/chi/v5"

	"github.com/tmac1973/llama-toolchest/internal/autotune"
	"github.com/tmac1973/llama-toolchest/internal/benchmark"
	"github.com/tmac1973/llama-toolchest/internal/models"
)

// currentSettingsProfile is the name the start dialog saves the live
// config under, for a model that has no profile to measure from yet.
const currentSettingsProfile = "Current settings"

// autotuneDialogData is what the autotune_dialog partial renders.
type autotuneDialogData struct {
	ModelID    string
	ModelName  string
	Profiles   []profileOption
	UseCases   []useCaseOption
	Cells      int
	Minutes    int
	Skipped    []string
	BusyReason string
	// HasAutoconfig reports an Autoconfigure profile among the choices.
	// Without one, nothing has checked that this model's context size and
	// memory settings fit the machine, and Autotune will not notice.
	HasAutoconfig bool
	// NoProfiles offers to save the live settings as a profile first:
	// autotune measures from a profile, not from a config that can change
	// under it.
	NoProfiles bool
	Banner     *panelBanner
}

type useCaseOption struct {
	Value, Label, Help string
	Checked            bool
}

func useCaseOptions() []useCaseOption {
	return []useCaseOption{
		{Value: string(autotune.UseChat), Label: "General chat",
			Help: "Questions, writing and conversation: answers that are mostly new text."},
		{Value: string(autotune.UseCode), Label: "Coding and editing",
			Help: "Changing code or text you give it. Answers repeat much of the prompt, which is where speculative decoding pays off most."},
		{Value: string(autotune.UseMixed), Label: "Mixed", Checked: true,
			Help: "Both kinds of work. Measures each setting twice, so it takes about twice as long."},
	}
}

// handleAutotuneDialog shows the start dialog, or the state of a run for
// this model that is going or waiting to be read.
func (s *Server) handleAutotuneDialog(w http.ResponseWriter, r *http.Request) {
	id := s.registry.ResolveID(chi.URLParam(r, "id"))
	// The button shows the model's latest run — its progress, or its
	// results — so nothing measured is hidden behind a fresh dialog.
	// "new" asks for the dialog itself, from "Run again" and from a run
	// that was stopped and will not be continued; without it a model
	// could never be measured again with different choices.
	if r.URL.Query().Get("new") != "1" {
		if rec, ok := s.tuneStore.LatestForModel(id); ok {
			s.renderAutotuneStatus(w, id, rec, nil)
			return
		}
	}
	s.renderAutotuneDialog(w, id, nil)
}

func (s *Server) renderAutotuneDialog(w http.ResponseWriter, id string, banner *panelBanner) {
	m, err := s.registry.Get(id)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	d := autotuneDialogData{ModelID: id, ModelName: m.PublicName(), UseCases: useCaseOptions(), Banner: banner}

	for _, p := range s.registry.Profiles(id) {
		d.Profiles = append(d.Profiles, profileOption{Name: p.Name, Label: profileLabel(p),
			Selected: p.Name == autoconfigProfileName})
	}
	// The Autoconfig profile first: it is the one most people will have,
	// and the one autoconfigure told them to tune.
	for i, p := range d.Profiles {
		if p.Name == autoconfigProfileName && i > 0 {
			d.Profiles = append([]profileOption{p}, append(d.Profiles[:i], d.Profiles[i+1:]...)...)
			break
		}
	}
	for _, p := range d.Profiles {
		if p.Name == autoconfigProfileName {
			d.HasAutoconfig = true
		}
	}
	d.NoProfiles = len(d.Profiles) == 0
	if d.NoProfiles {
		d.BusyReason = ""
	} else {
		d.Cells, d.Minutes, d.Skipped = s.autotuneEstimate(id, selectedProfile(d.Profiles), autotune.UseMixed)
	}
	if reason := s.gpuBusyReason(); reason != "" {
		d.BusyReason = "The GPU is in use: " + reason + ". Start Autotune when it finishes."
	} else if active := s.tuner.Active(); active != "" {
		d.BusyReason = "Autotune is already running for another model."
	} else if s.activeBuild() == "" && len(s.builder.List()) == 0 {
		d.BusyReason = "No llama.cpp build is available to measure with."
	}
	respondHTML(w)
	s.renderPartial(w, "autotune_dialog", d)
}

func selectedProfile(opts []profileOption) string {
	for _, p := range opts {
		if p.Selected {
			return p.Name
		}
	}
	if len(opts) > 0 {
		return opts[0].Name
	}
	return ""
}

// autotuneEstimate reports how much a run would measure, and what it
// cannot measure on this machine.
func (s *Server) autotuneEstimate(id, profileName string, uc autotune.UseCase) (cells, minutes int, skipped []string) {
	m, err := s.registry.Get(id)
	if err != nil {
		return 0, 0, nil
	}
	profile, err := s.registry.GetProfile(id, profileName)
	if err != nil {
		return 0, 0, nil
	}
	hw := s.hardware()
	cards := 0
	for _, g := range hw.GPUs {
		if !g.IsIGPU {
			cards++
		}
	}
	in := autotune.PlanInput{
		Model: m, Base: profile.Config, Cards: max(1, cards), Cores: hw.LogicalCores,
		DraftCandidates: func(mode string) []models.DraftCandidate {
			return s.registry.FindDraftCandidates(id, mode)
		},
		Finalists: map[string][]autotune.Candidate{},
	}
	cells, minutes = autotune.EstimateRun(in, uc, models.BytesToGiB(m.SizeBytes), 0)
	_, _, skipped = autotune.PlanStage(autotune.StageSpec, in)
	return cells, minutes, skipped
}

// handleAutotuneEstimate re-renders the dialog when the choices change,
// so the estimate matches what is selected.
func (s *Server) handleAutotuneEstimate(w http.ResponseWriter, r *http.Request) {
	id := s.registry.ResolveID(chi.URLParam(r, "id"))
	r.ParseForm()
	uc := autotune.UseCase(r.FormValue("use_case"))
	if _, ok := autotune.Workloads[uc]; !ok {
		uc = autotune.UseMixed
	}
	profile := r.FormValue("profile")
	cells, minutes, _ := s.autotuneEstimate(id, profile, uc)
	respondHTML(w)
	s.renderPartial(w, "autotune_estimate", struct {
		Cells, Minutes int
	}{cells, minutes})
}

// handleAutotuneSaveCurrent saves the live config as a profile, so a
// model with none has something to measure from.
func (s *Server) handleAutotuneSaveCurrent(w http.ResponseWriter, r *http.Request) {
	id := s.registry.ResolveID(chi.URLParam(r, "id"))
	if _, err := s.registry.SaveProfile(id, currentSettingsProfile, models.ProfileSourceUser, s.activeBuild()); err != nil {
		s.renderAutotuneDialog(w, id, &panelBanner{"error", "The settings could not be saved: " + err.Error()})
		return
	}
	s.renderAutotuneDialog(w, id, &panelBanner{"ok",
		fmt.Sprintf("Saved the current settings as the profile %q. Autotune will measure from it.", currentSettingsProfile)})
}

// handleAutotuneStart begins a run (form: profile, use_case).
func (s *Server) handleAutotuneStart(w http.ResponseWriter, r *http.Request) {
	id := s.registry.ResolveID(chi.URLParam(r, "id"))
	r.ParseForm()
	uc := autotune.UseCase(r.FormValue("use_case"))
	if _, ok := autotune.Workloads[uc]; !ok {
		uc = autotune.UseMixed
	}
	rec, err := s.tuner.Start(id, r.FormValue("profile"), uc)
	if err != nil {
		// A refusal renders as a message: htmx does not swap a non-2xx
		// response, and a button that appears to do nothing is worse.
		s.renderAutotuneDialog(w, id, &panelBanner{"error", strings.ToUpper(err.Error()[:1]) + err.Error()[1:] + "."})
		return
	}
	s.renderAutotuneStatus(w, id, rec, nil)
}

// handleAutotuneStatus is polled while a run is going, and returns the
// results once it has finished.
func (s *Server) handleAutotuneStatus(w http.ResponseWriter, r *http.Request) {
	id := s.registry.ResolveID(chi.URLParam(r, "id"))
	rec, ok := s.tuneStore.LatestForModel(id)
	if !ok {
		respondHTML(w) // nothing to show
		return
	}
	s.renderAutotuneStatus(w, id, rec, nil)
}

// handleAutotuneCancel stops a running autotune.
func (s *Server) handleAutotuneCancel(w http.ResponseWriter, r *http.Request) {
	id := s.registry.ResolveID(chi.URLParam(r, "id"))
	rec, ok := s.tuneStore.LatestForModel(id)
	if !ok {
		respondHTML(w)
		return
	}
	var banner *panelBanner
	if err := s.tuner.Cancel(rec.ID); err != nil {
		banner = &panelBanner{"error", "Not cancelled: " + err.Error()}
	} else {
		banner = &panelBanner{"ok", "Stopping after the measurement in progress. What it has measured is kept."}
	}
	rec, _ = s.tuneStore.LatestForModel(id)
	s.renderAutotuneStatus(w, id, rec, banner)
}

// handleAutotuneResume continues a cancelled or interrupted run.
func (s *Server) handleAutotuneResume(w http.ResponseWriter, r *http.Request) {
	id := s.registry.ResolveID(chi.URLParam(r, "id"))
	rec, ok := s.tuneStore.LatestForModel(id)
	if !ok {
		respondHTML(w)
		return
	}
	if _, err := s.tuner.Resume(rec.ID); err != nil {
		s.renderAutotuneStatus(w, id, rec, &panelBanner{"error", "Not resumed: " + err.Error()})
		return
	}
	rec, _ = s.tuneStore.LatestForModel(id)
	s.renderAutotuneStatus(w, id, rec, &panelBanner{"ok", "Continuing from where it stopped."})
}

// handleAutotuneRestore puts one of the measured profiles into use.
func (s *Server) handleAutotuneRestore(w http.ResponseWriter, r *http.Request) {
	id := s.registry.ResolveID(chi.URLParam(r, "id"))
	r.ParseForm()
	name := r.FormValue("profile")

	rec, _ := s.tuneStore.LatestForModel(id)
	p, err := s.registry.GetProfile(id, name)
	if err != nil {
		s.renderAutotuneStatus(w, id, rec, &panelBanner{"error", "That profile is no longer saved."})
		return
	}
	if err := models.ValidateProfileConfig(p.Config); err != nil {
		s.renderAutotuneStatus(w, id, rec, &panelBanner{"error", fmt.Sprintf("Profile %q was not restored: %s", p.Name, err)})
		return
	}
	if err := s.registry.ApplyProfile(id, p.Name); err != nil {
		s.renderAutotuneStatus(w, id, rec, &panelBanner{"error", fmt.Sprintf("Profile %q was not restored: %s", p.Name, err)})
		return
	}
	if cfg, err := s.registry.GetConfig(id); err == nil {
		s.afterConfigChange(w, r, id, cfg)
	}
	s.renderAutotuneStatus(w, id, rec, &panelBanner{"ok",
		fmt.Sprintf("Restored profile %q. It takes effect the next time this model loads.", p.Name)})
}

// autotuneProgressData is what the autotune_progress partial renders.
type autotuneProgressData struct {
	ModelID   string
	ModelName string
	Run       *autotune.Autotune
	Steps     []autotuneStep
	// Current is what is being measured right now, in plain language.
	Current       string
	CellsDone     int
	CellsTotal    int
	MinutesLeft   int
	Cancellable   bool
	Resumable     bool
	Banner        *panelBanner
	SkippedReason []string
}

// autotuneStep is one stage in the four-step indicator.
type autotuneStep struct {
	Title   string
	State   string // "done" | "running" | "waiting" | "stopped"
	JobID   string
	Summary string
}

// renderAutotuneStatus shows a run: its progress, or its results.
func (s *Server) renderAutotuneStatus(w http.ResponseWriter, id string, rec *autotune.Autotune, banner *panelBanner) {
	respondHTML(w)
	name := id
	if m, err := s.registry.Get(id); err == nil {
		name = m.PublicName()
	}
	if rec == nil {
		s.renderAutotuneDialog(w, id, banner)
		return
	}
	if rec.Status == autotune.StatusDone || rec.Status == autotune.StatusFailed {
		s.renderPartial(w, "autotune_results", s.autotuneResultsData(id, name, rec, banner))
		return
	}
	s.renderPartial(w, "autotune_progress", s.autotuneProgressData(id, name, rec, banner))
}

func (s *Server) autotuneProgressData(id, name string, rec *autotune.Autotune, banner *panelBanner) autotuneProgressData {
	d := autotuneProgressData{ModelID: id, ModelName: name, Run: rec, Banner: banner,
		SkippedReason: rec.Skipped}
	d.Cancellable = rec.Status == autotune.StatusRunning
	d.Resumable = rec.Status == autotune.StatusCancelled || rec.Status == autotune.StatusInterrupted

	for _, st := range autotune.StageOrder {
		step := autotuneStep{Title: st.Title, State: "waiting"}
		if rec, ok := rec.Stage(st.Key); ok {
			step.JobID = rec.JobID
			switch rec.Status {
			case autotune.StatusDone:
				step.State = "done"
				step.Summary = stageSummary(rec)
			case autotune.StatusRunning:
				step.State = "running"
			default:
				step.State = "stopped"
			}
		}
		d.Steps = append(d.Steps, step)
	}

	// The cell counts come from the stage job, which is where the work
	// actually is.
	for _, st := range rec.Stages {
		if st.Status != autotune.StatusRunning || st.JobID == "" {
			continue
		}
		job, err := s.bench.GetJob(st.JobID)
		if err != nil || job == nil {
			continue
		}
		d.CellsTotal = len(job.Cells)
		for _, c := range job.Cells {
			switch c.Status {
			case benchmark.CellStatusCompleted, benchmark.CellStatusFailed, benchmark.CellStatusSkipped:
				d.CellsDone++
			case benchmark.CellStatusRunning:
				d.Current = autotune.Describe(c.SweepValues)
			}
		}
		if rec.SecondsPerCell > 0 {
			left := float64(d.CellsTotal-d.CellsDone) * rec.SecondsPerCell
			d.MinutesLeft = int(left/60 + 0.5)
		}
	}
	return d
}

// stageSummary is what a finished stage found, in one line.
func stageSummary(st *autotune.StageRecord) string {
	if len(st.Finalists) == 0 {
		return ""
	}
	var parts []string
	for _, f := range st.Finalists {
		goals := make([]string, 0, len(f.Goals))
		for _, g := range f.Goals {
			goals = append(goals, strings.TrimPrefix(autotune.GoalLabel(g), "fastest "))
		}
		label := f.Label
		if len(goals) > 0 {
			label += " (" + strings.Join(goals, ", ") + ")"
		}
		parts = append(parts, label)
	}
	return strings.Join(parts, "; ")
}

// autotuneResultsData is what the autotune_results partial renders.
type autotuneResultsData struct {
	ModelID   string
	ModelName string
	// UseCaseLabel is the kind of work in the words the dialog used.
	UseCaseLabel string
	Run          *autotune.Autotune
	Results      []autotuneResult
	// ConfirmJob links to the benchmark job behind the numbers.
	ConfirmJob string
	Banner     *panelBanner
	Failed     []autotune.FailedCell
}

// autotuneResult is one goal's outcome as the screen shows it.
type autotuneResult struct {
	Goal        string
	Saved       bool
	ProfileName string
	Headline    string
	Detail      string
	Settings    []string
	Message     string
}

func (s *Server) autotuneResultsData(id, name string, rec *autotune.Autotune, banner *panelBanner) autotuneResultsData {
	d := autotuneResultsData{ModelID: id, ModelName: name, Run: rec, Banner: banner,
		UseCaseLabel: autotune.UseCaseLabel(rec.UseCase)}
	if st, ok := rec.Stage(autotune.StageConfirm); ok {
		d.ConfirmJob = st.JobID
		d.Failed = st.Failed
	}
	for _, g := range autotune.Goals {
		out, ok := rec.Results[g]
		if !ok {
			continue
		}
		res := autotuneResult{Goal: autotune.GoalLabel(g), Saved: out.Saved,
			ProfileName: out.ProfileName, Message: out.Message}
		if out.Saved {
			// The card is headed with the goal already; repeating it in
			// the sentence reads as a stutter.
			res.Headline = strings.TrimPrefix(out.Message, autotune.GoalLabel(g)+": ")
			res.Detail = resultDetail(out)
			for _, field := range sortedFields(out.Winner.Values) {
				res.Settings = append(res.Settings, autotune.DescribeValue(field, out.Winner.Values[field]))
			}
		}
		d.Results = append(d.Results, res)
	}
	return d
}

// resultDetail is the two measurements the headline does not name.
func resultDetail(out autotune.Outcome) string {
	var parts []string
	if v, ok := out.Winner.Scores[autotune.GoalGeneration]; ok {
		parts = append(parts, fmt.Sprintf("generation %.1f tokens per second", v.Value))
	}
	if v, ok := out.Winner.Scores[autotune.GoalPrompt]; ok {
		parts = append(parts, fmt.Sprintf("prompt %.0f tokens per second", v.Value))
	}
	if v, ok := out.Winner.Scores[autotune.GoalResponse]; ok {
		parts = append(parts, fmt.Sprintf("a full answer in %.1f seconds", -v.Value))
	}
	return strings.Join(parts, ", ")
}

func sortedFields(values map[string]string) []string {
	out := make([]string, 0, len(values))
	for k := range values {
		out = append(out, k)
	}
	for i := 1; i < len(out); i++ {
		for j := i; j > 0 && out[j] < out[j-1]; j-- {
			out[j], out[j-1] = out[j-1], out[j]
		}
	}
	return out
}
