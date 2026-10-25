package api

import (
	"fmt"
	"maps"
	"net/http"
	"slices"
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
		d.Cells, d.Minutes, d.Skipped = s.autotuneEstimate(id, selectedProfile(d.Profiles), autotune.UseMixed, autotune.DepthThorough)
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
func (s *Server) autotuneEstimate(id, profileName string, uc autotune.UseCase, depth autotune.Depth) (cells, minutes int, skipped []string) {
	m, err := s.registry.Get(id)
	if err != nil {
		return 0, 0, nil
	}
	profile, err := s.registry.GetProfile(id, profileName)
	if err != nil {
		return 0, 0, nil
	}
	cards, cores := s.autotuneHardware()
	in := autotune.PlanInput{
		Model: m, Base: profile.Config, Cards: cards, Cores: cores,
		DraftCandidates: func(mode string) []models.DraftCandidate {
			return s.registry.FindDraftCandidates(id, mode)
		},
		Finalists:   map[string][]autotune.Candidate{},
		MoECacheMiB: s.autotuneMoECache(m, profile.Config),
		Quick:       depth == autotune.DepthQuick,
	}
	// How long a cell takes on this machine, from the runs it has done;
	// the guess by size only when it has done none.
	size := models.BytesToGiB(m.SizeBytes)
	perCell := autotune.CellSecondsFromHistory(s.tuneStore.List(), id, size, func(other string) float64 {
		if om, err := s.registry.Get(other); err == nil {
			return models.BytesToGiB(om.SizeBytes)
		}
		return 0
	})
	cells, minutes = autotune.EstimateRun(in, uc, size, perCell)
	_, _, skipped = autotune.PlanStage(autotune.StageSpec, in)
	return cells, minutes, skipped
}

// autotuneMoECache is the largest MoE cache that fits beside a profile
// on this machine, or 0 when the active build has no such option.
func (s *Server) autotuneMoECache(m *models.Model, base models.ModelConfig) int {
	if !s.activeTarget().MoECacheOption() {
		return 0
	}
	return models.MaxMoECacheMiB(m, base, s.hardware())
}

// autotuneHardware is the card count and CPU cores Autotune plans with:
// discrete GPUs only, at least one. The dialog's estimate and the run use
// the same numbers, so the estimate describes the run that starts.
func (s *Server) autotuneHardware() (cards, cores int) {
	hw := s.hardware()
	for _, g := range hw.GPUs {
		if !g.IsIGPU {
			cards++
		}
	}
	return max(1, cards), hw.LogicalCores
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
	cells, minutes, _ := s.autotuneEstimate(id, profile, uc, autotune.Depth(r.FormValue("depth")))
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
	s.markConfigPanelStale(w, r, id) // the new profile belongs in Configure's picker
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
	rec, err := s.tuner.Start(id, r.FormValue("profile"), uc, autotune.Depth(r.FormValue("depth")))
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
	banner := s.restoreProfile(w, r, id, p)
	if banner.Kind != "error" {
		s.markConfigPanelStale(w, r, id)
	}
	s.renderAutotuneStatus(w, id, rec, &banner)
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

// expertCacheStageShown reports whether the expert-cache stage belongs in
// a run's progress list. Most models never have one to measure — no
// experts in system memory — and a step that is always skipped is noise.
func (s *Server) expertCacheStageShown(id string, rec *autotune.Autotune) bool {
	if st, ok := rec.Stage(autotune.StageMoECache); ok {
		return st.JobID != "" // measured, or being measured
	}
	p, err := s.registry.GetProfile(id, rec.BaseProfile)
	return err == nil && p.Config.CPUMoE > 0
}

// placementStageShown is the same for the placement stage, which has
// nothing to measure for a model on one GPU with nothing on the CPU.
func (s *Server) placementStageShown(id string, rec *autotune.Autotune) bool {
	if st, ok := rec.Stage(autotune.StagePlacement); ok {
		return st.JobID != ""
	}
	if cards, _ := s.autotuneHardware(); cards > 1 {
		return true
	}
	p, err := s.registry.GetProfile(id, rec.BaseProfile)
	if err != nil {
		return false
	}
	m, err := s.registry.Get(id)
	partial := err == nil && m.NLayers > 0 && p.Config.GPULayers > 0 && p.Config.GPULayers < m.NLayers
	return p.Config.CPUMoE > 0 || partial
}

func (s *Server) autotuneProgressData(id, name string, rec *autotune.Autotune, banner *panelBanner) autotuneProgressData {
	d := autotuneProgressData{ModelID: id, ModelName: name, Run: rec, Banner: banner,
		SkippedReason: rec.Skipped}
	d.Cancellable = rec.Status == autotune.StatusRunning
	d.Resumable = rec.Status == autotune.StatusCancelled || rec.Status == autotune.StatusInterrupted

	for _, st := range autotune.StageOrder {
		if st.Key == autotune.StageMoECache && !s.expertCacheStageShown(id, rec) {
			continue
		}
		if st.Key == autotune.StagePlacement && !s.placementStageShown(id, rec) {
			continue
		}
		if st.Key == autotune.StageSpecParams && rec.Depth == autotune.DepthQuick {
			continue // a quick run keeps the recommended draft settings
		}
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
	// Parts breaks the headline down by workload on a use case that
	// measures several; "" otherwise.
	Parts    string
	Detail   string
	Settings []string
	Message  string
	// Alternative is, when the starting profile won, the best other
	// setting measured and how far behind it came; "" when there is none.
	Alternative string
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
			res.Parts = autotune.PartsPhrase(rec.UseCase, g, out.Winner, out.Baseline)
			res.Detail = resultDetail(out)
			for _, field := range slices.Sorted(maps.Keys(out.Winner.Values)) {
				res.Settings = append(res.Settings, autotune.DescribeValue(field, out.Winner.Values[field]))
			}
		} else {
			// The profile won: nothing is saved, but what it measured, and
			// how close the next best came, are still worth knowing.
			// The profile's numbers are the baseline's: the winner can be
			// another setting that did not beat it by more than the noise.
			res.Parts = autotune.PartsPhrase(rec.UseCase, g, out.Baseline, autotune.Candidate{})
			res.Detail = profileDetail(g, out.Baseline)
			if alt, compared, ok := autotune.ClosestAlternative(rec, g); ok {
				var settings []string
				for _, field := range slices.Sorted(maps.Keys(alt.Values)) {
					settings = append(settings, autotune.DescribeValue(field, alt.Values[field]))
				}
				res.Alternative = fmt.Sprintf("%s, %s (%s)",
					autotune.MeasurementPhrase(g, alt.Scores[g]), compared, strings.Join(settings, ", "))
				if strings.HasSuffix(compared, "faster") {
					res.Alternative += ". " + autotune.MarginPhrase(g, out.Baseline, alt)
				}
			}
		}
		d.Results = append(d.Results, res)
	}
	return d
}

// profileDetail is every measurement of the starting profile, the goal's
// own first: when the profile wins there is no headline that names it.
func profileDetail(goal autotune.Goal, profile autotune.Candidate) string {
	var parts []string
	if v, ok := profile.Scores[goal]; ok {
		parts = append(parts, autotune.MeasurementPhrase(goal, v))
	}
	for _, g := range autotune.Goals {
		if v, ok := profile.Scores[g]; ok && g != goal {
			parts = append(parts, autotune.MeasurementPhrase(g, v))
		}
	}
	return strings.Join(parts, ", ")
}

// resultDetail is the two measurements the headline does not name: the
// headline already gives the card's own goal.
func resultDetail(out autotune.Outcome) string {
	var parts []string
	for _, g := range autotune.Goals {
		if v, ok := out.Winner.Scores[g]; ok && g != out.Goal {
			parts = append(parts, autotune.MeasurementPhrase(g, v))
		}
	}
	return strings.Join(parts, ", ")
}
