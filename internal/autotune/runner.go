package autotune

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/benchmark"
	"github.com/tmac1973/llama-toolchest/internal/models"
)

// Deps is what the runner needs from the server around it.
type Deps struct {
	Store    *Store
	Runs     *benchmark.Store
	Jobs     *benchmark.JobQueue
	Registry *models.Registry
	// ActiveBuild is the llama.cpp build the stages run on.
	ActiveBuild func() string
	// Hardware reports how many GPUs the machine has and how many
	// logical cores, which decide whether some settings are worth
	// measuring at all.
	Hardware func() (cards, cores int)
	// Busy returns a plain-language reason the GPU cannot be used now, or
	// "" — a benchmark job, or an autoconfigure run.
	Busy func() string
}

// Runner drives an autotune: it submits one ordinary benchmark job per
// stage, reads the results, and carries the winners forward.
type Runner struct {
	deps Deps

	mu     sync.Mutex
	active string             // the running record's ID, "" when idle
	cancel context.CancelFunc // cancels the running stage
}

// NewRunner wires a runner up. It holds no goroutines until Start.
func NewRunner(d Deps) *Runner { return &Runner{deps: d} }

// ErrBusy is returned when something else holds the GPU.
var ErrBusy = errors.New("busy")

// profileNamePrefix starts every profile autotune writes.
const profileNamePrefix = "Autotune"

// Start plans a run and begins it in the background.
func (r *Runner) Start(modelID, profileName string, uc UseCase) (*Autotune, error) {
	if _, ok := Workloads[uc]; !ok {
		return nil, fmt.Errorf("unknown use case %q", uc)
	}
	m, err := r.deps.Registry.Get(modelID)
	if err != nil {
		return nil, err
	}
	profile, err := r.deps.Registry.GetProfile(modelID, profileName)
	if err != nil {
		return nil, fmt.Errorf("the profile %q was not found for this model", profileName)
	}
	// The same check restoring a profile runs: a profile naming a draft
	// file that is gone would fail every cell, minutes apart.
	if err := models.ValidateProfileConfig(profile.Config); err != nil {
		return nil, fmt.Errorf("the profile %q cannot be run: %w", profileName, err)
	}
	if reason := r.busyReason(); reason != "" {
		return nil, fmt.Errorf("%w: %s", ErrBusy, reason)
	}

	rec := &Autotune{
		ID:          fmt.Sprintf("at-%d", time.Now().UnixMilli()),
		ModelID:     modelID,
		BaseProfile: profile.Name,
		UseCase:     uc,
		BuildID:     r.deps.ActiveBuild(),
		Status:      StatusPlanned,
		CreatedAt:   time.Now().UTC(),
	}
	if err := r.deps.Store.Save(rec); err != nil {
		return nil, err
	}
	slog.Info("autotune starting", "run", rec.ID, "model", modelID, "profile", profile.Name,
		"use_case", uc, "model_name", m.PublicName())
	if err := r.begin(rec); err != nil {
		return nil, err
	}
	return rec, nil
}

// Resume continues a run that was cancelled, interrupted by a restart, or
// stopped by a failure. Stages that finished are not measured again.
func (r *Runner) Resume(id string) (*Autotune, error) {
	rec, ok := r.deps.Store.Get(id)
	if !ok {
		return nil, fmt.Errorf("autotune run %s not found", id)
	}
	switch rec.Status {
	case StatusRunning:
		return nil, errors.New("that autotune run is already going")
	case StatusDone:
		return nil, errors.New("that autotune run has already finished")
	}
	if reason := r.busyReason(); reason != "" {
		return nil, fmt.Errorf("%w: %s", ErrBusy, reason)
	}
	rec.Error = ""
	slog.Info("autotune resuming", "run", rec.ID, "from_status", rec.Status)
	if err := r.begin(rec); err != nil {
		return nil, err
	}
	return rec, nil
}

// begin takes the run slot and starts the stages in the background.
func (r *Runner) begin(rec *Autotune) error {
	r.mu.Lock()
	if r.active != "" {
		r.mu.Unlock()
		return fmt.Errorf("%w: autotune is already running for another model", ErrBusy)
	}
	ctx, cancel := context.WithCancel(context.Background())
	r.active, r.cancel = rec.ID, cancel
	r.mu.Unlock()

	rec.Status = StatusRunning
	if err := r.deps.Store.Save(rec); err != nil {
		r.release(rec.ID)
		cancel()
		return err
	}
	go func() {
		defer cancel()
		defer r.release(rec.ID)
		r.run(ctx, rec)
	}()
	return nil
}

func (r *Runner) release(id string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.active == id {
		r.active, r.cancel = "", nil
	}
}

// busyReason reports what is holding the GPU, including autotune itself.
func (r *Runner) busyReason() string {
	r.mu.Lock()
	active := r.active
	r.mu.Unlock()
	if active != "" {
		return "another autotune run is going"
	}
	if r.deps.Busy != nil {
		return r.deps.Busy()
	}
	return ""
}

// Active returns the ID of the run in progress, or "".
func (r *Runner) Active() string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.active
}

// Cancel stops a running autotune after the cell in flight. Stages that
// finished are kept, so Resume continues from there.
func (r *Runner) Cancel(id string) error {
	r.mu.Lock()
	active, cancel := r.active, r.cancel
	r.mu.Unlock()
	if active != id || cancel == nil {
		return fmt.Errorf("autotune run %s is not going", id)
	}
	rec, ok := r.deps.Store.Get(id)
	if ok {
		if st := runningStage(rec); st != nil && st.JobID != "" {
			if err := r.deps.Jobs.Cancel(st.JobID); err != nil {
				slog.Debug("cancel stage job", "job", st.JobID, "error", err)
			}
		}
	}
	cancel()
	return nil
}

func runningStage(rec *Autotune) *StageRecord {
	for i := range rec.Stages {
		if rec.Stages[i].Status == StatusRunning {
			return &rec.Stages[i]
		}
	}
	return nil
}

// run walks the stages in order. Each one is an ordinary benchmark job,
// so it appears in the benchmark history and can be read there.
func (r *Runner) run(ctx context.Context, rec *Autotune) {
	for _, stage := range StageOrder {
		if st, ok := rec.Stage(stage.Key); ok && st.Status == StatusDone {
			continue // already measured, on an earlier attempt
		}
		if err := r.runStage(ctx, rec, stage.Key); err != nil {
			if ctx.Err() != nil {
				rec.Status = StatusCancelled
				slog.Info("autotune cancelled", "run", rec.ID, "stage", stage.Key)
			} else {
				rec.Status, rec.Error = StatusFailed, err.Error()
				slog.Warn("autotune failed", "run", rec.ID, "stage", stage.Key, "error", err)
			}
			r.save(rec)
			return
		}
	}
	r.conclude(rec)
}

// runStage measures one stage and records what it found.
func (r *Runner) runStage(ctx context.Context, rec *Autotune, stageKey string) error {
	in, err := r.planInput(rec)
	if err != nil {
		return err
	}
	cells, axes, skipped := PlanStage(stageKey, in)
	if stageKey == StageSpec {
		rec.Skipped = skipped
	}

	stage := StageRecord{Key: stageKey, Title: StageTitle(stageKey), Status: StatusRunning}
	if old, ok := rec.Stage(stageKey); ok {
		stage.JobID = old.JobID
	}
	// A stage with nothing to measure carries the previous finalists on:
	// a model with no speculative decoding has no draft settings to tune.
	if len(cells) == 0 || (len(cells) == 1 && len(cells[0]) == 0 && stageKey != StageBatch) {
		stage.Status = StatusDone
		stage.Finalists = in.Finalists[previousStage(stageKey)]
		rec.SetStage(stage)
		return r.save(rec)
	}

	profile, err := r.deps.Registry.GetProfile(rec.ModelID, rec.BaseProfile)
	if err != nil {
		return err
	}
	name := rec.ModelID
	if m, err := r.deps.Registry.Get(rec.ModelID); err == nil {
		name = m.PublicName()
	}
	if stage.JobID == "" {
		stage.JobID = fmt.Sprintf("%s-%s", rec.ID, stageKey)
	}
	job := BuildStageJob(rec, stageKey, StageJobInput{
		Profile:   benchmark.BaseProfile{Name: profile.Name, Config: profile.Config},
		Cells:     cells,
		Axes:      axes,
		ModelName: name,
		JobID:     stage.JobID,
	})

	// An earlier attempt may have measured part of this stage. Carry its
	// completed cells over, and the job runner skips them: a resumed
	// stage picks up where it stopped rather than measuring again.
	if existing, err := r.deps.Runs.GetJob(job.ID); err == nil && existing != nil {
		carryCompleted(&job, existing)
	}

	rec.SetStage(stage)
	if err := r.save(rec); err != nil {
		return err
	}
	slog.Info("autotune stage starting", "run", rec.ID, "stage", stageKey, "cells", len(job.Cells))
	if err := r.deps.Jobs.Submit(job); err != nil {
		return fmt.Errorf("starting the %s stage: %w", StageTitle(stageKey), err)
	}
	done, err := r.deps.Jobs.Wait(ctx, job.ID)
	if err != nil {
		return err
	}

	cands, failed := r.readStage(rec, done)
	stage.Candidates, stage.Failed = cands, failed
	stage.Finalists = finalists(cands, r.baseSnapshot(rec), stageKey)
	stage.Status = StatusDone
	if done.Status == benchmark.JobStatusCanceled {
		stage.Status = StatusCancelled
	}
	rec.SetStage(stage)
	if stageKey == StageBatch && rec.SecondsPerCell == 0 && done.Status == benchmark.JobStatusCompleted {
		// Only a stage that ran to the end divides its time by its cells;
		// a stage stopped after two of forty would put the time per cell
		// out by an order of magnitude, and it is never recomputed.
		rec.SecondsPerCell = secondsPerCell(done)
	}
	if err := r.save(rec); err != nil {
		return err
	}
	if stage.Status == StatusCancelled {
		return context.Canceled
	}
	if len(cands) == 0 {
		// Every cell failing almost always means one thing went wrong for
		// all of them — the model would not load at all, say. Put that
		// reason in the message rather than making the reader open the
		// list of failures to find it.
		if why := commonFailure(stage.Failed); why != "" {
			return fmt.Errorf("the %s stage measured nothing: every setting failed, each with the same problem — %s",
				StageTitle(stageKey), why)
		}
		return fmt.Errorf("the %s stage measured nothing: every setting failed", StageTitle(stageKey))
	}
	slog.Info("autotune stage done", "run", rec.ID, "stage", stageKey,
		"measured", len(cands), "failed", len(failed))
	return nil
}

// carryCompleted copies the cells an earlier attempt finished onto a
// freshly planned job, matched by preset and settings. A cell whose
// settings are no longer planned is simply not carried.
func carryCompleted(job *benchmark.BenchmarkJob, prev *benchmark.BenchmarkJob) {
	done := map[string]benchmark.JobCell{}
	for _, c := range prev.Cells {
		if c.Status == benchmark.CellStatusCompleted {
			done[cellKey(c)] = c
		}
	}
	for i := range job.Cells {
		if old, ok := done[cellKey(job.Cells[i])]; ok {
			job.Cells[i] = old
		}
	}
}

func cellKey(c benchmark.JobCell) string {
	return c.ModelID + "\x00" + c.BuildID + "\x00" + c.Preset + "\x00" + Candidate{Values: c.SweepValues}.Key()
}

func previousStage(key string) string {
	prev := ""
	for _, s := range StageOrder {
		if s.Key == key {
			return prev
		}
		prev = s.Key
	}
	return prev
}

// planInput gathers what the planner needs for this run.
func (r *Runner) planInput(rec *Autotune) (PlanInput, error) {
	m, err := r.deps.Registry.Get(rec.ModelID)
	if err != nil {
		return PlanInput{}, err
	}
	profile, err := r.deps.Registry.GetProfile(rec.ModelID, rec.BaseProfile)
	if err != nil {
		return PlanInput{}, err
	}
	cards, cores := 1, 0
	if r.deps.Hardware != nil {
		cards, cores = r.deps.Hardware()
	}
	in := PlanInput{
		Model: m, Base: profile.Config, Cards: cards, Cores: cores,
		DraftCandidates: func(mode string) []models.DraftCandidate {
			return r.deps.Registry.FindDraftCandidates(rec.ModelID, mode)
		},
		Finalists: map[string][]Candidate{},
	}
	for _, s := range rec.Stages {
		in.Finalists[s.Key] = s.Finalists
	}
	return in, nil
}

// baseSnapshot is the starting profile as the runs recorded it.
func (r *Runner) baseSnapshot(rec *Autotune) benchmark.ConfigSnapshot {
	profile, err := r.deps.Registry.GetProfile(rec.ModelID, rec.BaseProfile)
	if err != nil {
		return benchmark.ConfigSnapshot{}
	}
	return benchmark.SnapshotFromConfig(profile.Config, profile.Name, false)
}

// readStage turns a stage job's runs into scored candidates, and reports
// the settings that could not be measured.
func (r *Runner) readStage(rec *Autotune, job *benchmark.BenchmarkJob) ([]Candidate, []FailedCell) {
	byKey := map[string]*Candidate{}
	runsByKey := map[string]map[string]*benchmark.BenchmarkRun{}
	failures := map[string]string{}
	var order []string

	// A cell can fail before it produces a run at all — a config this
	// machine refuses never reaches the router — so the cells are read
	// first, and are the only record of those.
	for _, cell := range job.Cells {
		key := Candidate{Values: cell.SweepValues}.Key()
		if _, seen := byKey[key]; !seen {
			byKey[key] = &Candidate{Values: cell.SweepValues, Label: Describe(cell.SweepValues)}
			runsByKey[key] = map[string]*benchmark.BenchmarkRun{}
			order = append(order, key)
		}
		if cell.Status == benchmark.CellStatusFailed && cell.Error != "" {
			if _, have := failures[key]; !have {
				failures[key] = cell.Error
			}
		}
	}

	for _, run := range r.deps.Runs.RunsForJob(job.ID) {
		run := run
		key := Candidate{Values: run.SweepValues}.Key()
		if _, seen := byKey[key]; !seen {
			byKey[key] = &Candidate{Values: run.SweepValues, Label: Describe(run.SweepValues)}
			runsByKey[key] = map[string]*benchmark.BenchmarkRun{}
			order = append(order, key)
		}
		byKey[key].Config = run.Config
		if run.Status != benchmark.StatusCompleted {
			if _, have := failures[key]; !have {
				failures[key] = runError(&run)
			}
			continue
		}
		runsByKey[key][run.Preset] = &run
	}

	var cands []Candidate
	var failed []FailedCell
	for _, key := range order {
		c := byKey[key]
		c.Scores = map[Goal]Score{}
		for _, g := range Goals {
			if s, err := ScoreRuns(rec.UseCase, g, runsByKey[key]); err == nil {
				c.Scores[g] = s
			}
		}
		if len(c.Scores) == 0 {
			// Nothing usable: a setting this machine cannot run is not a
			// winner, and saying so is more use than leaving it out.
			reason := failures[key]
			if reason == "" {
				reason = "it produced no usable measurements"
			}
			failed = append(failed, FailedCell{Label: c.Label, Error: reason})
			continue
		}
		cands = append(cands, *c)
	}
	return cands, failed
}

func runError(run *benchmark.BenchmarkRun) string {
	if run.Error != "" {
		return run.Error
	}
	return "the run did not finish"
}

// finalists picks the winner for each goal and carries them forward, so
// the next stage builds on at most one setting per goal.
func finalists(cands []Candidate, base benchmark.ConfigSnapshot, stageKey string) []Candidate {
	// Indexes, not pointers: appending to out reallocates it, and a
	// pointer taken before that would update an abandoned copy — the
	// goal would silently vanish from the finalist.
	at := map[string]int{}
	var out []Candidate
	for _, g := range Goals {
		winner, _, ok := Pick(cands, g, base)
		if !ok {
			continue
		}
		k := winner.Key()
		if i, seen := at[k]; seen {
			out[i].Goals = append(out[i].Goals, g)
			continue
		}
		winner.Goals = []Goal{g}
		out = append(out, winner)
		at[k] = len(out) - 1
	}
	return out
}

// secondsPerCell is how long one cell took, measured rather than guessed,
// for the time remaining on the screens.
func secondsPerCell(job *benchmark.BenchmarkJob) float64 {
	if job == nil || job.StartedAt.IsZero() || job.FinishedAt.IsZero() || len(job.Cells) == 0 {
		return 0
	}
	return job.FinishedAt.Sub(job.StartedAt).Seconds() / float64(len(job.Cells))
}

func (r *Runner) save(rec *Autotune) error {
	if err := r.deps.Store.Save(rec); err != nil {
		slog.Warn("could not save the autotune record", "run", rec.ID, "error", err)
		return err
	}
	return nil
}

// conclude reads the confirming stage and saves a profile per goal. Only
// that stage's numbers decide: they were all measured in the same
// conditions, minutes apart rather than an hour.
func (r *Runner) conclude(rec *Autotune) {
	stage, ok := rec.Stage(StageConfirm)
	if !ok || len(stage.Candidates) == 0 {
		rec.Status, rec.Error = StatusFailed, "the confirming stage measured nothing"
		r.save(rec)
		return
	}
	base := r.baseSnapshot(rec)
	var baseline Candidate
	for _, c := range stage.Candidates {
		if c.Key() == "" {
			baseline = c
		}
	}

	rec.Results = map[Goal]Outcome{}
	winners := map[string][]Goal{}
	byKey := map[string]Candidate{}
	for _, g := range Goals {
		winner, _, ok := Pick(stage.Candidates, g, base)
		if !ok {
			continue
		}
		out := Outcome{Goal: g, Winner: winner, Baseline: baseline}
		if _, measured := baseline.Scores[g]; !measured {
			// Without the starting profile's own number there is nothing
			// to compare against, and claiming either way would be made
			// up.
			out.Message = fmt.Sprintf("The %q profile could not be measured for %s, so there is nothing to compare against.",
				rec.BaseProfile, GoalLabel(g))
			rec.Results[g] = out
			continue
		}
		if winner.Key() == "" || !Beats(winner.Scores[g], baseline.Scores[g]) {
			out.Message = fmt.Sprintf("Your %q profile is already the %s.", rec.BaseProfile, GoalLabel(g))
			rec.Results[g] = out
			continue
		}
		out.Saved = true
		rec.Results[g] = out
		winners[winner.Key()] = append(winners[winner.Key()], g)
		byKey[winner.Key()] = winner
	}

	for key, goals := range winners {
		winner := byKey[key]
		name := profileName(goals)
		if err := r.saveProfile(rec, name, winner, baseline, goals); err != nil {
			slog.Warn("could not save an autotune profile", "run", rec.ID, "profile", name, "error", err)
			rec.Error = "the results were measured, but a profile could not be saved: " + err.Error()
			continue
		}
		for _, g := range goals {
			out := rec.Results[g]
			out.ProfileName = name
			out.Message = outcomeMessage(g, winner, baseline, name)
			rec.Results[g] = out
		}
	}

	rec.Status = StatusDone
	r.save(rec)
	slog.Info("autotune done", "run", rec.ID, "profiles", len(winners))
}

// profileName names a profile after the goals it won, so one set of
// settings that wins two goals is saved once.
func profileName(goals []Goal) string {
	labels := make([]string, 0, len(goals))
	for _, g := range goals {
		labels = append(labels, strings.TrimPrefix(GoalLabel(g), "fastest "))
	}
	sort.Strings(labels)
	var joined string
	switch len(labels) {
	case 1:
		joined = labels[0]
	case 2:
		joined = labels[0] + " and " + labels[1]
	default:
		joined = strings.Join(labels[:len(labels)-1], ", ") + " and " + labels[len(labels)-1]
	}
	return fmt.Sprintf("%s – fastest %s", profileNamePrefix, joined)
}

// outcomeMessage is the one-line result the screens show.
func outcomeMessage(g Goal, winner, baseline Candidate, profile string) string {
	w, b := winner.Scores[g], baseline.Scores[g]
	switch g {
	case GoalResponse:
		return fmt.Sprintf("%s: %.1f → %.1f seconds (%s). Saved as %q.",
			GoalLabel(g), -b.Value, -w.Value, percentFaster(-b.Value, -w.Value, true), profile)
	default:
		return fmt.Sprintf("%s: %.1f → %.1f tokens per second (%s). Saved as %q.",
			GoalLabel(g), b.Value, w.Value, percentFaster(b.Value, w.Value, false), profile)
	}
}

// percentFaster renders a change as a percentage, in the direction that
// means faster for the measurement.
func percentFaster(from, to float64, lowerIsBetter bool) string {
	if from == 0 {
		return "no baseline"
	}
	change := (to - from) / from * 100
	if lowerIsBetter {
		change = -change
	}
	if change < 0 {
		return fmt.Sprintf("%.0f%% slower", -change)
	}
	return fmt.Sprintf("%.0f%% faster", change)
}

// saveProfile writes the winning settings as a profile of the model,
// with what was measured and why each setting is there.
func (r *Runner) saveProfile(rec *Autotune, name string, winner, baseline Candidate, goals []Goal) error {
	profile, err := r.deps.Registry.GetProfile(rec.ModelID, rec.BaseProfile)
	if err != nil {
		return err
	}
	cfg, err := benchmark.ConfigForValues(profile.Config, winner.Values, r.resolveModelPath)
	if err != nil {
		return err
	}
	models.NormalizeSpec(&cfg)

	goalNames := make([]string, 0, len(goals))
	for _, g := range goals {
		goalNames = append(goalNames, string(g))
	}
	measured := &models.ProfileMeasurement{
		Workload:            string(rec.UseCase),
		PPTokPerSec:         winner.Scores[GoalPrompt].Value,
		TGTokPerSec:         winner.Scores[GoalGeneration].Value,
		ResponseSec:         -winner.Scores[GoalResponse].Value,
		BaselinePP:          baseline.Scores[GoalPrompt].Value,
		BaselineTG:          baseline.Scores[GoalGeneration].Value,
		BaselineResponseSec: -baseline.Scores[GoalResponse].Value,
		Goals:               goalNames,
		AutotuneID:          rec.ID,
	}

	notes := []models.ProfileNote{{
		Origin: "autotune",
		Reason: fmt.Sprintf("Measured against the %q profile on the %s workload: %s.",
			rec.BaseProfile, UseCaseLabel(rec.UseCase), summaryLine(winner, baseline)),
	}}
	for _, field := range sortedKeys(winner.Values) {
		notes = append(notes, models.ProfileNote{
			Field:  field,
			Origin: "autotune",
			Reason: fmt.Sprintf("Measured as faster here: %s.", DescribeValue(field, winner.Values[field])),
		})
	}

	_, err = r.deps.Registry.SaveProfileFrom(rec.ModelID, name, models.ConfigProfile{
		Config:   cfg,
		Source:   models.ProfileSourceAutotune,
		BuildID:  rec.BuildID,
		Notes:    notes,
		Measured: measured,
	})
	return err
}

// summaryLine is the three measurements in one sentence.
func summaryLine(winner, baseline Candidate) string {
	var parts []string
	if w, ok := winner.Scores[GoalGeneration]; ok {
		parts = append(parts, fmt.Sprintf("generation %.1f → %.1f tokens per second",
			baseline.Scores[GoalGeneration].Value, w.Value))
	}
	if w, ok := winner.Scores[GoalPrompt]; ok {
		parts = append(parts, fmt.Sprintf("prompt %.0f → %.0f tokens per second",
			baseline.Scores[GoalPrompt].Value, w.Value))
	}
	if w, ok := winner.Scores[GoalResponse]; ok {
		parts = append(parts, fmt.Sprintf("a full answer %.1f → %.1f seconds",
			-baseline.Scores[GoalResponse].Value, -w.Value))
	}
	return strings.Join(parts, ", ")
}

func sortedKeys(m map[string]string) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// resolveModelPath turns a draft model's registry ID into its file, for
// the settings a winning cell carries.
func (r *Runner) resolveModelPath(id string) (string, error) {
	m, err := r.deps.Registry.Get(id)
	if err != nil {
		return "", err
	}
	return m.FilePath, nil
}

// commonFailure returns the error every failed cell reported, or "" when
// they did not all fail the same way. Used to explain a stage that
// measured nothing.
func commonFailure(failed []FailedCell) string {
	if len(failed) == 0 {
		return ""
	}
	first := strings.TrimSpace(failed[0].Error)
	if first == "" {
		return ""
	}
	for _, f := range failed[1:] {
		if strings.TrimSpace(f.Error) != first {
			return ""
		}
	}
	return first
}
