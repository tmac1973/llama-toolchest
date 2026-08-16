package api

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/tmac1973/llama-toolchest/internal/benchmark"
	"github.com/tmac1973/llama-toolchest/internal/evaluate"
	"github.com/tmac1973/llama-toolchest/internal/models"
)

// handleListJobs returns all benchmark jobs.
func (s *Server) handleListJobs(w http.ResponseWriter, r *http.Request) {
	jobs := s.bench.ListJobs()
	if isHTMX(r) {
		respondHTML(w)
		s.renderJobList(w, jobs)
		return
	}
	respondJSON(w, jobs)
}

// renderJobList renders the collapsible jobs list partial. Each job row
// shows status + cell-progress; expanding fetches the detail partial via
// HTMX so the matrix view doesn't render on every list refresh.
func (s *Server) renderJobList(w http.ResponseWriter, jobs []benchmark.BenchmarkJob) {
	enriched := make([]jobListEntry, 0, len(jobs))
	for _, j := range jobs {
		j := j
		var done, failed, total int
		for _, c := range j.Cells {
			total++
			switch c.Status {
			case benchmark.CellStatusCompleted:
				done++
			case benchmark.CellStatusFailed:
				failed++
			}
		}
		enriched = append(enriched, jobListEntry{
			Job:       &j,
			Done:      done,
			Failed:    failed,
			Total:     total,
			AdhocRuns: 0,
		})
	}
	if len(enriched) > 0 {
		// The synthetic adhoc job carries no Cells, so its progress
		// column should show the total run count instead.
		for i := range enriched {
			if enriched[i].Job.ID == benchmark.AdhocJobID {
				enriched[i].AdhocRuns = len(s.bench.RunsForJob(benchmark.AdhocJobID))
			}
		}
	}
	s.renderPartial(w, "job_list", enriched)
}

type jobListEntry struct {
	Job       *benchmark.BenchmarkJob
	Done      int
	Failed    int
	Total     int
	AdhocRuns int
}

// klRepoGroup is one repo's models in the job form's KL reference
// dropdown (installed models grouped by HF repo).
type klRepoGroup struct {
	Repo   string
	Models []*models.Model
}

// buildOpt is one build in the job form's build list.
type buildOpt struct {
	ID, Profile, GitRef, Tag string
}

// jobCreateRequest is the JSON body POST /api/benchmark-jobs accepts.
type jobCreateRequest struct {
	Name        string                     `json:"name"`
	Description string                     `json:"description,omitempty"`
	ModelIDs    []string                   `json:"model_ids"`
	BuildIDs    []string                   `json:"build_ids"`
	Presets     []string                   `json:"presets"`
	Overrides   *benchmark.ConfigOverrides `json:"overrides,omitempty"`
	Sweeps      []benchmark.SweepAxis      `json:"sweeps,omitempty"`
	// Params is the unified shape the job form posts: parameter name →
	// selected values. One value fixes the parameter, two or more sweep
	// it. Supersedes Overrides/Sweeps for form callers; the older fields
	// remain for JSON API callers and already-saved jobs.
	Params map[string][]string `json:"params,omitempty"`
	// KLReference names the registry model the kl-divergence cells
	// compare against; empty = automatic (the largest installed quant of
	// each model's own HF repo). Posted by the job form's KL dropdown.
	KLReference string `json:"kl_reference,omitempty"`
}

// resolveSweeps normalizes the accepted input shapes into overrides and
// axes. Params wins when present, since it is what the form sends and it
// expresses both concepts at once.
func resolveSweeps(req *jobCreateRequest) error {
	if len(req.Params) > 0 {
		derived, axes, err := benchmark.SplitParams(req.Params)
		if err != nil {
			return err
		}
		// Merge rather than replace, so any override field the form has
		// no control for would survive an edit. Every field currently
		// has a form control, making this a no-op, but the rule stays:
		// a field the params map can express is params' to decide,
		// including deciding to leave it unset.
		req.Overrides = benchmark.MergeOverrides(
			benchmark.KeepUnsweepable(req.Overrides), derived)
		req.Sweeps = axes
		return nil
	}
	return nil
}

// validateJobRequest checks the parts both create and update share.
// Sweeps are validated up front so a bad value fails at definition time
// rather than partway through a run that may already have taken hours.
func validateJobRequest(req jobCreateRequest) error {
	if strings.TrimSpace(req.Name) == "" {
		return errors.New("name is required")
	}
	if len(req.ModelIDs) == 0 || len(req.BuildIDs) == 0 || len(req.Presets) == 0 {
		return errors.New("model_ids, build_ids, and presets are all required")
	}
	if err := benchmark.ValidateSweeps(req.Sweeps); err != nil {
		return err
	}
	if err := benchmark.ValidateSamplingSupport(req.Presets, req.Overrides, req.Sweeps); err != nil {
		return err
	}
	// A sweep of every axis multiplies fast. Refuse obviously runaway
	// matrices rather than letting someone queue a week of work by
	// pasting a long list. The count applies the capability collapse
	// rule: a capability preset gets one cell per distinct eval-reaching
	// configuration, so axes without AffectsEval do not multiply it.
	cells := 0
	for _, name := range req.Presets {
		mult := 1
		capability := benchmark.GetPreset(name).EffectiveSource() == benchmark.PresetSourceCapability
		for _, sw := range req.Sweeps {
			if capability {
				if f, ok := benchmark.LookupSweepField(sw.Field); ok && !f.AffectsEval {
					continue
				}
			}
			mult *= len(sw.Values)
		}
		cells += len(req.ModelIDs) * len(req.BuildIDs) * mult
	}
	if cells > maxJobCells {
		return fmt.Errorf("this matrix expands to %d cells, above the %d limit — narrow a sweep or split the job", cells, maxJobCells)
	}
	if err := validateCapabilitySweeps(req); err != nil {
		return err
	}
	return nil
}

// validateCapabilitySweeps refuses a capability-only job whose only
// variation is a sweep that never reaches the evaluation: after the
// collapse every cell would be one identical run per preset.
//
// Refused ONLY when ALL FOUR hold — capability-only presets, at least
// one sweep axis configured, no configured axis has AffectsEval, and
// models × builds == 1. Precisely NOT refused: sweep-free capability
// jobs (the flagship four-quant job — zero sweeps is the normal case),
// and multi-model or multi-build jobs with inert sweeps (after collapse
// the cells still vary by model/build — meaningful).
//
// A job mixing benchy, capability, and a sampling axis gets the benchy
// refusal first (ValidateSamplingSupport), unchanged.
func validateCapabilitySweeps(req jobCreateRequest) error {
	allCapability := len(req.Presets) > 0
	for _, name := range req.Presets {
		if benchmark.GetPreset(name).EffectiveSource() != benchmark.PresetSourceCapability {
			allCapability = false
			break
		}
	}
	if !allCapability || len(req.Sweeps) == 0 || len(req.ModelIDs)*len(req.BuildIDs) != 1 {
		return nil
	}
	for _, sw := range req.Sweeps {
		if f, ok := benchmark.LookupSweepField(sw.Field); ok && f.AffectsEval {
			return nil
		}
	}
	return errors.New("the swept parameters do not affect capability evaluations — nothing would vary between cells; drop the sweeps or add a performance preset that uses them")
}

// maxJobCells caps matrix size. Each cell is a full benchmark and most
// carry a router reload, so four figures of them is a runaway job, not
// an experiment.
const maxJobCells = 500

// validateBatchMatrix rejects a job whose batch/micro-batch combinations
// can't load, before any of it runs.
//
// The apply-time check catches these too, but only once the cell is
// reached — a long sweep would run for an hour and then fail every cell
// above the batch size. Effective values depend on each model's saved
// config, so this needs the registry and can't live in the pure
// request validator.
func (s *Server) validateBatchMatrix(modelIDs []string, overrides *benchmark.ConfigOverrides, sweeps []benchmark.SweepAxis) error {
	// A parse failure here would silently degrade the check to "field not
	// set" and let a job through that fails every cell hours later.
	// ValidateSweeps has already parsed these, so an error is a bug.
	candidates := func(field string, fixed *int) ([]int, error) {
		for _, sw := range sweeps {
			if sw.Field != field {
				continue
			}
			out := make([]int, 0, len(sw.Values))
			for _, v := range sw.Values {
				n, err := strconv.Atoi(v)
				if err != nil {
					return nil, fmt.Errorf("%s: %q is not an integer", field, v)
				}
				out = append(out, n)
			}
			return out, nil
		}
		if fixed != nil {
			return []int{*fixed}, nil
		}
		return nil, nil // not set: fall back to the model's saved value
	}

	var fixedBatch, fixedUBatch *int
	if overrides != nil {
		fixedBatch, fixedUBatch = overrides.BatchSize, overrides.UBatchSize
	}
	batches, err := candidates("batch_size", fixedBatch)
	if err != nil {
		return err
	}
	ubatches, err := candidates("ubatch_size", fixedUBatch)
	if err != nil {
		return err
	}
	if len(batches) == 0 && len(ubatches) == 0 {
		return nil
	}

	// Reject only when *no* combination can run.
	//
	// A batch × micro-batch matrix legitimately contains invalid corners
	// — sweeping b=[1024,2048] against ub=[512,2048] means (1024, 2048)
	// can't load while the other three can. Failing the whole job for one
	// bad corner made that experiment impossible to create. Those cells
	// still fail individually at apply time, with a message naming the
	// pair, which is the right granularity.
	viable := func(bs, ubs []int) (int, int) {
		ok, total := 0, 0
		for _, b := range bs {
			for _, ub := range ubs {
				total++
				probe := models.ModelConfig{BatchSize: b, UBatchSize: ub}
				if probe.ValidateBatchSizes() == nil {
					ok++
				}
			}
		}
		return ok, total
	}

	if len(batches) > 0 && len(ubatches) > 0 {
		if ok, total := viable(batches, ubatches); ok == 0 && total > 0 {
			return fmt.Errorf("no batch / micro-batch combination in this job can run: every micro-batch value exceeds every batch value")
		}
	}

	// Then against each model's saved values, which fill in whichever
	// side the job left alone.
	for _, id := range modelIDs {
		saved, err := s.registry.GetConfig(id)
		if err != nil {
			continue // a missing model fails later with a clearer error
		}
		bs, ubs := batches, ubatches
		if len(bs) == 0 {
			bs = []int{saved.BatchSize}
		}
		if len(ubs) == 0 {
			ubs = []int{saved.UBatchSize}
		}
		if ok, total := viable(bs, ubs); ok == 0 && total > 0 {
			probe := models.ModelConfig{BatchSize: bs[0], UBatchSize: ubs[0]}
			return fmt.Errorf("%s: %w", id, probe.ValidateBatchSizes())
		}
	}
	return nil
}

// validateGPUAssignment refuses jobs whose GPU placement cannot produce
// distinct, correctly-labelled cells.
//
// gpu_assign and tensor_split are two ways to express the same thing:
// only tensor_split / split_mode / main_gpu reach llama-server, and
// gpu_assign is resolved into them. Setting both is contradictory, and
// whichever way it was resolved silently, some cells ended up
// byte-identical while being reported under different labels.
//
// A gpu_assign job also needs a GPU count to resolve against. Refusing
// here beats failing every cell partway through a run.
func (s *Server) validateGPUAssignment(overrides *benchmark.ConfigOverrides, sweeps []benchmark.SweepAxis) error {
	set := func(field string, fixed *string) bool {
		for _, sw := range sweeps {
			if sw.Field == field {
				return true
			}
		}
		return fixed != nil && *fixed != ""
	}
	var fixedAssign, fixedSplit *string
	if overrides != nil {
		fixedAssign, fixedSplit = overrides.GPUAssign, overrides.TensorSplit
	}
	assign := set("gpu_assign", fixedAssign)
	if !assign {
		return nil
	}
	if set("tensor_split", fixedSplit) {
		return errors.New("set GPU Assignment or Tensor Split, not both — GPU Assignment is resolved into a tensor split, so setting both makes some cells identical while reporting different values")
	}
	if len(s.monitor.Current().GPU) == 0 {
		return errors.New("cannot use GPU Assignment: no GPUs detected, so every value would resolve to the same configuration. Use Tensor Split directly, or check that GPU monitoring is working")
	}
	return nil
}

// validateKLJob refuses a KL job whose every cell would be skipped at
// run time (the runner skips a KL cell whose resolved reference IS the
// cell's own model — an expensive eval whose answer is known is waste).
//
// A reference that is merely AMONG the job's models is fine — that is
// the primary all-quants flow; only its own cell skips. Refused when
// every selected model resolves to itself: a single-model job whose
// model is the only installed quant of its repo, an explicit reference
// naming the only model, or every model being its own repo's largest
// (and only) quant. The message names what to add.
func (s *Server) validateKLJob(req jobCreateRequest) error {
	hasKL := false
	for _, name := range req.Presets {
		p := benchmark.GetPreset(name)
		if p.EffectiveSource() == benchmark.PresetSourceCapability && p.EvalMode == evaluate.ModeKLDiv {
			hasKL = true
			break
		}
	}
	if !hasKL {
		return nil
	}
	if req.KLReference != "" {
		if _, err := s.registry.Get(req.KLReference); err != nil {
			return fmt.Errorf("KL reference model %q is not an installed model", req.KLReference)
		}
	}
	// A fresh env: the resolution only reads the registry, and s.env
	// may be unwired (tests) while the router is not job-controlled.
	env := newJobEnv(s)
	for _, id := range req.ModelIDs {
		ref, err := env.resolveKLReference(id, req.KLReference)
		if err == nil && ref.ID != id {
			return nil // at least one model's cells would run
		}
	}
	return fmt.Errorf("every KL-divergence cell in this job would be the reference model's own cell — each selected model resolves to itself as its reference. Add a second quant of one of the repos, or pick a different KL reference model")
}

// handleCreateJob expands the matrix and submits the job to the queue.
// Returns 409 when another job is already running.
func (s *Server) handleCreateJob(w http.ResponseWriter, r *http.Request) {
	var req jobCreateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}
	if err := resolveSweeps(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if err := validateJobRequest(req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if err := s.validateBatchMatrix(req.ModelIDs, req.Overrides, req.Sweeps); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if err := s.validateGPUAssignment(req.Overrides, req.Sweeps); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if err := s.validateKLJob(req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	job := benchmark.BenchmarkJob{
		ID:          newJobID(),
		Name:        req.Name,
		Description: req.Description,
		Kind:        benchmark.JobKindBatch,
		Status:      benchmark.JobStatusPending,
		CreatedAt:   time.Now(),
		ModelIDs:    req.ModelIDs,
		BuildIDs:    req.BuildIDs,
		Presets:     req.Presets,
		Overrides:   req.Overrides,
		Sweeps:      req.Sweeps,
		KLReference: req.KLReference,
		Cells:       benchmark.ExpandCellsWithSweeps(req.ModelIDs, req.BuildIDs, req.Presets, req.Sweeps),
	}

	if err := s.jobs.Submit(job); err != nil {
		if errors.Is(err, benchmark.ErrJobAlreadyRunning) {
			http.Error(w, err.Error(), http.StatusConflict)
			return
		}
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	w.WriteHeader(http.StatusCreated)
	respondJSON(w, job)
}

// handleUpdateJob edits a job's definition (name/description/matrix/
// overrides) and re-submits it. Completed cells survive any matrix
// change that still includes their (model, build, preset) point; runs
// behind cells that no longer exist are reassigned to Ad-Hoc.
func (s *Server) handleUpdateJob(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	if id == benchmark.AdhocJobID {
		http.Error(w, "the Ad-Hoc job cannot be edited", http.StatusBadRequest)
		return
	}
	var req jobCreateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}
	if err := resolveSweeps(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if err := validateJobRequest(req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if err := s.validateBatchMatrix(req.ModelIDs, req.Overrides, req.Sweeps); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if err := s.validateGPUAssignment(req.Overrides, req.Sweeps); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if err := s.validateKLJob(req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	updated, err := s.bench.UpdateJobDefinition(id, benchmark.JobDefinition{
		Name:        req.Name,
		Description: req.Description,
		ModelIDs:    req.ModelIDs,
		BuildIDs:    req.BuildIDs,
		Presets:     req.Presets,
		Overrides:   req.Overrides,
		Sweeps:      req.Sweeps,
		KLReference: req.KLReference,
	})
	if err != nil {
		// "synthetic" is the adhoc-edit refusal; anything else means the
		// job wasn't found.
		if strings.Contains(err.Error(), "synthetic") {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}

	if err := s.jobs.Submit(*updated); err != nil {
		if errors.Is(err, benchmark.ErrJobAlreadyRunning) {
			http.Error(w, err.Error(), http.StatusConflict)
			return
		}
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	respondJSON(w, updated)
}

// handleGetJob returns one job (with its cells).
func (s *Server) handleGetJob(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	job, err := s.bench.GetJob(id)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	if isHTMX(r) {
		respondHTML(w)
		// The synthetic adhoc job has no cell matrix — render the
		// existing flat run list instead so legacy + quick-bench runs
		// stay visible.
		if job.ID == benchmark.AdhocJobID {
			s.renderBenchmarkList(w, s.bench.RunsForJob(benchmark.AdhocJobID))
			return
		}
		s.renderJobDetail(w, job)
		return
	}
	respondJSON(w, job)
}

// renderJobDetail enriches a job's cells with their linked run summary
// before rendering, so the matrix view can show TG t/s without a second
// round-trip per cell. Also tallies done/failed/total so the OOB
// summary update fragments at the top of the partial can patch the
// parent list row without re-rendering the entire list.
func (s *Server) renderJobDetail(w http.ResponseWriter, job *benchmark.BenchmarkJob) {
	type cellRow struct {
		Idx        int
		Cell       benchmark.JobCell
		ModelName  string
		Quant      string
		BuildLbl   string
		TGTPS      string // formatted, "—" when no summary
		PPTPS      string
		ErrorShort string
	}
	rows := make([]cellRow, 0, len(job.Cells))
	var done, failed int
	for i, c := range job.Cells {
		switch c.Status {
		case benchmark.CellStatusCompleted:
			done++
		case benchmark.CellStatusFailed:
			failed++
		}
		row := cellRow{Idx: i, Cell: c, ModelName: shortenModelName(c.ModelID), BuildLbl: c.BuildID, TGTPS: "—", PPTPS: "—"}
		// Pull Quant from the registry first so pending cells (no run
		// yet) still show it; the run's value wins once it exists.
		if m, err := s.registry.Get(c.ModelID); err == nil {
			row.Quant = m.Quant
		}
		if c.BenchmarkRunID != "" {
			if run, err := s.bench.Get(c.BenchmarkRunID); err == nil {
				if run.ModelName != "" {
					row.ModelName = run.ModelName
				}
				if run.Quant != "" {
					row.Quant = run.Quant
				}
				if run.BuildID != "" {
					row.BuildLbl = run.BuildID
				}
				if run.Summary != nil {
					row.TGTPS = fmt.Sprintf("%.1f", run.Summary.AvgGenTokPerSec)
					row.PPTPS = fmt.Sprintf("%.0f", run.Summary.AvgPromptTokPerSec)
				}
			}
		}
		if c.Error != "" {
			row.ErrorShort = c.Error
			if len(row.ErrorShort) > 80 {
				row.ErrorShort = row.ErrorShort[:80] + "…"
			}
		}
		rows = append(rows, row)
	}
	s.renderPartial(w, "job_detail", struct {
		Job    *benchmark.BenchmarkJob
		Rows   []cellRow
		Done   int
		Failed int
		Total  int
	}{Job: job, Rows: rows, Done: done, Failed: failed, Total: len(job.Cells)})
}

// handleJobForm renders the new-job modal contents (multi-select models,
// builds, presets, optional overrides, live cell-count preview).
func (s *Server) handleJobForm(w http.ResponseWriter, r *http.Request) {
	respondHTML(w)
	var enabled []*models.Model
	for _, m := range s.registry.List() {
		if cfg, err := s.registry.GetConfig(m.ID); err == nil && cfg.Enabled {
			enabled = append(enabled, m)
		}
	}
	var builds []buildOpt
	for _, b := range s.builder.List() {
		if b.Status != "success" {
			continue
		}
		builds = append(builds, buildOpt{ID: b.ID, Profile: b.Profile, GitRef: b.GitRef, Tag: b.Tag})
	}
	gpuList := s.monitor.Current().GPU
	numGPUs := len(gpuList)

	// The KL reference dropdown lists every installed model, grouped by
	// HF repo, so "the other quant of this repo" is one glance away.
	groups := map[string]*klRepoGroup{}
	var groupOrder []string
	for _, m := range enabled {
		g, ok := groups[m.ModelID]
		if !ok {
			g = &klRepoGroup{Repo: m.ModelID}
			groups[m.ModelID] = g
			groupOrder = append(groupOrder, m.ModelID)
		}
		g.Models = append(g.Models, m)
	}
	sort.Slice(groupOrder, func(i, j int) bool { return groupOrder[i] < groupOrder[j] })
	klOptions := make([]klRepoGroup, 0, len(groupOrder))
	for _, repo := range groupOrder {
		klOptions = append(klOptions, *groups[repo])
	}

	s.renderPartial(w, "job_form", struct {
		Models      []*models.Model
		Builds      []buildOpt
		Presets     []benchmark.Preset
		GPUOptions  []models.GPUOption
		Params      []paramView
		MaxCells    int
		Running     bool
		KLReference []klRepoGroup
	}{
		Models:     enabled,
		Builds:     builds,
		Presets:    benchmark.Presets(),
		GPUOptions: models.GPUAssignOptions(numGPUs, igpuFlags(gpuList)),
		Params:     paramViews(numGPUs, igpuFlags(gpuList)),
		MaxCells:   maxJobCells,
		Running:    s.process.IsRunning(),
		KLReference: klOptions,
	})
}

// handleDeleteJob removes a job. The runs query param controls what
// happens to its runs:
//   - cascade (default): runs deleted with the job
//   - orphan: runs reassigned to the synthetic adhoc job
func (s *Server) handleDeleteJob(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")

	// Deleting the record does not stop the run — the queue holds the job
	// in memory — so the job would keep driving the router while every
	// lookup by id failed. Cancel first.
	if cur, running := s.jobs.Status(); running && cur != nil && cur.ID == id {
		http.Error(w, "this job is running — cancel it before deleting", http.StatusConflict)
		return
	}
	disposition := benchmark.DeleteCascade
	if v := r.URL.Query().Get("runs"); v != "" {
		switch v {
		case "cascade":
			disposition = benchmark.DeleteCascade
		case "orphan":
			disposition = benchmark.DeleteOrphan
		default:
			http.Error(w, "runs must be 'cascade' or 'orphan'", http.StatusBadRequest)
			return
		}
	}
	if err := s.bench.DeleteJob(id, disposition); err != nil {
		// Refusing to delete the synthetic adhoc job is a 400 (the
		// caller passed something they shouldn't); not-found is a 404.
		if strings.Contains(err.Error(), "synthetic") {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

// handleCancelJob signals the queue to cancel the running job.
func (s *Server) handleCancelJob(w http.ResponseWriter, r *http.Request) {
	if err := s.jobs.Cancel(chi.URLParam(r, "id")); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

// handleRetryFailedCells re-queues only failed cells of the given job.
func (s *Server) handleRetryFailedCells(w http.ResponseWriter, r *http.Request) {
	if err := s.jobs.RetryFailed(chi.URLParam(r, "id")); err != nil {
		if errors.Is(err, benchmark.ErrJobAlreadyRunning) {
			http.Error(w, err.Error(), http.StatusConflict)
			return
		}
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	w.WriteHeader(http.StatusAccepted)
}

// handleJobProgress streams job snapshots over SSE. Uses a 500ms poll
// interval for now — the cell loop saves on every status transition, so
// polling the store is sufficient signal. A full pub/sub upgrade can
// land later if the dashboard ever shows multiple jobs at once.
//
// One event type:
//   - "snapshot": JSON-encoded BenchmarkJob with cells. Emitted only
//     when something changed since the previous send.
//
// Stream ends when the job reaches a terminal state.
func (s *Server) handleJobProgress(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	if _, err := s.bench.GetJob(id); err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	sse, err := NewSSEWriter(w)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	ctx := r.Context()
	ticker := time.NewTicker(500 * time.Millisecond)
	defer ticker.Stop()

	var lastSerialized []byte
	emit := func() bool {
		job, err := s.bench.GetJob(id)
		if err != nil {
			return false
		}
		data, err := json.Marshal(job)
		if err != nil {
			return false
		}
		if string(data) != string(lastSerialized) {
			lastSerialized = data
			_ = sse.SendEvent("snapshot", string(data))
		}
		return jobInTerminalState(job.Status)
	}

	if emit() {
		return
	}
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			if emit() {
				return
			}
		}
	}
}

// handleExportJob serves a per-job export. Both formats accept the
// same scope param (?scope=cells|summary) since CSV honours it and
// JSON is always full-fidelity.
func (s *Server) handleExportJob(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	job, err := s.bench.GetJob(id)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	format, err := parseExportFormat(r)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	scope, err := parseExportScope(r)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	runs := s.bench.RunsForJob(id)
	jobs := newJobLookup([]*benchmark.BenchmarkJob{job})

	switch format {
	case exportFormatJSON:
		_ = writeJSONExport(w, fmt.Sprintf("job-%s.json", id), ExportEnvelope{
			Version: exportEnvelopeVersion,
			Jobs:    []*benchmark.BenchmarkJob{job},
			Runs:    runs,
		})
	default: // csv
		_ = writeCSVExport(w, fmt.Sprintf("job-%s-%s.csv", id, scope), runs, jobs, scope)
	}
}

func jobInTerminalState(status string) bool {
	switch status {
	case benchmark.JobStatusCompleted, benchmark.JobStatusFailed, benchmark.JobStatusCanceled:
		return true
	}
	return false
}

// newJobID returns a short, time-prefixed random ID. Time prefix keeps
// IDs sortable; the random suffix avoids collisions when a UI submits
// twice within the same millisecond.
func newJobID() string {
	var buf [4]byte
	_, _ = rand.Read(buf[:])
	return fmt.Sprintf("job-%d-%s", time.Now().UnixMilli(), hex.EncodeToString(buf[:]))
}

// paramView is a sweep field with its choices resolved for rendering.
type paramView struct {
	benchmark.SweepField
	Choices []benchmark.SweepChoice
}

// paramViews resolves each parameter's choice list, filling in the sets
// that depend on the host — currently just GPU assignment, which can't
// be a static table because it depends on how many GPUs are installed.
func paramViews(numGPUs int, igpu []bool) []paramView {
	fields := benchmark.SweepFields()
	out := make([]paramView, 0, len(fields))
	for _, f := range fields {
		pv := paramView{SweepField: f, Choices: f.Choices}
		if f.DynamicChoices == "gpu_assign" {
			for _, o := range models.GPUAssignOptions(numGPUs, igpu) {
				if o.Disabled {
					continue
				}
				pv.Choices = append(pv.Choices, benchmark.SweepChoice{Value: o.Value, Label: o.Label})
			}
		}
		out = append(out, pv)
	}
	return out
}
