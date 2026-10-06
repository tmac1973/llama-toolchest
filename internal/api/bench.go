package api

import (
	"errors"
	"fmt"
	"html"
	"log/slog"
	"net/http"
	"sort"
	"strings"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/tmac1973/llama-toolchest/internal/benchmark"
	"github.com/tmac1973/llama-toolchest/internal/builder"
	"github.com/tmac1973/llama-toolchest/internal/models"
)

// builderResolver adapts the builder's List() into a benchmark.BuildResolver
// closure so the benchmark package can backfill Build snapshots without
// importing builder. Returns the zero value when the build is no longer
// known, which the migration treats as "fall back to the legacy flat
// fields."
func builderResolver(b *builder.Builder) benchmark.BuildResolver {
	return func(buildID string) benchmark.BuildSnapshot {
		br, ok := b.Find(buildID)
		if !ok {
			return benchmark.BuildSnapshot{}
		}
		return benchmark.BuildSnapshot{
			ID:         br.ID,
			Tag:        br.Tag,
			Profile:    br.Profile,
			Vendor:     br.Profile,
			GitSHA:     br.GitSHA,
			GitRef:     br.GitRef,
			CMakeFlags: br.CMakeFlags,
			BinaryPath: br.BinaryPath,
		}
	}
}

// handleListBenchmarks returns benchmark runs, optionally filtered:
//   - ?job=<id>          returns only runs belonging to that job
//   - ?scope=adhoc       returns only runs in the synthetic adhoc job
//   - ?scope=batch       returns only runs belonging to a real batch job
func (s *Server) handleListBenchmarks(w http.ResponseWriter, r *http.Request) {
	runs := s.bench.List()

	if jobID := r.URL.Query().Get("job"); jobID != "" {
		runs = filterRunsByJob(runs, jobID)
	}
	if scope := r.URL.Query().Get("scope"); scope != "" {
		switch scope {
		case "adhoc":
			runs = filterRunsByJob(runs, benchmark.AdhocJobID)
		case "batch":
			batchJobIDs := map[string]bool{}
			for _, j := range s.bench.ListJobs() {
				if j.ID != benchmark.AdhocJobID {
					batchJobIDs[j.ID] = true
				}
			}
			filtered := runs[:0]
			for _, run := range runs {
				if batchJobIDs[run.JobID] {
					filtered = append(filtered, run)
				}
			}
			runs = filtered
		default:
			http.Error(w, "scope must be 'adhoc' or 'batch'", http.StatusBadRequest)
			return
		}
	}

	if isHTMX(r) {
		respondHTML(w)
		s.renderBenchmarkList(w, runs)
		return
	}

	respondJSON(w, runs)
}

func filterRunsByJob(runs []benchmark.BenchmarkRun, jobID string) []benchmark.BenchmarkRun {
	out := runs[:0]
	for _, r := range runs {
		if r.JobID == jobID {
			out = append(out, r)
		}
	}
	return out
}

// benchListData feeds the benchmark_list partial.
type benchListData struct {
	Groups []benchListGroup
	// HasEval adds the Score column: at least one run in the list is a
	// capability run (run.Eval set).
	HasEval bool
	Cols    int // column count, for the colspan cells
}

// benchListGroup is one model's runs, collapsed by default.
type benchListGroup struct {
	Name string
	Runs []benchListRow
}

// benchListRow is one run with its cell text worked out. An empty PP,
// TG, TTFT or Score shows as an em-dash.
type benchListRow struct {
	Run          *benchmark.BenchmarkRun
	Search       string
	PP, TG, TTFT string
	Score        string
	ScoreNote    string // tooltip warning that the score is not comparable
}

// renderBenchmarkList emits the grouped, searchable benchmarks table.
// Groups by ModelName; each group is collapsed by default. The wrapping
// JS in benchmarks.html drives toggle/filter/compare/export/delete using
// data-model and data-search attributes plus the .bench-runs-container
// scope so the same renderer can serve multiple list contexts (today
// just the adhoc job's expanded view; tomorrow per-job filtered views).
func (s *Server) renderBenchmarkList(w http.ResponseWriter, runs []benchmark.BenchmarkRun) {
	data := benchListData{HasEval: hasEvalRuns(runs), Cols: 10}
	if data.HasEval {
		data.Cols++
	}

	idx := map[string]int{}
	for i := range runs {
		run := &runs[i]
		key := run.ModelName
		if key == "" {
			key = "(unknown)"
		}
		g, ok := idx[key]
		if !ok {
			g = len(data.Groups)
			idx[key] = g
			data.Groups = append(data.Groups, benchListGroup{Name: key})
		}
		data.Groups[g].Runs = append(data.Groups[g].Runs, benchListRowFor(run, data.HasEval))
	}
	sort.SliceStable(data.Groups, func(i, j int) bool {
		return strings.ToLower(data.Groups[i].Name) < strings.ToLower(data.Groups[j].Name)
	})

	s.renderPartial(w, "benchmark_list", data)
}

// benchListRowFor works out one run's cells. A capability run shows its
// score and no timings; the Score cell exists only when hasEval.
func benchListRowFor(run *benchmark.BenchmarkRun, hasEval bool) benchListRow {
	row := benchListRow{
		Run:    run,
		Search: strings.ToLower(strings.Join([]string{run.ModelName, run.Quant, run.BuildID, run.BuildRef, run.Preset}, " ")),
	}
	if run.Summary != nil {
		row.PP = fmt.Sprintf("%.0f", run.Summary.AvgPromptTokPerSec)
		row.TG = fmt.Sprintf("%.1f", run.Summary.AvgGenTokPerSec)
		row.TTFT = fmt.Sprintf("%.0f ms", run.Summary.AvgTTFTMs)
	}
	if hasEval && run.Eval != nil {
		row.Score = evalScoreText(run.Eval)
		if row.Score == "" {
			row.Score = "score unavailable"
		}
		// The caveat travels with the number: a score measured through
		// a compressed memory cache reads exactly like a comparable one
		// otherwise.
		row.ScoreNote = evalComparabilityNote(run.Config)
		row.PP, row.TG, row.TTFT = "", "", ""
	}
	return row
}

// handleGetBenchmark returns a single benchmark run.
func (s *Server) handleGetBenchmark(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	run, err := s.bench.Get(id)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}

	if isHTMX(r) {
		respondHTML(w)
		s.renderPartial(w, "benchmark_detail", run)
		return
	}

	respondJSON(w, run)
}

// handleStartBenchmark is the Quick Benchmark entry point: it builds a
// 1-cell job from a (model, preset) pair and submits it to the JobQueue,
// using the currently-active build. Equivalent to the new-job form with
// {ModelIDs:[modelID], BuildIDs:[active], Presets:[preset]}.
func (s *Server) handleStartBenchmark(w http.ResponseWriter, r *http.Request) {
	r.ParseForm()
	modelID := r.FormValue("model_id")
	presetName := r.FormValue("preset")

	if modelID == "" {
		http.Error(w, "model_id is required", http.StatusBadRequest)
		return
	}

	model, err := s.registry.Get(modelID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}

	if !s.process.IsRunning() {
		http.Error(w, "router is not running — start the server first", http.StatusBadRequest)
		return
	}

	// Resolve active build the same way startRouter does. resolveActiveBuild
	// also validates the selected build actually succeeded, falling back to
	// the latest successful one otherwise.
	buildID := ""
	if b := s.resolveActiveBuild(); b != nil {
		buildID = b.ID
	}
	if buildID == "" {
		http.Error(w, "no compiled build available — build llama.cpp first", http.StatusBadRequest)
		return
	}

	preset := benchmark.GetPreset(presetName)

	jobName := fmt.Sprintf("Quick: %s / %s", models.ShortModelName(model.ModelID), preset.Name)
	job := benchmark.BenchmarkJob{
		ID:        newJobID(),
		Name:      jobName,
		Kind:      benchmark.JobKindBatch,
		Status:    benchmark.JobStatusPending,
		CreatedAt: time.Now(),
		ModelIDs:  []string{modelID},
		BuildIDs:  []string{buildID},
		Presets:   []string{preset.Name},
		Cells:     benchmark.ExpandCells([]string{modelID}, []string{buildID}, []string{preset.Name}),
	}

	if !s.submitJob(w, job) {
		return
	}

	if isHTMX(r) {
		respondHTML(w)
		// Tell the jobs list to refresh so the user sees the new job
		// appear; the response body itself is just a brief confirmation.
		w.Header().Set("HX-Trigger", "jobChanged")
		fmt.Fprintf(w, `<small style="color:var(--pico-muted-color);">Started: %s</small>`, html.EscapeString(jobName))
		return
	}

	respondJSON(w, job)
}

// handleDeleteBenchmark removes a benchmark run.
func (s *Server) handleDeleteBenchmark(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	if err := s.bench.Delete(id); err != nil {
		http.Error(w, err.Error(), benchErrorStatus(err, http.StatusNotFound))
		return
	}

	if isHTMX(r) {
		s.handleListBenchmarks(w, r)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

// handleBatchDeleteBenchmarks deletes multiple benchmark runs at once.
func (s *Server) handleBatchDeleteBenchmarks(w http.ResponseWriter, r *http.Request) {
	idsParam := r.URL.Query().Get("ids")
	if idsParam == "" {
		http.Error(w, "ids parameter required", http.StatusBadRequest)
		return
	}
	for _, id := range strings.Split(idsParam, ",") {
		id = strings.TrimSpace(id)
		if id == "" {
			continue
		}
		// A run that is already gone is not a failure here; a refusal or
		// a failed save is.
		if err := s.bench.Delete(id); err != nil {
			if status := benchErrorStatus(err, http.StatusOK); status != http.StatusOK {
				http.Error(w, err.Error(), status)
				return
			}
		}
	}
	w.WriteHeader(http.StatusOK)
}

// benchErrorStatus maps a benchmark store error to an HTTP status: a
// read-only store is a conflict, a failed save is a server error, and
// anything else gets fallback.
func benchErrorStatus(err error, fallback int) int {
	switch {
	case errors.Is(err, benchmark.ErrStoreReadOnly):
		return http.StatusConflict
	case errors.Is(err, benchmark.ErrSaveFailed):
		return http.StatusInternalServerError
	}
	return fallback
}

// handleExportBenchmarks exports an arbitrary selection of runs in
// either CSV (cells or summary scope) or JSON. Routes through the
// shared exporter so the column schema matches /api/benchmark-jobs/{id}/export.
func (s *Server) handleExportBenchmarks(w http.ResponseWriter, r *http.Request) {
	idsParam := r.URL.Query().Get("ids")
	if idsParam == "" {
		http.Error(w, "ids parameter required", http.StatusBadRequest)
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

	var runs []benchmark.BenchmarkRun
	jobIDs := map[string]struct{}{}
	for _, id := range strings.Split(idsParam, ",") {
		id = strings.TrimSpace(id)
		if id == "" {
			continue
		}
		if run, err := s.bench.Get(id); err == nil {
			runs = append(runs, *run)
			if run.JobID != "" {
				jobIDs[run.JobID] = struct{}{}
			}
		}
	}

	jobs := make([]*benchmark.BenchmarkJob, 0, len(jobIDs))
	for jid := range jobIDs {
		if j, err := s.bench.GetJob(jid); err == nil {
			jobs = append(jobs, j)
		}
	}
	jobLU := newJobLookup(jobs)

	switch format {
	case exportFormatJSON:
		_ = writeJSONExport(w, "benchmarks.json", ExportEnvelope{
			Version: exportEnvelopeVersion,
			Jobs:    jobs,
			Runs:    runs,
		})
	default:
		_ = writeCSVExport(w, fmt.Sprintf("benchmarks-%s.csv", scope), runs, jobLU, scope)
	}
}

// handleCompareBenchmarks returns comparison data for selected runs.
func (s *Server) handleCompareBenchmarks(w http.ResponseWriter, r *http.Request) {
	idsParam := r.URL.Query().Get("ids")
	if idsParam == "" {
		http.Error(w, "ids parameter required", http.StatusBadRequest)
		return
	}

	ids := strings.Split(idsParam, ",")
	var runs []benchmark.BenchmarkRun
	var missing []string
	for _, id := range ids {
		id = strings.TrimSpace(id)
		if id == "" {
			continue
		}
		run, err := s.bench.Get(id)
		if err != nil {
			// Selected but no longer in the store — deleted between
			// selecting and comparing. Recorded rather than dropped: a
			// comparison holding fewer runs than the user picked, with
			// nothing said about it, is how a missing contender goes
			// unnoticed.
			missing = append(missing, id)
			continue
		}
		runs = append(runs, *run)
	}

	if len(runs) < 2 {
		http.Error(w, "need at least 2 runs to compare", http.StatusBadRequest)
		return
	}

	comparison := benchmark.BuildComparison(runs)
	comparison.MissingRunIDs = missing

	if isHTMX(r) {
		respondHTML(w)
		s.renderPartial(w, "benchmark_compare", comparison)
		return
	}

	respondJSON(w, comparison)
}

// handleTimings returns passive timing data.
func (s *Server) handleTimings(w http.ResponseWriter, r *http.Request) {
	modelID := chi.URLParam(r, "model_id")

	if isHTMX(r) {
		respondHTML(w)
		summary := s.bench.TimingSummary()
		s.renderPartial(w, "timings_summary", struct {
			Summary []benchmark.TimingModelSummary
		}{Summary: summary})
		return
	}

	samples := s.bench.Timings(modelID)
	respondJSON(w, samples)
}

// handleBenchmarkForm returns the benchmark form options (model list, presets).
func (s *Server) handleBenchmarkForm(w http.ResponseWriter, r *http.Request) {
	respondHTML(w)

	// Helper models are not benchmark targets: their settings are fixed,
	// so there is nothing to compare.
	allModels := s.registry.ListServing()
	var enabledModels []*models.Model
	for _, m := range allModels {
		if cfg, err := s.registry.GetConfig(m.ID); err == nil && cfg.Enabled {
			enabledModels = append(enabledModels, m)
		}
	}

	// Capability presets are filtered out of the quick-benchmark form as
	// a product decision: the path itself does NOT bypass the cell loop
	// (handleStartBenchmark builds a 1-cell job for the same JobQueue, so
	// a capability preset POSTed here or via the JSON API would run
	// correctly), but the form has no KL-reference selector, its "router
	// is not running — start the server first" gate is meaningless for
	// capability cells (they stop the router), and its single-run
	// framing is built around timing results.
	var quickPresets []benchmark.Preset
	for _, p := range benchmark.VisiblePresets() {
		if p.EffectiveSource() == benchmark.PresetSourceCapability {
			continue
		}
		quickPresets = append(quickPresets, p)
	}

	s.renderPartial(w, "benchmark_form", struct {
		Models  []*models.Model
		Presets []benchmark.Preset
		Running bool
	}{
		Models:  enabledModels,
		Presets: quickPresets,
		Running: s.process.IsRunning(),
	})
}

// captureTimings is called by the proxy to record passive timing data.
func (s *Server) captureTimings(modelID string, timings map[string]any) {
	promptN, _ := timings["prompt_n"].(float64)
	predictedN, _ := timings["predicted_n"].(float64)
	promptPerSec, _ := timings["prompt_per_second"].(float64)
	predictedPerSec, _ := timings["predicted_per_second"].(float64)

	if predictedN == 0 {
		return
	}

	s.bench.RecordTiming(benchmark.TimingSample{
		Timestamp:       time.Now(),
		ModelID:         modelID,
		PromptTokens:    int(promptN),
		GenTokens:       int(predictedN),
		PromptTokPerSec: promptPerSec,
		GenTokPerSec:    predictedPerSec,
	})

	slog.Debug("captured timing", "model", modelID,
		"prompt_tps", fmt.Sprintf("%.1f", promptPerSec),
		"gen_tps", fmt.Sprintf("%.1f", predictedPerSec))
}
