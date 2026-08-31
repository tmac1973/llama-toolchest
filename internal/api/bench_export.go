package api

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"strconv"
	"strings"

	"github.com/tmac1973/llama-toolchest/internal/benchmark"
	"github.com/tmac1973/llama-toolchest/internal/evaluate"
)

// ExportEnvelope is the on-disk JSON shape for both per-job and
// per-selection exports. Always includes a `version` so we can grow
// the shape without breaking re-importers.
//
//   - Per-job exports: Jobs has one entry (the requested job), Runs
//     contains every run belonging to it.
//   - Per-selection exports: Jobs is the unique set of jobs the runs
//     belong to (lookup by id), Runs is the user's exact selection.
type ExportEnvelope struct {
	Version int                       `json:"version"`
	Jobs    []*benchmark.BenchmarkJob `json:"jobs,omitempty"`
	Runs    []benchmark.BenchmarkRun  `json:"runs"`
}

// v2: runs carry size_gib / gpus[].vram_total_mib instead of the
// misnamed size_gb / vram_total_mb (the values were always binary units).
// v3: runs may carry `memory` — what the load actually allocated,
// itemised. Absent on every run recorded before it was measured, and on
// any run whose router was below log verbosity 4.
const exportEnvelopeVersion = 3

const (
	exportFormatCSV  = "csv"
	exportFormatJSON = "json"
	exportScopeCells = "cells"
	exportScopeSum   = "summary"
)

// formatCMakeFlags renders a build's CMake flags as a single stable
// cell value. Matches effectiveCMakeFlags' "-DK=V" form and sorting so
// an exported row can be compared against the Builds tab preview
// verbatim. Without this column, two builds off the same git ref that
// differ only in flags are indistinguishable in CSV — which is exactly
// the flag-comparison workflow benchmarks exist to support.
func formatCMakeFlags(flags map[string]string) string {
	if len(flags) == 0 {
		return ""
	}
	parts := make([]string, 0, len(flags))
	for k, v := range flags {
		parts = append(parts, fmt.Sprintf("-D%s=%s", k, v))
	}
	sort.Strings(parts)
	return strings.Join(parts, " ")
}

// formatSweepValues renders a run's sweep point as a stable cell value.
// Without it, a sweep's rows are identical apart from a throughput
// number — the same reason cmake_flags had to be exported.
func formatSweepValues(values map[string]string) string {
	if len(values) == 0 {
		return ""
	}
	parts := make([]string, 0, len(values))
	for k, v := range values {
		parts = append(parts, fmt.Sprintf("%s=%s", k, v))
	}
	sort.Strings(parts)
	return strings.Join(parts, " ")
}

// memoryExportFields renders a run's measured footprint as the six
// memory columns, in header order. A run with nothing measured gets six
// empty strings rather than zeros: llama.cpp reports its buffers only at
// log verbosity 4, and a zero would read as "this model used no memory"
// instead of "nobody looked".
//
// card_gib is empty for a load that overlapped another, because that
// figure covers both models. The per-instance columns beside it are
// still that model's own.
func memoryExportFields(m *benchmark.MemorySnapshot) []string {
	if m == nil {
		return []string{"", "", "", "", "", ""}
	}
	card := ""
	if m.CardDeltaGiB > 0 {
		card = ftoa(m.CardDeltaGiB)
	}
	return []string{
		ftoa(m.GPUGiB), ftoa(m.WeightsGiB), ftoa(m.KVGiB),
		ftoa(m.ComputeGiB), ftoa(m.HostGiB), card,
	}
}

// memoryExportHeader names those columns. GiB throughout, matching the
// model sizes already in these exports.
var memoryExportHeader = []string{
	"mem_gpu_gib", "mem_weights_gib", "mem_kv_gib",
	"mem_compute_gib", "mem_host_gib", "mem_card_gib",
}

// jobByID looks up a job pointer in a slice without scanning twice.
type jobLookup map[string]*benchmark.BenchmarkJob

func newJobLookup(jobs []*benchmark.BenchmarkJob) jobLookup {
	m := make(jobLookup, len(jobs))
	for _, j := range jobs {
		m[j.ID] = j
	}
	return m
}

// writeJSONExport serializes the envelope with stable indentation.
func writeJSONExport(w http.ResponseWriter, filename string, env ExportEnvelope) error {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Content-Disposition", fmt.Sprintf(`attachment; filename=%q`, filename))
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(env)
}

// writeCSVExport writes either per-cell rows (one per result) or per-cell
// summary rows (one per run), with full job/build/preset/source context
// per the column schema in plan/batch-benchmarks.md.
func writeCSVExport(w http.ResponseWriter, filename string, runs []benchmark.BenchmarkRun, jobs jobLookup, scope string) error {
	w.Header().Set("Content-Type", "text/csv; charset=utf-8")
	w.Header().Set("Content-Disposition", fmt.Sprintf(`attachment; filename=%q`, filename))
	cw := csv.NewWriter(w)
	defer cw.Flush()

	switch scope {
	case exportScopeSum:
		return writeCSVSummary(cw, runs, jobs)
	case exportScopeCells, "":
		return writeCSVCells(cw, runs, jobs)
	default:
		return fmt.Errorf("unknown scope %q (want cells or summary)", scope)
	}
}

// evalExportFields renders the run's capability scores as the seven
// export columns, in header order: mode, dataset, score, error,
// tasks/chunks, KL statistics, reference identity. Performance runs
// (Eval == nil) get seven empty strings — matching how the per-mode
// throughput columns already behave (empty, not zero) — and the JSON
// export carries the same data through the run's own EvalScores.
//
// Values keep the tool's own precision (the parser pins it): perplexity
// 4+5 decimals, KL 6 decimals, accuracy 4 decimals. The score column is
// the single headline number in raw form; the display's formatted form
// (plan step 1) lives in evalScoreText.
func evalExportFields(e *benchmark.EvalScores) []string {
	out := make([]string, 7)
	if e == nil {
		return out
	}
	out[0] = e.Mode
	out[1] = e.Dataset
	switch e.Mode {
	case string(evaluate.ModePerplexity):
		out[2] = strconv.FormatFloat(e.Perplexity, 'f', 5, 64)
		out[3] = strconv.FormatFloat(e.PerplexityErr, 'f', 5, 64)
		out[4] = evalCountOrFull(e.Chunks)
	case string(evaluate.ModeKLDiv):
		out[2] = strconv.FormatFloat(e.KLMean, 'f', 6, 64)
		out[3] = strconv.FormatFloat(e.KLMeanErr, 'f', 6, 64)
		out[4] = evalCountOrFull(e.Chunks)
		out[5] = fmt.Sprintf("max=%s p999=%s same_top_pct=%.3f±%.3f",
			strconv.FormatFloat(e.KLMax, 'f', 6, 64),
			strconv.FormatFloat(e.KLP999, 'f', 6, 64),
			e.SameTopPct, e.SameTopPctErr)
	case string(evaluate.ModeHellaSwag), string(evaluate.ModeWinogrande):
		out[2] = strconv.FormatFloat(e.Accuracy, 'f', 4, 64)
		if e.Mode == string(evaluate.ModeWinogrande) {
			// Winogrande's error is symmetric: the half-width of its CI.
			out[3] = strconv.FormatFloat((e.AccuracyCIHigh-e.AccuracyCILow)/2, 'f', 4, 64)
		}
		out[4] = strconv.Itoa(e.Tasks)
		out[5] = fmt.Sprintf("ci_low=%.4f ci_high=%.4f", e.AccuracyCILow, e.AccuracyCIHigh)
	}
	out[6] = e.Reference
	return out
}

// evalCountOrFull renders the chunk count the way the display does: the
// number, or "full" when the run recorded none.
func evalCountOrFull(n int) string {
	if n <= 0 {
		return "full"
	}
	return strconv.Itoa(n)
}

func writeCSVCells(cw *csv.Writer, runs []benchmark.BenchmarkRun, jobs jobLookup) error {
	header := []string{
		"job_id", "job_name", "run_id", "created_at",
		"model_id", "model_name", "quant",
		"build_id", "build_profile", "git_ref", "cmake_flags",
		"preset", "sweep",
		"eval_mode", "eval_dataset", "eval_score", "eval_error",
		"eval_tasks_chunks", "eval_kl_stats", "eval_reference",
		"mem_gpu_gib", "mem_weights_gib", "mem_kv_gib",
		"mem_compute_gib", "mem_host_gib", "mem_card_gib",
		"source",
		"prompt_tokens", "gen_tokens", "depth", "concurrency", "repetition",
		"pp_throughput", "pp_throughput_std",
		"tg_throughput", "tg_throughput_std",
		"peak_throughput", "peak_throughput_std",
		"ttft_ms", "ttfr_ms", "e2e_ttft_ms", "total_ms",
	}
	if err := cw.Write(header); err != nil {
		return err
	}

	for _, run := range runs {
		jobName := ""
		if j := jobs[run.JobID]; j != nil {
			jobName = j.Name
		}
		build := run.EffectiveBuild()
		base := []string{
			run.JobID, jobName, run.ID, run.CreatedAt.UTC().Format("2006-01-02T15:04:05Z"),
			run.ModelID, run.ModelName, run.Quant,
			build.ID, build.Profile, build.GitRef, formatCMakeFlags(build.CMakeFlags),
			run.Preset, formatSweepValues(run.SweepValues),
		}
		base = append(base, evalExportFields(run.Eval)...)
		// Memory is a property of the load, so it repeats on every row
		// of a run — the same way build and preset do.
		base = append(base, memoryExportFields(run.Memory)...)

		if len(run.Results) > 0 {
			for _, r := range run.Results {
				row := append([]string(nil), base...)
				row = append(row,
					"internal",
					itoa(r.PromptTokens), itoa(r.GenTokens), "", "1", itoa(r.Repetition),
					ftoa(r.PromptTokPerSec), "",
					ftoa(r.GenTokPerSec), "",
					"", "",
					ftoa(r.TTFTMs), "", "", ftoa(r.TotalMs),
				)
				if err := cw.Write(row); err != nil {
					return err
				}
			}
		}

		for _, b := range run.LlamaBenchy {
			row := append([]string(nil), base...)
			row = append(row,
				"benchy",
				itoa(b.PromptSize), itoa(b.ResponseSize), itoa(b.ContextSize), itoa(b.Concurrency), "",
				metricMean(b.PPThroughput), metricStd(b.PPThroughput),
				metricMean(b.TGThroughput), metricStd(b.TGThroughput),
				metricMean(b.PeakThroughput), metricStd(b.PeakThroughput),
				"", metricMean(b.TTFR), metricMean(b.E2ETTFT), "",
			)
			if err := cw.Write(row); err != nil {
				return err
			}
		}

		// If a run has neither internal results nor benchy results
		// (e.g. failed before producing any), emit a single
		// header-only row so the run still shows up in the export.
		if len(run.Results) == 0 && len(run.LlamaBenchy) == 0 {
			row := append([]string(nil), base...)
			source := "internal"
			if run.BenchyCommand != "" {
				source = "benchy"
			}
			row = append(row, source, "", "", "", "", "", "", "", "", "", "", "", "", "", "", "")
			if err := cw.Write(row); err != nil {
				return err
			}
		}
	}
	return nil
}

func writeCSVSummary(cw *csv.Writer, runs []benchmark.BenchmarkRun, jobs jobLookup) error {
	header := []string{
		"job_id", "job_name", "run_id", "created_at",
		"model_id", "model_name", "quant",
		"build_id", "build_profile", "git_ref", "cmake_flags",
		"preset", "sweep",
		"eval_mode", "eval_dataset", "eval_score", "eval_error",
		"eval_tasks_chunks", "eval_kl_stats", "eval_reference",
		"mem_gpu_gib", "mem_weights_gib", "mem_kv_gib",
		"mem_compute_gib", "mem_host_gib", "mem_card_gib",
		"source", "status",
		"avg_pp_throughput", "avg_tg_throughput", "avg_ttft_ms",
		"min_tg_throughput", "max_tg_throughput",
		"result_count", "duration_ms",
	}
	if err := cw.Write(header); err != nil {
		return err
	}
	for _, run := range runs {
		jobName := ""
		if j := jobs[run.JobID]; j != nil {
			jobName = j.Name
		}
		build := run.EffectiveBuild()
		source := "internal"
		if len(run.LlamaBenchy) > 0 || run.BenchyCommand != "" {
			source = "benchy"
		}
		count := len(run.Results) + len(run.LlamaBenchy)

		var avgPP, avgTG, avgTTFT, minTG, maxTG string
		if run.Summary != nil {
			avgPP = ftoa(run.Summary.AvgPromptTokPerSec)
			avgTG = ftoa(run.Summary.AvgGenTokPerSec)
			avgTTFT = ftoa(run.Summary.AvgTTFTMs)
			minTG = ftoa(run.Summary.MinGenTokPerSec)
			maxTG = ftoa(run.Summary.MaxGenTokPerSec)
		}

		row := []string{
			run.JobID, jobName, run.ID, run.CreatedAt.UTC().Format("2006-01-02T15:04:05Z"),
			run.ModelID, run.ModelName, run.Quant,
			build.ID, build.Profile, build.GitRef, formatCMakeFlags(build.CMakeFlags),
			run.Preset, formatSweepValues(run.SweepValues),
		}
		row = append(row, evalExportFields(run.Eval)...)
		row = append(row, memoryExportFields(run.Memory)...)
		row = append(row, source, run.Status,
			avgPP, avgTG, avgTTFT,
			minTG, maxTG,
			itoa(count), strconv.FormatInt(run.DurationMs, 10),
		)
		if err := cw.Write(row); err != nil {
			return err
		}
	}
	return nil
}

func itoa(n int) string { return strconv.Itoa(n) }

func ftoa(f float64) string {
	if f == 0 {
		return "0"
	}
	return strconv.FormatFloat(f, 'f', -1, 64)
}

func metricMean(m *benchmark.LlamaBenchyMetric) string {
	if m == nil {
		return ""
	}
	return ftoa(m.Mean)
}
func metricStd(m *benchmark.LlamaBenchyMetric) string {
	if m == nil {
		return ""
	}
	return ftoa(m.Std)
}

// parseEnumParam normalizes a two-valued enum query param: empty → the
// first valid value, either valid value → itself (lowercased), anything
// else → an error naming the valid values.
func parseEnumParam(r *http.Request, key, valid1, valid2 string) (string, error) {
	switch v := strings.ToLower(r.URL.Query().Get(key)); v {
	case "":
		return valid1, nil
	case valid1, valid2:
		return v, nil
	default:
		return "", fmt.Errorf("%s must be %q or %q", key, valid1, valid2)
	}
}

// parseExportFormat normalizes the format query param. Defaults to csv
// because the typical browser flow downloads a spreadsheet-friendly file.
func parseExportFormat(r *http.Request) (string, error) {
	return parseEnumParam(r, "format", exportFormatCSV, exportFormatJSON)
}

func parseExportScope(r *http.Request) (string, error) {
	return parseEnumParam(r, "scope", exportScopeCells, exportScopeSum)
}
