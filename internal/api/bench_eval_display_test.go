package api

import (
	"bytes"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"html/template"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/benchmark"
	"github.com/tmac1973/llama-toolchest/internal/config"
	"github.com/tmac1973/llama-toolchest/internal/evaluate"
	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/web"
)

// Phase 04 render + export tests: the conditional Score column in the
// results list, the job cell matrix's score + skip-reason cells, the
// compare view's per-mode scores, the Evaluation Data card, and the
// export's eval columns.

// evalRun builds a completed capability run with a PPL quick score.
func evalRun(id string) benchmark.BenchmarkRun {
	return benchmark.BenchmarkRun{
		ID:        id,
		JobID:     "job-eval",
		CreatedAt: time.Unix(0, 0).UTC(),
		Status:    benchmark.StatusCompleted,
		ModelID:   "m4",
		ModelName: "M-4B",
		Quant:     "Q4_K_XL",
		Preset:    "perplexity-quick",
		Eval: &benchmark.EvalScores{
			Mode:          "perplexity",
			Dataset:       "wikitext-2",
			ContextSize:   512,
			Chunks:        100,
			Perplexity:    6.234,
			PerplexityErr: 0.04,
		},
	}
}

// perfRun builds a completed performance run with a summary.
func perfRun(id string) benchmark.BenchmarkRun {
	return benchmark.BenchmarkRun{
		ID:        id,
		JobID:     "job-perf",
		CreatedAt: time.Unix(0, 0).UTC(),
		Status:    benchmark.StatusCompleted,
		ModelID:   "m4",
		ModelName: "M-4B",
		Quant:     "Q4_K_XL",
		Preset:    "internal-standard",
		Summary: &benchmark.BenchmarkSummary{
			AvgPromptTokPerSec: 100,
			AvgGenTokPerSec:    40,
			AvgTTFTMs:          10,
		},
	}
}

// benchListServer builds the minimal server a render path needs. The
// pages map is populated because renderPartial (job_detail, the eval
// data card) executes through it — without it a render silently writes
// nothing.
func benchListServer(t *testing.T) *Server {
	t.Helper()
	s := &Server{registry: models.NewRegistry(t.TempDir(), t.TempDir())}
	s.pages = s.parseTemplates()
	return s
}

// A list containing any run with Eval set gains the Score column; the
// capability run renders its mode-appropriate score and em-dashes in
// the timing columns; the performance run keeps its timings and shows
// an em-dash in the score column.
func TestBenchListRendersScoreColumnAndEvalRow(t *testing.T) {
	s := benchListServer(t)
	runs := []benchmark.BenchmarkRun{perfRun("r-perf"), evalRun("r-eval")}
	rec := httptest.NewRecorder()
	s.renderBenchmarkList(rec, runs)
	out := rec.Body.String()

	if !strings.Contains(out, ">Score</th>") {
		t.Errorf("the Score column header is missing\n%s", out)
	}
	// The plan's exact rendering form (phase 04 step 1).
	if !strings.Contains(out, "PPL 6.234 ±0.04 (100 chunks)") {
		t.Errorf("the perplexity score cell is not rendered in its plan form\n%s", out)
	}
	// The capability row: score present, timing columns em-dash.
	// Find the row holding the score and check its PP/TG/TTFT cells.
	idx := strings.Index(out, "PPL 6.234")
	if idx == -1 {
		t.Fatal("score cell not found")
	}
	row := out[strings.LastIndex(out[:idx], "<tr>"):]
	// The row has 11 cells now (10 base + score); the three timing
	// cells (PP/TG/TTFT, positions 3-5 after check/model/quant) must be
	// em-dashes because a capability run produced no t/s.
	cells := strings.Split(row, "</td>")
	// 11 cells now (10 base + score). After the split: cells[0] checkbox,
	// cells[1] model, cells[2] quant, cells[3-5] the timing cells
	// (PP/TG/TTFT) — all em-dash for a capability run (no t/s).
	if len(cells) < 11 {
		t.Fatalf("eval row has %d cells, want 11\n%s", len(cells), row)
	}
	// Splitting on "</td>" consumes the closing tag, so a bare em-dash
	// cell reads "<td>—" (the leading tag stays with the content).
	for i := 3; i <= 5; i++ {
		if !strings.Contains(cells[i], "<td>—") {
			t.Errorf("eval row timing cell %d = %q, want an em-dash", i-2, cells[i])
		}
	}
	// The performance row keeps its timings and shows an em-dash score.
	pidx := strings.Index(out, "<strong>40.0</strong>")
	if pidx == -1 {
		t.Fatal("the performance row's TG value is missing")
	}
	prow := out[strings.LastIndex(out[:pidx], "<tr>"):]
	pcells := strings.Split(prow, "</td>")
	if len(pcells) < 11 {
		t.Fatalf("perf row has %d cells, want 11\n%s", len(pcells), prow)
	}
	// Score is the 7th td (after check/model/quant/PP/TG/TTFT).
	if !strings.Contains(pcells[6], "<td>—") {
		t.Errorf("perf row score cell = %q, want an em-dash", pcells[6])
	}
}

// A pure performance list renders exactly as before: no Score column,
// ten cells per row.
func TestBenchListPerformanceOnlyUnchanged(t *testing.T) {
	s := benchListServer(t)
	rec := httptest.NewRecorder()
	s.renderBenchmarkList(rec, []benchmark.BenchmarkRun{perfRun("r1"), perfRun("r2")})
	out := rec.Body.String()

	if strings.Contains(out, ">Score</th>") {
		t.Error("a pure performance list must not gain a Score column")
	}
	// The group header and detail rows keep their ten columns.
	if !strings.Contains(out, `colspan="10"`) {
		t.Errorf("performance rows must keep colspan 10\n%s", out)
	}
	if strings.Contains(out, "PPL ") {
		t.Error("no score text in a performance-only list")
	}
}

// The job cell matrix: a capability cell renders its score in the new
// Score column (em-dash timings), and a KL cell skipped as the
// reference model renders its SkipReason informationally in the status
// cell — not as an error.
func TestJobDetailRendersScoreAndSkipReason(t *testing.T) {
	s := benchListServer(t)
	dir := t.TempDir()
	store := benchmark.NewStore(dir, nil)
	s.bench = store

	const skipReason = "this is the reference model — its difference from itself is zero"
	job := benchmark.BenchmarkJob{
		ID: "job-skip", Name: "kl", Kind: benchmark.JobKindBatch,
		Status: benchmark.JobStatusCompleted, CreatedAt: time.Unix(0, 0).UTC(),
		Cells: []benchmark.JobCell{
			{ModelID: "m4", BuildID: "b1", Preset: "kl-divergence-quick",
				Status: benchmark.CellStatusCompleted, BenchmarkRunID: "run-kl", Attempt: 1},
			{ModelID: "m8", BuildID: "b1", Preset: "kl-divergence-quick",
				Status: benchmark.CellStatusCompleted, Attempt: 1, SkipReason: skipReason},
		},
	}
	klRun := benchmark.BenchmarkRun{
		ID: "run-kl", JobID: "job-skip", CreatedAt: time.Unix(0, 0).UTC(),
		Status: benchmark.StatusCompleted, ModelID: "m4", ModelName: "M-4B",
		Quant: "Q4_K_XL", Preset: "kl-divergence-quick",
		Eval: &benchmark.EvalScores{
			Mode: "kl-divergence", Dataset: "wikitext-2", ContextSize: 512,
			Chunks: 100, KLMean: 0.012, KLMeanErr: 0.001,
			SameTopPct: 97.4, Reference: "UD-Q8_K_XL",
		},
	}
	store.SaveJob(job)
	store.Save(klRun)

	saved, err := store.GetJob("job-skip")
	if err != nil {
		t.Fatal(err)
	}
	rec := httptest.NewRecorder()
	s.renderJobDetail(rec, saved)
	out := rec.Body.String()

	// The conditional Score column (the job has a capability run).
	if !strings.Contains(out, ">Score</th>") {
		t.Errorf("the job matrix is missing the Score column\n%s", out)
	}
	// The KL cell renders the plan's exact form.
	if !strings.Contains(out, "KLD 0.012 ±0.001 · same top token 97.4% (vs UD-Q8_K_XL)") {
		t.Errorf("the KL score cell is not rendered in its plan form\n%s", out)
	}
	// The skipped reference cell shows its note, informationally
	// (muted, not the error color) and not in the error slot.
	if !strings.Contains(out, skipReason) {
		t.Errorf("the skip note is missing\n%s", out)
	}
	skipIdx := strings.Index(out, skipReason)
	if skipIdx == -1 {
		t.Fatalf("the skip note is missing\n%s", out)
	}
	tdStart := strings.LastIndex(out[:skipIdx], "<td>")
	if tdStart == -1 {
		t.Fatalf("skip note has no containing cell\n%s", out)
	}
	skipCell := out[tdStart : skipIdx+50]
	if strings.Contains(skipCell, "var(--pico-del-color)") {
		t.Errorf("the skip note is styled as an error:\n%s", skipCell)
	}
	// The detail-row colspan accounts for the extra column.
	if !strings.Contains(out, `colspan="12"`) {
		t.Errorf("the detail row colspan does not account for the Score column\n%s", out)
	}
}

// The compare view: comparing runs of different modes shows each row's
// own metric (the mode is part of the row identity — no cross-metric
// math), and a performance row shows an em-dash in the Score column.
func TestCompareRendersPerModeScores(t *testing.T) {
	base, err := template.New("").Funcs(testFuncMap).ParseFS(web.Templates,
		"templates/layout.html", "templates/partials/*.html")
	if err != nil {
		t.Fatalf("parse templates: %v", err)
	}
	hs := benchmark.BenchmarkRun{
		ID: "r-hs", JobID: "j", CreatedAt: time.Unix(0, 0).UTC(),
		Status: benchmark.StatusCompleted, ModelID: "m4", ModelName: "M-4B",
		Quant: "Q4_K_XL", Preset: "hellaswag-quick",
		Eval: &benchmark.EvalScores{
			Mode: "hellaswag", Dataset: "hellaswag", ContextSize: 512,
			Accuracy: 77.2, AccuracyCILow: 75.9, AccuracyCIHigh: 78.5, Tasks: 400,
		},
	}
	ppl := benchmark.BenchmarkRun{
		ID: "r-ppl", JobID: "j", CreatedAt: time.Unix(0, 0).UTC(),
		Status: benchmark.StatusCompleted, ModelID: "m4", ModelName: "M-4B",
		Quant: "Q4_K_XL", Preset: "perplexity-quick",
		Eval: &benchmark.EvalScores{
			Mode: "perplexity", Dataset: "wikitext-2", ContextSize: 512,
			Chunks: 100, Perplexity: 6.234, PerplexityErr: 0.04,
		},
	}
	perf := perfRun("r-perf")

	data := benchmark.BuildComparison([]benchmark.BenchmarkRun{hs, ppl, perf})
	if !data.HasEval {
		t.Fatal("BuildComparison did not flag the comparison as containing evals")
	}
	var buf bytes.Buffer
	if err := base.ExecuteTemplate(&buf, "benchmark_compare", data); err != nil {
		t.Fatalf("execute: %v", err)
	}
	out := buf.String()

	// Each row shows its own metric, in the plan's forms.
	if !strings.Contains(out, "HellaSwag 77.2% [75.9–78.5] (400)") {
		t.Errorf("the HellaSwag row is not rendered in its plan form\n%s", out)
	}
	if !strings.Contains(out, "PPL 6.234 ±0.04 (100 chunks)") {
		t.Errorf("the perplexity row is not rendered in its plan form\n%s", out)
	}
	// The performance table row shows an em-dash in the Score column
	// (the bar-chart divs carry the same data-run-id; anchor on the
	// table row itself).
	idx := strings.Index(out, `<tr data-run-id="r-perf"`)
	if idx == -1 {
		t.Fatal("the performance table row is missing")
	}
	rowEnd := strings.Index(out[idx:], "</tr>")
	row := out[idx : idx+rowEnd]
	// Cells after the "</td>" split: 0 model, 1 quant, 2 sweep, 3 test,
	// 4 PP, 5 TG, 6 TTFT, 7 score. The score cell is an em-dash.
	rcells := strings.Split(row, "</td>")
	if len(rcells) < 9 {
		t.Fatalf("compare row has %d cells, want 9+\n%s", len(rcells), row)
	}
	if !strings.Contains(rcells[7], "<td>—") {
		t.Errorf("compare row score cell = %q, want an em-dash", rcells[7])
	}
	// The score sort button only appears when there is a score to sort.
	if !strings.Contains(out, `data-sort-key="score"`) {
		t.Error("the Score sort button is missing")
	}
}

// A comparison of performance-only runs renders no Score column and no
// score sort button — unchanged from before capability presets.
func TestComparePerformanceOnlyNoScoreColumn(t *testing.T) {
	base, err := template.New("").Funcs(testFuncMap).ParseFS(web.Templates,
		"templates/layout.html", "templates/partials/*.html")
	if err != nil {
		t.Fatalf("parse templates: %v", err)
	}
	data := benchmark.BuildComparison([]benchmark.BenchmarkRun{perfRun("r1"), perfRun("r2")})
	if data.HasEval {
		t.Fatal("a performance-only comparison must not be flagged HasEval")
	}
	var buf bytes.Buffer
	if err := base.ExecuteTemplate(&buf, "benchmark_compare", data); err != nil {
		t.Fatalf("execute: %v", err)
	}
	out := buf.String()
	if strings.Contains(out, ">Score</th>") {
		t.Error("performance-only comparison must not show a Score column")
	}
	if strings.Contains(out, `data-sort-key="score"`) {
		t.Error("performance-only comparison must not offer score sorting")
	}
}

// The run detail shows the score block (same rendered form) plus the
// raw precision the tool reported; performance runs are unchanged
// (no block at all).
func TestBenchmarkDetailRendersEvalBlock(t *testing.T) {
	base, err := template.New("").Funcs(testFuncMap).ParseFS(web.Templates,
		"templates/layout.html", "templates/partials/*.html")
	if err != nil {
		t.Fatalf("parse templates: %v", err)
	}
	// Pointer receivers (EffectiveBuild) require a *BenchmarkRun, the
	// same shape the handler passes.
	var buf bytes.Buffer
	evRun := evalRun("r1")
	if err := base.ExecuteTemplate(&buf, "benchmark_detail", &evRun); err != nil {
		t.Fatalf("execute: %v", err)
	}
	out := buf.String()
	if !strings.Contains(out, "PPL 6.234 ±0.04 (100 chunks)") {
		t.Errorf("the detail score is not in the plan form\n%s", out)
	}
	if !strings.Contains(out, "6.2340 ± 0.04000") {
		t.Errorf("the raw perplexity precision is missing\n%s", out)
	}

	buf.Reset()
	perfRun2 := perfRun("r1")
	if err := base.ExecuteTemplate(&buf, "benchmark_detail", &perfRun2); err != nil {
		t.Fatalf("execute: %v", err)
	}
	if strings.Contains(buf.String(), "Capability evaluation") {
		t.Error("a performance run must not show the evaluation block")
	}
}

// The Evaluation Data card renders both sections: the pinned datasets
// with their on-disk state, and the KL logits list with per-row delete.
// The dummy dataset files do not match the pinned hashes, so their
// state is "present, hash mismatch" — the point of the test is that
// every row renders with license, size, and a state.
func TestEvalDataCardRendersBothSections(t *testing.T) {
	dir := t.TempDir()
	root := evaluate.EvalDataRoot(dir)

	if err := os.MkdirAll(evaluate.DatasetsDir(root), 0o755); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"wikitext-2", "hellaswag", "winogrande"} {
		if err := os.WriteFile(evaluate.DatasetPath(root, name), []byte("dummy"), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	// A KL logits entry with a known mtime.
	logitsDir := evaluate.LogitsDir(root)
	if err := os.MkdirAll(logitsDir, 0o755); err != nil {
		t.Fatal(err)
	}
	key := evaluate.KLBaseKey{ModelID: "u--M-GGUF", Quant: "Q8_0", Dataset: "wikitext-2", Chunks: 100, Ctx: 512}
	base := filepath.Join(logitsDir, key.Filename())
	if err := os.WriteFile(base, make([]byte, 5<<20), 0o644); err != nil {
		t.Fatal(err)
	}
	when := time.Now().Add(-3 * time.Hour)
	if err := os.Chtimes(base, when, when); err != nil {
		t.Fatal(err)
	}

	s := benchListServer(t)
	s.cfg = &config.Config{DataDir: dir}
	req := httptest.NewRequest("GET", "/api/benchmarks/eval-data", nil)
	req.Header.Set("HX-Request", "true")
	rec := httptest.NewRecorder()
	s.handleEvalData(rec, req)
	out := rec.Body.String()

	// Datasets section: all three pinned rows with license + state.
	if !strings.Contains(out, ">Evaluation Data<") {
		t.Errorf("the card title is missing\n%s", out)
	}
	for _, name := range []string{">wikitext-2<", ">hellaswag<", ">winogrande<"} {
		if !strings.Contains(out, name) {
			t.Errorf("dataset row %s is missing", name)
		}
	}
	for _, lic := range []string{"CC BY-SA 3.0", "MIT", "Apache-2.0"} {
		if !strings.Contains(out, lic) {
			t.Errorf("license %s is missing (must render verbatim from the pinned table)", lic)
		}
	}
	// Logits section: the entry with its size, chunks, and delete form.
	if !strings.Contains(out, "u--M-GGUF (Q8_0)") {
		t.Errorf("the logits row is missing\n%s", out)
	}
	if !strings.Contains(out, "100 chunks") {
		t.Error("the logits row's chunk count is missing")
	}
	if !strings.Contains(out, "5.0 MiB") {
		t.Errorf("the logits row's size is missing\n%s", out)
	}
	if !strings.Contains(out, "logits cache: 5.0 MiB") {
		t.Errorf("the total cache size is missing\n%s", out)
	}
	if !strings.Contains(out, `hx-post="/api/benchmarks/eval-data/delete-logits"`) {
		t.Error("the per-row delete form is missing")
	}
	for _, f := range []string{"name=\"model_id\"", "name=\"quant\"", "name=\"dataset\"", "name=\"chunks\"", "name=\"ctx\""} {
		if !strings.Contains(out, f) {
			t.Errorf("the delete form is missing the cache key field %s", f)
		}
	}
	// The regenerate-automatically note is present (deleting is safe).
	if !strings.Contains(out, "regenerate") {
		t.Error("the card does not say the data regenerates automatically")
	}
}

// Deleting a KL logits entry removes the file and the refreshed card
// no longer lists it.
func TestEvalDataDeleteLogitsRemovesEntry(t *testing.T) {
	dir := t.TempDir()
	root := evaluate.EvalDataRoot(dir)
	logitsDir := evaluate.LogitsDir(root)
	if err := os.MkdirAll(logitsDir, 0o755); err != nil {
		t.Fatal(err)
	}
	key := evaluate.KLBaseKey{ModelID: "u--M-GGUF", Quant: "Q8_0", Dataset: "wikitext-2", Chunks: 100, Ctx: 512}
	if err := os.WriteFile(filepath.Join(logitsDir, key.Filename()), make([]byte, 1024), 0o644); err != nil {
		t.Fatal(err)
	}

	s := benchListServer(t)
	s.cfg = &config.Config{DataDir: dir}
	form := "model_id=u--M-GGUF&quant=Q8_0&dataset=wikitext-2&chunks=100&ctx=512"
	req := httptest.NewRequest("POST", "/api/benchmarks/eval-data/delete-logits", strings.NewReader(form))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("HX-Request", "true")
	rec := httptest.NewRecorder()
	s.handleDeleteKLLogits(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("delete status = %d, want 200; body: %s", rec.Code, rec.Body.String())
	}
	if _, err := os.Stat(filepath.Join(logitsDir, key.Filename())); !os.IsNotExist(err) {
		t.Fatal("the logits file still exists after delete")
	}
	out := rec.Body.String()
	if !strings.Contains(out, "No cached reference logits yet") {
		t.Errorf("the refreshed card is missing the empty-state message\n%s", out)
	}
	if i := strings.Index(out, "u--M-GGUF (Q8_0)"); i != -1 {
		t.Errorf("the deleted entry is still in the refreshed card; context: %q", out[max(0, i-200):i+80])
	}
}

// Deleting while a job runs is refused with the same busy message the
// other job-conflicting actions use (409).
func TestEvalDataDeleteRefusedWhileJobBusy(t *testing.T) {
	dir := t.TempDir()
	s := &Server{cfg: &config.Config{DataDir: dir}}
	s.env = newJobEnv(s)
	s.env.mu.Lock()
	s.env.ownsRouter = true
	s.env.mu.Unlock()

	req := httptest.NewRequest("POST", "/api/benchmarks/eval-data/delete-logits",
		strings.NewReader("model_id=x&quant=Q&dataset=d&chunks=1&ctx=1"))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	rec := httptest.NewRecorder()
	s.handleDeleteKLLogits(rec, req)

	if rec.Code != http.StatusConflict {
		t.Fatalf("busy delete status = %d, want 409", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), "a benchmark job is currently running") {
		t.Errorf("the busy message is not the shared one: %q", rec.Body.String())
	}
}

// A capability run's eval fields round-trip the export: the CSV cells
// and summary scopes carry the seven eval columns, and the JSON
// envelope carries the run's EvalScores intact.
func TestExportRoundTripsCapabilityRun(t *testing.T) {
	run := evalRun("run-cap")
	run.Eval.Perplexity = 6.198
	run.Eval.PerplexityErr = 0.03
	run.Eval.Chunks = 0 // the full variant renders "full"
	perf := perfRun("run-perf")
	perf.JobID = "job-eval"
	runs := []benchmark.BenchmarkRun{run, perf}
	job := &benchmark.BenchmarkJob{ID: "job-eval", Name: "eval job"}
	jobs := jobLookup{"job-eval": job}

	// CSV cells scope: the capability row carries the eval fields; the
	// performance row's eval columns are empty (the per-mode-column
	// behavior — empty, not zero).
	rows := parseCSV(t, func(cw *csv.Writer) error {
		return writeCSVCells(cw, runs, jobs)
	})
	header := rows[0]
	for _, col := range []string{
		"eval_mode", "eval_dataset", "eval_score", "eval_error",
		"eval_tasks_chunks", "eval_kl_stats", "eval_reference",
	} {
		if _, err := columnIndexErr(header, col); err != nil {
			t.Fatalf("the cells export is missing the column %q: %v", col, err)
		}
	}
	idx := columnIndex(t, header, "eval_score")
	capRow := rows[1]
	// The raw value at the tool's precision (6.198 is exactly
	// representable; the fixed precision pads it).
	if got, want := capRow[idx], "6.19800"; got != want {
		t.Errorf("capability row eval_score = %q, want %q", got, want)
	}
	modeIdx := columnIndex(t, header, "eval_mode")
	if got, want := capRow[modeIdx], "perplexity"; got != want {
		t.Errorf("capability row eval_mode = %q, want %q", got, want)
	}
	chunksIdx := columnIndex(t, header, "eval_tasks_chunks")
	if got, want := capRow[chunksIdx], "full"; got != want {
		t.Errorf("full run eval_tasks_chunks = %q, want %q", got, want)
	}
	refIdx := columnIndex(t, header, "eval_reference")
	if got := capRow[refIdx]; got != "" {
		t.Errorf("non-KL row eval_reference = %q, want empty", got)
	}
	perfRow := rows[2]
	if perfRow[modeIdx] != "" || perfRow[idx] != "" || perfRow[chunksIdx] != "" {
		t.Errorf("performance row eval columns must be empty: mode=%q score=%q chunks=%q",
			perfRow[modeIdx], perfRow[idx], perfRow[chunksIdx])
	}

	// CSV summary scope: same columns, one row per run.
	sumRows := parseCSV(t, func(cw *csv.Writer) error {
		return writeCSVSummary(cw, runs, jobs)
	})
	sumHeader := sumRows[0]
	for _, col := range []string{
		"eval_mode", "eval_dataset", "eval_score", "eval_error",
		"eval_tasks_chunks", "eval_kl_stats", "eval_reference",
	} {
		if _, err := columnIndexErr(sumHeader, col); err != nil {
			t.Fatalf("the summary export is missing the column %q: %v", col, err)
		}
	}
	sIdx := columnIndex(t, sumHeader, "eval_score")
	if got, want := sumRows[1][sIdx], "6.19800"; got != want {
		t.Errorf("summary capability row eval_score = %q, want %q", got, want)
	}
	if got := sumRows[2][sIdx]; got != "" {
		t.Errorf("summary performance row eval_score = %q, want empty", got)
	}

	// JSON scope: the envelope carries the EvalScores field-for-field.
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	env := ExportEnvelope{Version: exportEnvelopeVersion, Jobs: []*benchmark.BenchmarkJob{job}, Runs: runs}
	if err := enc.Encode(env); err != nil {
		t.Fatal(err)
	}
	var back ExportEnvelope
	if err := json.Unmarshal(buf.Bytes(), &back); err != nil {
		t.Fatal(err)
	}
	var gotEval *benchmark.EvalScores
	for _, r := range back.Runs {
		if r.ID == "run-cap" {
			gotEval = r.Eval
		}
		if r.ID == "run-perf" && r.Eval != nil {
			t.Error("the performance run's Eval must round-trip as nil")
		}
	}
	if gotEval == nil {
		t.Fatal("the capability run's EvalScores are missing from the JSON export")
	}
	if gotEval.Mode != "perplexity" || gotEval.Dataset != "wikitext-2" ||
		gotEval.Perplexity != 6.198 || gotEval.PerplexityErr != 0.03 || gotEval.Chunks != 0 {
		t.Errorf("the EvalScores did not round-trip intact: %+v", gotEval)
	}
}

// columnIndexErr is the error-returning form of columnIndex (which
// t.Fatal's), for header scans in loops.
func columnIndexErr(header []string, name string) (int, error) {
	for i, h := range header {
		if h == name {
			return i, nil
		}
	}
	return -1, fmt.Errorf("column %q not in header", name)
}
