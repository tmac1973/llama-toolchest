package api

import (
	"bytes"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"html/template"
	"math"
	"net/http"
	"net/http/httptest"
	"net/url"
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
	// The label restores the "/" that SafeModelID turned into "--", so
	// the row reads as the repo path it is.
	if !strings.Contains(out, "u/M-GGUF (Q8_0)") {
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
	// The delete form names the file on disk; the handler re-validates
	// it rather than trusting it as a path.
	if !strings.Contains(out, `name="filename" value="`+key.Filename()+`"`) {
		t.Errorf("the delete form does not carry the row's filename\n%s", out)
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
	form := "filename=" + url.QueryEscape(key.Filename())
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
	if i := strings.Index(out, "u/M-GGUF (Q8_0)"); i != -1 {
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
		strings.NewReader("filename=x~Q~d~c1~ctx1~fnone.kld"))
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

// countCells returns the number of <th> in the table head and the
// number of <td> in the first data row of the rendered list, ignoring
// the full-width detail row that follows each one.
func countCells(t *testing.T, out string) (headers, cells int) {
	t.Helper()
	head := out[strings.Index(out, "<thead>"):strings.Index(out, "</thead>")]
	// "<th>" and "<th ..." only — a bare "<th" prefix count would also
	// match the enclosing "<thead>".
	headers = strings.Count(head, "<th>") + strings.Count(head, "<th ")

	i := strings.Index(out, `class="bench-row-group"`)
	if i < 0 {
		t.Fatalf("no data row in the rendered list\n%s", out)
	}
	row := out[i:]
	row = row[:strings.Index(row, "</tr>")]
	cells = strings.Count(row, "<td")
	return headers, cells
}

// A table whose rows carry more cells than its head carries headers
// silently shifts every column after the extra one. The Score cell is
// conditional on the same fact as the Score header, and this checks
// they agree — in BOTH directions, since the performance-only list is
// the common case and the one that regressed.
func TestBenchListColumnCountsMatch(t *testing.T) {
	s := benchListServer(t)

	rec := httptest.NewRecorder()
	s.renderBenchmarkList(rec, []benchmark.BenchmarkRun{perfRun("r1"), perfRun("r2")})
	headers, cells := countCells(t, rec.Body.String())
	if headers != cells {
		t.Errorf("performance-only list: %d headers but %d cells per row — every column after the mismatch is shifted", headers, cells)
	}

	rec = httptest.NewRecorder()
	s.renderBenchmarkList(rec, []benchmark.BenchmarkRun{perfRun("r1"), evalRun("r2")})
	headersEval, cellsEval := countCells(t, rec.Body.String())
	if headersEval != cellsEval {
		t.Errorf("mixed list: %d headers but %d cells per row", headersEval, cellsEval)
	}
	if headersEval != headers+1 {
		t.Errorf("the Score column should add exactly one column: %d vs %d", headersEval, headers)
	}
}

// The KL score names the reference the way a person reads it, not by
// registry ID — "(vs unsloth--Qwen3.5-9B-MTP-GGUF--Qwen3.5-9B-IQ4_NL)"
// is not a thing to put in a table cell. Runs stored before the label
// existed fall back to the ID rather than losing the reference.
func TestEvalScoreTextUsesReferenceLabel(t *testing.T) {
	withLabel := &benchmark.EvalScores{
		Mode: "kl-divergence", KLMean: 0.0123, KLMeanErr: 0.0004, SameTopPct: 97.4,
		Reference:      "unsloth--Qwen3.5-9B-MTP-GGUF--Qwen3.5-9B-IQ4_NL",
		ReferenceLabel: "Qwen3.5-9B (IQ4_NL)",
	}
	got := evalScoreText(withLabel)
	if !strings.Contains(got, "(vs Qwen3.5-9B (IQ4_NL))") {
		t.Errorf("score text does not use the label: %q", got)
	}
	if strings.Contains(got, "unsloth--") {
		t.Errorf("score text leaked the registry ID: %q", got)
	}

	legacy := &benchmark.EvalScores{Mode: "kl-divergence", KLMean: 0.0123, Reference: "old-id"}
	if got := evalScoreText(legacy); !strings.Contains(got, "(vs old-id)") {
		t.Errorf("a run without a label lost its reference: %q", got)
	}
}

// The score sort is a rank, not a raw value: one ascending sort has to
// put the best result first for every mode at once, and accuracy runs
// the other way from perplexity and KLD.
func TestEvalScoreValueRanksBestFirst(t *testing.T) {
	betterPPL := &benchmark.EvalScores{Mode: "perplexity", Perplexity: 6.1}
	worsePPL := &benchmark.EvalScores{Mode: "perplexity", Perplexity: 9.4}
	if evalScoreValue(betterPPL) >= evalScoreValue(worsePPL) {
		t.Error("lower perplexity must rank first")
	}

	betterAcc := &benchmark.EvalScores{Mode: "hellaswag", Accuracy: 79.0}
	worseAcc := &benchmark.EvalScores{Mode: "hellaswag", Accuracy: 71.5}
	if evalScoreValue(betterAcc) >= evalScoreValue(worseAcc) {
		t.Error("higher accuracy must rank first")
	}

	// A performance run has no score and belongs at the end, not at the
	// top on a zero.
	if evalScoreValue(nil) <= evalScoreValue(worsePPL) {
		t.Error("a run with no score must sort after every scored run")
	}
	// The rank is rendered into an attribute and read back with
	// parseFloat, which cannot handle "+Inf".
	if math.IsInf(evalScoreValue(nil), 0) || math.IsNaN(evalScoreValue(nil)) {
		t.Error("the no-score rank must be finite")
	}
}

// Every capability preset's mode must carry interpretation guidance. A
// score with no explanation is the thing this table exists to prevent,
// so a new mode arriving without an entry has to fail here rather than
// ship as a bare number.
func TestEveryCapabilityModeHasGuidance(t *testing.T) {
	for _, p := range benchmark.Presets() {
		if p.EffectiveSource() != benchmark.PresetSourceCapability {
			continue
		}
		d, ok := evalDocForPreset(p)
		if !ok {
			t.Errorf("preset %s (mode %s) has no interpretation guidance", p.Name, p.EvalMode)
			continue
		}
		if d.Headline == "" {
			t.Errorf("%s: no headline", p.EvalMode)
		}
		if len(d.Reading) == 0 {
			t.Errorf("%s: no reading rules", p.EvalMode)
		}
		// "Is 8.043 good?" is the question a reader actually arrives
		// with, and only a scale of worked examples answers it.
		if len(d.Examples) < 3 {
			t.Errorf("%s: %d worked examples — too few to show a reader where their score sits",
				p.EvalMode, len(d.Examples))
		}
		for _, ex := range d.Examples {
			if ex.Value == "" || ex.Verdict == "" || ex.Meaning == "" {
				t.Errorf("%s: incomplete example %+v", p.EvalMode, ex)
			}
		}
		if len(d.Links) == 0 {
			t.Errorf("%s: no sources to read further", p.EvalMode)
		}
		for _, l := range d.Links {
			if !strings.HasPrefix(l.URL, "https://") {
				t.Errorf("%s: link %q is not an https URL", p.EvalMode, l.URL)
			}
			if l.Label == "" || l.Note == "" {
				// The note is what stops the shared llama.cpp page from
				// looking like the wrong link under a KL score.
				t.Errorf("%s: link %s has no label or no note saying why to open it", p.EvalMode, l.URL)
			}
		}
	}
}

// The run detail view is where someone goes to find out what their
// score means, so the guidance and its authoritative link render there
// with the number — not only in the help page.
func TestBenchmarkDetailRendersScoreGuidance(t *testing.T) {
	base, err := template.New("").Funcs(testFuncMap).ParseFS(web.Templates,
		"templates/layout.html", "templates/partials/*.html")
	if err != nil {
		t.Fatalf("parse templates: %v", err)
	}

	var buf bytes.Buffer
	evRun := evalRun("r1")
	if err := base.ExecuteTemplate(&buf, "benchmark_detail", &evRun); err != nil {
		t.Fatalf("execute: %v", err)
	}
	out := buf.String()

	doc, _ := evalDocFor("perplexity")
	// The one-line "what this measures" is a tooltip, per the project's
	// convention that explanatory text does not sit as static prose.
	if !strings.Contains(out, `title="`+doc.Headline+`"`) {
		t.Errorf("the headline explaining the number is missing\n%s", out)
	}
	for _, l := range doc.Links {
		if !strings.Contains(out, l.URL) {
			t.Errorf("the %s link is missing\n%s", l.Label, out)
		}
	}
	if !strings.Contains(out, "How to read this score") {
		t.Error("the reading rules are missing")
	}
	// The reader has to know what a good value looks like, not just
	// which direction is better.
	if !strings.Contains(out, "Is my score good?") {
		t.Error("the good-versus-bad table is missing")
	}
	if !strings.Contains(out, "Broken") {
		t.Error("the examples table did not render its rows")
	}

	// A performance run has no score to explain.
	buf.Reset()
	pr := perfRun("r2")
	if err := base.ExecuteTemplate(&buf, "benchmark_detail", &pr); err != nil {
		t.Fatalf("execute: %v", err)
	}
	if strings.Contains(buf.String(), "How to read this score") {
		t.Error("a performance run must not show score guidance")
	}
}

// A score measured through a compressed memory cache reads exactly like
// a comparable one, so the caveat has to travel with the number rather
// than living only in the run detail.
func TestScoreCellCarriesComparabilityWarning(t *testing.T) {
	s := benchListServer(t)

	// f16: no marker.
	plain := evalRun("r1")
	rec := httptest.NewRecorder()
	s.renderBenchmarkList(rec, []benchmark.BenchmarkRun{plain})
	if strings.Contains(rec.Body.String(), "&#9888;") {
		t.Error("an f16 run should carry no warning marker")
	}

	// Compressed KV cache: marker with the reason in its tooltip.
	compressed := evalRun("r2")
	compressed.Config.KVCacheQuant = "q8_0"
	rec = httptest.NewRecorder()
	s.renderBenchmarkList(rec, []benchmark.BenchmarkRun{compressed})
	out := rec.Body.String()
	if !strings.Contains(out, "&#9888;") {
		t.Errorf("no warning marker beside a non-comparable score\n%s", out)
	}
	if !strings.Contains(out, "kv_cache_quant = q8_0") {
		t.Errorf("the marker does not say what setting caused it\n%s", out)
	}
	// The columns must still line up — the marker lives inside the
	// existing Score cell, it does not add one.
	headers, cells := countCells(t, out)
	if headers != cells {
		t.Errorf("%d headers but %d cells per row", headers, cells)
	}
}

// The same-top-token scale has to cover every value a run can produce.
// A real result of 91.8% fell in a gap between the "above 99%" and
// "below 90%" rows, leaving the reader with nothing to compare against.
func TestSameTopTokenScaleHasNoGaps(t *testing.T) {
	d, ok := evalDocFor("kl-divergence")
	if !ok {
		t.Fatal("no KL guidance")
	}
	var bands []string
	for _, ex := range d.Examples {
		if strings.HasPrefix(ex.Value, "Same top token") {
			bands = append(bands, ex.Value)
		}
	}
	if len(bands) < 4 {
		t.Errorf("only %d same-top-token bands (%v) — a value between the extremes has nothing to compare against", len(bands), bands)
	}
	// The real 91.8% result must fall inside a named band.
	if !strings.Contains(strings.Join(bands, " "), "90% to 95%") {
		t.Errorf("no band covers the 90-95%% range: %v", bands)
	}
}

// The compare view's details table renders ONE ROW PER PROMPT SIZE, so
// a run measured at three sizes contributes three rows sharing a
// data-run-id. The sort code assumed one row per run, which is what
// made sorting appear not to work; this checks the markup the fixed
// code depends on is actually there.
func TestCompareTableRowsShareRunID(t *testing.T) {
	base, err := template.New("").Funcs(testFuncMap).ParseFS(web.Templates,
		"templates/layout.html", "templates/partials/*.html")
	if err != nil {
		t.Fatalf("parse templates: %v", err)
	}

	multi := perfRun("multi")
	multi.Summary.PerSize = []benchmark.SizeSummary{
		{PromptTokens: 128, TGMean: 40, Count: 3},
		{PromptTokens: 512, TGMean: 38, Count: 3},
		{PromptTokens: 2048, TGMean: 30, Count: 3},
	}
	single := perfRun("single")
	single.Summary.PerSize = []benchmark.SizeSummary{{PromptTokens: 512, TGMean: 44, Count: 3}}

	var buf bytes.Buffer
	if err := base.ExecuteTemplate(&buf, "benchmark_compare",
		benchmark.BuildComparison([]benchmark.BenchmarkRun{multi, single})); err != nil {
		t.Fatalf("execute: %v", err)
	}
	out := buf.String()
	if n := strings.Count(out, `data-run-id="multi"`); n < 4 {
		t.Errorf(`data-run-id="multi" appears %d times, want 4 (one bar row per chart plus three size rows)`, n)
	}
	// The sort has to be able to read a value for every run from the
	// markup, or a run has no position to sort into.
	if !strings.Contains(out, `data-sort-gen=`) {
		t.Error("no sort values in the rendered comparison")
	}
}

// Bar labels must differ from one another, and the tooltip must carry
// the settings the visible label leaves out.
func TestCompareBarsAreDistinguishable(t *testing.T) {
	base, err := template.New("").Funcs(testFuncMap).ParseFS(web.Templates,
		"templates/layout.html", "templates/partials/*.html")
	if err != nil {
		t.Fatalf("parse templates: %v", err)
	}

	mk := func(id string, gen float64, ub string) benchmark.BenchmarkRun {
		r := perfRun(id)
		r.Summary.AvgGenTokPerSec = gen
		r.SweepValues = map[string]string{"ubatch_size": ub}
		r.Summary.PerSize = []benchmark.SizeSummary{{PromptTokens: 512, TGMean: gen, Count: 1}}
		r.Config = benchmark.ConfigSnapshot{GPULayers: 999, ContextSize: 8192, Threads: 8}
		return r
	}
	runs := []benchmark.BenchmarkRun{mk("a", 85.6, "64"), mk("b", 100.5, "256")}

	var buf bytes.Buffer
	if err := base.ExecuteTemplate(&buf, "benchmark_compare", benchmark.BuildComparison(runs)); err != nil {
		t.Fatalf("execute: %v", err)
	}
	out := buf.String()

	for _, want := range []string{"micro-batch 64", "micro-batch 256"} {
		if !strings.Contains(out, want) {
			t.Errorf("bar labels do not name the swept value %q\n", want)
		}
	}
	// The winner is marked rather than left to be judged by bar length.
	if !strings.Contains(out, ">best<") {
		t.Error("the fastest run is not marked")
	}
	// And the others say how far behind they are.
	if !strings.Contains(out, "&minus;") {
		t.Error("no shortfall shown against the leader")
	}
	// The tooltip carries the full configuration.
	if !strings.Contains(out, "threads:") {
		t.Errorf("the label tooltip does not carry the configuration\n%s", out[:400])
	}
}
