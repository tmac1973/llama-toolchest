package api

import (
	"encoding/json"
	"net/http/httptest"
	"regexp"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/benchmark"
	"github.com/tmac1973/llama-toolchest/internal/config"
	"github.com/tmac1973/llama-toolchest/internal/models"
)

// vizServer builds a server whose store holds the given runs.
func vizServer(t *testing.T, runs ...benchmark.BenchmarkRun) *Server {
	t.Helper()
	dir := t.TempDir()
	s := &Server{
		bench:    benchmark.NewStore(dir, nil),
		registry: models.NewRegistry(t.TempDir(), t.TempDir()),
		cfg:      &config.Config{DataDir: dir},
	}
	s.pages = s.parseTemplates()
	for _, r := range runs {
		s.bench.Save(r)
	}
	return s
}

func vizSweepRun(id string, gen float64, ub string) benchmark.BenchmarkRun {
	r := perfRun(id)
	r.Summary.AvgGenTokPerSec = gen
	r.SweepValues = map[string]string{"ubatch_size": ub}
	r.Config = benchmark.ConfigSnapshot{GPULayers: 999, ContextSize: 8192, Threads: 8}
	return r
}

// dataPayload pulls the embedded JSON back out of the rendered page.
func dataPayload(t *testing.T, body string) string {
	t.Helper()
	m := regexp.MustCompile(`(?s)<script id="viz-data" type="application/json">(.*?)</script>`).FindStringSubmatch(body)
	if m == nil {
		t.Fatalf("no data script tag in the page:\n%s", body)
	}
	return m[1]
}

// The regression that cost the most to find: html/template escapes a
// plain string placed in a script element into a QUOTED JAVASCRIPT
// STRING LITERAL. JSON.parse then hands the page a string rather than
// an object, every field read off it is undefined, and the page renders
// perfectly with no chart and no error anywhere.
func TestVisualizePayloadParsesAsAnObject(t *testing.T) {
	s := vizServer(t, vizSweepRun("a", 85.6, "64"), vizSweepRun("b", 100.5, "256"))
	rec := httptest.NewRecorder()
	s.handleVisualizePage(rec, httptest.NewRequest("GET", "/benchmarks/visualize?ids=a,b", nil))

	raw := strings.TrimSpace(dataPayload(t, rec.Body.String()))
	if strings.HasPrefix(raw, `"`) {
		t.Fatalf("the payload was escaped into a string literal, so JSON.parse yields a string:\n%s", raw[:120])
	}
	var v benchmark.VizData
	if err := json.Unmarshal([]byte(raw), &v); err != nil {
		t.Fatalf("the embedded payload is not valid JSON: %v", err)
	}
	if len(v.Points) != 2 {
		t.Errorf("points = %d, want 2", len(v.Points))
	}
	if len(v.Dimensions) == 0 {
		t.Error("no dimensions offered as axes")
	}
}

// Nothing in a run's text may close the script element. encoding/json
// escapes the characters that could, and this checks it rather than
// trusting it.
func TestVisualizePayloadCannotBreakOutOfTheScriptTag(t *testing.T) {
	a := vizSweepRun("a", 85.6, "64")
	a.ModelName = `</script><script>alert(1)</script>`
	s := vizServer(t, a, vizSweepRun("b", 100.5, "256"))

	rec := httptest.NewRecorder()
	s.handleVisualizePage(rec, httptest.NewRequest("GET", "/benchmarks/visualize?ids=a,b", nil))
	raw := dataPayload(t, rec.Body.String())
	if strings.Contains(raw, "</script>") {
		t.Errorf("a model name closed the script element:\n%s", raw)
	}
	// The text survives as inert data, with every angle bracket escaped
	// to its \u form — that is what makes it inert.
	if strings.Contains(raw, "<") || strings.Contains(raw, ">") {
		t.Errorf("a raw angle bracket reached the payload:\n%s", raw[:200])
	}
	if !strings.Contains(raw, `\u003c`) {
		t.Error("the model name was dropped rather than escaped")
	}
}

// A selection that cannot be plotted gets an explanation, not an empty
// chart area.
func TestVisualizeExplainsWhenThereIsNothingToPlot(t *testing.T) {
	cap1 := benchmark.BenchmarkRun{ID: "c1", ModelName: "M", Preset: "hellaswag-quick",
		Eval: &benchmark.EvalScores{Mode: "hellaswag", Accuracy: 78}}
	cap2 := benchmark.BenchmarkRun{ID: "c2", ModelName: "M", Preset: "winogrande-quick",
		Eval: &benchmark.EvalScores{Mode: "winogrande", Accuracy: 70}}

	for _, tc := range []struct{ name, ids, want string }{
		{"nothing selected", "", "Select some benchmark runs"},
		{"only one run", "c1", "at least two runs"},
		{"no timings at all", "c1,c2", "recorded speeds"},
	} {
		s := vizServer(t, cap1, cap2)
		rec := httptest.NewRecorder()
		s.handleVisualizePage(rec, httptest.NewRequest("GET", "/benchmarks/visualize?ids="+tc.ids, nil))
		body := rec.Body.String()
		if !strings.Contains(body, tc.want) {
			t.Errorf("%s: page does not explain itself (looking for %q)", tc.name, tc.want)
		}
		if strings.Contains(body, `id="viz-plot"`) {
			t.Errorf("%s: an empty chart area was rendered anyway", tc.name)
		}
	}
}

// The page says how many runs it left out rather than quietly plotting
// fewer points than were selected.
func TestVisualizeReportsSkippedRuns(t *testing.T) {
	failed := benchmark.BenchmarkRun{ID: "f", ModelName: "M", Preset: "perplexity-quick",
		Status: benchmark.StatusFailed}
	s := vizServer(t, vizSweepRun("a", 85.6, "64"), vizSweepRun("b", 100.5, "256"), failed)

	rec := httptest.NewRecorder()
	s.handleVisualizePage(rec, httptest.NewRequest("GET", "/benchmarks/visualize?ids=a,b,f", nil))
	body := rec.Body.String()
	if !strings.Contains(body, "2 runs") || !strings.Contains(body, "1 left out") {
		t.Errorf("the page does not report what it plotted and what it skipped:\n%s",
			regexp.MustCompile(`(?s)<header.*?</header>`).FindString(body))
	}
}

// The JSON endpoint returns the same payload for anyone plotting it
// somewhere else.
func TestVisualizeDataEndpoint(t *testing.T) {
	s := vizServer(t, vizSweepRun("a", 85.6, "64"), vizSweepRun("b", 100.5, "256"))
	rec := httptest.NewRecorder()
	s.handleVisualizeData(rec, httptest.NewRequest("GET", "/api/benchmarks/visualize?ids=a,b", nil))
	if rec.Code != 200 {
		t.Fatalf("status = %d", rec.Code)
	}
	var v benchmark.VizData
	if err := json.Unmarshal(rec.Body.Bytes(), &v); err != nil {
		t.Fatalf("bad JSON: %v", err)
	}
	if len(v.Points) != 2 || len(v.Metrics) == 0 {
		t.Errorf("payload = %d points, %d metrics", len(v.Points), len(v.Metrics))
	}
}
