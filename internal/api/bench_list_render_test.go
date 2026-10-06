package api

import (
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/benchmark"
	"github.com/tmac1973/llama-toolchest/internal/evaluate"
)

// A KL score names its reference model, a label the runner recorded.
// The list used to write the score text into the page unescaped; it is
// escaped once now, like every other value in the row.
func TestBenchListEscapesScoreText(t *testing.T) {
	s := benchListServer(t)
	run := perfRun("r1")
	run.Eval = &benchmark.EvalScores{Mode: string(evaluate.ModeKLDiv), KLMean: 0.1, ReferenceLabel: `<b>ref</b> & "x"`}
	rec := httptest.NewRecorder()
	s.renderBenchmarkList(rec, []benchmark.BenchmarkRun{run})
	out := rec.Body.String()
	if strings.Contains(out, "<b>ref</b>") {
		t.Errorf("score text written as markup:\n%s", out)
	}
	if !strings.Contains(out, `(vs &lt;b&gt;ref&lt;/b&gt; &amp; &#34;x&#34;)</td>`) {
		t.Errorf("score text not escaped once:\n%s", out)
	}
}
