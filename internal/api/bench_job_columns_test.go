package api

import (
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/benchmark"
)

// A job's results table shares its width between eleven columns, and how
// the browser shares it follows the narrowest each column can be. The
// sweep column's labels may break anywhere, so without a floor its
// narrowest is one character and it hands its width to any other column
// that asks — which is how a sweep of four settings came to be rendered
// three characters wide, one letter per line, beside a status column
// holding a skip reason. The floor, the cap over the status column and
// the table's tighter cell padding all hang off these three class names.
func TestJobDetailTableCarriesItsWidthClasses(t *testing.T) {
	s := benchListServer(t)
	job := &benchmark.BenchmarkJob{
		ID: "job-1",
		Cells: []benchmark.JobCell{{
			ModelID:    "m4",
			BuildID:    "b1",
			Preset:     "internal-standard",
			Status:     benchmark.CellStatusSkipped,
			SkipReason: "an earlier attempt at these settings could not allocate the KV cache",
			SweepValues: map[string]string{
				"batch_size":      "2048",
				"flash_attention": "true",
			},
		}},
	}
	rec := httptest.NewRecorder()
	s.renderJobDetail(rec, job)
	out := rec.Body.String()

	if !strings.Contains(out, `class="job-cells has-sweep"`) {
		t.Errorf("the results table is missing its job-cells or has-sweep class\n%s", out)
	}
	if !strings.Contains(out, `<th class="sweep-cell">Sweep</th>`) ||
		!strings.Contains(out, `<td class="sweep-cell">`) {
		t.Errorf("the sweep column is missing its class on the header or the cells\n%s", out)
	}
	if !strings.Contains(out, `<th class="status-cell">Status</th>`) ||
		!strings.Contains(out, `<td class="status-cell">`) {
		t.Errorf("the status column is missing its class on the header or the cells\n%s", out)
	}
	// One label per setting, so the column wraps between settings rather
	// than inside one. See benchmark.SweepChips.
	if !strings.Contains(out, "<kbd>batch_size=2048</kbd>") || !strings.Contains(out, "<kbd>flash_attention=true</kbd>") {
		t.Errorf("the sweep point is not rendered as one label per setting\n%s", out)
	}
}

// A job that swept nothing has an em-dash in every sweep cell, and the
// column's minimum width would be spent on that. The class that carries
// the minimum is only on a table that has a sweep point to show.
func TestJobDetailWithoutASweepDoesNotReserveTheColumn(t *testing.T) {
	s := benchListServer(t)
	job := &benchmark.BenchmarkJob{
		ID:    "job-2",
		Cells: []benchmark.JobCell{{ModelID: "m4", BuildID: "b1", Preset: "internal-standard", Status: benchmark.CellStatusCompleted}},
	}
	rec := httptest.NewRecorder()
	s.renderJobDetail(rec, job)
	if out := rec.Body.String(); strings.Contains(out, "has-sweep") {
		t.Errorf("a job with no sweep reserves the sweep column's width\n%s", out)
	}
}
