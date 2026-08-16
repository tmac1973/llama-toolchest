package api

import (
	"encoding/json"
	"html/template"
	"net/http"
	"strings"

	"github.com/tmac1973/llama-toolchest/internal/benchmark"
)

// The visualization page.
//
// "Compare Selected" is the quick read: which of these won. This is the
// slower one — where the ridge is in a two-parameter sweep, whether a
// setting has a knee, whether two parameters interact. It opens in its
// own tab so its chart library, which is larger than everything else
// the application serves put together, is never loaded by the pages
// people use all day.
//
// The data is embedded in the page rather than fetched: the payload is
// a few kilobytes for any realistic selection, and embedding means the
// tab survives being reloaded, bookmarked, or kept open while the
// selection behind it changes.

// visualizeView is the page template's data.
type visualizeView struct {
	pageData
	// DataJSON is the VizData payload, ready to drop into a script tag.
	//
	// template.JS, not string: html/template escapes a plain string
	// placed in a script element into a quoted JavaScript string
	// literal, so JSON.parse would hand the page a string rather than
	// an object and every field read off it would be undefined. The
	// page rendered and the chart silently never appeared.
	//
	// Marking it safe is safe here: encoding/json escapes "<", ">" and
	// "&" to their \u equivalents, so the payload cannot close the
	// script element or open a tag, whatever a model name contains.
	DataJSON template.JS
	// RunCount and Skipped describe the selection in the page heading.
	RunCount int
	Skipped  int
	// Error is set when the selection cannot be visualized at all.
	Error string
}

// handleVisualizePage renders the visualization page for a selection of
// runs, identified the same way the compare and export endpoints
// identify them.
//
// GET /benchmarks/visualize?ids=a,b,c
func (s *Server) handleVisualizePage(w http.ResponseWriter, r *http.Request) {
	view := visualizeView{pageData: pageData{Title: "Visualize benchmarks", Nav: "benchmarks"}}

	runs := s.runsByIDs(r.URL.Query().Get("ids"))
	switch {
	case len(runs) == 0:
		view.Error = "No runs to visualize. Select some benchmark runs and choose Visualize."
	case len(runs) < 2:
		view.Error = "Visualizing needs at least two runs — a chart of one point has nothing to show. Select more and try again."
	}

	if view.Error == "" {
		data := benchmark.BuildVisualization(runs)
		view.RunCount = len(data.Points)
		view.Skipped = data.Skipped
		if len(data.Points) < 2 {
			view.Error = "None of the selected runs recorded speeds, so there is nothing to plot. Capability evaluations and failed runs have no throughput figures."
			view.RunCount, view.Skipped = 0, 0
		} else {
			encoded, err := json.Marshal(data)
			if err != nil {
				view.Error = "Could not prepare the data for plotting: " + err.Error()
			} else {
				view.DataJSON = template.JS(encoded)
			}
		}
	}

	s.render(w, "visualize.html", view)
}

// handleVisualizeData returns the same payload as JSON, for anyone
// wanting to plot it somewhere else.
//
// GET /api/benchmarks/visualize?ids=a,b,c
func (s *Server) handleVisualizeData(w http.ResponseWriter, r *http.Request) {
	runs := s.runsByIDs(r.URL.Query().Get("ids"))
	if len(runs) == 0 {
		http.Error(w, "ids parameter required, naming runs that exist", http.StatusBadRequest)
		return
	}
	respondJSON(w, benchmark.BuildVisualization(runs))
}

// runsByIDs resolves a comma-separated id list, skipping ids that no
// longer exist. Shared by the page and the JSON endpoint so the two
// cannot disagree about what a selection means.
func (s *Server) runsByIDs(idsParam string) []benchmark.BenchmarkRun {
	if idsParam == "" {
		return nil
	}
	var runs []benchmark.BenchmarkRun
	for _, id := range strings.Split(idsParam, ",") {
		id = strings.TrimSpace(id)
		if id == "" {
			continue
		}
		if run, err := s.bench.Get(id); err == nil {
			runs = append(runs, *run)
		}
	}
	return runs
}
