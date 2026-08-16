package benchmark

import (
	"sort"
	"strconv"
)

// Data for the visualization page.
//
// The compare view answers "which of these won" at a glance. This
// answers the questions that need a shape rather than a number: where
// the ridge is in a two-parameter sweep, whether a setting has a knee,
// whether two parameters interact or act independently. Those need axes
// the reader chooses, so this hands the page every dimension the runs
// vary on and lets it decide what goes where.
//
// It deliberately reuses the compare view's dimension machinery
// (comparisonDimensions, BuildRunLabels). One definition of "what makes
// these runs different" serves both, so an axis in a chart is the same
// thing as a term in a bar label, and a parameter that appears in one
// cannot go missing from the other.

// VizDimension is one axis the page can plot against: a name, the
// values present across the compared runs, and whether those values are
// numbers.
//
// Numeric matters for more than sorting. A batch size of 1024 sits
// eight times further from 128 than 512 does, and a scatter plot that
// spaces them evenly hides that; a model name has no such spacing and
// must be treated as a category.
type VizDimension struct {
	Name    string   `json:"name"`
	Values  []string `json:"values"`
	Numeric bool     `json:"numeric"`
}

// VizMetric is one measurement the page can plot.
type VizMetric struct {
	Key   string `json:"key"`
	Label string `json:"label"`
	Unit  string `json:"unit"`
	// HigherIsBetter tells the page which end of the scale is good, so
	// it can colour a heatmap in the right direction rather than making
	// the reader work out whether dark means fast or slow.
	HigherIsBetter bool `json:"higher_is_better"`
}

// VizPoint is one run: its position on every dimension, and its value
// for every metric.
type VizPoint struct {
	RunID   string             `json:"run_id"`
	Label   string             `json:"label"`
	Detail  string             `json:"detail"`
	Dims    map[string]string  `json:"dims"`
	Metrics map[string]float64 `json:"metrics"`
}

// VizData is the whole payload the page renders from.
type VizData struct {
	Dimensions []VizDimension `json:"dimensions"`
	Metrics    []VizMetric    `json:"metrics"`
	Points     []VizPoint     `json:"points"`
	// Skipped counts runs that carried no timings — capability runs and
	// failures. They have no place on a throughput chart, and saying how
	// many were left out is better than quietly plotting fewer points
	// than the user selected.
	Skipped int `json:"skipped"`
}

// vizMetrics is the fixed list of measurements a performance run
// carries. Capability scores are deliberately absent: a perplexity and
// a HellaSwag score share no scale, so putting them on one axis would
// invite a comparison that means nothing.
var vizMetrics = []VizMetric{
	{Key: "gen", Label: "Generation speed", Unit: "tok/s", HigherIsBetter: true},
	{Key: "prompt", Label: "Prompt speed", Unit: "tok/s", HigherIsBetter: true},
	{Key: "ttft", Label: "Time to first token", Unit: "ms", HigherIsBetter: false},
}

// BuildVisualization turns a set of runs into the visualization
// payload: the dimensions they vary on, the metrics available, and one
// point per run that produced timings.
func BuildVisualization(runs []BenchmarkRun) VizData {
	labels := BuildRunLabels(runs)
	dims := comparisonDimensions(runs)

	out := VizData{Metrics: vizMetrics}

	// Only runs with timings can be plotted. Collect them first so the
	// dimensions describe what is actually on the chart rather than
	// offering an axis whose values all belong to points that were
	// dropped.
	plotted := make([]BenchmarkRun, 0, len(runs))
	for _, r := range runs {
		if r.Summary == nil {
			out.Skipped++
			continue
		}
		plotted = append(plotted, r)
	}

	for _, d := range dims {
		seen := map[string]bool{}
		var values []string
		for _, r := range plotted {
			v := d.Value(r)
			if v == "" || seen[v] {
				continue
			}
			seen[v] = true
			values = append(values, v)
		}
		// A dimension every run agrees on is not an axis: plotting
		// against it draws every point in one line.
		if len(values) < 2 {
			continue
		}
		numeric := allNumeric(values)
		sortDimensionValues(values, numeric)
		out.Dimensions = append(out.Dimensions, VizDimension{
			Name: d.Name, Values: values, Numeric: numeric,
		})
	}

	for _, r := range plotted {
		p := VizPoint{
			RunID:  r.ID,
			Label:  labels[r.ID].Short,
			Detail: labels[r.ID].Full,
			Dims:   make(map[string]string, len(dims)),
			Metrics: map[string]float64{
				"gen":    r.Summary.AvgGenTokPerSec,
				"prompt": r.Summary.AvgPromptTokPerSec,
				"ttft":   r.Summary.AvgTTFTMs,
			},
		}
		for _, d := range dims {
			if v := d.Value(r); v != "" {
				p.Dims[d.Name] = v
			}
		}
		out.Points = append(out.Points, p)
	}
	return out
}

// allNumeric reports whether every value parses as a number.
func allNumeric(values []string) bool {
	for _, v := range values {
		if _, err := strconv.ParseFloat(v, 64); err != nil {
			return false
		}
	}
	return len(values) > 0
}

// sortDimensionValues orders an axis: by value when the values are
// numbers, alphabetically otherwise. Sorting "1024, 128, 2048, 512" as
// text puts the axis in an order that makes a curve look like noise.
func sortDimensionValues(values []string, numeric bool) {
	if numeric {
		sort.Slice(values, func(i, j int) bool {
			a, _ := strconv.ParseFloat(values[i], 64)
			b, _ := strconv.ParseFloat(values[j], 64)
			return a < b
		})
		return
	}
	sort.Strings(values)
}
