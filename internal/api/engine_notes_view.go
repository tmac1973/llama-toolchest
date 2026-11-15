package api

import (
	"fmt"
	"hash/fnv"
	"net/http"
	"strings"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/enginenotes"
	"github.com/tmac1973/llama-toolchest/internal/models"
)

// timeView is a time as the page shows it: UTC text that a script on the
// page rewrites to the browser's local time.
type timeView struct {
	ISO   string
	Label string
}

func newTimeView(t, now time.Time) timeView {
	t, now = t.UTC(), now.UTC()
	label := t.Format("15:04 UTC")
	if t.Format("2006-01-02") != now.Format("2006-01-02") {
		label = t.Format("2006-01-02 15:04 UTC")
	}
	return timeView{ISO: t.Format(time.RFC3339), Label: label}
}

// notesStripView is the one line above the Server log.
type notesStripView struct {
	// Warn is set when the last 24 hours hold a crash or a failed load
	// that a job was not there to find (enginenotes.Expected).
	Warn bool
	Text string
	// LastAt is when the latest failure happened, shown after Text.
	LastAt *timeView
	// Sig changes when the set of failures in the strip does, so the page
	// can draw attention to a new one and not to an unchanged poll.
	Sig string
}

// count is "1 crash" or "3 crashes".
func count(n int, one, many string) string {
	return fmt.Sprintf("%d %s", n, plural(n, one, many))
}

// newNotesStripView sums up the last 24 hours. A failure a job was there
// to find — a setting that did not fit or that llama.cpp refused while
// Autotune or a benchmark tried it — is counted but does not warn: an
// Autotune run can leave dozens, and none of them is a problem.
func newNotesStripView(sum enginenotes.Summary) notesStripView {
	t := sum.Totals
	var notable []enginenotes.Event
	for _, f := range sum.Failures {
		if !f.Expected {
			notable = append(notable, f)
		}
	}
	v := notesStripView{Sig: failureSig(notable)}
	if len(notable) == 0 {
		switch {
		case t == (enginenotes.Totals{}) && len(sum.Models) == 0:
			v.Text = "Engine notes — nothing recorded yet"
		case t.ExpectedFailures > 0:
			v.Text = "Engine notes — " + count(t.Loads, "load", "loads") + " in the last 24 h, no crashes outside the " +
				count(t.ExpectedFailures, "setting", "settings") + " that jobs found would not run"
		default:
			v.Text = "Engine notes — " + count(t.Loads, "load", "loads") + " in the last 24 h, no crashes"
		}
		return v
	}
	v.Warn = true
	var crashes, failedLoads, faults, inJobs int
	for _, f := range notable {
		if f.JobID != "" {
			inJobs++
		}
		if f.Kind == enginenotes.KindLoadFailed {
			failedLoads++
		} else {
			crashes++
		}
		if f.Cause == enginenotes.CauseGPUFault {
			faults++
		}
	}
	var parts []string
	if crashes > 0 {
		parts = append(parts, count(crashes, "crash", "crashes"))
	}
	if failedLoads > 0 {
		parts = append(parts, count(failedLoads, "failed load", "failed loads"))
	}
	text := strings.Join(parts, " and ") + " in the last 24 h"
	var notes []string
	if faults > 0 {
		notes = append(notes, count(faults, "GPU fault", "GPU faults"))
	}
	if inJobs > 0 {
		notes = append(notes, fmt.Sprintf("%d in jobs", inJobs))
	}
	if len(notes) > 0 {
		text += " (" + strings.Join(notes, ", ") + ")"
	}
	last := notable[0] // newest first
	text += " — last: " + failureModel(last)
	var about []string
	if enginenotes.IsGeneration(last.Source) {
		about = append(about, last.Source)
	}
	if last.Cause != "" && last.Cause != enginenotes.CauseOther {
		about = append(about, last.Cause.Label())
	}
	if len(about) > 0 {
		text += " (" + strings.Join(about, ", ") + ")"
	}
	at := newTimeView(last.Time, sum.To)
	v.Text, v.LastAt = text, &at
	return v
}

// failureSig is an FNV-1a hash of the failures' times and instances.
func failureSig(fs []enginenotes.Event) string {
	h := fnv.New64a()
	for _, f := range fs {
		fmt.Fprintf(h, "%d %s %s\n", f.Time.UnixNano(), f.Source, failureModel(f))
	}
	return fmt.Sprintf("%016x", h.Sum64())
}

func failureModel(e enginenotes.Event) string {
	switch {
	case e.ModelName != "":
		return e.ModelName
	case e.ModelID != "":
		return e.ModelID
	}
	return e.RouterName
}

// notesBodyView is the Review dialog's content for one period.
type notesBodyView struct {
	Period  string
	Periods []notesPeriodButton
	From    timeView
	Since   timeView
	Totals  enginenotes.Totals
	Models  []notesModelView
	// GenInstances are the generation backends' instances, kept apart
	// from the models: they have no requests or tokens to count here.
	GenInstances []notesModelView
	Failures     []notesFailureView
	Empty        bool
	// Result is the line after a delete.
	Result string
}

type notesPeriodButton struct {
	Key, Label string
	Active     bool
}

type notesModelView struct {
	Name string
	// Source is the backend, for a generation instance.
	Source                               string
	Loads, FailedLoads, Crashes, Unloads int
	UnloadsForRoom                       int
	Loaded, AvgLoad                      string
	Requests, FailedRequests             string
	PromptTokens, GenTokens              string
	LastUsed                             *timeView
}

// notesFailureView is one failure, or a run of the same failure: the
// same model, kind, cause, job and first error line, one after another.
// An Autotune stage can fail the same way twenty times in two minutes.
type notesFailureView struct {
	At timeView
	// Count is how many failures the entry stands for, and FirstAt when
	// the oldest of them happened; At and the rest are the newest one's.
	Count   int
	FirstAt *timeView
	Model   string
	// Source is the program that failed: llama-server, sd-server or
	// ace-server.
	Source     string
	What       string
	Cause      string
	CauseLabel string
	Expected   bool
	ExitStatus int
	Build      string
	GPUs       string
	Job        string
	ErrorLines []string
	Tail       []string
}

// notesPeriodOrder is the period picker, in order.
var notesPeriodOrder = []notesPeriodButton{
	{Key: "24h", Label: "24 h"},
	{Key: "7d", Label: "7 days"},
	{Key: "30d", Label: "30 days"},
	{Key: "all", Label: "All"},
}

func newNotesBodyView(period string, sum enginenotes.Summary) notesBodyView {
	v := notesBodyView{
		Period: period,
		From:   newTimeView(sum.From, sum.To),
		Since:  newTimeView(sum.CountingSince, sum.To),
		Totals: sum.Totals,
		Empty:  sum.Totals == (enginenotes.Totals{}) && len(sum.Models) == 0,
	}
	for _, p := range notesPeriodOrder {
		p.Active = p.Key == period
		v.Periods = append(v.Periods, p)
	}
	for _, m := range sum.Models {
		row := notesModelView{
			Name: m.Name, Loads: m.Loads, FailedLoads: m.FailedLoads, Crashes: m.Crashes, Unloads: m.Unloads,
			UnloadsForRoom: m.UnloadsForRoom,
			Loaded:         "—", AvgLoad: "—",
			Requests:       models.GroupDigits(m.Requests),
			FailedRequests: models.GroupDigits(m.FailedRequests),
			PromptTokens:   models.GroupDigits(m.PromptTokens),
			GenTokens:      models.GroupDigits(m.GenTokens),
		}
		if m.LoadedSeconds > 0 {
			row.Loaded = formatSeconds(m.LoadedSeconds)
		}
		if m.Loads > 0 {
			row.AvgLoad = formatSeconds(m.AvgLoadSeconds)
		}
		if !m.LastUsed.IsZero() {
			at := newTimeView(m.LastUsed, sum.To)
			row.LastUsed = &at
		}
		if enginenotes.IsGeneration(m.Source) {
			row.Source = m.Source
			v.GenInstances = append(v.GenInstances, row)
			continue
		}
		v.Models = append(v.Models, row)
	}
	var prev *enginenotes.Event
	for i, f := range sum.Failures {
		if prev != nil && sameFailure(*prev, f) {
			last := &v.Failures[len(v.Failures)-1]
			last.Count++
			// A range within one minute would read "07:13 to 07:13".
			if at := newTimeView(f.Time, sum.To); at.Label != last.At.Label {
				last.FirstAt = &at
			} else {
				last.FirstAt = nil
			}
			continue
		}
		prev = &sum.Failures[i]
		what := "crash"
		if f.Kind == enginenotes.KindLoadFailed {
			what = "failed to load"
			if enginenotes.IsGeneration(f.Source) {
				what = "failed to start"
			}
		}
		source := f.Source
		if source == "" {
			source = enginenotes.SourceLlamaServer
		}
		v.Failures = append(v.Failures, notesFailureView{
			At: newTimeView(f.Time, sum.To), Count: 1, Model: failureModel(f), Source: source, What: what,
			Cause: string(f.Cause), CauseLabel: f.Cause.Label(), Expected: f.Expected,
			ExitStatus: f.ExitStatus, Build: f.Build, GPUs: f.GPUs, Job: f.JobName,
			ErrorLines: f.ErrorLines, Tail: f.Tail,
		})
	}
	return v
}

// sameFailure reports whether b repeats a: the same program, model,
// kind, cause, job and first error line.
func sameFailure(a, b enginenotes.Event) bool {
	first := func(e enginenotes.Event) string {
		if len(e.ErrorLines) > 0 {
			return e.ErrorLines[0]
		}
		return ""
	}
	return a.Source == b.Source && failureModel(a) == failureModel(b) && a.Kind == b.Kind && a.Cause == b.Cause &&
		a.JobName == b.JobName && first(a) == first(b)
}

// formatSeconds is a duration as the dialog shows it: "35 s", "42 m",
// "3 h 12 m" or "2 d 3 h".
func formatSeconds(sec float64) string {
	s := int(sec + 0.5)
	switch {
	case s < 60:
		return fmt.Sprintf("%d s", s)
	case s < 3600:
		return fmt.Sprintf("%d m", s/60)
	case s < 86400:
		return fmt.Sprintf("%d h %d m", s/3600, s%3600/60)
	}
	return fmt.Sprintf("%d d %d h", s/86400, s%86400/3600)
}

// handleEngineNotesStrip answers GET /api/service/notes/strip with the
// line above the Server log, for the last 24 hours.
func (s *Server) handleEngineNotesStrip(w http.ResponseWriter, r *http.Request) {
	if s.notes == nil {
		s.renderPartial(w, "engine_notes_strip", notesStripView{Text: "Engine notes — not available"})
		return
	}
	sum, _ := s.notesSummary("24h")
	s.renderPartial(w, "engine_notes_strip", newNotesStripView(sum))
}

// renderNotesBody renders the Review dialog's content for period.
func (s *Server) renderNotesBody(w http.ResponseWriter, period, result string) {
	if period == "" {
		period = "24h"
	}
	sum, ok := s.notesSummary(period)
	if !ok {
		http.Error(w, "period must be 24h, 7d, 30d or all", http.StatusBadRequest)
		return
	}
	v := newNotesBodyView(period, sum)
	v.Result = result
	s.renderPartial(w, "engine_notes_body", v)
}
