package api

import (
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/enginenotes"
)

var viewNow = time.Date(2026, 10, 30, 15, 0, 0, 0, time.UTC)

func crashAt(ago time.Duration, name, job string) enginenotes.Event {
	return enginenotes.Event{Kind: enginenotes.KindCrash, Time: viewNow.Add(-ago),
		ModelName: name, RouterName: name, JobName: job, JobID: job}
}

func TestStripText(t *testing.T) {
	cases := []struct {
		name string
		sum  enginenotes.Summary
		want string
		warn bool
	}{
		{"nothing yet", enginenotes.Summary{}, "Engine notes — nothing recorded yet", false},
		{"quiet", enginenotes.Summary{Totals: enginenotes.Totals{Loads: 14, Unloads: 13}},
			"Engine notes — 14 loads in the last 24 h, no crashes", false},
		{"one load", enginenotes.Summary{Totals: enginenotes.Totals{Loads: 1}},
			"Engine notes — 1 load in the last 24 h, no crashes", false},
		{"one crash", enginenotes.Summary{
			Totals:   enginenotes.Totals{Loads: 2, Crashes: 1},
			Failures: []enginenotes.Event{crashAt(time.Hour, "Qwen3.6-35B-A3B", "")},
		}, "1 crash in the last 24 h — last: Qwen3.6-35B-A3B", true},
		{"crashes in jobs", enginenotes.Summary{
			Totals: enginenotes.Totals{Crashes: 3, FailuresInJobs: 2},
			Failures: []enginenotes.Event{crashAt(time.Hour, "A", "j"), crashAt(2*time.Hour, "B", "j"),
				crashAt(3*time.Hour, "C", "")},
		}, "3 crashes in the last 24 h (2 in jobs) — last: A", true},
		{"crash and failed load", enginenotes.Summary{
			Totals: enginenotes.Totals{Crashes: 1, FailedLoads: 1},
			Failures: []enginenotes.Event{
				{Kind: enginenotes.KindLoadFailed, Time: viewNow.Add(-time.Minute), RouterName: "b"},
				crashAt(time.Hour, "A", "")},
		}, "1 crash and 1 failed load in the last 24 h — last: b", true},
	}
	for _, c := range cases {
		c.sum.To = viewNow
		v := newNotesStripView(c.sum)
		if v.Text != c.want || v.Warn != c.warn {
			t.Errorf("%s: %q warn=%v, want %q warn=%v", c.name, v.Text, v.Warn, c.want, c.warn)
		}
		if c.warn && (v.LastAt == nil || v.LastAt.Label == "") {
			t.Errorf("%s: no time for the last failure", c.name)
		}
	}
}

func TestStripSigChangesOnlyWithTheFailures(t *testing.T) {
	a := []enginenotes.Event{crashAt(time.Hour, "A", "")}
	same := []enginenotes.Event{crashAt(time.Hour, "A", "")}
	more := append([]enginenotes.Event{crashAt(time.Minute, "B", "")}, a...)
	if failureSig(a) != failureSig(same) {
		t.Error("the same failures gave a different fingerprint")
	}
	if failureSig(a) == failureSig(more) {
		t.Error("a new failure did not change the fingerprint")
	}
	// Other events do not touch it: the strip only passes failures.
	if failureSig(nil) == failureSig(a) {
		t.Error("no failures and one failure share a fingerprint")
	}
}

func TestFormatSeconds(t *testing.T) {
	for sec, want := range map[float64]string{
		35: "35 s", 59.4: "59 s", 60: "1 m", 42 * 60: "42 m",
		3*3600 + 12*60: "3 h 12 m", 2*86400 + 3*3600 + 59: "2 d 3 h",
	} {
		if got := formatSeconds(sec); got != want {
			t.Errorf("formatSeconds(%v) = %q, want %q", sec, got, want)
		}
	}
}

func TestTimeViewShowsTheDateOnlyForOtherDays(t *testing.T) {
	if got := newTimeView(viewNow.Add(-time.Hour), viewNow).Label; got != "14:00 UTC" {
		t.Errorf("today = %q", got)
	}
	if got := newTimeView(viewNow.Add(-24*time.Hour), viewNow).Label; got != "2026-10-29 15:00 UTC" {
		t.Errorf("yesterday = %q", got)
	}
}

func TestNotesBodyRenders(t *testing.T) {
	s := newTestServer(t)
	crash := crashAt(time.Hour, "Qwen", "night run")
	crash.ExitStatus, crash.Build, crash.GPUs = 1, "b9000", "gpu:1"
	crash.ErrorLines = []string{"CUDA error: an illegal memory access"}
	crash.Tail = []string{"last line written"}
	sum := enginenotes.Summary{
		CountingSince: viewNow.Add(-48 * time.Hour), From: viewNow.Add(-24 * time.Hour), To: viewNow,
		Totals: enginenotes.Totals{Loads: 3, Crashes: 1, FailuresInJobs: 1},
		Models: []enginenotes.ModelRow{{
			ModelID: "q", Name: "Qwen", Loads: 3, Crashes: 1, LoadedSeconds: 3*3600 + 12*60,
			AvgLoadSeconds: 35, LastUsed: viewNow.Add(-time.Hour), Requests: 1200, PromptTokens: 1234567, GenTokens: 89,
		}},
		Failures: []enginenotes.Event{crash},
	}
	rec := httptest.NewRecorder()
	s.renderPartial(rec, "engine_notes_body", newNotesBodyView("7d", sum))
	html := rec.Body.String()
	for _, want := range []string{
		`hx-get="/api/service/notes?period=7d"`, `aria-current="true"`,
		"<td>Qwen</td>", "3 h 12 m", "35 s", "1,234,567", "1,200",
		"CUDA error: an illegal memory access", "last line written", "Build b9000 · gpu:1 · during job “night run”",
		"exit status 1", `<time datetime="2026-10-30T14:00:00Z">14:00 UTC</time>`,
		`name="period" value="7d"`,
	} {
		if !strings.Contains(html, want) {
			t.Errorf("body is missing %q", want)
		}
	}
	if strings.Contains(html, "Nothing recorded in this period") {
		t.Error("a period with data says it has none")
	}

	rec = httptest.NewRecorder()
	s.renderPartial(rec, "engine_notes_body", newNotesBodyView("24h", enginenotes.Summary{To: viewNow}))
	if !strings.Contains(rec.Body.String(), "Nothing recorded in this period.") {
		t.Error("an empty period does not say so")
	}
}

// The HTMX delete answers with the dialog body and its result line, and
// tells the strip to update.
func TestNotesDeleteHTMX(t *testing.T) {
	path := filepath.Join(t.TempDir(), "engine-notes.json")
	since := time.Now().Add(-60 * 24 * time.Hour).UTC().Format(time.RFC3339)
	if err := os.WriteFile(path, []byte(`{"version":1,"counting_since":"`+since+`","events":[]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	st, err := enginenotes.Open(path, time.Now)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { st.Close() })
	s := newTestServer(t)
	s.notes = st
	s.notes.Record(enginenotes.Event{Kind: enginenotes.KindLoad, ModelID: "m", Time: time.Now().Add(-10 * 24 * time.Hour)})

	req := httptest.NewRequest(http.MethodPost, "/api/service/notes/delete",
		strings.NewReader(url.Values{"older_than": {"7d"}, "period": {"30d"}}.Encode()))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("HX-Request", "true")
	rec := httptest.NewRecorder()
	s.handleEngineNotesDelete(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	if got := rec.Header().Get("HX-Trigger"); got != "engineNotesChanged" {
		t.Errorf("HX-Trigger = %q", got)
	}
	body := rec.Body.String()
	if !strings.Contains(body, "Deleted 1 event and 0 hours of usage.") {
		t.Errorf("no result line in %s", body)
	}
	if !strings.Contains(body, `name="period" value="30d"`) {
		t.Error("the body after a delete lost the period it showed")
	}
}

func TestNotesStripEndpoint(t *testing.T) {
	s := newTestServer(t)
	openTestNotes(t, s)
	s.notes.Record(enginenotes.Event{Kind: enginenotes.KindCrash, ModelName: "Qwen", RouterName: "q"})
	rec := httptest.NewRecorder()
	s.handleEngineNotesStrip(rec, httptest.NewRequest(http.MethodGet, "/api/service/notes/strip", nil))
	html := rec.Body.String()
	if !strings.Contains(html, `data-level="warning"`) || !strings.Contains(html, "1 crash in the last 24 h — last: Qwen at <time") {
		t.Errorf("strip = %s", html)
	}
}
