package api

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/generation"
)

// The note after a successful measurement says what was found and what
// happened to the reservation.
func TestMeasureNote(t *testing.T) {
	inst := generation.Instance{MeasuredPeakMiB: 19032, ReservedMiB: 20992}
	for _, c := range []struct {
		name      string
		before    int
		estimated bool
		restarted bool
		want      string
	}{
		{"first measurement", 0, false, false, "Peak 18.6 GiB. Reservation set to 20.5 GiB."},
		{"kept, as suggested", 20992, false, false, "Peak 18.6 GiB. Reservation stays at 20.5 GiB, as suggested."},
		{"kept, suggests other", 30720, false, false, "Peak 18.6 GiB. Reservation stays at 30.0 GiB; this measurement suggests 20.5 GiB."},
		{"estimated and restarted", 20992, true, true, "Peak 18.6 GiB (estimated from the whole GPU's use). Reservation stays at 20.5 GiB, as suggested. Restarted."},
	} {
		in := inst
		in.MeasureEstimated = c.estimated
		if c.before != 0 && c.before != 20992 {
			in.ReservedMiB = c.before
		}
		if got := measureNote(c.before, in, c.restarted); got != c.want {
			t.Errorf("%s:\n got  %q\n want %q", c.name, got, c.want)
		}
	}
}

// The instance table shows the last measurement's result, or its full
// failure message, in the instance's row, plus the measured peak.
func TestInstanceTableShowsMeasurement(t *testing.T) {
	s := instanceServer(t)
	if err := s.gen.AddInstance(generation.Instance{ID: "m", Name: "m", ModelSetID: "ming-image", Port: 8081, Enabled: true,
		ReservedMiB: 30720, MeasuredPeakMiB: 19032, MeasuredAt: time.Date(2026, 10, 25, 2, 17, 0, 0, time.UTC)}); err != nil {
		t.Fatal(err)
	}
	table := func() string {
		rec := httptest.NewRecorder()
		s.handleGenInstances(rec, instanceReq("GET", "/", "", "", nil))
		return rec.Body.String()
	}

	s.genMeasure.set("m", measureOutcome{Note: "Peak 18.6 GiB. Reservation stays at 30.0 GiB; this measurement suggests 20.5 GiB.", At: time.Date(2026, 10, 25, 2, 17, 0, 0, time.Local)})
	out := table()
	for _, want := range []string{
		"Measured at 02:17: Peak 18.6 GiB. Reservation stays at 30.0 GiB; this measurement suggests 20.5 GiB.",
		"peak 18.6 GiB, measured 2026-10-25 02:17",
		"Use suggested 20.5 GiB",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("table missing %q", want)
		}
	}

	s.genMeasure.set("m", measureOutcome{Err: "the test generation failed: sd-server exited", At: time.Date(2026, 10, 25, 2, 20, 0, 0, time.Local)})
	if out := table(); !strings.Contains(out, "Measurement failed at 02:20: the test generation failed: sd-server exited") {
		t.Errorf("failure not shown in full:\n%s", out)
	}
}

// The edit form is filled with the instance's current values and saves
// with PUT; it is refused while the instance runs.
func TestGenInstanceEditForm(t *testing.T) {
	s := instanceServer(t)
	if err := s.gen.AddInstance(generation.Instance{ID: "m", Name: "My Ming", ModelSetID: "ming-image", BuildID: "sd-master-1-abc-vulkan",
		GPU: 0, Port: 8081, Enabled: true, AutoStart: true, ExtraArgs: []string{"--backend", "te=cpu"}}); err != nil {
		t.Fatal(err)
	}
	rec := httptest.NewRecorder()
	s.handleGenInstanceEditForm(rec, instanceReq("GET", "/", "m", "", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("edit form: %d %s", rec.Code, rec.Body.String())
	}
	out := rec.Body.String()
	for _, want := range []string{
		`hx-put="/api/gen/instances/m"`,
		`value="My Ming" disabled`,
		`<option value="ming-image" selected>`,
		`<option value="sd-master-1-abc-vulkan" selected>`,
		`value="--backend te=cpu"`,
		`name="auto_start" checked`,
		"Save changes",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("edit form missing %q", want)
		}
	}
	if strings.Contains(out, `hx-post="/api/gen/instances"`) {
		t.Error("edit form would create a new instance")
	}

	s.genMgr.SetMeasuring("m", true)
	rec = httptest.NewRecorder()
	s.handleGenInstanceEditForm(rec, instanceReq("GET", "/", "m", "", nil))
	if rec.Code != http.StatusConflict {
		t.Errorf("edit form while measuring: %d, want 409", rec.Code)
	}
}
