package api

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/builder"
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
		started   bool
		want      string
	}{
		{"first measurement", 0, false, false, false, "Peak 18.6 GiB. Reservation set to 20.5 GiB."},
		{"first measurement from Start", 0, false, false, true, "Peak 18.6 GiB. Reservation set to 20.5 GiB. Started."},
		{"kept, as suggested", 20992, false, false, false, "Peak 18.6 GiB. Reservation stays at 20.5 GiB, as suggested."},
		{"kept, suggests other", 30720, false, false, false, "Peak 18.6 GiB. Reservation stays at 30.0 GiB; this measurement suggests 20.5 GiB."},
		{"estimated and restarted", 20992, true, true, false, "Peak 18.6 GiB (estimated from the whole GPU's use). Reservation stays at 20.5 GiB, as suggested. Restarted."},
	} {
		in := inst
		in.MeasureEstimated = c.estimated
		if c.before != 0 && c.before != 20992 {
			in.ReservedMiB = c.before
		}
		if got := measureNote(c.before, in, c.restarted, c.started); got != c.want {
			t.Errorf("%s:\n got  %q\n want %q", c.name, got, c.want)
		}
	}
	// LTX-2.3 on an RTX A4000: the room was below the peak, so the
	// reservation is the room and leaves no slack.
	a4000 := generation.Instance{MeasuredPeakMiB: 15320, ReservedMiB: 14676, MeasuredSuggestedMiB: 14676, MeasuredFrames: 33}
	want := "Peak 15.0 GiB. Reservation capped at 14.3 GiB, all this card can give the instance. The peak is 0.6 GiB above it, within the 1.0 GiB allowed for memory the backend's own limit does not count, such as the GPU runtime's."
	if got := measureNote(0, a4000, false, false); got != want {
		t.Errorf("capped below the peak:\n got  %q\n want %q", got, want)
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

// Start on an instance with no reservation asks to measure it first
// instead of posting a start that can only be refused.
func TestStartWithoutReservationAsksToMeasure(t *testing.T) {
	s := instanceServer(t)
	for _, in := range []generation.Instance{
		{ID: "new", Name: "new", ModelSetID: "ming-image", Port: 8081, Enabled: true},
		{ID: "ok", Name: "ok", ModelSetID: "ming-image", Port: 8082, Enabled: true, ReservedMiB: 20992},
	} {
		if err := s.gen.AddInstance(in); err != nil {
			t.Fatal(err)
		}
	}
	rec := httptest.NewRecorder()
	s.handleGenInstances(rec, instanceReq("GET", "/", "", "", nil))
	out := rec.Body.String()
	if !strings.Contains(out, `openGenMeasureStart('new'`) || strings.Contains(out, `hx-post="/api/gen/instances/new/start"`) {
		t.Errorf("unmeasured instance's Start should open the measure dialog:\n%s", out)
	}
	if !strings.Contains(out, `hx-post="/api/gen/instances/ok/start"`) || strings.Contains(out, `openGenMeasureStart('ok'`) {
		t.Errorf("measured instance's Start should start it directly:\n%s", out)
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

// While an LLM measurement holds an instance's GPU, its row says so.
func TestInstanceTableShowsGPUHold(t *testing.T) {
	s := instanceServer(t)
	if err := s.gen.AddInstance(generation.Instance{ID: "m", Name: "m", ModelSetID: "ming-image", GPU: 0, Port: 8081, Enabled: true}); err != nil {
		t.Fatal(err)
	}
	release, err := s.genQueue.Hold(context.Background(), []int{0}, "Autotune: Qwen3-4B")
	if err != nil {
		t.Fatal(err)
	}
	rec := httptest.NewRecorder()
	s.handleGenInstances(rec, instanceReq("GET", "/", "", "", nil))
	if !strings.Contains(rec.Body.String(), "GPU in use by: Autotune: Qwen3-4B") {
		t.Errorf("table does not show the hold:\n%s", rec.Body.String())
	}
	release()
	rec = httptest.NewRecorder()
	s.handleGenInstances(rec, instanceReq("GET", "/", "", "", nil))
	if strings.Contains(rec.Body.String(), "GPU in use by") {
		t.Error("hold still shown after release")
	}
}

// A row with no web UI link says why, and how to get one.
func TestGenWebUILinkReasons(t *testing.T) {
	s := instanceServer(t)
	s.cfg.ExternalURL = "http://compute2:3000"
	inst := generation.Instance{ID: "m", Port: 8083}
	withUI := &builder.BuildResult{ID: "sd-1", Project: builder.ProjectSDCpp, HasWebUI: true}
	noUI := &builder.BuildResult{ID: "sd-0", Project: builder.ProjectSDCpp}

	if link, note := s.genWebUILink(withUI, inst, true); link != "http://compute2:8083" || note != "" {
		t.Errorf("running, with web UI: %q %q", link, note)
	}
	if link, note := s.genWebUILink(withUI, inst, false); link != "" || note != "" {
		t.Errorf("stopped: %q %q, want neither", link, note)
	}
	for _, running := range []bool{true, false} {
		if link, note := s.genWebUILink(noUI, inst, running); link != "" || !strings.Contains(note, "Built-in web UI") || !strings.Contains(note, "stable-diffusion.cpp") {
			t.Errorf("build without web UI (running %v): %q %q", running, link, note)
		}
	}

	s.inContainerFn = func() bool { return true }
	t.Setenv("LLAMA_TOOLCHEST_GEN_PORTS", "")
	if _, note := s.genWebUILink(withUI, inst, true); !strings.Contains(note, "setup.sh install") {
		t.Errorf("ports not published: %q", note)
	}
	s.inContainerFn = func() bool { return false }
	s.cfg.ExternalURL = ""
	if _, note := s.genWebUILink(withUI, inst, true); !strings.Contains(note, "Server URL") {
		t.Errorf("no server URL: %q", note)
	}
}
