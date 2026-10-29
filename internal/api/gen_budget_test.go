package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/generation"
)

func TestMeasureBudget(t *testing.T) {
	video := generation.ModelSet{Kind: generation.KindVideo}
	req := func(v string) *http.Request {
		return httptest.NewRequest("POST", "/", strings.NewReader(url.Values{"budget_gib": {v}}.Encode()))
	}
	for v, want := range map[string]int{"": 0, " ": 0, "14": 14336, "14.5": 14848} {
		r := req(v)
		r.Header.Set("Content-Type", "application/x-www-form-urlencoded")
		if got, err := measureBudget(r, video); err != nil || got != want {
			t.Errorf("budget %q = %d, %v; want %d", v, got, err, want)
		}
	}
	for _, v := range []string{"3", "abc", "-2"} {
		r := req(v)
		r.Header.Set("Content-Type", "application/x-www-form-urlencoded")
		if _, err := measureBudget(r, video); err == nil {
			t.Errorf("budget %q was taken", v)
		}
	}
	r := req("14")
	r.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	if got, _ := measureBudget(r, generation.ModelSet{Kind: generation.KindSD}); got != 0 {
		t.Errorf("an image set took a budget: %d", got)
	}
}

func TestPresetNeed(t *testing.T) {
	short := generation.MeasurePoint{Frames: 33, Width: 768, Height: 512, PeakMiB: 18000}
	medium := generation.MeasurePoint{Frames: 65, Width: 1280, Height: 720, PeakMiB: 22000}
	full := generation.VideoSize{Frames: 121, Width: 1280, Height: 720}
	room := 28000
	if n := presetNeed(nil, full, room, true); n != (sizeNeed{}) {
		t.Errorf("nothing measured: %+v, want nothing", n)
	}
	if n := presetNeed([]generation.MeasurePoint{short, medium}, generation.VideoSize{Frames: 65, Width: 1280, Height: 720}, room, true); !n.Measured || n.Fit != "fits" {
		t.Errorf("a measured size: %+v", n)
	}
	// About 26.4 GiB predicted for Full: within 2 GiB of the 27.3 GiB room.
	if n := presetNeed([]generation.MeasurePoint{short, medium}, full, room, true); n.Measured || n.Fit != "close" || n.GiB < 25 || n.GiB > 27 {
		t.Errorf("Full predicted: %+v, want close to the limit", n)
	}
	if n := presetNeed([]generation.MeasurePoint{short, medium}, full, 22000, true); n.Fit != "over" {
		t.Errorf("Full with a small room: %+v, want over", n)
	}
	stopped := generation.MeasurePoint{Frames: 65, Width: 1280, Height: 720, PeakMiB: 30000, Over: true}
	if n := presetNeed([]generation.MeasurePoint{short, stopped}, full, room, true); n.Fit != "over" {
		t.Errorf("above a stopped size: %+v, want over", n)
	}
	inBudget := short
	inBudget.BudgetMiB = 14336
	if n := presetNeed([]generation.MeasurePoint{inBudget}, generation.VideoSize{Frames: 33, Width: 768, Height: 512}, room, true); n.Measured {
		t.Errorf("a size measured inside a budget counted for the whole room: %+v", n)
	}
}

// The Measure dialog's data carries each preset's need, the number of
// measurements, the room and the current budget.
func TestVideoMeasureJSONNeeds(t *testing.T) {
	set := ti2vSet()
	presets := generation.VideoPresets(set)
	inst := generation.Instance{BudgetMiB: 14336, MeasurePoints: []generation.MeasurePoint{
		{Frames: presets[0].Frames, Width: presets[0].Width, Height: presets[0].Height, PeakMiB: 18000},
	}}
	var info struct {
		Needs     []sizeNeed `json:"needs"`
		Points    int        `json:"points"`
		RoomGiB   float64    `json:"roomGiB"`
		BudgetGiB float64    `json:"budgetGiB"`
	}
	if err := json.Unmarshal([]byte(videoMeasureJSON(set, inst, 28672, true)), &info); err != nil {
		t.Fatal(err)
	}
	if len(info.Needs) != len(presets) || !info.Needs[0].Measured || info.Needs[1].GiB != 0 || info.Points != 1 || info.RoomGiB != 28 || info.BudgetGiB != 14 {
		t.Errorf("dialog data = %+v", info)
	}
}

// A reservation set by hand makes it sd-server's limit again.
func TestSetReservationClearsBudget(t *testing.T) {
	s := vramServer(t)
	addInstance(t, s, generation.Instance{ID: "vid", Port: 8082, Enabled: true, ReservedMiB: 16384, BudgetMiB: 14336})
	rec := httptest.NewRecorder()
	s.handleGenSetReservation(rec, instanceReq("PUT", "/", "vid", "", url.Values{"reserved_gib": {"12"}}))
	inst, _ := s.gen.Instance("vid")
	if rec.Code != http.StatusOK || inst.BudgetMiB != 0 || inst.ReservedMiB != 12288 {
		t.Errorf("HTTP %d, budget %d, reservation %d; want the budget cleared", rec.Code, inst.BudgetMiB, inst.ReservedMiB)
	}
}

func TestMeasureNoteMentionsBudget(t *testing.T) {
	inst := generation.Instance{MeasuredPeakMiB: 15000, ReservedMiB: 16896, MeasuredSuggestedMiB: 16896, MeasuredFrames: 81, BudgetMiB: 14336}
	if note := measureNote(0, inst, false, false); !strings.Contains(note, "inside a budget of 14.0 GiB") {
		t.Errorf("note = %q", note)
	}
}
