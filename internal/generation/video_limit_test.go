package generation

import "testing"

func TestVideoPresets(t *testing.T) {
	catalog := ModelSet{CatalogID: "wan22-ti2v-5b", DefaultFrames: 33, DefaultWidth: 832, DefaultHeight: 480}
	if p := VideoPresets(catalog); len(p) != 3 || p[0].Label != "Short" || p[2].Label != "Full" {
		t.Errorf("catalog set (copy or not) = %+v, want the catalog's three", p)
	}
	stored := ModelSet{CatalogID: "gone-from-catalog", VideoSizes: []VideoSize{{Label: "Only", Frames: 9, Width: 64, Height: 64}}}
	if p := VideoPresets(stored); len(p) != 1 || p[0].Label != "Only" {
		t.Errorf("set whose entry is gone = %+v, want its own copy", p)
	}
	custom := ModelSet{DefaultFrames: 17, DefaultWidth: 640, DefaultHeight: 360}
	if p := VideoPresets(custom); len(p) != 1 || p[0] != (VideoSize{Label: "Short", Frames: 17, Width: 640, Height: 360}) {
		t.Errorf("custom set = %+v, want one Short preset at its test size", p)
	}
}

func TestVideoLimit(t *testing.T) {
	set := ModelSet{CatalogID: "wan22-ti2v-5b", DefaultFrames: 33, DefaultWidth: 832, DefaultHeight: 480}
	medium := Instance{ReservedMiB: 20000, MeasuredSuggestedMiB: 20000, MeasuredFrames: 81, MeasuredWidth: 1280, MeasuredHeight: 704}
	if got, m := VideoLimit(medium, set); !m || got != (VideoSize{Label: "Medium", Frames: 81, Width: 1280, Height: 704}) {
		t.Errorf("measured at Medium = %+v, %v", got, m)
	}
	custom := medium
	custom.MeasuredFrames = 49
	if got, _ := VideoLimit(custom, set); got.Label != "custom size" {
		t.Errorf("custom measured size label = %q", got.Label)
	}
	low := medium
	low.ReservedMiB = 16000
	if got, m := VideoLimit(low, set); m || got.Frames != 33 || got.Width != 832 {
		t.Errorf("reservation set below the suggestion = %+v, %v; want the test size", got, m)
	}
	if got, m := VideoLimit(Instance{ReservedMiB: 26000}, set); m || got.Frames != 33 {
		t.Errorf("never measured = %+v, %v; want the test size", got, m)
	}
}

func TestCoveringPreset(t *testing.T) {
	presets := VideoPresets(ModelSet{CatalogID: "wan22-ti2v-5b"})
	for _, tc := range []struct {
		frames, w, h int
		want         string
		ok           bool
	}{
		{33, 832, 480, "Short", true},
		{121, 832, 480, "Medium", true}, // less video than Medium's 81 frames at 1280×704
		{121, 1280, 704, "Full", true},
		{121, 1280, 720, "", false},
	} {
		p, ok := CoveringPreset(presets, tc.frames, tc.w, tc.h)
		if ok != tc.ok || (ok && p.Label != tc.want) {
			t.Errorf("%d×%dx%d = %q, %v; want %q, %v", tc.frames, tc.w, tc.h, p.Label, ok, tc.want, tc.ok)
		}
	}
}

// A video set's presets start at its test size, end within the model's
// limit, and use frame counts sd-server does not round (4n+1).
func TestCatalogVideoPresets(t *testing.T) {
	for _, e := range Catalog() {
		if e.Kind != KindVideo {
			if len(e.VideoSizes) > 0 {
				t.Errorf("%s: presets on a %s set", e.ID, e.Kind)
			}
			continue
		}
		if len(e.VideoSizes) == 0 {
			t.Errorf("%s: no measure presets", e.ID)
			continue
		}
		first := e.VideoSizes[0]
		if first.Frames != e.DefaultFrames || first.Width != e.DefaultWidth || first.Height != e.DefaultHeight {
			t.Errorf("%s: first preset %+v is not the test size", e.ID, first)
		}
		for _, p := range e.VideoSizes {
			if p.Frames > e.MaxFrames || max(p.Width, p.Height) > max(e.MaxWidth, e.MaxHeight) || min(p.Width, p.Height) > min(e.MaxWidth, e.MaxHeight) {
				t.Errorf("%s: preset %+v is past the model's limit", e.ID, p)
			}
			if (p.Frames-1)%4 != 0 {
				t.Errorf("%s: preset %s has %d frames, not 4n+1", e.ID, p.Label, p.Frames)
			}
			if p.Label == "" || p.Width%8 != 0 || p.Height%8 != 0 {
				t.Errorf("%s: preset %+v needs a label and sizes in multiples of 8", e.ID, p)
			}
		}
		if e.FullMeasureTime == "" {
			t.Errorf("%s: no FullMeasureTime for the Measure dialog", e.ID)
		}
	}
}

func TestAddMeasurePoint(t *testing.T) {
	short := MeasurePoint{Frames: 33, Width: 832, Height: 480, PeakMiB: 18000}
	var pts []MeasurePoint
	pts = AddMeasurePoint(pts, short)
	pts = AddMeasurePoint(pts, MeasurePoint{Frames: 33, Width: 832, Height: 480, PeakMiB: 18000, BudgetMiB: 14336})
	pts = AddMeasurePoint(pts, MeasurePoint{Frames: 33, Width: 832, Height: 480, PeakMiB: 17500})
	if len(pts) != 2 || pts[1].PeakMiB != 17500 || pts[0].BudgetMiB != 14336 {
		t.Errorf("a new measurement of a size inside the same budget should replace the old one: %+v", pts)
	}
	for f := 1; f <= 10; f++ {
		pts = AddMeasurePoint(pts, MeasurePoint{Frames: f, Width: 64, Height: 64, PeakMiB: f})
	}
	if len(pts) != maxMeasurePoints || pts[len(pts)-1].Frames != 10 {
		t.Errorf("want the %d newest kept: %+v", maxMeasurePoints, pts)
	}
}

func TestPredictPeak(t *testing.T) {
	// LTX-2.3 on the R9700 (2026-10-27): Short 27.7 GiB, Medium 29.0 GiB;
	// Full measured 29.7 GiB (30413 MiB). The line through Short and
	// Medium gives about 30.4 GiB (31160 MiB): a little above, within
	// 1.5 GiB.
	short := MeasurePoint{Frames: 33, Width: 768, Height: 512, PeakMiB: 28365}
	medium := MeasurePoint{Frames: 65, Width: 1280, Height: 720, PeakMiB: 29696}
	full := VideoSize{Frames: 121, Width: 1280, Height: 720}
	got, over, ok := PredictPeak([]MeasurePoint{short, medium}, full, 0)
	if !ok || over || got < 31160-50 || got > 31160+50 || got-30413 > 1536 {
		t.Errorf("PredictPeak = %d, %v, %v; want about 31160, within 1.5 GiB of the measured 30413", got, over, ok)
	}
	if _, _, ok := PredictPeak([]MeasurePoint{short}, full, 0); ok {
		t.Error("predicted from one measurement")
	}
	if _, _, ok := PredictPeak([]MeasurePoint{short, medium}, full, 14336); ok {
		t.Error("predicted inside a budget nothing was measured in")
	}
	tiled := medium
	tiled.Fallback = true
	if _, _, ok := PredictPeak([]MeasurePoint{short, tiled}, full, 0); ok {
		t.Error("predicted across a decode that fitted only in smaller pieces")
	}
	// A size stopped for going past the room: it and anything larger will
	// not fit, whatever the line says.
	stopped := MeasurePoint{Frames: 65, Width: 1280, Height: 720, PeakMiB: 30106, Over: true}
	if _, over, _ := PredictPeak([]MeasurePoint{short, stopped}, full, 0); !over {
		t.Error("a size above one stopped for going past the room was not marked over")
	}
	if _, over, _ := PredictPeak([]MeasurePoint{short, stopped}, VideoSize{Frames: 49, Width: 832, Height: 480}, 0); over {
		t.Error("a smaller size than the stopped one was marked over")
	}
}
