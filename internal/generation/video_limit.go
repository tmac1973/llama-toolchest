package generation

// SuggestReservation is the reservation a measured peak suggests: 10%
// more, rounded up to the next 512 MiB.
func SuggestReservation(peakMiB int) int {
	n := peakMiB * 11 / 10
	return (n + 511) / 512 * 512
}

// PixelFrames is how much video a size is: frames × width × height. Video
// memory grows with it, so a request is compared with an instance's
// measured size by this one number, whatever its shape.
func PixelFrames(frames, width, height int) int64 {
	return int64(frames) * int64(width) * int64(height)
}

// VideoPresets are the sizes Measure offers a video set, smallest first:
// the current catalog entry's, when the set's entry is still in the
// catalog (so sets downloaded before presets existed get them too); else
// the set's own copy; else one "Short" preset at the set's test size.
func VideoPresets(set ModelSet) []VideoSize {
	if set.CatalogID != "" {
		if e, ok := LookupCatalog(set.CatalogID); ok && len(e.VideoSizes) > 0 {
			return e.VideoSizes
		}
	}
	if len(set.VideoSizes) > 0 {
		return set.VideoSizes
	}
	return []VideoSize{{Label: "Short", Frames: set.DefaultFrames, Width: set.DefaultWidth, Height: set.DefaultHeight}}
}

// VideoLimit is the largest video an instance may make: the size it was
// last measured at, with measured true, while its reservation still covers
// what that measurement suggested; otherwise the set's test size, with
// measured false (never measured at a larger size, or the reservation set
// by hand below what the measurement needs). Label names the matching
// preset, or "custom size".
func VideoLimit(inst Instance, set ModelSet) (size VideoSize, measured bool) {
	presets := VideoPresets(set)
	if inst.MeasuredFrames > 0 && inst.MeasuredWidth > 0 && inst.MeasuredHeight > 0 &&
		inst.MeasuredSuggestedMiB > 0 && inst.ReservedMiB >= inst.MeasuredSuggestedMiB {
		size = VideoSize{Frames: inst.MeasuredFrames, Width: inst.MeasuredWidth, Height: inst.MeasuredHeight}
		measured = true
	} else {
		size = VideoSize{Frames: set.DefaultFrames, Width: set.DefaultWidth, Height: set.DefaultHeight}
	}
	size.Label = PresetLabel(presets, size)
	return size, measured
}

// PresetLabel is the label of the preset with exactly this size, or
// "custom size".
func PresetLabel(presets []VideoSize, size VideoSize) string {
	for _, p := range presets {
		if p.Frames == size.Frames && p.Width == size.Width && p.Height == size.Height {
			return p.Label
		}
	}
	return "custom size"
}

// CoveringPreset is the smallest preset with at least this much video, or
// false when none has.
func CoveringPreset(presets []VideoSize, frames, width, height int) (VideoSize, bool) {
	need := PixelFrames(frames, width, height)
	var best VideoSize
	found := false
	for _, p := range presets {
		pf := PixelFrames(p.Frames, p.Width, p.Height)
		if pf >= need && (!found || pf < PixelFrames(best.Frames, best.Width, best.Height)) {
			best, found = p, true
		}
	}
	return best, found
}

// HasHighNoiseModel says the set is a Wan2.2 A14B-style pair, with a
// high-noise expert besides the low-noise one.
func (m ModelSet) HasHighNoiseModel() bool {
	for _, c := range m.Components {
		if c.Slot == "high_noise_diffusion_model" {
			return true
		}
	}
	return false
}

// maxMeasurePoints is how many measurements an instance keeps.
const maxMeasurePoints = 6

// AddMeasurePoint adds p to points, replacing an earlier measurement of
// the same size inside the same budget, and dropping the oldest beyond
// maxMeasurePoints.
func AddMeasurePoint(points []MeasurePoint, p MeasurePoint) []MeasurePoint {
	out := make([]MeasurePoint, 0, len(points)+1)
	for _, q := range points {
		if q.Frames != p.Frames || q.Width != p.Width || q.Height != p.Height || q.BudgetMiB != p.BudgetMiB {
			out = append(out, q)
		}
	}
	out = append(out, p)
	return out[max(0, len(out)-maxMeasurePoints):]
}

// PredictPeak predicts the peak of a video of size, from an instance's
// measurements inside budget (0 for the whole room). A peak has two parts:
// the weights, which do not change with the video's size, and working
// memory, which grows with frames × width × height (PixelFrames). So a
// line through two measurements at different sizes gives the rest: the
// two largest such sizes, both measured to the end, whose decodes fitted
// the same way. ok is false without two.
//
// A larger size measured and stopped for going past the room is known to
// need at least that much: over is true for any size at or above it.
func PredictPeak(points []MeasurePoint, size VideoSize, budget int) (peak int, over, ok bool) {
	pf := PixelFrames(size.Frames, size.Width, size.Height)
	var a, b *MeasurePoint
	for i := range points {
		p := &points[i]
		if p.BudgetMiB != budget {
			continue
		}
		if p.Over {
			if pf >= PixelFrames(p.Frames, p.Width, p.Height) {
				over = true
			}
			continue
		}
		switch {
		case a == nil || PixelFrames(p.Frames, p.Width, p.Height) > PixelFrames(a.Frames, a.Width, a.Height):
			a, b = p, a
		case PixelFrames(p.Frames, p.Width, p.Height) != PixelFrames(a.Frames, a.Width, a.Height) &&
			(b == nil || PixelFrames(p.Frames, p.Width, p.Height) > PixelFrames(b.Frames, b.Width, b.Height)):
			b = p
		}
	}
	if a == nil || b == nil || a.Fallback != b.Fallback {
		return 0, over, false
	}
	pa, pb := PixelFrames(a.Frames, a.Width, a.Height), PixelFrames(b.Frames, b.Width, b.Height)
	if pa == pb {
		return 0, over, false
	}
	rate := float64(a.PeakMiB-b.PeakMiB) / float64(pa-pb)
	peak = a.PeakMiB + int(rate*float64(pf-pa)+0.5)
	return max(peak, 0), over, true
}
