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
