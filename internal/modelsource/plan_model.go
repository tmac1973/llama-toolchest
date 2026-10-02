package modelsource

import "github.com/tmac1973/llama-toolchest/internal/models"

// MetaProbeFile picks the file whose header describes a repository: the
// largest model file. Every quant carries the same description, and a
// header read costs the same whatever the file's size, so the largest is
// chosen because it is certainly the main model — a draft model or MTP
// head kept in the same repository is always smaller.
func MetaProbeFile(files []File) (File, bool) {
	var best File
	found := false
	for _, f := range files {
		if f.IsMMProj || f.Size <= 0 {
			continue
		}
		if !found || f.Size > best.Size {
			best, found = f, true
		}
	}
	return best, found
}

// Bits per weight a real quant of a model can have: from the 1.x-bit
// IQ1 quants to F32 with a little room. A file outside this range against
// a repository's parameter count is some other model.
const (
	minPlausibleBPW = 1.0
	maxPlausibleBPW = 34
)

// BitsPerWeight is a file's average stored bits per parameter, or 0 when
// the parameter count is not known.
func BitsPerWeight(f File, params int64) float64 {
	if params <= 0 {
		return 0
	}
	return float64(f.Size) * 8 / float64(params)
}

// PlausibleFile reports whether a file plausibly holds the model the
// parameter count describes. A repository sometimes carries a small draft
// model beside the main one; against the main model's parameter count its
// bits per weight come out far below any real quant, and the shared
// description would misdescribe it.
func PlausibleFile(f File, params int64) bool {
	if params <= 0 {
		return true
	}
	bpw := BitsPerWeight(f, params)
	return bpw >= minPlausibleBPW && bpw <= maxPlausibleBPW
}

// PlanModel builds the Model the fit planner needs for one file of a
// repository, from the repository's shared description (see ProbeMeta).
// Tensor sizes are derived for this file (see GGUFMeta.DerivedFor); a
// per-layer embedding table measured by ProbePLE replaces the derived one.
func PlanModel(meta *models.GGUFMeta, f File, params int64) *models.Model {
	d := meta.DerivedFor(f.Size, params)
	if f.StreamProbed {
		d.PLEBytes = f.StreamedBytes
	}
	m := &models.Model{SizeBytes: f.Size}
	d.ApplyTo(m)
	return m
}
