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

// nominalBPW is the usual bits per weight of a quant, for working out a
// model's parameter count from a file when the host's count is wrong.
// Only quants common as a repo's largest file are listed.
var nominalBPW = map[string]float64{
	"F32": 32, "F16": 16, "BF16": 16,
	"Q8_0": 8.5, "UD_Q8_K_XL": 8.5, "Q8_K_XL": 8.5,
	"Q6_K": 6.56, "Q6_K_L": 6.56, "UD_Q6_K_XL": 6.56,
	"Q5_K_M": 5.69, "Q5_K_S": 5.54, "Q4_K_M": 4.85,
}

// ParamsFromFile estimates a model's parameter count from one file and
// its quant, or 0 when the quant's usual size is not known.
func ParamsFromFile(f File) int64 {
	bpw, ok := nominalBPW[f.Quant]
	if !ok {
		bpw, ok = nominalBPW[models.ParseQuant(f.Filename)]
	}
	if !ok || f.Size <= 0 {
		return 0
	}
	return int64(float64(f.Size) * 8 / bpw)
}

// RepoParams is a repository's parameter count: the host's, when its
// largest model file is a plausible quant of a model that size, and
// otherwise one worked out from that file (0 when it cannot be).
//
// HuggingFace's count comes from its summary of one GGUF file of its
// choosing, which is sometimes the image reader or a draft model: a
// 35B repository reported as 0.45B parameters of architecture "clip".
// Believed, it made every real file look implausible.
func RepoParams(files []File, hostParams int64) int64 {
	probe, ok := MetaProbeFile(files)
	if !ok {
		return hostParams
	}
	if hostParams > 0 && PlausibleFile(probe, hostParams) {
		return hostParams
	}
	return ParamsFromFile(probe)
}
