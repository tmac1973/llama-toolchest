package autoconfig

import (
	"context"
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/tmac1973/llama-toolchest/internal/models"
)

// Checking a proposal by loading it.
//
// The hardware fit is an estimate, and it compares the memory a model
// needs with the memory of every GPU added together. llama.cpp does not
// fill the cards evenly. On the machine this is from — three 16 GiB cards
// and a 27B model with its own draft layers — the plan fitted in total
// with 5 GiB to spare, and the third card ran out: it held the largest
// share of the weights and the whole draft context, while the second
// card had 3 GiB free. The profile that came out of it could not answer
// a single request.
//
// How llama.cpp lays a model out across cards is its own business and
// changes between versions, so the answer is not a better guess. It is to
// load the proposal once, send it a request, and adjust it when that
// fails, before anyone is asked to save it.

// Check is what a test load of a proposal found.
type Check struct {
	// OK means the model loaded and answered a request.
	OK bool
	// Reason is what went wrong, in the words llama-server used.
	Reason string
	// OutOfMemory says the failure was a GPU running out of memory,
	// which is the only kind of failure a smaller plan can fix.
	OutOfMemory bool
	// Device is the GPU that ran out, as llama.cpp numbers them, or -1
	// when the failure did not name one.
	Device int
}

// VerifyFunc loads cfg for the model and sends it one request. An error
// means the check itself could not be run — the GPU was busy, the server
// would not start — and says nothing about the settings.
type VerifyFunc func(ctx context.Context, modelID string, cfg models.ModelConfig) (Check, error)

// Verification statuses.
const (
	VerifyPassed  = "passed"
	VerifyFailed  = "failed"
	VerifySkipped = "skipped"
)

// Verification is what the review screen says about the test load.
type Verification struct {
	// Status is "" when no check was asked for, else one of the
	// Verify* constants.
	Status string
	// Attempts is how many test loads were run.
	Attempts int
	// Adjusted counts the settings changed to make the proposal run.
	Adjusted int
	// Reason is why the check failed or could not be run.
	Reason string
}

const (
	// maxChecks bounds the test loads of one run. A failed load costs
	// about two minutes, so this is also what bounds how long a run can
	// take on a machine where nothing fits.
	maxChecks = 6
	// maxSplitMoves is how many times layers are moved off a full card
	// before giving up on moving them. Two covers the case where the
	// first move fills the card they were moved to.
	maxSplitMoves = 2
	// splitShare is what a full card's share of the layers is multiplied
	// by. 0.85 moved about 1 GiB off the full card on the machine this
	// is from, which is the size of what did not fit.
	splitShare = 0.85
	// splitUnit is the share each card starts with when the split is
	// written out for the first time. Whole numbers, because a person
	// will read them in the config form.
	splitUnit = 20
	// checkedUBatch is the prompt batch tried when the default does not
	// fit: the smallest one Autotune measures.
	checkedUBatch = 256
	// minCheckedContext is the smallest context the check will reduce
	// to, the same floor the hardware fit has.
	minCheckedContext = 4096
)

// originTestLoad marks a note for a setting the test load changed.
const originTestLoad = "test load"

// verifyProposal loads the proposal and, while it runs out of memory,
// adjusts it and loads it again. It returns the settings that ran (or the
// proposal unchanged when none did), the notes that explain each change,
// and the outcome for the review screen.
func verifyProposal(ctx context.Context, d Deps, m *models.Model, proposed models.ModelConfig, progress func(string)) (models.ModelConfig, []models.ProfileNote, Verification) {
	cfg := proposed
	var notes []models.ProfileNote
	v := Verification{}
	splitMoves := 0

	for v.Attempts < maxChecks {
		if v.Attempts == 0 {
			progress("Loading the proposed settings to check that they run")
		} else {
			progress(fmt.Sprintf("Those settings ran out of memory. Trying again with an adjustment (test load %d)", v.Attempts+1))
		}
		chk, err := d.Verify(ctx, m.ID, cfg)
		if err != nil {
			v.Status, v.Reason = VerifySkipped, err.Error()
			return proposed, nil, v
		}
		v.Attempts++
		if chk.OK {
			v.Status, v.Adjusted = VerifyPassed, len(notes)
			return cfg, notes, v
		}
		v.Reason = chk.Reason
		if !chk.OutOfMemory {
			// Not a problem a smaller plan solves: a file that will not
			// parse, a build that cannot run this model.
			v.Status = VerifyFailed
			return proposed, nil, v
		}
		next, note, ok := stepDown(cfg, chk, len(d.Hardware.GPUs), &splitMoves)
		if !ok {
			break
		}
		cfg = next
		notes = replaceNote(notes, note)
	}
	v.Status = VerifyFailed
	return proposed, nil, v
}

// replaceNote adds a note, replacing an earlier one for the same setting:
// a context reduced twice is explained once, by its final size.
func replaceNote(notes []models.ProfileNote, n models.ProfileNote) []models.ProfileNote {
	for i := range notes {
		if notes[i].Field == n.Field {
			notes[i] = n
			return notes
		}
	}
	return append(notes, n)
}

// stepDown returns the next smaller plan to try after cfg ran out of
// memory, and the note that explains the change. The steps are ordered by
// what they cost the user, least first:
//
//  1. Move layers off the card that ran out. Costs nothing: the same
//     model, context and speed, laid out differently.
//  2. A smaller prompt batch. Costs some prompt speed at most, and frees
//     working memory on every card.
//  3. Half the context. Costs what the model can hold at once, so it is
//     last.
func stepDown(cfg models.ModelConfig, chk Check, numGPUs int, splitMoves *int) (models.ModelConfig, models.ProfileNote, bool) {
	if *splitMoves < maxSplitMoves {
		if next, device, ok := moveLayersOff(cfg, chk.Device, numGPUs); ok {
			*splitMoves++
			return next, models.ProfileNote{
				Field:  "gpu_assign",
				Origin: originTestLoad,
				Reason: fmt.Sprintf("A test load ran out of memory on GPU %d while other GPUs had room. Fewer layers are placed on GPU %d (split %s) so that the model fits. This does not change the context size or the speed.",
					device, device, next.TensorSplit),
			}, true
		}
	}
	if cfg.EffectiveUBatchSize() > checkedUBatch {
		next := cfg
		next.UBatchSize = checkedUBatch
		if next.BatchSize == 0 {
			next.BatchSize = models.DefaultBatchSize
		}
		return next, models.ProfileNote{
			Field:  "ubatch_size",
			Origin: originTestLoad,
			Reason: fmt.Sprintf("A test load ran out of memory with a prompt batch of %d. A prompt batch of %d needs less working memory on each GPU. Autotune can measure whether a larger one fits and is faster.",
				cfg.EffectiveUBatchSize(), checkedUBatch),
		}, true
	}
	if half := cfg.ContextSize / 2; half >= minCheckedContext {
		next := cfg
		next.ContextSize = half
		return next, models.ProfileNote{
			Field:  "context_size",
			Origin: originTestLoad,
			Reason: fmt.Sprintf("A test load ran out of memory at a larger context; reduced to %s tokens, the largest that loaded and ran.", groupThousands(half)),
		}, true
	}
	return cfg, models.ProfileNote{}, false
}

// moveLayersOff gives the card that ran out a smaller share of the
// model's layers, and reports which card that was. It applies only to a
// layer split over every card in the machine: a tensor-parallel split
// puts the same share on every card by design, and one card has nowhere
// to move anything.
//
// When the failure did not name a card, the last one is assumed for a
// model that drafts, because llama.cpp puts the whole draft context on
// the card holding the model's last layers. Without a draft context there
// is no reason to prefer any card, and nothing is moved.
func moveLayersOff(cfg models.ModelConfig, device, numGPUs int) (models.ModelConfig, int, bool) {
	if cfg.SplitMode == "tensor" || strings.HasPrefix(cfg.GPUAssign, "tensor") || numGPUs < 2 {
		return cfg, 0, false
	}
	shares := splitShares(cfg, numGPUs)
	used := 0
	last := -1
	for i, s := range shares {
		if s > 0 {
			used++
			last = i
		}
	}
	// Every card has to be in use. A config that leaves a card out does
	// so with a device list, which keeps that card out of the process
	// altogether; a custom split cannot say that, and would bring it
	// back with a share of nothing.
	if used < 2 || used != len(shares) {
		return cfg, 0, false
	}
	if device < 0 || device >= len(shares) || shares[device] <= 0 {
		if !models.IsDraftMode(cfg.SpecType) {
			return cfg, 0, false
		}
		device = last
	}
	reduced := math.Round(shares[device] * splitShare)
	if reduced < 1 || reduced == shares[device] {
		return cfg, 0, false
	}
	shares[device] = reduced

	parts := make([]string, len(shares))
	for i, s := range shares {
		parts[i] = strconv.Itoa(int(s))
	}
	next := cfg
	// "custom" is the assignment that keeps a hand-written split: every
	// other value is turned back into an even one when the config is
	// saved.
	next.GPUAssign = "custom"
	next.TensorSplit = strings.Join(parts, ",")
	if next.SplitMode == "" {
		next.SplitMode = "layer"
	}
	return next, device, true
}

// splitShares returns each card's share of the layers as whole numbers.
// A config with no split written out, or one that only switches cards on
// and off ("1,1,0"), is an even split, written here as splitUnit for
// each card in use so that a share can be reduced by less than a whole.
func splitShares(cfg models.ModelConfig, numGPUs int) []float64 {
	shares := make([]float64, numGPUs)
	ts := strings.TrimSpace(cfg.TensorSplit)
	if ts == "" {
		for i := range shares {
			shares[i] = splitUnit
		}
		return shares
	}
	onOff := true
	for i, part := range strings.Split(ts, ",") {
		if i >= numGPUs {
			break
		}
		v, err := strconv.ParseFloat(strings.TrimSpace(part), 64)
		if err != nil || v < 0 {
			v = 0
		}
		shares[i] = v
		if v != 0 && v != 1 {
			onOff = false
		}
	}
	if onOff {
		for i := range shares {
			shares[i] *= splitUnit
		}
	}
	return shares
}

// groupThousands writes n with thousands separators.
func groupThousands(n int) string {
	s := strconv.Itoa(n)
	for i := len(s) - 3; i > 0; i -= 3 {
		s = s[:i] + "," + s[i:]
	}
	return s
}
