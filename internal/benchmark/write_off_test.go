package benchmark

import (
	"errors"
	"strings"
	"testing"
)

// A tensor split whose GPUs cannot reach each other fails identically
// every time. Measuring it once per remaining combination cost a model
// load and a timeout each, for half a stage, to prove the same thing.
func TestWriteOffAbandonsASettingThatAlwaysFails(t *testing.T) {
	w := newWriteOffTracker()
	const nccl = "warmup failed: CUDA error: unhandled system error"

	tensor := map[string]string{"split_mode": "tensor", "ubatch_size": "256"}
	layer := map[string]string{"split_mode": "layer", "ubatch_size": "256"}

	w.recordSuccess(layer)
	if _, skip := w.writeOff(tensor, true); skip {
		t.Fatal("wrote a setting off before it had failed at all")
	}

	w.recordFailure(tensor, nccl)
	if _, skip := w.writeOff(map[string]string{"split_mode": "tensor", "ubatch_size": "512"}, true); skip {
		t.Error("wrote a setting off after a single failure, which can be a fluke")
	}

	w.recordFailure(map[string]string{"split_mode": "tensor", "ubatch_size": "512"}, nccl)
	reason, skip := w.writeOff(map[string]string{"split_mode": "tensor", "ubatch_size": "1024"}, true)
	if !skip {
		t.Fatal("a setting that failed twice the same way was measured again")
	}
	if !strings.Contains(reason, "split_mode = tensor") || !strings.Contains(reason, nccl) {
		t.Errorf("reason = %q, want it to name the setting and the error", reason)
	}

	// The other value of the same field must be unaffected.
	if _, skip := w.writeOff(map[string]string{"split_mode": "layer", "ubatch_size": "1024"}, true); skip {
		t.Error("a working split mode was written off alongside the broken one")
	}
}

// Until something has been measured, a run of failures says the job is
// wrong rather than any one setting, so blaming a setting would put a
// misleading reason on every remaining cell.
func TestWriteOffWaitsForSomethingToWork(t *testing.T) {
	w := newWriteOffTracker()
	cell := map[string]string{"batch_size": "2048"}
	w.recordFailure(cell, "failed to load model: HTTP 404")
	w.recordFailure(cell, "failed to load model: HTTP 404")

	if _, skip := w.writeOff(cell, false); skip {
		t.Error("wrote a setting off although nothing in the job had worked yet")
	}
	if _, skip := w.writeOff(cell, true); !skip {
		t.Error("did not write the setting off once something else had worked")
	}
}

// A value that has worked is never the reason a later cell failed.
func TestWriteOffKeepsASettingThatHasWorked(t *testing.T) {
	w := newWriteOffTracker()
	cell := map[string]string{"ubatch_size": "4096"}
	w.recordSuccess(cell)
	w.recordFailure(cell, "out of memory")
	w.recordFailure(cell, "out of memory")

	if _, skip := w.writeOff(cell, true); skip {
		t.Error("wrote off a setting that had already been measured successfully")
	}
}

// Two failures that say different things do not convict the value: the
// errors have to agree before it is the value itself that is at fault.
func TestWriteOffNeedsTheSameErrorTwice(t *testing.T) {
	w := newWriteOffTracker()
	cell := map[string]string{"split_mode": "tensor"}
	w.recordFailure(cell, "out of memory")
	w.recordFailure(cell, "the router did not come back")

	if _, skip := w.writeOff(cell, true); skip {
		t.Error("wrote a setting off on two unrelated failures")
	}
	w.recordFailure(cell, "the router did not come back")
	if _, skip := w.writeOff(cell, true); !skip {
		t.Error("did not write off after the same error twice in a row")
	}
}

// End to end through the cell loop: a sweep where one value of one axis
// cannot run must abandon that value after it has failed twice, not
// reload the model for every remaining combination carrying it.
func TestJobSkipsRemainingCellsOfASettingThatKeepsFailing(t *testing.T) {
	router := newFakeRouter(t)
	env := &fakeEnv{
		routerURL: router.URL,
		saved:     ConfigSnapshot{GPULayers: 999, ContextSize: 8192, Threads: 8},
		applyErrFor: func(cfg ConfigSnapshot) error {
			if cfg.SplitMode == "tensor" {
				return errors.New("CUDA error: unhandled system error")
			}
			return nil
		},
	}

	sweeps := []SweepAxis{
		{Field: "split_mode", Values: []string{"layer", "tensor"}},
		{Field: "ubatch_size", Values: []string{"256", "512", "1024", "2048"}},
	}
	job := BenchmarkJob{
		ID: "job-write-off", Name: "test", Kind: JobKindBatch,
		ModelIDs: []string{"m"}, BuildIDs: []string{"b"}, Presets: []string{"internal-quick"},
		Sweeps: sweeps,
		Cells:  ExpandCellsWithSweeps([]string{"m"}, []string{"b"}, []string{"internal-quick"}, sweeps),
	}

	done, _ := runJob(t, job, env)

	var tensorFailed, tensorSkipped, layerCompleted int
	for _, c := range done.Cells {
		isTensor := c.SweepValues["split_mode"] == "tensor"
		switch {
		case isTensor && c.Status == CellStatusFailed:
			tensorFailed++
		case isTensor && c.Status == CellStatusSkipped:
			tensorSkipped++
			if !strings.Contains(c.Error, "split_mode = tensor") {
				t.Errorf("skipped cell does not say why: %q", c.Error)
			}
		case !isTensor && c.Status == CellStatusCompleted:
			layerCompleted++
		}
	}

	if layerCompleted != 4 {
		t.Errorf("%d of 4 layer-split cells completed", layerCompleted)
	}
	if tensorFailed != failuresBeforeWriteOff {
		t.Errorf("tensor cells measured %d times, want %d before being written off",
			tensorFailed, failuresBeforeWriteOff)
	}
	if want := 4 - failuresBeforeWriteOff; tensorSkipped != want {
		t.Errorf("%d tensor cells skipped, want %d", tensorSkipped, want)
	}
	// The job still counts as done: something was measured.
	if done.Status != JobStatusCompleted {
		t.Errorf("job status = %s, want completed", done.Status)
	}
}

// Two cells that fail for one reason can quote different lines from the
// server log — the buffer that did not fit is a different size under a
// different split mode. That difference must not make them look like
// two unrelated failures.
func TestWriteOffComparesErrorsWithoutTheQuotedServerLine(t *testing.T) {
	w := newWriteOffTracker()
	const head = "warmup failed after retries: HTTP 500: model failed to load"
	cell := map[string]string{"ubatch_size": "1024"}
	w.recordFailure(cell, withServerReason(head, []string{"allocating 5120.00 MiB on device 2: cudaMalloc failed: out of memory"}))
	w.recordFailure(cell, withServerReason(head, []string{"allocating 3072.00 MiB on device 0: cudaMalloc failed: out of memory"}))

	reason, skip := w.writeOff(cell, true)
	if !skip {
		t.Fatal("two failures with the same error and different quoted lines were treated as unrelated")
	}
	if !strings.Contains(reason, "out of memory") {
		t.Errorf("reason = %q, want it to keep what the server reported", reason)
	}
}

// Autotune measures a draft model alone and then paired with each n-gram
// assist. Every pairing is a different swept value, so a draft model that
// cannot load used to fail twice per pairing — twelve loads and timeouts
// on the machine this is from — before each was abandoned in turn.
func TestWriteOffAbandonsADraftFileAcrossItsPairings(t *testing.T) {
	w := newWriteOffTracker()
	const loadFailed = "warmup failed after retries: HTTP 500: model failed to load"
	const file = "draft_model=org--small--Q4"

	spec := func(v string) map[string]string {
		return map[string]string{"spec_type": v, "ubatch_size": "256"}
	}

	// What works on this machine: no speculative decoding, an n-gram
	// assist alone, and MTP from the model's own layers.
	w.recordSuccess(spec("none"))
	w.recordSuccess(spec("ngram-mod:assist_n_max=64"))
	w.recordSuccess(spec("draft-mtp:draft_max=6"))

	// The draft model alone, once per workload.
	alone := spec("draft:draft_max=16," + file)
	w.recordFailure(alone, loadFailed)
	w.recordFailure(alone, loadFailed)

	paired := spec("draft+ngram-mod:assist_n_max=64,draft_max=16," + file)
	reason, skip := w.writeOff(paired, true)
	if !skip {
		t.Fatal("a pairing of a draft file that never loaded was measured again")
	}
	if !strings.Contains(reason, "org--small--Q4") || !strings.Contains(reason, loadFailed) {
		t.Errorf("reason = %q, want it to name the draft file and the error", reason)
	}

	// Everything that does not load that file is unaffected.
	for _, v := range []string{
		"draft-mtp+ngram-mod:assist_n_max=64,draft_max=6",
		"ngram-simple:assist_size_n=12",
		"draft+ngram-mod:assist_n_max=64,draft_max=16,draft_model=org--other--Q4",
	} {
		if _, skip := w.writeOff(spec(v), true); skip {
			t.Errorf("%q was written off alongside a draft file it does not load", v)
		}
	}
}

// A draft file that has loaded once is not the reason a later cell
// failed, whatever it was paired with.
func TestWriteOffKeepsADraftFileThatHasWorked(t *testing.T) {
	w := newWriteOffTracker()
	const file = "draft_model=org--small--Q4"
	w.recordSuccess(map[string]string{"spec_type": "draft:draft_max=16," + file})

	paired := map[string]string{"spec_type": "draft+ngram-mod:assist_n_max=64,draft_max=16," + file}
	w.recordFailure(paired, "out of memory")
	if _, skip := w.writeOff(map[string]string{"spec_type": "draft+ngram-cache:draft_max=16," + file}, true); skip {
		t.Error("wrote off a draft file that had already been measured successfully")
	}
}
