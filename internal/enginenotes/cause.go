package enginenotes

import (
	"slices"
	"strings"
)

// Cause is why a crash or a failed load happened, as far as the lines
// llama-server wrote can tell. It is worked out when the notes are read,
// not saved, so events recorded before a cause existed get one too.
type Cause string

const (
	// CauseGPUFault is a CUDA or HIP fault: an illegal or misaligned
	// memory access, or a kernel that failed to launch. A setting does
	// not cause these; a bug in llama.cpp, the driver or the hardware
	// does. They are the ones worth a look wherever they happen.
	CauseGPUFault Cause = "gpu_fault"
	// CauseOutOfVRAM is a GPU that ran out of memory, at load or later.
	CauseOutOfVRAM Cause = "out_of_vram"
	// CauseOutOfRAM is system memory that ran out, or a process killed
	// for it.
	CauseOutOfRAM Cause = "out_of_ram"
	// CauseUnsupported is a setting llama.cpp refused: a failed
	// assertion, or an option it does not support. It fails the same
	// way every time.
	CauseUnsupported Cause = "unsupported"
	// CauseOther is anything the lines do not explain.
	CauseOther Cause = "other"
)

// causeLabels name each cause as the page shows it.
var causeLabels = map[Cause]string{
	CauseGPUFault:    "GPU fault",
	CauseOutOfVRAM:   "out of VRAM",
	CauseOutOfRAM:    "out of system memory",
	CauseUnsupported: "setting not supported",
	CauseOther:       "other",
}

// Label is the cause as the page shows it.
func (c Cause) Label() string {
	if l, ok := causeLabels[c]; ok {
		return l
	}
	return string(c)
}

// Markers for each cause, matched in lower case. The order in CauseOf
// matters: a GPU fault can make later allocations fail, so it is checked
// before memory.
var (
	gpuFaultMarkers = []string{
		"illegal memory access",
		"misaligned address",
		"unspecified launch failure",
		"illegal address",
		"memory access fault", // ROCm
		"hipErrorIllegalAddress",
	}
	ramMarkers = []string{
		"std::bad_alloc",
		"cannot allocate memory",
	}
	vramMarkers = []string{
		"out of memory",
		"cannot make enough memory available", // sd.cpp's model manager
		"cudamalloc failed",
		"hipmalloc failed",
		"failed to allocate",
		"erroroutofmemory",
	}
	// computeFailedMarker is ggml's line for an operation that failed on
	// the GPU ("ggml_cuda_compute_forward: CONCAT failed"). The line
	// after it says why; when that is lost, a failed operation with no
	// memory error is a fault.
	computeFailedMarker = "compute_forward:"
	unsupportedMarkers  = []string{
		"ggml_assert(",
		"not supported",
		"unsupported",
		"invalid argument:",
		"error: unknown argument",
	}
)

// killedStatuses are the exit statuses of a process ended by SIGKILL,
// which on Linux is nearly always the kernel's out-of-memory killer
// when nobody asked for the stop.
var killedStatuses = map[int]bool{137: true, -9: true, 9: true}

// computeFailed reports whether line is ggml's line for a failed GPU
// operation.
func computeFailed(line string) bool {
	lower := strings.ToLower(line)
	i := strings.Index(lower, computeFailedMarker)
	return i >= 0 && strings.Contains(lower[i:], " failed")
}

// retryMarker starts the line sd.cpp writes when a step ran out of
// memory and it tries again with less ("VAE decode ran out of memory;
// retrying with temporal tiling").
const retryMarker = "retrying with"

// afterRetry returns the lines after the last retry line, and whether
// there was one. The memory errors before it were recovered from.
func afterRetry(lines []string) ([]string, bool) {
	for i := len(lines) - 1; i >= 0; i-- {
		if strings.Contains(strings.ToLower(lines[i]), retryMarker) {
			return lines[i+1:], true
		}
	}
	return lines, false
}

// CauseOf works out why a failure happened. Events that are not failures
// have no cause.
//
// For sd-server only the lines after its last retry count (afterRetry).
// When sd.cpp retried a step for memory and then aborted, and the lines
// after the retry show no fault, the abort is the end of running out of
// memory: out of VRAM. A fault after the retry is still a fault.
func CauseOf(e Event) Cause {
	if !e.IsFailure() {
		return ""
	}
	errLines, tail, retried := e.ErrorLines, e.Tail, false
	if e.Source == SourceSDServer {
		var r1, r2 bool
		errLines, r1 = afterRetry(errLines)
		tail, r2 = afterRetry(tail)
		retried = r1 || r2
	}
	errs := append(append([]string{}, errLines...), e.RouterError)
	all := append(append([]string{}, errs...), tail...)
	// The tail holds ordinary lines too, so the words that also turn up
	// in those ("not supported") are looked for in the error lines only.
	in := func(lines []string, markers []string) bool {
		for _, l := range lines {
			lower := strings.ToLower(l)
			for _, m := range markers {
				if strings.Contains(lower, strings.ToLower(m)) {
					return true
				}
			}
		}
		return false
	}
	switch {
	case in(all, gpuFaultMarkers):
		return CauseGPUFault
	case in(all, ramMarkers):
		return CauseOutOfRAM
	case in(all, vramMarkers):
		return CauseOutOfVRAM
	case killedStatuses[e.ExitStatus]:
		return CauseOutOfRAM
	case in(errs, unsupportedMarkers):
		return CauseUnsupported
	case retried:
		return CauseOutOfVRAM
	case slices.ContainsFunc(all, computeFailed):
		return CauseGPUFault
	}
	return CauseOther
}

// Expected reports whether a failure is one a job is there to find: a
// setting that did not fit or that llama.cpp refused, while a benchmark
// or Autotune tried it, or a size that did not fit while Measure tried
// it on a generation instance. These are counted but not warned about.
func Expected(e Event, c Cause) bool {
	if e.JobID == "" {
		return false
	}
	switch c {
	case CauseOutOfVRAM, CauseOutOfRAM, CauseUnsupported:
		return true
	}
	return false
}
