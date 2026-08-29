// Package memreport parses the per-buffer memory report llama.cpp prints
// while loading a model, which is the only place the split between model
// weights, KV cache, recurrent state and compute buffers is stated.
//
// The report appears at log verbosity 4 and above (LLAMA_ARG_LOG_VERBOSITY,
// a curated runtime-env option). At the default of 3 llama.cpp emits
// warnings from its core library but not these INFO lines, so nothing here
// will match. See plan/ple-vram-findings.md for why the split matters: the
// VRAM estimate is currently accurate on one model only because a large
// over-count of weights cancels a large under-count of everything else,
// and separating the terms is what makes that testable.
package memreport

import (
	"regexp"
	"strconv"
	"strings"
)

// Kind is the sort of allocation a buffer holds.
type Kind string

const (
	KindModel     Kind = "model"     // weights
	KindKV        Kind = "kv"        // attention KV cache
	KindRecurrent Kind = "recurrent" // recurrent/linear-attention state
	KindCompute   Kind = "compute"   // scratch for the graph
	KindOutput    Kind = "output"    // logits
)

// Entry is one "<device> <kind> buffer size = N MiB" observation.
type Entry struct {
	// Instance is the child process id llama.cpp's router prefixes onto
	// lines from the model instance that is doing the loading. Empty for
	// lines the router itself emitted. Reports from concurrently loading
	// models interleave in one stream, so this is what separates them.
	Instance string
	// Category is the llama.cpp function that logged the line
	// (load_tensors, llama_kv_cache, sched_reserve, ...). Kept because
	// the same Kind is reported by more than one of them.
	Category string
	Device   string
	Kind     Kind
	MiB      float64
}

// OnGPU reports whether the device holds accelerator memory.
//
// Host buffers are named for the CPU: "CPU", "CPU_Mapped" (weights left
// in the mmap rather than copied), and the pinned staging buffers each
// backend names "<backend>_Host" — ROCm_Host, CUDA_Host. Everything else
// is a device: ROCm0, CUDA1, Vulkan0, and "Meta()", which is what
// llama.cpp calls the aggregate when one logical device spans several
// cards.
func (e Entry) OnGPU() bool { return !isHostDevice(e.Device) }

func isHostDevice(device string) bool {
	return device == "CPU" ||
		strings.HasPrefix(device, "CPU_") ||
		strings.HasSuffix(device, "_Host")
}

// lineRE matches an optional "[pid] " prefix, llama.cpp's timestamp and
// level, the logging function, and then the buffer report itself. The
// device name is deliberately \S+ rather than a list: "Meta()" carries
// parentheses and backend names grow, so an allowlist would silently
// drop the very numbers this exists to collect.
var lineRE = regexp.MustCompile(
	`^(?:\[(\d+)\] )?[\d.]+ [A-Z] (\w+):\s+(\S+)\s+(model|KV|compute|output)\s+buffer size\s*=\s*([\d.]+)\s*MiB`)

// recurrentRE is separate because llama.cpp spells recurrent state "RS"
// in the buffer line while naming the function llama_memory_recurrent.
var recurrentRE = regexp.MustCompile(
	`^(?:\[(\d+)\] )?[\d.]+ [A-Z] (\w+):\s+(\S+)\s+RS\s+buffer size\s*=\s*([\d.]+)\s*MiB`)

// ParseLine extracts one entry, reporting false for any line that is not
// a buffer report — which is the overwhelming majority of the stream.
func ParseLine(line string) (Entry, bool) {
	line = strings.TrimPrefix(line, "data: ")
	if !strings.Contains(line, "buffer size") {
		return Entry{}, false
	}
	if m := lineRE.FindStringSubmatch(line); m != nil {
		mib, err := strconv.ParseFloat(m[5], 64)
		if err != nil {
			return Entry{}, false
		}
		kind := KindModel
		switch m[4] {
		case "KV":
			kind = KindKV
		case "compute":
			kind = KindCompute
		case "output":
			kind = KindOutput
		}
		return Entry{Instance: m[1], Category: m[2], Device: m[3], Kind: kind, MiB: mib}, true
	}
	if m := recurrentRE.FindStringSubmatch(line); m != nil {
		mib, err := strconv.ParseFloat(m[4], 64)
		if err != nil {
			return Entry{}, false
		}
		return Entry{Instance: m[1], Category: m[2], Device: m[3], Kind: KindRecurrent, MiB: mib}, true
	}
	return Entry{}, false
}

// Report collects the entries seen for one model load.
type Report struct {
	Entries []Entry
}

// Add parses a log line and keeps it when it is a buffer report.
func (r *Report) Add(line string) bool {
	e, ok := ParseLine(line)
	if ok {
		r.Entries = append(r.Entries, e)
	}
	return ok
}

// ForInstance returns the entries a single model instance logged. The
// router interleaves output from every child, so totals taken without
// this are the sum of whatever happened to be loading.
func (r Report) ForInstance(pid string) Report {
	var out Report
	for _, e := range r.Entries {
		if e.Instance == pid {
			out.Entries = append(out.Entries, e)
		}
	}
	return out
}

// Instances lists the child process ids present, in first-seen order.
func (r Report) Instances() []string {
	seen := map[string]bool{}
	var out []string
	for _, e := range r.Entries {
		if !seen[e.Instance] {
			seen[e.Instance] = true
			out = append(out, e.Instance)
		}
	}
	return out
}

// SumMiB totals every entry matching the filter.
//
// Summing is right for weights, KV and recurrent state: each line is a
// distinct allocation on a distinct device. It is NOT known to be right
// for compute buffers. llama.cpp reserves more than once per load — a
// hybrid model reserves separately for each memory module — and whether
// those reservations are separate allocations or one buffer resized to
// the largest graph is not something the log states. MaxComputeMiB
// exists for the other reading; plan/ple-vram-findings.md step 3 settles
// which by comparing both against measured VRAM.
func (r Report) SumMiB(match func(Entry) bool) float64 {
	var total float64
	for _, e := range r.Entries {
		if match == nil || match(e) {
			total += e.MiB
		}
	}
	return total
}

// ByKind totals each kind, summing.
func (r Report) ByKind() map[Kind]float64 {
	out := map[Kind]float64{}
	for _, e := range r.Entries {
		out[e.Kind] += e.MiB
	}
	return out
}

// ByDevice totals each device, summing.
func (r Report) ByDevice() map[string]float64 {
	out := map[string]float64{}
	for _, e := range r.Entries {
		out[e.Device] += e.MiB
	}
	return out
}

// MaxComputeMiB totals the largest compute reservation seen per device,
// the reading in which repeated reserves share one buffer rather than
// each adding to the footprint. See SumMiB for why both exist.
func (r Report) MaxComputeMiB(onGPU bool) float64 {
	peak := map[string]float64{}
	for _, e := range r.Entries {
		if e.Kind != KindCompute || e.OnGPU() != onGPU {
			continue
		}
		if e.MiB > peak[e.Device] {
			peak[e.Device] = e.MiB
		}
	}
	var total float64
	for _, v := range peak {
		total += v
	}
	return total
}

// GPUMiB is everything allocated on an accelerator, the figure a VRAM
// estimate is trying to predict. compute uses whichever aggregation the
// caller asks for, since that is the open question.
func (r Report) GPUMiB(sumCompute bool) float64 {
	var total float64
	for _, e := range r.Entries {
		if !e.OnGPU() || (e.Kind == KindCompute && !sumCompute) {
			continue
		}
		total += e.MiB
	}
	if !sumCompute {
		total += r.MaxComputeMiB(true)
	}
	return total
}

// HostMiB is everything allocated in system memory: weights left mapped,
// pinned staging buffers, and any layer that did not fit a device.
func (r Report) HostMiB(sumCompute bool) float64 {
	var total float64
	for _, e := range r.Entries {
		if e.OnGPU() || (e.Kind == KindCompute && !sumCompute) {
			continue
		}
		total += e.MiB
	}
	if !sumCompute {
		total += r.MaxComputeMiB(false)
	}
	return total
}
