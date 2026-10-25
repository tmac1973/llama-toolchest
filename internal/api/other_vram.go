package api

import (
	"regexp"
	"strconv"
	"sync"

	"github.com/tmac1973/llama-toolchest/internal/memreport"
	"github.com/tmac1973/llama-toolchest/internal/monitor"
	"github.com/tmac1973/llama-toolchest/internal/process"
)

// GPU memory other programs hold: the desktop, ComfyUI, another server.
// The planner budgets each card from what is free for llama.cpp, not from
// its total — on a workstation the desktop alone holds 1.6-1.8 GiB, and
// a plan made against the whole card left 0.17 GiB spare.
//
// The tools cannot say whose memory it is. rocm-smi lists no process for
// what a desktop holds, and nvidia-smi inside a container names processes
// by host PIDs the container cannot see. So it is worked out from what is
// known about the toolchest's own use:
//
//   - With no model of ours loaded, everything in use is someone else's.
//     Exact: a router with nothing loaded holds a few MiB.
//   - With models loaded, it is what is in use less what llama.cpp said it
//     allocated for them. The report leaves out llama.cpp's own context
//     (a few hundred MiB per card), so this errs high, which is the side
//     to err on.
//   - With neither — no report, or a load still under way — the last
//     reading taken while nothing of ours was loaded.
//
// That last reading also stands on its own (idleHardware), for plans of
// a launch that starts with nothing of ours loaded: the split a router
// start writes into the preset, and a benchmark cell or a load that
// replaces the loaded model. The figure with models loaded counts their
// unreported memory (the context, the image reader) as someone else's,
// which makes the cards look up to 2 GiB smaller than that launch finds
// them. On three 16 GiB cards with 10.5 and 7.5 GiB reserved on two, it
// moved a layer onto the card with the least room, and a benchmark of
// the loaded model was refused for want of memory the run frees itself.

// otherVRAMState is the latest figure per GPU index, in MiB, refreshed on
// every monitor poll.
type otherVRAMState struct {
	mu      sync.Mutex
	current []int // per GPU index; nil until first worked out
	idle    []int // the last reading with nothing of ours loaded
	// idleOther is idle less what generation backends used at the time
	// of that reading, which is the same figure as current: other
	// programs only.
	idleOther []int
}

// otherVRAMMiB is the memory other programs hold on each GPU, by index,
// or nil when it is not known yet.
func (s *Server) otherVRAMMiB() []int {
	s.otherVRAM.mu.Lock()
	defer s.otherVRAM.mu.Unlock()
	return append([]int(nil), s.otherVRAM.current...)
}

// idleOtherVRAMMiB is otherVRAMMiB as of the last reading with nothing of
// ours loaded, or nil when there has been none.
func (s *Server) idleOtherVRAMMiB() []int {
	s.otherVRAM.mu.Lock()
	defer s.otherVRAM.mu.Unlock()
	return append([]int(nil), s.otherVRAM.idleOther...)
}

// watchOtherVRAM refreshes the figure on every monitor poll, for the life
// of the server. Started once, from NewServer.
func (s *Server) watchOtherVRAM() {
	ch := s.monitor.Subscribe()
	defer s.monitor.Unsubscribe(ch)
	for m := range ch {
		s.updateOtherVRAM(m)
	}
}

func (s *Server) updateOtherVRAM(m monitor.Metrics) {
	used := make([]int, len(m.GPU))
	for i, g := range m.GPU {
		used[i] = g.VRAMUsedMB
	}
	other, idle := s.attributeVRAM(used)
	var idleOther []int
	if idle {
		idleOther = withoutGenUse(used, m.GPU, s.genVRAMByGPU(""))
	}

	s.otherVRAM.mu.Lock()
	defer s.otherVRAM.mu.Unlock()
	if idle {
		s.otherVRAM.idle = append([]int(nil), used...)
		s.otherVRAM.idleOther = idleOther
	}
	switch {
	case other != nil:
		s.otherVRAM.current = other
	case len(s.otherVRAM.idle) == len(used):
		s.otherVRAM.current = append([]int(nil), s.otherVRAM.idle...)
	}
}

// withoutGenUse is used (MiB per GPU, in the monitor's order) less what
// the generation backends hold on each card.
func withoutGenUse(used []int, gpus []monitor.GPUInfo, gen map[int]*genVRAM) []int {
	out := append([]int(nil), used...)
	for i := range out {
		if i < len(gpus) {
			if v := gen[gpus[i].Index]; v != nil {
				out[i] = max(0, out[i]-v.Using)
			}
		}
	}
	return out
}

// attributeVRAM gathers what the attribution needs from the server and
// hands it to attributeUse.
func (s *Server) attributeVRAM(used []int) (other []int, idle bool) {
	// A benchmark or evaluation runs its own llama.cpp processes, which the
	// router knows nothing about.
	if s.routerBusyWithJob() {
		return nil, false
	}
	running := s.process != nil && s.process.IsRunning()
	var loaded []process.ModelStatus
	if running {
		var err error
		if loaded, err = s.process.ListModels(); err != nil {
			return nil, false
		}
	}
	return attributeUse(used, running, loaded, func(name string) (memreport.Report, bool) {
		meas, ok := s.memory.Latest(name)
		return meas.Report, ok && len(meas.Report.Entries) > 0
	})
}

// attributeUse works out other programs' share of used (MiB per GPU
// index), given whether the router is running, what it has loaded, and
// each loaded model's buffer report. idle reports that nothing of ours
// was loaded, so used is wholly theirs. A nil result means it could not
// be worked out from this reading, and the last idle one stands.
func attributeUse(used []int, running bool, loaded []process.ModelStatus, report func(string) (memreport.Report, bool)) (other []int, idle bool) {
	if !running {
		return used, true
	}
	var names []string
	for _, ms := range loaded {
		switch ms.Status.Value {
		case "loading":
			return nil, false // its report is not complete yet
		case "loaded":
			names = append(names, ms.ID)
		}
	}
	if len(names) == 0 {
		return used, true
	}

	ours := make([]float64, len(used))
	for _, name := range names {
		r, ok := report(name)
		if !ok {
			return nil, false
		}
		for dev, mib := range r.ByDevice() {
			if !(memreport.Entry{Device: dev}).OnGPU() {
				continue
			}
			i, ok := deviceIndex(dev)
			if !ok || i >= len(used) {
				// An aggregate over several cards (a tensor-parallel
				// split) cannot be put on one of them.
				return nil, false
			}
			ours[i] += mib
		}
	}
	other = make([]int, len(used))
	for i := range used {
		other[i] = max(0, used[i]-int(ours[i]))
	}
	return other, false
}

var deviceNumber = regexp.MustCompile(`^(?:CUDA|ROCm|Vulkan|SYCL)(\d+)$`)

// deviceIndex is the GPU index of a llama.cpp device name ("CUDA1" → 1).
// The app runs CUDA with CUDA_DEVICE_ORDER=PCI_BUS_ID, the order the
// monitor lists cards in, and HIP enumerates in the same order as
// rocm-smi.
func deviceIndex(device string) (int, bool) {
	m := deviceNumber.FindStringSubmatch(device)
	if m == nil {
		return 0, false
	}
	i, err := strconv.Atoi(m[1])
	return i, err == nil
}
