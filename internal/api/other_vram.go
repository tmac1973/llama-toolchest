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

// otherVRAMState is the latest figure per GPU index, in MiB, refreshed on
// every monitor poll.
type otherVRAMState struct {
	mu      sync.Mutex
	current []int // per GPU index; nil until first worked out
	idle    []int // the last reading with nothing of ours loaded
}

// otherVRAMMiB is the memory other programs hold on each GPU, by index,
// or nil when it is not known yet.
func (s *Server) otherVRAMMiB() []int {
	s.otherVRAM.mu.Lock()
	defer s.otherVRAM.mu.Unlock()
	return append([]int(nil), s.otherVRAM.current...)
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

	s.otherVRAM.mu.Lock()
	defer s.otherVRAM.mu.Unlock()
	if idle {
		s.otherVRAM.idle = append([]int(nil), used...)
	}
	switch {
	case other != nil:
		s.otherVRAM.current = other
	case len(s.otherVRAM.idle) == len(used):
		s.otherVRAM.current = append([]int(nil), s.otherVRAM.idle...)
	}
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
