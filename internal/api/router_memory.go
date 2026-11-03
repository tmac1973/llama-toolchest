package api

import (
	"fmt"

	"github.com/tmac1973/llama-toolchest/internal/models"
)

// routerLimits returns --models-max and --models-memory-margin for a
// router start, and the memory mode they put it in. Several models stay
// loaded only for serving: a benchmark or Autotune start measures one
// model at a time, as it always has. While generation backends or forks
// hold VRAM outside the router, toolchest decides what to unload
// (loadLLMWithRoom, any build) and the router gets no limit of its own;
// otherwise the router decides by memory, on builds that have the option.
func routerLimits(modelsMax int, routerMemory, holdsVRAMOutside, job, buildHasOption bool, marginMiB int) (int, int, memoryMode) {
	switch {
	case !routerMemory:
		return modelsMax, 0, memoryCount
	case job:
		return 1, 0, memoryCount
	case holdsVRAMOutside:
		return 0, 0, memoryToolchest
	case !buildHasOption:
		return modelsMax, 0, memoryCount
	}
	return modelsMax, marginMiB, memoryRouter
}

// routerMemoryMarginMiB is the memory the router keeps free on each GPU
// when it manages memory (models.RouterMemoryMarginMiB).
func (s *Server) routerMemoryMarginMiB() int {
	var totals []int
	for _, g := range s.monitor.Current().GPU {
		totals = append(totals, g.VRAMTotalMB)
	}
	return models.RouterMemoryMarginMiB(totals)
}

// routerMemoryView is what the Server page says about the option:
// Toolchest is set when toolchest decides (generation backends or forks
// on), Restart when the running router was started in another mode than
// a start now would use, Margin the router's effective margin, and Note
// what is in the way of the router option on the active build.
type routerMemoryView struct {
	On        bool
	Toolchest bool
	Restart   bool
	Margin    string
	Note      string
}

func (s *Server) routerMemoryInfo() routerMemoryView {
	cfg := s.cfgCopy()
	v := routerMemoryView{On: cfg.RouterMemory, Toolchest: cfg.HoldsVRAMOutsideRouter()}
	hasOption := s.activeTarget().RouterMemoryOption()
	if s.process.IsRunning() && !s.routerBusyWithJob() {
		_, _, want := routerLimits(cfg.ModelsMax, cfg.RouterMemory, v.Toolchest, false, hasOption, 1)
		v.Restart = want != memoryMode(s.memoryMode.Load())
	}
	if v.Toolchest {
		return v
	}
	if mib := s.routerMemoryMarginMiB(); mib > 0 {
		v.Margin = fmt.Sprintf("%.1f GiB free on each GPU", float64(mib)/1024)
	}
	if !hasOption {
		v.Note = "The active llama.cpp build is older than b11930 and does not use this setting."
	}
	return v
}

// applyRouterMemoryLocked applies the "keep several models loaded"
// option a settings request asked for (nil: unchanged), with cfgMu held.
// Turning it on with Max Loaded at 1 lifts the count limit, since memory
// decides now. It stays on when generation backends or forks are turned
// on: toolchest decides from the next router start (routerLimits).
func (s *Server) applyRouterMemoryLocked(on *bool) {
	if on == nil {
		return
	}
	if *on && !s.cfg.RouterMemory && s.cfg.ModelsMax == 1 {
		s.cfg.ModelsMax = 0
	}
	s.cfg.RouterMemory = *on
}
