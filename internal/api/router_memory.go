package api

import (
	"errors"
	"fmt"

	"github.com/tmac1973/llama-toolchest/internal/models"
)

// routerLimits returns --models-max and --models-memory-margin for a
// router start. The router keeps several models loaded and unloads by
// memory only for serving, and only on builds that have the option: a
// benchmark or Autotune start measures one model at a time, as it always
// has.
func routerLimits(modelsMax int, routerMemory, job, buildHasOption bool, marginMiB int) (int, int) {
	if !routerMemory {
		return modelsMax, 0
	}
	if job {
		return 1, 0
	}
	if !buildHasOption {
		return modelsMax, 0
	}
	return modelsMax, marginMiB
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

// routerMemoryView is what the Server page says about the router option:
// Blocked is why it cannot be turned on (empty when it can), Margin the
// effective margin, and Note what is in the way of it working on the
// active build.
type routerMemoryView struct {
	On      bool
	Blocked string
	Margin  string
	Note    string
}

func (s *Server) routerMemoryInfo() routerMemoryView {
	cfg := s.cfgCopy()
	v := routerMemoryView{On: cfg.RouterMemory}
	if cfg.HoldsVRAMOutsideRouter() {
		v.Blocked = "Turn off generation backends and llama.cpp forks in Settings to use this. The router cannot see their VRAM reservations."
	}
	if mib := s.routerMemoryMarginMiB(); mib > 0 {
		v.Margin = fmt.Sprintf("%.1f GiB free on each GPU", float64(mib)/1024)
	}
	if !s.activeTarget().RouterMemoryOption() {
		v.Note = "The active llama.cpp build is older than b11930 and does not use this setting."
	}
	return v
}

// errRouterMemoryWithGeneration refuses the router option while
// generation backends or llama.cpp forks are on.
var errRouterMemoryWithGeneration = errors.New("the router can keep several models loaded only while generation backends and llama.cpp forks are off; turn them off in Settings first")

// applyRouterMemoryLocked applies the router memory option a settings
// request asked for (nil: unchanged), with cfgMu held. Turning it on with
// Max Loaded at 1 lifts the count limit, since memory decides now; and
// turning generation backends on turns it off, so toolchest's own
// unloading and the router's never both run. Turning llama.cpp forks on
// does the same, since the router cannot see their GPUs either.
func (s *Server) applyRouterMemoryLocked(on *bool) {
	if on != nil {
		if *on && !s.cfg.RouterMemory && s.cfg.ModelsMax == 1 {
			s.cfg.ModelsMax = 0
		}
		s.cfg.RouterMemory = *on
	}
	if s.cfg.HoldsVRAMOutsideRouter() && s.cfg.RouterMemory {
		s.cfg.RouterMemory = false
	}
}
