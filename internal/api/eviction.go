package api

import (
	"fmt"
	"net/http"

	"github.com/tmac1973/llama-toolchest/internal/config"
	"github.com/tmac1973/llama-toolchest/internal/models"
)

// routerLimits returns --models-max and --models-memory-margin for a
// router start, and the memory mode they put it in, from the eviction
// mode in effect (config.EffectiveEviction). Several models stay loaded
// only for serving: a benchmark or Autotune start measures one model at
// a time, as it always has. When toolchest decides what to unload
// (loadLLMWithRoom, any build) the router gets no limit of its own;
// llama.cpp decides by memory only on builds that have the option, and
// by count on older ones.
func routerLimits(modelsMax int, eviction string, job, buildHasOption bool, marginMiB int) (int, int, memoryMode) {
	switch {
	case eviction == config.EvictionOff:
		return modelsMax, 0, memoryCount
	case job:
		return 1, 0, memoryCount
	case eviction == config.EvictionToolchest:
		return 0, 0, memoryToolchest
	case !buildHasOption:
		return modelsMax, 0, memoryCount
	}
	return modelsMax, marginMiB, memoryRouter
}

// routerMemoryMarginMiB is the memory the router keeps free on each GPU
// when it manages memory (models.RouterMemoryMarginMiB).
func (s *Server) routerMemoryMarginMiB() int {
	return models.RouterMemoryMarginMiB(s.hardware().GPUs)
}

// evictionView is what the Settings card (eviction_card partial) and the
// Server page say about eviction.
type evictionView struct {
	// Saved is the chosen mode (config.Eviction), and Checked the radio
	// the card checks: toolchest while llama.cpp gives way to it.
	Saved   string
	Checked string
	// Label names the mode a router start uses now, for the Server page.
	Label string
	// LlamaCppBlocked says why llama.cpp cannot decide now (backends or
	// forks on); GaveWay is set when it was chosen anyway.
	LlamaCppBlocked string
	GaveWay         bool
	// Margin is the router's margin when llama.cpp decides; Note says
	// the active build is too old for it.
	Margin string
	Note   string

	ModelsMax          int
	GenerationEnabled  bool
	GenEvict           bool
	GenKeepPromptCache bool
	// Running is set while the router runs, and Restart when it was
	// started in another mode than a start now would use.
	Running bool
	Restart bool
}

// evictionInfo builds the eviction view from the current settings.
func (s *Server) evictionInfo() evictionView {
	cfg := s.cfgCopy()
	hasOption := s.activeTarget().RouterMemoryOption()
	effective := cfg.EffectiveEviction()
	v := evictionView{
		Saved:              cfg.Eviction,
		Checked:            effective,
		ModelsMax:          cfg.ModelsMax,
		GenerationEnabled:  cfg.GenerationEnabled,
		GenEvict:           cfg.GenEvict,
		GenKeepPromptCache: cfg.GenKeepPromptCache,
		Running:            s.process.IsRunning(),
	}
	switch {
	case cfg.GenerationEnabled && cfg.ForksEnabled:
		v.LlamaCppBlocked = "Not with generation backends or forks on"
	case cfg.GenerationEnabled:
		v.LlamaCppBlocked = "Not with generation backends on"
	case cfg.ForksEnabled:
		v.LlamaCppBlocked = "Not with llama.cpp forks on"
	}
	v.GaveWay = cfg.Eviction == config.EvictionLlamaCpp && v.LlamaCppBlocked != ""

	switch effective {
	case config.EvictionToolchest:
		v.Label = "llama-toolchest"
	case config.EvictionLlamaCpp:
		v.Label = "llama.cpp"
		if !hasOption {
			v.Label = "llama.cpp (by count: build too old)"
			v.Note = "The active llama.cpp build is older than b11930, so models unload by count until a newer build is active."
		} else if mib := s.routerMemoryMarginMiB(); mib > 0 {
			v.Margin = fmt.Sprintf("%.1f GiB", float64(mib)/1024)
		}
	default:
		v.Label = "Off"
	}

	if v.Running && !s.routerBusyWithJob() {
		_, _, want := routerLimits(cfg.ModelsMax, effective, false, hasOption, 1)
		v.Restart = want != memoryMode(s.memoryMode.Load())
	}
	return v
}

// handleEvictionCard renders the Settings page's Eviction card, which
// fetches itself again after each save (the settingsSaved event).
func (s *Server) handleEvictionCard(w http.ResponseWriter, r *http.Request) {
	respondHTML(w)
	s.renderPartial(w, "eviction_card", s.evictionInfo())
}

// applyEvictionLocked applies the eviction mode a settings request asked
// for (nil: unchanged), with cfgMu held. Leaving off with Max Loaded at 1
// lifts the count limit, since memory decides now. llama.cpp stays chosen
// when generation backends or forks are turned on: toolchest decides
// from the next router start (config.EffectiveEviction).
func (s *Server) applyEvictionLocked(mode *string) error {
	if mode == nil {
		return nil
	}
	if !config.ValidEviction(*mode) {
		return fmt.Errorf("eviction must be %s, %s or %s, not %q",
			config.EvictionOff, config.EvictionLlamaCpp, config.EvictionToolchest, *mode)
	}
	if *mode != config.EvictionOff && s.cfg.Eviction == config.EvictionOff && s.cfg.ModelsMax == 1 {
		s.cfg.ModelsMax = 0
	}
	s.cfg.Eviction = *mode
	return nil
}

// routerMemoryEviction maps the older router_memory switch of the API and
// of backups to an eviction mode: on is llama.cpp, as a config from
// before the setting reads (config.migrateEviction).
func routerMemoryEviction(on bool) string {
	if on {
		return config.EvictionLlamaCpp
	}
	return config.EvictionOff
}
