package api

import (
	"slices"
	"strings"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/models"
)

// idleLLM is a loaded LLM that no request uses now: its registry ID, its
// last use and its planned memory per GPU.
type idleLLM struct {
	id      string
	lastUse time.Time
	mib     map[int]int
}

// idleLLMsByLRU lists the LLMs the router has loaded and no request uses,
// least recently used first, leaving out except. A model the router is
// still loading is not idle.
func (s *Server) idleLLMsByLRU(except string) []idleLLM {
	if !s.process.IsRunning() {
		return nil
	}
	list, err := s.process.ListModels()
	if err != nil {
		return nil
	}
	room := models.RoomOf(s.idleHardware())
	var out []idleLLM
	for _, rm := range list {
		if rm.Status.Value != "loaded" {
			continue
		}
		reg, cfg := s.resolveRouterModel(rm)
		if reg == nil || cfg == nil || reg.ID == except || s.llmUse.busy(reg.ID) {
			continue
		}
		mib := map[int]int{}
		addLLMMiB(mib, reg, cfg, room)
		out = append(out, idleLLM{id: reg.ID, lastUse: s.llmUse.lastUse(reg.ID), mib: mib})
	}
	slices.SortFunc(out, func(a, b idleLLM) int {
		if c := a.lastUse.Compare(b.lastUse); c != 0 {
			return c
		}
		return strings.Compare(a.id, b.id)
	})
	return out
}

// loadedLLMCount is how many of our models the router has loaded or is
// loading: what Max Loaded limits.
func (s *Server) loadedLLMCount() int {
	if !s.process.IsRunning() {
		return 0
	}
	list, err := s.process.ListModels()
	if err != nil {
		return 0
	}
	n := 0
	for _, rm := range list {
		if rm.Status.Value != "loaded" && rm.Status.Value != "loading" {
			continue
		}
		if reg, _ := s.resolveRouterModel(rm); reg != nil {
			n++
		}
	}
	return n
}
