package api

import (
	"context"
	"fmt"
	"log/slog"
	"maps"
	"slices"
	"strings"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/monitor"
)

// memoryMode is how the running router was told to share GPU memory
// between LLMs, fixed at its start (startRouterWith).
type memoryMode int32

const (
	memoryCount     memoryMode = iota // --models-max decides, by count
	memoryRouter                      // --models-memory-margin: the router decides
	memoryToolchest                   // toolchest decides (generation backends or forks on)
)

// String names the mode for the log: count, router or toolchest.
func (m memoryMode) String() string {
	switch m {
	case memoryRouter:
		return "router"
	case memoryToolchest:
		return "toolchest"
	}
	return "count"
}

// toolchestMemory reports whether the running router leaves it to
// toolchest to unload LLMs by memory (loadLLMWithRoom).
func (s *Server) toolchestMemory() bool {
	return memoryMode(s.memoryMode.Load()) == memoryToolchest
}

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

// unloadForLLM unloads idle, an idle LLM, to make room for the model
// named forName. ok is false, with no error, when a request began to use
// it since the idle list was made; the caller then takes the next one.
func (s *Server) unloadForLLM(ctx context.Context, idle idleLLM, forName string) (ok bool, err error) {
	m, err := s.registry.Get(idle.id)
	if err != nil || m == nil {
		return false, fmt.Errorf("model %s: %w", idle.id, err)
	}
	done, ok := s.llmUse.tryEvict(idle.id)
	if !ok {
		return false, nil
	}
	defer done()
	var gpus []int
	for g, mib := range idle.mib {
		if mib > 0 {
			gpus = append(gpus, g)
		}
	}
	slices.Sort(gpus)
	unloaded, err := s.unloadLLMEvicting(ctx, m, "room for "+forName, gpus)
	if err != nil {
		return false, err
	}
	if unloaded {
		slog.Info("unloaded LLM to make room for another LLM", "model", m.PublicName(), "for", forName)
	}
	return true, nil
}

// llmPlannedNeed is the cards and the planned LLM memory per GPU with m
// loaded beside the LLMs the router has loaded or is loading.
func (s *Server) llmPlannedNeed(m *models.Model) (models.Hardware, map[int]int) {
	hw := s.plannedLLMHardware()
	return hw, s.loadedLLMMiB(models.RoomOf(hw), m.ID, false)
}

// llmShortGPUs lists the GPUs where need is more than the planner lets
// models use of the card, beside the generation reservations that count
// now. Unlike reservationShortGPUs it checks every GPU, also one with no
// reservation: in toolchest memory mode nothing else checks a load.
func (s *Server) llmShortGPUs(hw models.Hardware, need map[int]int) []int {
	byGPU := s.genFitVRAMByGPU("")
	var out []int
	for _, g := range hw.GPUs {
		g.ReservedMiB = 0
		if v := byGPU[g.Index]; v != nil {
			g.ReservedMiB = v.Reserved
		}
		if n, ok := need[g.Index]; ok && n > models.CardBudgetMiB(g) {
			out = append(out, g.Index)
		}
	}
	return out
}

// gpuReadings is the system monitor's latest GPU readings.
func (s *Server) gpuReadings() []monitor.GPUInfo {
	if s.testGPUReadings != nil {
		return s.testGPUReadings()
	}
	return s.monitor.Current().GPU
}

// realFreeShort is one GPU short of real free memory for a load: free
// what the card shows free, needed what the load needs of it.
type realFreeShort struct {
	gpu, free, needed int
}

// realFreeShortGPUs checks m's load against the memory the cards show
// free, for each GPU m's plan uses: m's plan, the full plan of each model
// still loading (whose memory may not be allocated yet), the reservations
// of backends that count but do not hold their memory now, llama.cpp's
// working memory and the GPU memory margin the user set. The reading already counts the desktop and
// other programs. Without readings it finds nothing short.
func (s *Server) realFreeShortGPUs(m *models.Model) []realFreeShort {
	cfg, err := s.registry.GetConfig(m.ID)
	if err != nil || cfg == nil {
		return nil
	}
	room := models.RoomOf(s.idleHardware())
	plan := map[int]int{}
	addLLMMiB(plan, m, cfg, room)
	loading := s.loadingLLMMiB(room, m.ID)
	byGPU := s.genFitVRAMByGPU("")
	sc := s.cfgCopy()
	var out []realFreeShort
	for _, g := range s.gpuReadings() {
		if plan[g.Index] <= 0 {
			continue
		}
		needed := plan[g.Index] + loading[g.Index] + models.WorkingMemoryMiB + sc.VRAMMarginFor(g.Index)
		if v := byGPU[g.Index]; v != nil {
			needed += max(0, v.Reserved-v.Using)
		}
		if free := g.VRAMTotalMB - g.VRAMUsedMB; free < needed {
			out = append(out, realFreeShort{gpu: g.Index, free: free, needed: needed})
		}
	}
	return out
}

// loadingLLMMiB is the planned memory per GPU of the LLMs the router is
// loading now, leaving out except.
func (s *Server) loadingLLMMiB(room models.CardRoom, except string) map[int]int {
	out := map[int]int{}
	if !s.process.IsRunning() {
		return out
	}
	list, err := s.process.ListModels()
	if err != nil {
		return out
	}
	for _, rm := range list {
		if rm.Status.Value != "loading" {
			continue
		}
		if reg, cfg := s.resolveRouterModel(rm); reg != nil && cfg != nil && reg.ID != except {
			addLLMMiB(out, reg, cfg, room)
		}
	}
	return out
}

// unloadIdleOn unloads the least recently used idle LLM that holds
// memory on one of gpus (any GPU when gpus is nil), to make room for m.
// It reports false, with no error, when there is none left.
func (s *Server) unloadIdleOn(ctx context.Context, m *models.Model, gpus []int) (bool, error) {
	for _, idle := range s.idleLLMsByLRU(m.ID) {
		if gpus != nil && !slices.ContainsFunc(gpus, func(g int) bool { return idle.mib[g] > 0 }) {
			continue
		}
		ok, err := s.unloadForLLM(ctx, idle, m.PublicName())
		if err != nil {
			return false, fmt.Errorf("make room for %s: %w", m.PublicName(), err)
		}
		if ok {
			return true, nil
		}
	}
	return false, nil
}

// makeRoomByUnloading unloads idle LLMs, least recently used first, until
// a load of m stays under Max Loaded and, with byVRAM, its plan fits on
// every GPU. It returns the GPUs still short of planned memory, or the
// error that stopped it.
func (s *Server) makeRoomByUnloading(ctx context.Context, m *models.Model, byVRAM bool) ([]int, error) {
	if n := s.cfgCopy().ModelsMax; n > 0 {
		for s.loadedLLMCount() >= n {
			ok, err := s.unloadIdleOn(ctx, m, nil)
			if err != nil {
				return nil, err
			}
			if !ok {
				return nil, fmt.Errorf("Max loaded models (%d) is reached and every loaded model is answering a request; try again when one is idle", n)
			}
		}
	}
	for byVRAM {
		short := s.llmShortGPUs(s.llmPlannedNeed(m))
		if len(short) == 0 {
			return nil, nil
		}
		ok, err := s.unloadIdleOn(ctx, m, short)
		if err != nil {
			return nil, err
		}
		if !ok {
			return short, nil
		}
	}
	return nil, nil
}

// neverFits checks whether m's plan cannot fit even once everything that
// may go is gone: every idle LLM, and with eviction on every backend not
// kept running. What stays is the LLMs that answer a request or are
// loading, the reservations that cannot be freed (eviction off, kept
// running, fork instances) and m itself. Checked before anything is
// unloaded.
//
// Only a reservation refuses the load (err): it is a promise to a
// backend. A plan is an estimate, so otherwise llama-server decides
// whether the model fits. busyBlocked is set when LLMs that answer a
// request or are loading are what leave the plan short; unloading the
// idle ones would then free memory for nothing, so the load is sent
// with nothing unloaded.
func (s *Server) neverFits(m *models.Model) (busyBlocked bool, err error) {
	cfg, err := s.registry.GetConfig(m.ID)
	if err != nil || cfg == nil {
		return false, nil
	}
	room := models.RoomOf(s.idleHardware())
	stay := map[int]int{}
	addLLMMiB(stay, m, cfg, room)
	plan := maps.Clone(stay)
	busy := map[int][]string{}
	if s.process.IsRunning() {
		if list, err := s.process.ListModels(); err == nil {
			for _, rm := range list {
				st := rm.Status.Value
				if st != "loaded" && st != "loading" {
					continue
				}
				reg, rcfg := s.resolveRouterModel(rm)
				if reg == nil || rcfg == nil || reg.ID == m.ID {
					continue
				}
				if st == "loaded" && !s.llmUse.busy(reg.ID) {
					continue // idle: it would be unloaded
				}
				own := map[int]int{}
				addLLMMiB(own, reg, rcfg, room)
				for g, mib := range own {
					stay[g] += mib
					busy[g] = append(busy[g], reg.PublicName())
				}
			}
		}
	}
	byGPU := s.genFitVRAMByGPU("")
	evict := s.genEvictOn()
	for _, g := range s.plannedLLMHardware().GPUs {
		if plan[g.Index] <= 0 {
			continue
		}
		g.ReservedMiB = 0
		var kept []string
		if v := byGPU[g.Index]; v != nil {
			g.ReservedMiB = v.Reserved
			if evict {
				g.ReservedMiB = v.Reserved - v.Evictable
			}
			if g.ReservedMiB > 0 {
				kept = v.Names
			}
		}
		budget := models.CardBudgetMiB(g)
		if stay[g.Index] <= budget {
			continue
		}
		if g.ReservedMiB <= 0 {
			if len(busy[g.Index]) > 0 {
				busyBlocked = true
			}
			continue
		}
		var why []string
		if len(busy[g.Index]) > 0 {
			why = append(why, "answering a request or loading: "+strings.Join(busy[g.Index], ", "))
		}
		what := "reserved for generation backends"
		if evict {
			what = "reserved for generation backends that are kept running or are forks"
		}
		why = append(why, fmt.Sprintf("%s %s (%s)", mibText(g.ReservedMiB), what, strings.Join(kept, ", ")))
		msg := fmt.Sprintf("%s does not fit: GPU %d has %s for models and needs %s, even with every idle model unloaded; %s",
			m.PublicName(), g.Index, mibText(max(0, budget-(stay[g.Index]-plan[g.Index]))), mibText(plan[g.Index]), strings.Join(why, "; "))
		return false, &genLLMFitError{msg: msg}
	}
	return busyBlocked, nil
}

// loadLLMWithRoom loads m in toolchest memory mode: it unloads idle LLMs,
// least recently used first, until m fits under Max Loaded and in VRAM;
// then, with eviction on, it stops generation backends; then it checks
// the cards' real free memory. Only Max Loaded and generation
// reservations refuse a load: when the plan or the free memory is still
// short with nothing left to unload, the load is sent and llama-server
// decides (issue #561). Loads run one at a time, from the first check
// until the router lists m as loading. release ends the hold on
// generation GPUs that makeRoomForLLM took; call it once the load has
// finished.
func (s *Server) loadLLMWithRoom(ctx context.Context, m *models.Model, routerName string) (release func(), err error) {
	release = func() {}
	fail := func(err error) (func(), error) {
		release()
		return func() {}, err
	}
	s.llmLoadMu.Lock()
	locked := true
	defer func() {
		if locked {
			s.llmLoadMu.Unlock()
		}
	}()

	busyBlocked, err := s.neverFits(m)
	if err != nil {
		return fail(err)
	}
	short, err := s.makeRoomByUnloading(ctx, m, !busyBlocked)
	if err != nil {
		return fail(err)
	}
	if len(short) > 0 {
		// The idle LLMs are not enough: generation backends may be in the
		// way. makeRoomForLLM stops them with eviction on, and may wait
		// for a job to finish first, so other loads are not held up
		// behind it.
		s.llmLoadMu.Unlock()
		locked = false
		rel, err := s.makeRoomForLLM(ctx, m)
		if err != nil {
			return func() {}, err
		}
		release = rel
		s.llmLoadMu.Lock()
		locked = true
		if short, err = s.makeRoomByUnloading(ctx, m, true); err != nil {
			return fail(err)
		}
		if len(short) > 0 {
			// Not "unload a model first": every idle one is gone already.
			hw, need := s.llmPlannedNeed(m)
			if err := s.reservationFitError(hw, need, false); err != nil {
				return fail(err)
			}
			slog.Info("loading a model past its planned room; llama-server decides whether it fits",
				"model", m.PublicName(), "gpus", short)
		}
	}

	for !busyBlocked {
		short := s.realFreeShortGPUs(m)
		if len(short) == 0 {
			break
		}
		var gpus []int
		for _, r := range short {
			gpus = append(gpus, r.gpu)
		}
		ok, err := s.unloadIdleOn(ctx, m, gpus)
		if err != nil {
			return fail(err)
		}
		if !ok {
			r := short[0]
			slog.Info("loading a model with less VRAM free than its plan; llama-server decides whether it fits",
				"model", m.PublicName(), "gpu", r.gpu, "free_mib", r.free, "planned_mib", r.needed)
			break
		}
	}

	s.prepareNgramCacheForLoad(m.ID)
	if err := s.process.LoadModel(routerName); err != nil {
		return fail(fmt.Errorf("load %s: %w", m.PublicName(), err))
	}
	deadline := time.Now().Add(llmUnloadTimeout)
	for {
		if state, _ := s.lookupRouterState(routerName, m); state == "loading" || state == "loaded" {
			break
		}
		if time.Now().After(deadline) {
			return fail(fmt.Errorf("the router did not start loading %s within %s", m.PublicName(), llmUnloadTimeout))
		}
		select {
		case <-ctx.Done():
			return fail(ctx.Err())
		case <-time.After(autoLoadPollInterval):
		}
	}
	s.llmUse.touch(m.ID)
	return release, nil
}
