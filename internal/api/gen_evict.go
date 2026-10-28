package api

import (
	"context"
	"fmt"
	"log/slog"
	"slices"
	"sync"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/generation"
	"github.com/tmac1973/llama-toolchest/internal/models"
)

// Eviction (Settings → Generation backends, config.GenEvict): when an LLM
// and a generation backend do not both fit on a GPU, the newest request
// gets the card. A chat request for an LLM that is not loaded stops the
// backends in its way (makeRoomForLLM), and a job for a stopped backend
// unloads LLMs and starts it (genReady). Neither side cuts work short:
// the LLM side holds the GPU's job queue, which waits for the running
// job, and the backend side waits for the LLM's requests in progress
// (llmUse). A reservation counts only while its backend runs
// (genFitVRAMByGPU), so a stopped backend is not in the way. Making room
// stops only as many backends as it needs, largest first, and never one
// marked keep-running (Instance.KeepRunning), whose reservation always
// counts.

// genStartTimeout caps how long a job waits for its backend to start.
// Loading an image model set from disk can take a minute or two.
const genStartTimeout = 5 * time.Minute

// llmUnloadTimeout caps how long an unload may take before the router
// still reports the model as loaded.
const llmUnloadTimeout = time.Minute

// genEvictOn reports whether LLMs and backends may take a GPU from each
// other.
func (s *Server) genEvictOn() bool {
	if s.gen == nil || s.genMgr == nil {
		return false
	}
	cfg := s.cfgCopy()
	return cfg.GenerationEnabled && cfg.GenEvict
}

// genFitVRAMByGPU is genVRAMByGPU as the fit checks count it: with
// eviction on, only the backends that are running or kept running, since
// any other stopped one gives its memory up.
func (s *Server) genFitVRAMByGPU(exclude string) map[int]*genVRAM {
	return s.genVRAM(exclude, s.genEvictOn())
}

// llmUse counts the requests in progress for each LLM (by registry ID),
// so an unload for a backend waits for them, and holds new ones back
// while the model is being unloaded.
type llmUse struct {
	mu       sync.Mutex
	n        map[string]int
	evicting map[string]bool
	// changed is closed and replaced whenever a count or an eviction
	// ends, waking the waiters.
	changed chan struct{}
}

func (u *llmUse) initLocked() {
	if u.n == nil {
		u.n, u.evicting, u.changed = map[string]int{}, map[string]bool{}, make(chan struct{})
	}
}

func (u *llmUse) notifyLocked() {
	close(u.changed)
	u.changed = make(chan struct{})
}

// waitLocked waits, with u.mu held, until ready() or ctx ends.
func (u *llmUse) waitLocked(ctx context.Context, ready func() bool) error {
	for !ready() {
		ch := u.changed
		u.mu.Unlock()
		select {
		case <-ch:
			u.mu.Lock()
		case <-ctx.Done():
			u.mu.Lock()
			return ctx.Err()
		}
	}
	return nil
}

// begin counts a request for model id, after any unload of it under way
// has finished. end must be called when the request is done.
func (u *llmUse) begin(ctx context.Context, id string) (end func(), err error) {
	u.mu.Lock()
	defer u.mu.Unlock()
	u.initLocked()
	if err := u.waitLocked(ctx, func() bool { return !u.evicting[id] }); err != nil {
		return nil, err
	}
	u.n[id]++
	var once sync.Once
	return func() {
		once.Do(func() {
			u.mu.Lock()
			defer u.mu.Unlock()
			if u.n[id]--; u.n[id] <= 0 {
				delete(u.n, id)
			}
			u.notifyLocked()
		})
	}, nil
}

// evict waits until model id has no requests in progress, then holds new
// ones back until done is called.
func (u *llmUse) evict(ctx context.Context, id string) (done func(), err error) {
	u.mu.Lock()
	defer u.mu.Unlock()
	u.initLocked()
	if err := u.waitLocked(ctx, func() bool { return u.n[id] == 0 && !u.evicting[id] }); err != nil {
		return nil, err
	}
	u.evicting[id] = true
	var once sync.Once
	return func() {
		once.Do(func() {
			u.mu.Lock()
			defer u.mu.Unlock()
			delete(u.evicting, id)
			u.notifyLocked()
		})
	}, nil
}

// requestModelID is the registry ID of the model a proxied request body
// names, or "" when it names none we know.
func (s *Server) requestModelID(body []byte) string {
	if m, _ := s.modelFromBody(body); m != nil {
		return m.ID
	}
	return ""
}

// makeRoomForLLM checks that m fits beside the generation reservations,
// as checkLLMFitsReservations does. With eviction on, when it does not,
// it holds the job queues of every GPU the LLMs would use, so their
// running jobs finish and no new one starts, and stops backends on them
// until m fits. Every GPU, not only the ones short now: while the hold
// waits for one job, a job arriving for another of the model's GPUs would
// otherwise start there and leave the load short again.
// release ends the hold; call it once the load has finished, so a job
// waiting for those GPUs cannot take them back before the model is in.
func (s *Server) makeRoomForLLM(ctx context.Context, m *models.Model) (release func(), err error) {
	noop := func() {}
	hw, need, replacing := s.llmReservationNeed(m)
	short := s.reservationShortGPUs(hw, need)
	if len(short) == 0 {
		return noop, nil
	}
	err = s.reservationFitError(hw, need, !replacing)
	if !s.genEvictOn() || s.genQueue == nil {
		return noop, err
	}
	var gpus []int
	for gpu, mib := range need {
		if mib > 0 {
			gpus = append(gpus, gpu)
		}
	}
	release, herr := s.genQueue.Hold(ctx, gpus, "loading "+m.PublicName())
	if herr != nil {
		return noop, herr
	}
	for _, gpu := range gpus {
		s.stopGenUntil(gpu, "", "loading "+m.PublicName(), func() bool {
			hw, need, _ := s.llmReservationNeed(m)
			return !slices.Contains(s.reservationShortGPUs(hw, need), gpu)
		})
	}
	// Checked again whether or not anything was stopped: the hold waited
	// for a job that was starting its backend, which may have failed to
	// start, and then there is room with nothing to stop.
	if err := s.checkLLMFitsReservations(m); err != nil {
		release()
		return noop, err
	}
	return release, nil
}

// stopGenUntil stops running backends on gpu other than except, largest
// first, until fits() reports there is room, waiting after each for the
// card to show its memory free. Backends marked keep-running are never
// stopped. It returns how many it stopped. Its callers hold the GPU's job
// queue, so none is mid-job.
func (s *Server) stopGenUntil(gpu int, except, reason string, fits func() bool) int {
	type candidate struct {
		inst  generation.Instance
		share int
	}
	var cands []candidate
	for _, inst := range s.gen.Instances() {
		if inst.GPU != gpu || inst.ID == except || inst.KeepRunning || !s.genMgr.IsActive(inst.ID) {
			continue
		}
		share := inst.ReservedMiB
		if pid := s.genMgr.PID(inst.ID); pid > 0 {
			if mib, ok := s.genProcVRAM.read(pid); ok {
				share = max(share, mib)
			}
		}
		cands = append(cands, candidate{inst, share})
	}
	slices.SortStableFunc(cands, func(a, b candidate) int { return b.share - a.share })
	stopped := 0
	for _, c := range cands {
		if fits() {
			break
		}
		inst, share := c.inst, c.share
		before := s.gpuUsedMiB(gpu)
		if err := s.genMgr.Stop(inst.ID); err != nil {
			slog.Warn("failed to stop generation backend to make room", "instance", inst.ID, "error", err)
			continue
		}
		slog.Info("stopped generation backend to make room", "instance", inst.ID, "gpu", gpu, "for", reason)
		s.waitGPUFreed(gpu, max(0, before-share))
		stopped++
	}
	return stopped
}

// genReady makes sure inst is running before its job is sent; the job
// holds inst's GPU turn. A stopped instance is started, and a starting
// one waited for. With eviction off, an enabled instance's reservation
// counts while it is stopped, so starting it takes no memory promised to
// anything else; the fit check in startGenInstance still refuses it when
// another program has taken the room. With eviction on, room is made
// first: by stopping other backends on the GPU (only as many as needed,
// and none kept running), then by unloading LLMs from the GPU, one at a
// time until it fits.
func (s *Server) genReady(ctx context.Context, inst generation.Instance) error {
	if s.genMgr.IsRunning(inst.ID) {
		return nil
	}
	if !s.genMgr.IsActive(inst.ID) {
		if s.genEvictOn() {
			if err := s.makeRoomForGen(ctx, inst); err != nil {
				return err
			}
		}
		slog.Info("starting generation backend for a job", "instance", inst.ID)
		if err := s.startGenInstance(inst.ID); err != nil {
			return err
		}
	}
	wctx, cancel := context.WithTimeout(ctx, genStartTimeout)
	defer cancel()
	if err := s.genMgr.WaitRunning(wctx, inst.ID); err != nil {
		return fmt.Errorf("backend %s did not start: %w", inst.ID, err)
	}
	if !s.genMgr.IsRunning(inst.ID) {
		return fmt.Errorf("backend %s stopped while starting; see its log on the Server page", inst.ID)
	}
	return nil
}

// startGenFromUI starts inst for the Start and Restart buttons. When
// inst fits, or eviction is off, it starts it as before (and is refused
// when it does not fit). With eviction on and inst in the way of an LLM or
// another backend, it makes room the way a job does: it takes the GPU's
// turn in the queue, so no job is mid-run there, stops the backends and
// unloads the LLMs in the way, and starts inst. The turn is kept until
// inst runs, so a chat waiting for its LLM cannot load it into the room
// first.
func (s *Server) startGenFromUI(ctx context.Context, inst generation.Instance) error {
	if !s.genEvictOn() || s.genQueue == nil || s.genFitCheck(inst, inst.ReservedMiB) == nil {
		return s.startGenInstance(inst.ID)
	}
	set, err := s.gen.ModelSet(inst.ModelSetID)
	if err != nil {
		return err
	}
	ticket, err := s.genQueue.Submit(inst.GPU, inst.ID, set.Kind)
	if err != nil {
		return err
	}
	if err := ticket.Wait(ctx); err != nil {
		return err
	}
	if err := s.makeRoomForGen(ctx, inst); err != nil {
		ticket.Done()
		return err
	}
	if err := s.startGenInstance(inst.ID); err != nil {
		ticket.Done()
		return err
	}
	go func() {
		defer ticket.Done()
		wctx, cancel := context.WithTimeout(context.Background(), genStartTimeout)
		defer cancel()
		s.genMgr.WaitRunning(wctx, inst.ID)
	}()
	return nil
}

// makeRoomForGen frees inst's GPU until inst's reservation fits.
func (s *Server) makeRoomForGen(ctx context.Context, inst generation.Instance) error {
	err := s.genFitCheck(inst, inst.ReservedMiB)
	if err == nil {
		return nil
	}
	fits := func() bool { return s.genFitCheck(inst, inst.ReservedMiB) == nil }
	if s.stopGenUntil(inst.GPU, inst.ID, "starting "+inst.Name, fits) > 0 {
		if err = s.genFitCheck(inst, inst.ReservedMiB); err == nil {
			return nil
		}
	}
	for _, id := range s.loadedLLMsOn(inst.GPU) {
		if uerr := s.unloadForGen(ctx, id, inst); uerr != nil {
			return uerr
		}
		if err = s.genFitCheck(inst, inst.ReservedMiB); err == nil {
			return nil
		}
	}
	return err
}

// loadedLLMsOn lists the registry IDs of the LLMs the router has loaded
// (or is loading) with planned memory on gpu, largest share first.
func (s *Server) loadedLLMsOn(gpu int) []string {
	if !s.process.IsRunning() {
		return nil
	}
	list, err := s.process.ListModels()
	if err != nil {
		return nil
	}
	room := models.RoomOf(s.idleHardware())
	type entry struct {
		id  string
		mib int
	}
	var out []entry
	for _, rm := range list {
		if rm.Status.Value != "loaded" && rm.Status.Value != "loading" {
			continue
		}
		reg, cfg := s.resolveRouterModel(rm)
		if reg == nil || cfg == nil {
			continue
		}
		on := map[int]int{}
		addLLMMiB(on, reg, cfg, room)
		if on[gpu] > 0 {
			out = append(out, entry{reg.ID, on[gpu]})
		}
	}
	ids := make([]string, 0, len(out))
	for len(out) > 0 {
		best := 0
		for i := range out {
			if out[i].mib > out[best].mib {
				best = i
			}
		}
		ids = append(ids, out[best].id)
		out = append(out[:best], out[best+1:]...)
	}
	return ids
}

// unloadForGen unloads model id for inst once its requests in progress
// have finished, and waits for the router and then the card to show it
// gone. New requests for the model wait until it is gone, then load it
// again, taking their turn on the GPU after inst's job.
func (s *Server) unloadForGen(ctx context.Context, id string, inst generation.Instance) error {
	m, err := s.registry.Get(id)
	if err != nil || m == nil {
		return fmt.Errorf("model %s: %w", id, err)
	}
	done, err := s.llmUse.evict(ctx, id)
	if err != nil {
		return err
	}
	defer done()
	routerName := s.registry.RouterName(id)
	// Another job's unload may have taken it out while this one waited
	// its turn. Then there is nothing to save or unload, and asking the
	// router for the model's slots would load it again.
	if state, _ := s.lookupRouterState(routerName, m); state != "loaded" && state != "loading" {
		return nil
	}
	// Its requests are done and new ones wait, so the slots are idle.
	s.savePromptCache(ctx, m, routerName)
	before := s.gpuUsedMiB(inst.GPU)
	planned := s.loadedLLMMiB(models.RoomOf(s.idleHardware()), "", false)[inst.GPU]
	if err := s.process.UnloadModel(routerName); err != nil {
		return err
	}
	slog.Info("unloaded LLM to make room for a generation backend", "model", m.PublicName(), "instance", inst.ID)
	deadline := time.Now().Add(llmUnloadTimeout)
	for {
		state, _ := s.lookupRouterState(routerName, m)
		if state != "loaded" && state != "loading" && state != "unloading" {
			break
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("model %s was still loaded %s after the unload", m.PublicName(), llmUnloadTimeout)
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(autoLoadPollInterval):
		}
	}
	after := s.loadedLLMMiB(models.RoomOf(s.idleHardware()), "", false)[inst.GPU]
	s.waitGPUFreed(inst.GPU, max(0, before-(planned-after)))
	return nil
}

// gpuFreedTimeout caps the wait for the card to show memory free after a
// stop or an unload.
const gpuFreedTimeout = 20 * time.Second

// waitGPUFreed waits for the card to show what a stopped backend or an
// unloaded LLM held as free, reading the monitor's fresh samples (see
// gpuFreed). Without GPU readings there is nothing to wait for.
func (s *Server) waitGPUFreed(gpu, target int) {
	if len(s.monitor.Current().GPU) == 0 {
		return
	}
	ch := s.monitor.Subscribe()
	defer s.monitor.Unsubscribe(ch)
	readings := make(chan int)
	stop := make(chan struct{})
	defer close(stop)
	go func() {
		for {
			select {
			case m := <-ch:
				for _, g := range m.GPU {
					if g.Index == gpu {
						select {
						case readings <- g.VRAMUsedMB:
						case <-stop:
							return
						}
					}
				}
			case <-stop:
				return
			}
		}
	}()
	if !gpuFreed(readings, target, gpuFreedTimeout) {
		slog.Warn("the card did not show the memory free in time", "gpu", gpu, "used_mib", s.gpuUsedMiB(gpu), "expected_mib", target)
	}
}

// gpuFreed takes the card's used memory from readings, each sampled
// after the process exited or the model was unloaded, and reports true
// once a reading is within gpuReleasedSlackMiB of target, or once two
// readings in a row show it no longer falling: the process held less
// than planned (an idle backend holds far less than its reservation,
// which is its peak). False when timeout passes first.
func gpuFreed(readings <-chan int, target int, timeout time.Duration) bool {
	deadline := time.After(timeout)
	prev := -1
	for {
		select {
		case used := <-readings:
			if used <= target+gpuReleasedSlackMiB || (prev >= 0 && used >= prev-gpuReleasedSlackMiB) {
				return true
			}
			prev = used
		case <-deadline:
			return false
		}
	}
}
