package api

import (
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	"github.com/go-chi/chi/v5"
	"github.com/tmac1973/llama-toolchest/internal/builder"
	"github.com/tmac1973/llama-toolchest/internal/forks"
	"github.com/tmac1973/llama-toolchest/internal/generation"
	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/process"
)

// forkInstanceView is one fork instance as the API reports it.
type forkInstanceView struct {
	forks.Instance
	State string `json:"state"`
	Error string `json:"error,omitempty"`
}

func (s *Server) forkInstanceViews(store *forks.Store) []forkInstanceView {
	var out []forkInstanceView
	for _, inst := range store.Instances() {
		st := s.forkMgr.Status(inst.ID)
		out = append(out, forkInstanceView{Instance: inst, State: string(st.State), Error: st.Error})
	}
	return out
}

func (s *Server) handleForkInstances(w http.ResponseWriter, r *http.Request) {
	store := s.forkStore(w)
	if store == nil {
		return
	}
	if !isHTMX(r) {
		respondJSON(w, s.forkInstanceViews(store))
		return
	}
	respondHTML(w)
	s.renderPartial(w, "fork_instances_table", s.forkRows(store))
}

// forkInstanceFromForm reads an instance's fields from the form into
// inst, checking what depends on this host: the fork, its model set, the
// build, the GPUs, the extra arguments and the environment lines. The
// port is set by forkPortFromForm. On an error it returns the HTTP status.
func (s *Server) forkInstanceFromForm(r *http.Request, store *forks.Store, inst *forks.Instance) (int, error) {
	if err := r.ParseForm(); err != nil {
		return http.StatusBadRequest, fmt.Errorf("invalid form: %w", err)
	}
	f, ok := findFork(r.FormValue("fork"))
	if !ok {
		return http.StatusBadRequest, errors.New("choose a fork")
	}
	inst.ForkID = f.ID
	buildID := r.FormValue("build")
	if buildID != "" {
		if b := s.resolveForkBuild(f.ID, buildID); b == nil || b.ID != buildID {
			return http.StatusBadRequest, fmt.Errorf("build %s is not a successful %s build; choose another build or Newest", buildID, f.Name)
		}
	}
	inst.BuildID = buildID
	gpu, err := strconv.Atoi(r.FormValue("gpu"))
	if err != nil || gpu < 0 {
		return http.StatusBadRequest, errors.New("choose a GPU")
	}
	inst.GPU, inst.ExtraGPUs = gpu, nil
	// second_gpu is empty for an instance on one GPU.
	if raw := strings.TrimSpace(r.FormValue("second_gpu")); raw != "" {
		second, err := strconv.Atoi(raw)
		if err != nil || second < 0 {
			return http.StatusBadRequest, fmt.Errorf("second GPU %q is not a GPU number", raw)
		}
		if f.GPULimit() < 2 {
			return http.StatusBadRequest, fmt.Errorf("%s runs on one GPU only", f.Name)
		}
		if second == gpu {
			return http.StatusBadRequest, errors.New("choose a second GPU that is not the first one")
		}
		inst.ExtraGPUs = []int{second}
	}
	if gpus := s.monitor.Current().GPU; len(gpus) > 0 {
		for _, g := range inst.GPUs() {
			if g >= len(gpus) {
				return http.StatusBadRequest, fmt.Errorf("this machine has no GPU %d", g)
			}
		}
	}
	args, err := generation.ValidateExtraArgs(r.FormValue("extra_args"))
	if err != nil {
		return http.StatusBadRequest, err
	}
	inst.ExtraArgs = args
	inst.ExtraEnv = nil
	for _, line := range strings.Split(r.FormValue("extra_env"), "\n") {
		if line = strings.TrimSpace(line); line != "" {
			inst.ExtraEnv = append(inst.ExtraEnv, line)
		}
	}
	if err := forks.CheckEnv(inst.ExtraEnv); err != nil {
		return http.StatusBadRequest, err
	}
	inst.AutoStart = r.FormValue("auto_start") == "on" || r.FormValue("auto_start") == "true"
	return 0, nil
}

// forkNameTaken reports why id cannot be a fork instance's name: it is
// the name clients send, so it must not be a model the router or a
// generation instance already answers to.
func (s *Server) forkNameTaken(id string) error {
	if m, err := s.registry.Get(id); err == nil && m != nil {
		return fmt.Errorf("%q is already the name of a model", id)
	}
	if s.gen != nil {
		if _, err := s.gen.Instance(id); err == nil {
			return fmt.Errorf("%q is already the name of a generation instance", id)
		}
	}
	return nil
}

// instancePorts maps each port a generation or fork instance holds to
// the instance, leaving out the instance being saved.
func (s *Server) instancePorts(skipGen, skipFork string) map[int]string {
	used := map[int]string{}
	if s.gen != nil {
		for _, inst := range s.gen.Instances() {
			if inst.ID != skipGen {
				used[inst.Port] = inst.ID
			}
		}
	}
	if s.forks != nil {
		for _, inst := range s.forks.Instances() {
			if inst.ID != skipFork {
				used[inst.Port] = inst.ID + " (fork)"
			}
		}
	}
	return used
}

// forkPortFromForm is portFromForm for a fork instance.
func (s *Server) forkPortFromForm(r *http.Request, inst *forks.Instance) (int, error) {
	return s.portFromForm(r, &inst.Port, "", inst.ID)
}

// portFromForm sets *port from the form's port field, for the generation
// or fork instance skipGen or skipFork ("" for the other kind). Empty
// keeps the instance's port, or gives a new instance the lowest port of
// the generation port range that no generation or fork instance uses. A
// chosen port must be in that range and not used by another instance; the
// instance's own port is always kept, also when the range changed so that
// it is now outside. On an error it returns the HTTP status to send.
func (s *Server) portFromForm(r *http.Request, port *int, skipGen, skipFork string) (int, error) {
	cfg := s.cfgCopy()
	start, count := cfg.GenPorts(s.inContainer())
	used := s.instancePorts(skipGen, skipFork)
	raw := strings.TrimSpace(r.FormValue("port"))
	if raw == "" {
		if *port != 0 {
			return 0, nil
		}
		for p := start; p < start+count; p++ {
			if _, ok := used[p]; !ok {
				*port = p
				return 0, nil
			}
		}
		return http.StatusConflict, fmt.Errorf("%d–%d: %w; widen the range in Settings, or delete an instance", start, start+count-1, generation.ErrNoPort)
	}
	p, err := strconv.Atoi(raw)
	if err != nil {
		return http.StatusBadRequest, fmt.Errorf("port %q is not a number", raw)
	}
	if p == *port {
		return 0, nil
	}
	if p < start || p >= start+count {
		return http.StatusBadRequest, fmt.Errorf("port %d is outside the generation port range %d–%d", p, start, start+count-1)
	}
	if who, ok := used[p]; ok {
		return http.StatusConflict, fmt.Errorf("port %d is %w (%s)", p, generation.ErrPortUsed, who)
	}
	*port = p
	return 0, nil
}

func (s *Server) handleForkAddInstance(w http.ResponseWriter, r *http.Request) {
	store := s.forkStore(w)
	if store == nil {
		return
	}
	var inst forks.Instance
	if code, err := s.forkInstanceFromForm(r, store, &inst); err != nil {
		http.Error(w, err.Error(), code)
		return
	}
	inst.ID = strings.TrimSpace(r.FormValue("id"))
	if err := s.forkNameTaken(inst.ID); err != nil {
		http.Error(w, err.Error(), http.StatusConflict)
		return
	}
	if code, err := s.forkPortFromForm(r, &inst); err != nil {
		http.Error(w, err.Error(), code)
		return
	}
	if err := store.AddInstance(inst); err != nil {
		code := http.StatusBadRequest
		if errors.Is(err, forks.ErrIDTaken) {
			code = http.StatusConflict
		}
		http.Error(w, err.Error(), code)
		return
	}
	s.forkInstancesChanged(w, r)
}

func (s *Server) handleForkEditInstance(w http.ResponseWriter, r *http.Request) {
	store := s.forkStore(w)
	if store == nil {
		return
	}
	inst, ok := s.forkInstance(w, r, store)
	if !ok {
		return
	}
	if s.forkMgr.IsActive(inst.ID) {
		http.Error(w, "stop the instance before editing it", http.StatusConflict)
		return
	}
	if code, err := s.forkInstanceFromForm(r, store, &inst); err != nil {
		http.Error(w, err.Error(), code)
		return
	}
	if code, err := s.forkPortFromForm(r, &inst); err != nil {
		http.Error(w, err.Error(), code)
		return
	}
	if err := store.UpdateInstance(inst); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	s.forkInstancesChanged(w, r)
}

func (s *Server) handleForkDeleteInstance(w http.ResponseWriter, r *http.Request) {
	store := s.forkStore(w)
	if store == nil {
		return
	}
	inst, ok := s.forkInstance(w, r, store)
	if !ok {
		return
	}
	if deleteStoppedInstance(w, s.forkMgr.Instances, inst.ID, store.DeleteInstance) {
		s.forkInstancesChanged(w, r)
	}
}

// forkInstance reads the {id} instance, writing a 404 when it is unknown.
func (s *Server) forkInstance(w http.ResponseWriter, r *http.Request, store *forks.Store) (forks.Instance, bool) {
	return urlInstance(w, r, store.Instance)
}

// forkStartResult is a start's answer: the needs the host falls short of
// (the start goes on regardless) and the instance's state.
type forkStartResult struct {
	Warnings []string `json:"warnings,omitempty"`
	State    string   `json:"state"`
}

func (s *Server) handleForkInstanceAction(w http.ResponseWriter, r *http.Request) {
	store := s.forkStore(w)
	if store == nil {
		return
	}
	inst, ok := s.forkInstance(w, r, store)
	if !ok {
		return
	}
	var res forkStartResult
	switch chi.URLParam(r, "action") {
	case "start":
		warnings, err := s.startForkInstance(inst.ID)
		if err != nil {
			code := http.StatusBadRequest
			var busy *forkGPUBusyError
			if errors.As(err, &busy) {
				code = http.StatusConflict
			}
			http.Error(w, err.Error(), code)
			return
		}
		res.Warnings = warnings
		s.forkNotes.set(inst.ID, warnings)
	case "stop":
		if err := s.forkMgr.Stop(inst.ID); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		s.forkNotes.set(inst.ID, nil)
	default:
		http.Error(w, "unknown action", http.StatusNotFound)
		return
	}
	res.State = string(s.forkMgr.Status(inst.ID).State)
	if isHTMX(r) {
		// The card's table, with the start's warnings under the state.
		respondHTML(w)
		s.renderPartial(w, "fork_instances_table", s.forkRows(store))
		return
	}
	respondJSON(w, res)
}

func (s *Server) handleForkInstanceLogs(w http.ResponseWriter, r *http.Request) {
	if s.forkStore(w) == nil {
		return
	}
	streamInstanceLogs(w, r, s.forkMgr.Instances)
}

// forkInstancesChanged answers a change with the instance list, and tells
// htmx pages to refresh it.
func (s *Server) forkInstancesChanged(w http.ResponseWriter, r *http.Request) {
	instancesChanged(w, r, "forkInstancesChanged", s.handleForkInstances)
}

// forkGPUBusyError is a start refused because something else is on one
// of the instance's GPUs.
type forkGPUBusyError struct{ msg string }

func (e *forkGPUBusyError) Error() string { return e.msg }

// gpuOccupants names what holds memory on a GPU, for a fork start: the
// router's loaded models planned there, generation instances that run or
// reserve memory there, and other fork instances starting or running
// there. skipFork is the instance being started.
func (s *Server) gpuOccupants(gpu int, skipFork string) []string {
	var out []string
	if s.process != nil && s.process.IsRunning() {
		room := models.RoomOf(s.idleHardware())
		if list, err := s.process.ListModels(); err == nil {
			for _, rm := range list {
				if rm.Status.Value != "loaded" && rm.Status.Value != "loading" {
					continue
				}
				m, cfg := s.resolveRouterModel(rm)
				if m == nil || cfg == nil {
					continue
				}
				if models.CardLoadsMiB(m, cfg, room.For(m, cfg))[gpu] > 0 {
					out = append(out, m.ID+" (router)")
				}
			}
		}
	}
	if s.generationOn() && s.gen != nil && s.genMgr != nil {
		// A stopped instance holds its GPU only when its reservation
		// counts while stopped: always with auto-evict off, and with it on
		// only when the instance is locked (kept running). With auto-evict
		// on, an unlocked one reserves only while it runs, and the fork's
		// whole-GPU reservation then keeps it from starting.
		evict := s.genEvictOn()
		for _, inst := range s.gen.Instances() {
			if inst.GPU != gpu {
				continue
			}
			holds := inst.Enabled && (!evict || inst.KeepRunning)
			if holds || s.genMgr.IsActive(inst.ID) {
				out = append(out, inst.Name+" (generation)")
			}
		}
	}
	if s.forks != nil {
		for _, inst := range s.forks.Instances() {
			if inst.ID != skipFork && slices.Contains(inst.GPUs(), gpu) && s.forkMgr.IsActive(inst.ID) {
				out = append(out, inst.ID+" (fork)")
			}
		}
	}
	return out
}

// forkHostInfo describes this host for a fork's needs: RAM, AVX2 and the
// architectures of the GPUs. Disk space is left out: it only matters for
// the download, which is done by the time an instance starts.
func (s *Server) forkHostInfo(gpus []int, backend string) forks.HostInfo {
	if s.testForkHostInfo != nil {
		info := *s.testForkHostInfo
		info.Backend = backend
		return info
	}
	info := forks.HostInfo{Backend: backend}
	if s.monitor != nil {
		m := s.monitor.Current()
		info.RAMGiB = m.Memory.TotalMiB / 1024
		for _, gpu := range gpus {
			arch := ""
			for _, g := range m.GPU {
				if g.Index == gpu {
					arch = g.Arch
				}
			}
			info.GPUArchs = append(info.GPUArchs, arch)
		}
	}
	if raw, err := os.ReadFile("/proc/cpuinfo"); err == nil {
		info.AVX2Known = true
		info.AVX2 = strings.Contains(string(raw), " avx2")
	}
	return info
}

// startForkInstance starts a fork instance after its checks. Every start
// goes through here (the button and auto-start). It returns the needs the
// host falls short of, which do not stop the start.
func (s *Server) startForkInstance(id string) ([]string, error) {
	if !s.forksEnabled() {
		return nil, errors.New("llama.cpp forks are off")
	}
	inst, err := s.forks.Instance(id)
	if err != nil {
		return nil, err
	}
	if s.forkMgr.IsActive(id) {
		return nil, nil
	}
	f, ok := findFork(inst.ForkID)
	if !ok {
		return nil, fmt.Errorf("instance %s uses fork %q, which this toolchest does not know", id, inst.ForkID)
	}
	set, err := s.forks.ModelSet(f.ID)
	if err != nil || !set.Complete {
		return nil, fmt.Errorf("%s is not fully downloaded yet; download it on the Models page", f.ModelName)
	}
	build := s.resolveForkBuild(f.ID, inst.BuildID)
	if build == nil {
		return nil, fmt.Errorf("there is no successful %s build; build one on the Builds page", f.Name)
	}
	backend, err := generation.BuildBackend(*build)
	if err != nil {
		return nil, err
	}
	if err := generation.CheckGPUVendor(backend, s.monitor.GPUVendor()); err != nil {
		return nil, err
	}
	var held []string
	for _, gpu := range inst.GPUs() {
		if busy := s.gpuOccupants(gpu, id); len(busy) > 0 {
			held = append(held, fmt.Sprintf("GPU %d holds %s.", gpu, strings.Join(busy, ", ")))
		}
	}
	if len(held) > 0 {
		return nil, &forkGPUBusyError{msg: strings.Join(held, " ") + " A fork instance uses each of its GPUs whole; unload or stop them first."}
	}
	warnings := forks.CheckNeeds(f, s.forkHostInfo(inst.GPUs(), backend))

	env := process.BuildEnv(filepath.Dir(build.BinaryPath), nil)
	// The GPU selector and the instance's own lines must not follow
	// BuildEnv's "inherited wins" rule: a stray HIP_VISIBLE_DEVICES around
	// toolchest would put the instance on the wrong GPU.
	var devices []int
	for _, gpu := range inst.GPUs() {
		devices = append(devices, models.BackendDeviceIndex(backend, gpu))
	}
	for _, line := range forks.Env(f, inst, backend, devices) {
		name, value, _ := strings.Cut(line, "=")
		env = builder.SetEnv(env, name, value)
	}
	cfg := process.SupervisorConfig{
		Name:          "fork:" + inst.ID,
		BinaryPath:    build.BinaryPath,
		Args:          forks.Args(f, set, inst),
		Env:           env,
		WorkDir:       set.Dir,
		HealthURL:     inst.BaseURL() + "/health",
		HealthTimeout: forks.StartTimeout,
	}
	if err := s.forkMgr.Start(inst.ID, cfg); err != nil {
		return warnings, err
	}
	for _, w := range warnings {
		slog.Warn("fork instance started on a host short of its needs", "instance", id, "warning", w)
	}
	return warnings, nil
}

// forksSwitched follows the llama.cpp forks switch: off stops every fork
// instance, on starts the auto-start ones. In the background: a stop can
// wait for a server to exit, and a start for a long model load.
func (s *Server) forksSwitched(was, now bool) {
	switch {
	case was && !now && s.forkMgr != nil:
		go func() {
			s.forkSwitchMu.Lock()
			defer s.forkSwitchMu.Unlock()
			s.forkMgr.StopAll()
		}()
	case !was && now:
		go func() {
			s.forkSwitchMu.Lock()
			defer s.forkSwitchMu.Unlock()
			if s.forkAutoStartFn != nil {
				s.forkAutoStartFn()
				return
			}
			s.autoStartForkInstances(false)
		}()
	}
}

// autoStartForkInstances starts every fork instance set to start with
// toolchest, one after another. When the router is auto-starting too, it
// waits until the router is running or has failed, so the router's GPU
// use is known to the busy-GPU check.
func (s *Server) autoStartForkInstances(afterRouter bool) {
	if !s.forksEnabled() {
		return
	}
	var ids []string
	for _, inst := range s.forks.Instances() {
		if inst.AutoStart {
			ids = append(ids, inst.ID)
		}
	}
	s.autoStartInstances("fork", afterRouter, ids, func(id string) error {
		warnings, err := s.startForkInstance(id)
		if err == nil {
			s.forkNotes.set(id, warnings)
		}
		return err
	})
}

// forkVRAM adds the fork instances that are starting or running to the
// per-GPU reservations: each one reserves each of its GPUs whole, and is
// never stopped to make room. exclude is left out.
func (s *Server) forkVRAM(out map[int]*genVRAM, exclude string) {
	if !s.forksEnabled() || s.forkMgr == nil {
		return
	}
	for _, inst := range s.forks.Instances() {
		if inst.ID == exclude || !s.forkMgr.IsActive(inst.ID) {
			continue
		}
		// The process's memory is one number for all its GPUs; on two it
		// is split evenly, which is close enough under a whole-GPU hold.
		gpus := inst.GPUs()
		using := 0
		if pid := s.forkMgr.PID(inst.ID); pid > 0 {
			using, _ = s.genProcVRAM.read(pid)
		}
		using /= len(gpus)
		for _, gpu := range gpus {
			v := out[gpu]
			if v == nil {
				v = &genVRAM{}
				out[gpu] = v
			}
			v.Using += using
			v.Reserved += max(s.gpuTotalMiB(gpu), using)
			v.Names = append(v.Names, inst.ID+" (fork, whole GPU)")
		}
	}
}

// gpuTotalMiB is a GPU's memory, from the test hardware or the monitor;
// 0 when unknown. Not from hardware(), which counts the reservations and
// so calls back into genVRAM.
func (s *Server) gpuTotalMiB(gpu int) int {
	if s.testHardware != nil {
		for _, g := range s.testHardware.GPUs {
			if g.Index == gpu {
				return g.VRAMTotalMiB
			}
		}
		return 0
	}
	if s.monitor == nil {
		return 0
	}
	for _, g := range s.monitor.Current().GPU {
		if g.Index == gpu {
			return g.VRAMTotalMiB
		}
	}
	return 0
}
