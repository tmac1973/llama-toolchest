package api

import (
	"net/http"
	"strconv"
	"strings"
	"sync"

	"github.com/tmac1973/llama-toolchest/internal/forks"
	"github.com/tmac1973/llama-toolchest/internal/generation"
	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/process"
)

// forkStartNotes keeps, per fork instance, the hardware warnings of its
// last start, so the Server page can show them under its state. Cleared
// on stop; lost on restart.
type forkStartNotes struct {
	mu   sync.Mutex
	byID map[string][]string
}

func (n *forkStartNotes) set(id string, notes []string) {
	n.mu.Lock()
	defer n.mu.Unlock()
	if n.byID == nil {
		n.byID = map[string][]string{}
	}
	if len(notes) == 0 {
		delete(n.byID, id)
		return
	}
	n.byID[id] = notes
}

func (n *forkStartNotes) get(id string) []string {
	n.mu.Lock()
	defer n.mu.Unlock()
	return n.byID[id]
}

// forkRow is one fork instance in the Server page card.
type forkRow struct {
	forks.Instance
	ForkName string
	// BuildLabel is the build the instance would run: the chosen one, or
	// the newest of the fork; empty when there is none. Untested marks a
	// build of another commit than the fork's pinned one.
	BuildLabel string
	Untested   bool
	GPUName    string
	Status     process.Status
	// SetReady reports whether the fork's model set is downloaded.
	SetReady bool
	Warnings []string
}

// Active reports whether the instance is starting or running.
func (r forkRow) Active() bool {
	return r.Status.State == process.StateStarting || r.Status.State == process.StateRunning
}

func (s *Server) forkRows(store *forks.Store) []forkRow {
	gpuNames := map[int]string{}
	for _, g := range s.monitor.Current().GPU {
		gpuNames[g.Index] = g.Name
	}
	var rows []forkRow
	for _, inst := range store.Instances() {
		row := forkRow{Instance: inst, ForkName: inst.ForkID, GPUName: gpuNames[inst.GPU],
			Status: s.forkMgr.Status(inst.ID), Warnings: s.forkNotes.get(inst.ID)}
		if f, ok := findFork(inst.ForkID); ok {
			row.ForkName = f.Name
			if b := s.resolveForkBuild(f.ID, inst.BuildID); b != nil {
				row.BuildLabel = b.ID
				row.Untested = b.GitSHA != f.PinnedSHA
			}
		}
		if set, err := store.ModelSet(inst.ForkID); err == nil {
			row.SetReady = set.Complete
		}
		rows = append(rows, row)
	}
	return rows
}

// forkFormData feeds the fork_instance_form partial.
type forkFormData struct {
	Forks []forkChoice
	GPUs  []genGPUChoice
	// PortStart and PortEnd are the shared generation port range.
	PortStart, PortEnd int
	// Editing is set when the form changes a stopped instance.
	Editing bool
	Inst    forks.Instance
	// ExtraArgs and ExtraEnv are the instance's values as the fields
	// show them.
	ExtraArgs, ExtraEnv string
}

// forkChoice is one fork in the form, with what the form says about it.
type forkChoice struct {
	forks.Fork
	SetReady bool
	Builds   []forkBuildChoice
}

type forkBuildChoice struct {
	ID       string
	Untested bool
}

func (s *Server) forkFormChoices(store *forks.Store) forkFormData {
	cfg := s.cfgCopy()
	start, count := cfg.GenPorts(s.inContainer())
	data := forkFormData{PortStart: start, PortEnd: start + count - 1}
	for _, g := range s.monitor.Current().GPU {
		data.GPUs = append(data.GPUs, genGPUChoice{Index: g.Index, Name: g.Name})
	}
	vendor := s.monitor.GPUVendor()
	for _, f := range curatedForks() {
		c := forkChoice{Fork: f}
		if set, err := store.ModelSet(f.ID); err == nil {
			c.SetReady = set.Complete
		}
		for _, b := range s.builder.SuccessfulBuildsRankedFor(f.ID) {
			// Only builds this machine's GPUs can run.
			if backend, err := generation.BuildBackend(b); err == nil && generation.CheckGPUVendor(backend, vendor) != nil {
				continue
			}
			c.Builds = append(c.Builds, forkBuildChoice{ID: b.ID, Untested: b.GitSHA != f.PinnedSHA})
		}
		data.Forks = append(data.Forks, c)
	}
	return data
}

// handleForkInstanceForm is the Add form, or with ?id= the Edit form of
// a stopped instance.
func (s *Server) handleForkInstanceForm(w http.ResponseWriter, r *http.Request) {
	store := s.forkStore(w)
	if store == nil {
		return
	}
	data := s.forkFormChoices(store)
	if id := r.URL.Query().Get("id"); id != "" {
		inst, err := store.Instance(id)
		if err != nil {
			http.Error(w, err.Error(), http.StatusNotFound)
			return
		}
		if s.forkMgr.IsActive(id) {
			http.Error(w, "stop the instance before editing it", http.StatusConflict)
			return
		}
		data.Editing, data.Inst = true, inst
		data.ExtraArgs = joinArgs(inst.ExtraArgs)
		data.ExtraEnv = strings.Join(inst.ExtraEnv, "\n")
	}
	respondHTML(w)
	s.renderPartial(w, "fork_instance_form", data)
}

// joinArgs writes arguments back as one line, quoting those with spaces,
// so ValidateExtraArgs reads them back the same.
func joinArgs(args []string) string {
	out := make([]string, len(args))
	for i, a := range args {
		if strings.ContainsAny(a, " \t\"") {
			a = "'" + a + "'"
		}
		out[i] = a
	}
	return strings.Join(out, " ")
}

// handleForkNeeds lists the needs of a fork that this host falls short of
// on a GPU, for the form's warning box.
func (s *Server) handleForkNeeds(w http.ResponseWriter, r *http.Request) {
	if s.forkStore(w) == nil {
		return
	}
	f, ok := findFork(r.URL.Query().Get("fork"))
	if !ok {
		respondHTML(w)
		return
	}
	gpu, _ := strconv.Atoi(r.URL.Query().Get("gpu"))
	backend := ""
	if b := s.resolveForkBuild(f.ID, r.URL.Query().Get("build")); b != nil {
		backend, _ = generation.BuildBackend(*b)
	} else if s.monitor.GPUVendor() == "nvidia" {
		backend = "cuda"
	} else if s.monitor.GPUVendor() == "rocm" {
		backend = "rocm"
	}
	respondHTML(w)
	s.renderPartial(w, "fork_needs", forks.CheckNeeds(f, s.forkHostInfo(gpu, backend)))
}

// forkReservationAllocations adds the running fork instances to the GPU
// map, each as its whole GPU, striped as reserved.
func (s *Server) forkReservationAllocations(firstColor int) []models.GPUAllocation {
	if !s.forksOn() || s.forks == nil {
		return nil
	}
	var out []models.GPUAllocation
	for _, inst := range s.forks.Instances() {
		if !s.forkMgr.IsActive(inst.ID) {
			continue
		}
		gb := float64(s.gpuTotalMiB(inst.GPU)) / 1024
		if gb == 0 {
			continue
		}
		out = append(out, models.GPUAllocation{
			ModelID: "fork:" + inst.ID, ModelName: inst.ID + " (fork, whole GPU)",
			GPUs: []int{inst.GPU}, PerGPUGB: gb, TotalGB: gb,
			Color: models.AllocationColor(firstColor + len(out)), Reserved: true,
		})
	}
	return out
}
