package api

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/tmac1973/llama-toolchest/internal/builder"
	"github.com/tmac1973/llama-toolchest/internal/generation"
	"github.com/tmac1973/llama-toolchest/internal/process"
)

// genListenHost is the address generation backends listen on: the same
// as the router's default, so each web UI is reachable on its published
// port the way the chat UI is on 8080.
const genListenHost = "0.0.0.0"

// genInstancesData feeds the gen_instances_table partial.
type genInstancesData struct {
	Rows []genInstanceRow `json:"instances"`
}

type genInstanceRow struct {
	generation.Instance
	SetName string                 `json:"set_name"`
	Kind    generation.BackendKind `json:"kind"`
	// BuildLabel is the build the instance runs, or the one it would
	// start with when it has none pinned.
	BuildLabel string         `json:"build"`
	GPUName    string         `json:"gpu_name"`
	Status     process.Status `json:"status"`
	// Measuring is true during a measured test run; MeasureError is how
	// the last one failed, if it did.
	Measuring    bool   `json:"measuring"`
	MeasureError string `json:"measure_error,omitempty"`
	// SuggestedMiB is the reservation the last measurement suggests; 0
	// before any measurement.
	SuggestedMiB int `json:"suggested_mib,omitempty"`
}

// ReservedGiB and the others render the VRAM figures for the table.
func (r genInstanceRow) ReservedGiB() float64  { return float64(r.ReservedMiB) / 1024 }
func (r genInstanceRow) PeakGiB() float64      { return float64(r.MeasuredPeakMiB) / 1024 }
func (r genInstanceRow) SuggestedGiB() float64 { return float64(r.SuggestedMiB) / 1024 }

// genFormData feeds the gen_instance_form partial: the choices a new
// instance can be made from.
type genFormData struct {
	Sets   []generation.ModelSet
	Builds map[generation.BackendKind][]builder.BuildResult
	GPUs   []genGPUChoice
	// Missing explains why an instance cannot be made yet, e.g. no
	// downloaded set or no build.
	Missing []string
}

type genGPUChoice struct {
	Index int
	Name  string
}

func (s *Server) genInstanceRows(store *generation.Store) []genInstanceRow {
	sets := map[string]generation.ModelSet{}
	for _, m := range store.ModelSets() {
		sets[m.ID] = m
	}
	gpuNames := map[int]string{}
	for _, g := range s.monitor.Current().GPU {
		gpuNames[g.Index] = g.Name
	}
	var rows []genInstanceRow
	for _, inst := range store.Instances() {
		row := genInstanceRow{
			Instance: inst, SetName: sets[inst.ModelSetID].Name, Kind: sets[inst.ModelSetID].Kind,
			GPUName: gpuNames[inst.GPU], Status: s.genMgr.Status(inst.ID),
			Measuring: s.genMgr.IsMeasuring(inst.ID), MeasureError: s.genMeasure.get(inst.ID),
		}
		if inst.MeasuredPeakMiB > 0 {
			row.SuggestedMiB = suggestReservation(inst.MeasuredPeakMiB)
		}
		if b := s.resolveBackendBuild(string(row.Kind), inst.BuildID); b != nil {
			row.BuildLabel = b.ID
		}
		rows = append(rows, row)
	}
	return rows
}

func (s *Server) handleGenInstances(w http.ResponseWriter, r *http.Request) {
	store := s.genStore(w)
	if store == nil {
		return
	}
	data := genInstancesData{Rows: s.genInstanceRows(store)}
	if !isHTMX(r) {
		respondJSON(w, data)
		return
	}
	respondHTML(w)
	s.renderPartial(w, "gen_instances_table", data)
}

func (s *Server) handleGenInstanceForm(w http.ResponseWriter, r *http.Request) {
	store := s.genStore(w)
	if store == nil {
		return
	}
	data := genFormData{Builds: map[generation.BackendKind][]builder.BuildResult{}}
	for _, m := range store.ModelSets() {
		if m.Complete {
			data.Sets = append(data.Sets, m)
		}
	}
	for _, kind := range []generation.BackendKind{generation.KindSD, generation.KindAce} {
		for _, p := range builder.Projects() {
			if p.Kind == string(kind) {
				data.Builds[kind] = s.builder.SuccessfulBuildsRankedFor(p.ID)
			}
		}
	}
	for _, g := range s.monitor.Current().GPU {
		data.GPUs = append(data.GPUs, genGPUChoice{Index: g.Index, Name: g.Name})
	}
	if len(data.Sets) == 0 {
		data.Missing = append(data.Missing, "a fully downloaded model set (Models page, Generation models)")
	}
	if len(data.Builds[generation.KindSD]) == 0 && len(data.Builds[generation.KindAce]) == 0 {
		data.Missing = append(data.Missing, "a successful stable-diffusion.cpp or acestep.cpp build (Builds page)")
	}
	respondHTML(w)
	s.renderPartial(w, "gen_instance_form", data)
}

// genInstanceFromForm reads the instance fields the create and edit forms
// share, checking them against the store, builds and GPUs.
func (s *Server) genInstanceFromForm(r *http.Request, store *generation.Store, inst *generation.Instance) error {
	r.ParseForm()
	set, err := store.ModelSet(r.FormValue("model_set"))
	if err != nil {
		return fmt.Errorf("choose a model set: %w", err)
	}
	if !set.Complete {
		return fmt.Errorf("model set %s is not fully downloaded yet", set.Name)
	}
	buildID := r.FormValue("build")
	build := s.resolveBackendBuild(string(set.Kind), buildID)
	if build == nil {
		return fmt.Errorf("there is no successful %s build to run %s; build one on the Builds page", kindProject(set.Kind), set.Name)
	}
	backend, err := generation.BuildBackend(*build)
	if err != nil {
		return err
	}
	if err := generation.CheckGPUVendor(backend, s.monitor.GPUVendor()); err != nil {
		return err
	}
	gpu, err := strconv.Atoi(r.FormValue("gpu"))
	if err != nil || gpu < 0 {
		return errors.New("choose a GPU")
	}
	if gpus := s.monitor.Current().GPU; len(gpus) > 0 && gpu >= len(gpus) {
		return fmt.Errorf("this machine has no GPU %d", gpu)
	}
	args, err := generation.ValidateExtraArgs(r.FormValue("extra_args"))
	if err != nil {
		return err
	}
	// Running it differently invalidates any VRAM measurement.
	if inst.ModelSetID != set.ID || !slicesEqual(inst.ExtraArgs, args) || inst.BuildID != buildID || inst.GPU != gpu {
		inst.ReservedMiB, inst.MeasuredPeakMiB, inst.MeasuredAt, inst.MeasureEstimated = 0, 0, time.Time{}, false
	}
	inst.ModelSetID, inst.BuildID, inst.GPU, inst.ExtraArgs = set.ID, buildID, gpu, args
	inst.AutoStart = r.FormValue("auto_start") == "on"
	return nil
}

func slicesEqual(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func kindProject(kind generation.BackendKind) string {
	for _, p := range builder.Projects() {
		if p.Kind == string(kind) {
			return p.ID
		}
	}
	return string(kind)
}

func (s *Server) handleGenCreateInstance(w http.ResponseWriter, r *http.Request) {
	store := s.genStore(w)
	if store == nil {
		return
	}
	inst := generation.Instance{Enabled: true}
	if err := s.genInstanceFromForm(r, store, &inst); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	name := strings.TrimSpace(r.FormValue("name"))
	if name == "" {
		name = s.freeInstanceName(store, inst.ModelSetID)
	}
	inst.Name, inst.ID = name, generation.Slug(name)
	if inst.ID == "" {
		http.Error(w, "the instance needs a name", http.StatusBadRequest)
		return
	}
	start, count := s.cfg.GenPorts(s.inContainer())
	port, err := store.AssignPort(start, count)
	if err != nil {
		http.Error(w, err.Error()+"; widen the range in Settings, or delete an instance", http.StatusConflict)
		return
	}
	inst.Port = port
	if err := store.AddInstance(inst); err != nil {
		code := http.StatusInternalServerError
		if errors.Is(err, generation.ErrIDTaken) {
			code = http.StatusConflict
		}
		http.Error(w, err.Error(), code)
		return
	}
	s.genInstancesChanged(w, r)
}

// freeInstanceName is the set's ID, or the set's ID with -2, -3 and so on
// when that is taken.
func (s *Server) freeInstanceName(store *generation.Store, base string) string {
	name := base
	for n := 2; ; n++ {
		if _, err := store.Instance(generation.Slug(name)); errors.Is(err, generation.ErrNotFound) {
			return name
		}
		name = fmt.Sprintf("%s-%d", base, n)
	}
}

// handleGenEditInstance changes a stopped instance's set, build, GPU,
// auto-start or extra arguments.
func (s *Server) handleGenEditInstance(w http.ResponseWriter, r *http.Request) {
	store := s.genStore(w)
	if store == nil {
		return
	}
	inst, ok := s.genInstance(w, r, store)
	if !ok {
		return
	}
	if s.genMgr.IsActive(inst.ID) || s.genMgr.IsMeasuring(inst.ID) {
		http.Error(w, "stop the instance before editing it", http.StatusConflict)
		return
	}
	if err := s.genInstanceFromForm(r, store, &inst); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if err := store.UpdateInstance(inst); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	s.genInstancesChanged(w, r)
}

// genInstance reads the {id} instance, writing a 404 when it is unknown.
func (s *Server) genInstance(w http.ResponseWriter, r *http.Request, store *generation.Store) (generation.Instance, bool) {
	inst, err := store.Instance(chi.URLParam(r, "id"))
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return inst, false
	}
	return inst, true
}

// startGenInstance starts an enabled instance that has a VRAM
// reservation, after checking the reservation still fits. Every start
// goes through here (the button, auto-start, the restart after a
// measurement), which is what keeps an enabled instance startable.
func (s *Server) startGenInstance(id string) error {
	store := s.gen
	inst, err := store.Instance(id)
	if err != nil {
		return err
	}
	if !inst.Enabled {
		return fmt.Errorf("instance %s is disabled; enable it first", inst.Name)
	}
	if s.genMgr.IsMeasuring(id) {
		return fmt.Errorf("instance %s is being measured", inst.Name)
	}
	if inst.ReservedMiB == 0 {
		return fmt.Errorf("instance %s has no VRAM reservation yet; press Measure to work one out", inst.Name)
	}
	if err := s.genFitCheck(inst, inst.ReservedMiB); err != nil {
		return err
	}
	set, err := store.ModelSet(inst.ModelSetID)
	if err != nil {
		return err
	}
	return s.launchGenInstance(inst, set, generation.LaunchOptions{})
}

// launchGenInstance runs an instance with no checks of its own.
func (s *Server) launchGenInstance(inst generation.Instance, set generation.ModelSet, opts generation.LaunchOptions) error {
	build := s.resolveBackendBuild(string(set.Kind), inst.BuildID)
	if build == nil {
		return fmt.Errorf("there is no successful %s build to run %s", kindProject(set.Kind), inst.Name)
	}
	cfg, err := generation.LaunchSpec(inst, set, *build, genListenHost, s.cfg.DataDir, opts)
	if err != nil {
		return err
	}
	return s.genMgr.Start(inst.ID, set.Kind, cfg)
}

func (s *Server) handleGenInstanceAction(w http.ResponseWriter, r *http.Request) {
	store := s.genStore(w)
	if store == nil {
		return
	}
	inst, ok := s.genInstance(w, r, store)
	if !ok {
		return
	}
	var err error
	switch chi.URLParam(r, "action") {
	case "start":
		err = s.startGenInstance(inst.ID)
	case "stop":
		err = s.genMgr.Stop(inst.ID)
	case "restart":
		if err = s.genMgr.Stop(inst.ID); err == nil {
			err = s.startGenInstance(inst.ID)
		}
	case "enable":
		// Its reservation starts to count now, so it has to fit.
		if err = s.genFitCheck(inst, inst.ReservedMiB); err == nil {
			inst.Enabled = true
			err = store.UpdateInstance(inst)
		}
	case "disable":
		if err = s.genMgr.Stop(inst.ID); err == nil {
			inst.Enabled = false
			err = store.UpdateInstance(inst)
		}
	default:
		http.Error(w, "unknown action", http.StatusNotFound)
		return
	}
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	s.genInstancesChanged(w, r)
}

func (s *Server) handleGenDeleteInstance(w http.ResponseWriter, r *http.Request) {
	store := s.genStore(w)
	if store == nil {
		return
	}
	inst, ok := s.genInstance(w, r, store)
	if !ok {
		return
	}
	if s.genMgr.IsActive(inst.ID) {
		http.Error(w, "stop the instance before deleting it", http.StatusConflict)
		return
	}
	if err := store.DeleteInstance(inst.ID); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	s.genMgr.Forget(inst.ID)
	s.genInstancesChanged(w, r)
}

func (s *Server) handleGenInstanceLogs(w http.ResponseWriter, r *http.Request) {
	if s.genStore(w) == nil {
		return
	}
	id := chi.URLParam(r, "id")
	ch := s.genMgr.Subscribe(id)
	if ch == nil {
		http.Error(w, "this instance has not run yet", http.StatusNotFound)
		return
	}
	defer s.genMgr.Unsubscribe(id, ch)
	StreamLines(w, r.Context(), ch, "Instance stopped")
}

// genInstancesChanged answers a change: the refreshed table for htmx, the
// instance list as JSON otherwise.
func (s *Server) genInstancesChanged(w http.ResponseWriter, r *http.Request) {
	if isHTMX(r) {
		w.Header().Set("HX-Trigger", "genInstancesChanged")
	}
	s.handleGenInstances(w, r)
}

// autoStartGenInstances starts every enabled auto-start instance. It does
// not depend on the router's own auto-start setting; when the router is
// auto-starting too, it waits until the router is running or has failed.
func (s *Server) autoStartGenInstances(afterRouter bool) {
	if !s.cfg.GenerationEnabled || s.gen == nil {
		return
	}
	if afterRouter {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
		_ = s.process.WaitRunning(ctx)
		cancel()
	}
	for _, inst := range s.gen.Instances() {
		if !inst.Enabled || !inst.AutoStart {
			continue
		}
		if err := s.startGenInstance(inst.ID); err != nil {
			slog.Warn("generation instance auto-start failed", "instance", inst.ID, "error", err)
		} else {
			slog.Info("auto-started generation instance", "instance", inst.ID)
		}
	}
}
