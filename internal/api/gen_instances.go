package api

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"net/url"
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
	// EvictOn shows the keep-running buttons: without eviction every
	// instance already keeps its reservation and is never stopped for
	// another.
	EvictOn bool `json:"evict_on"`
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
	// Measuring is true during a measured test run. MeasureNote is what
	// the last one found, MeasureError how it failed, and MeasureAt when it
	// ended; all empty before the first one since llama-toolchest started.
	Measuring    bool      `json:"measuring"`
	MeasureNote  string    `json:"measure_note,omitempty"`
	MeasureError string    `json:"measure_error,omitempty"`
	MeasureAt    time.Time `json:"measure_at,omitzero"`
	// Video instances only: the size the instance may make (VideoLimit),
	// shown in the row, and VideoJSON, what the Measure dialog offers.
	Limit         generation.VideoSize `json:"video_limit,omitzero"`
	LimitMeasured bool                 `json:"video_limit_measured,omitempty"`
	VideoJSON     string               `json:"-"`
	// SuggestedMiB is the reservation the last measurement suggests; 0
	// before any measurement.
	SuggestedMiB int `json:"suggested_mib,omitempty"`
	// WebUIURL opens the backend's own web page; empty unless it is
	// running and its build has one. WebUINote says why there is no link
	// and how to get one, when that is not simply that it is stopped.
	WebUIURL  string `json:"web_ui_url,omitempty"`
	WebUINote string `json:"web_ui_note,omitempty"`
	// GPUJobsRunning and GPUJobsWaiting are the image and music jobs on
	// the instance's GPU, whichever instance they are for.
	GPUJobsRunning int `json:"gpu_jobs_running"`
	GPUJobsWaiting int `json:"gpu_jobs_waiting"`
	// GPUHeldBy names the LLM measurement (a benchmark, Autotune or
	// Autoconfigure's test load) that has the GPU while its jobs wait.
	GPUHeldBy string `json:"gpu_held_by,omitempty"`
}

// ReservedGiB and the others render the VRAM figures for the table.
func (r genInstanceRow) ReservedGiB() float64  { return float64(r.ReservedMiB) / 1024 }
func (r genInstanceRow) PeakGiB() float64      { return float64(r.MeasuredPeakMiB) / 1024 }
func (r genInstanceRow) SuggestedGiB() float64 { return float64(r.SuggestedMiB) / 1024 }

// genPortsPublished reports whether a browser can reach the generation
// backend ports. On the host it always can. A container publishes them
// only when setup.sh was told to, and docker-compose.gen-ports.yml (or the
// Quadlet unit) then sets LLAMA_TOOLCHEST_GEN_PORTS=1.
func (s *Server) genPortsPublished() bool {
	return !s.inContainer() || s.cfg.GenPortsPublished
}

// genWebUIURL is where a browser reaches the backend listening on port:
// the external URL's host, as for the chat UI link, on the port as seen
// from outside (a container may publish the range on other host ports).
// Empty when the external URL has no host, or the ports are not published.
func (s *Server) genWebUIURL(port int) string {
	if !s.genPortsPublished() {
		return ""
	}
	cfg := s.cfgCopy()
	u, err := url.Parse(cfg.ExternalURL)
	if err != nil || u.Hostname() == "" {
		return ""
	}
	inC := s.inContainer()
	start, _ := cfg.GenPorts(inC)
	return fmt.Sprintf("%s://%s:%d", u.Scheme, u.Hostname(), port-start+cfg.GenPublicStart(inC))
}

// genWebUILink is the link to an instance's own web page, or why there is
// none and what to do about it. A stopped instance has neither: the link
// appears once it runs. A build without the web page is reported either
// way, since starting the instance would not help.
func (s *Server) genWebUILink(b *builder.BuildResult, inst generation.Instance, running bool) (link, note string) {
	if !b.HasWebUI {
		return "", "No web UI: this build was made without the \"Built-in web UI\" option, which needs pnpm on the machine that builds. " +
			"To get one, build " + b.ProjectID() + " again on the Builds page with that option on, then use Edit to switch this instance to the new build."
	}
	if !running {
		return "", ""
	}
	if link := s.genWebUIURL(inst.Port); link != "" {
		return link, ""
	}
	if !s.genPortsPublished() {
		return "", "No web UI link: this container does not publish the web UI ports. Run setup.sh install again and answer yes to publishing them."
	}
	return "", "No web UI link: set the Server URL in Settings, which gives the host name the link uses."
}

// genFormData feeds the gen_instance_form partial: the choices a new
// instance can be made from, and when editing, the instance's current
// values to fill the form with.
type genFormData struct {
	Sets []generation.ModelSet
	// Builds are the successful builds that can run on this machine's
	// GPUs, by backend kind (sd, which video sets use too, and ace).
	Builds map[generation.BackendKind][]builder.BuildResult
	GPUs   []genGPUChoice
	// ForSet is the set a new instance is for, when the form was opened
	// from that set on the Models page. Sets and Builds then hold only it
	// and the builds of its backend.
	ForSet *generation.ModelSet
	// Missing explains why an instance cannot be made yet, e.g. no
	// downloaded set or no build.
	Missing []string
	// Editing is true for the edit form; the Edit fields hold the
	// instance's current values.
	Editing       bool
	EditID        string
	EditName      string
	EditSet       string
	EditBuild     string
	EditGPU       int
	EditArgs      string
	EditAutoStart bool
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
			Measuring: s.genMgr.IsMeasuring(inst.ID),
		}
		if o := s.genMeasure.get(inst.ID); !row.Measuring {
			row.MeasureNote, row.MeasureError, row.MeasureAt = o.Note, o.Err, o.At
		}
		if inst.MeasuredPeakMiB > 0 {
			row.SuggestedMiB = generation.SuggestReservation(inst.MeasuredPeakMiB)
			if inst.MeasuredSuggestedMiB > 0 {
				row.SuggestedMiB = inst.MeasuredSuggestedMiB
			}
		}
		if set := sets[inst.ModelSetID]; set.Kind == generation.KindVideo {
			row.Limit, row.LimitMeasured = generation.VideoLimit(inst, set)
			row.VideoJSON = videoMeasureJSON(set)
		}
		if b := s.resolveBackendBuild(string(row.Kind), inst.BuildID); b != nil {
			row.BuildLabel = b.ID
			row.WebUIURL, row.WebUINote = s.genWebUILink(b, inst, row.Status.State == process.StateRunning)
		}
		snap := s.genQueue.Snapshot(inst.GPU)
		if snap.Running != nil {
			row.GPUJobsRunning = 1
		}
		row.GPUJobsWaiting, row.GPUHeldBy = len(snap.Waiting), snap.HeldBy
		rows = append(rows, row)
	}
	return rows
}

// videoMeasureJSON is what the Measure dialog offers a video set: its
// presets, frame rate and limits, and how long measuring the largest took.
func videoMeasureJSON(set generation.ModelSet) string {
	info := map[string]any{
		"presets":   generation.VideoPresets(set),
		"fps":       set.DefaultFPS,
		"maxFrames": set.MaxFrames,
		"maxSide":   max(set.MaxWidth, set.MaxHeight),
	}
	if e, ok := generation.LookupCatalog(set.CatalogID); ok {
		info["fullMeasureTime"] = e.FullMeasureTime
	}
	b, _ := json.Marshal(info)
	return string(b)
}

func (s *Server) handleGenInstances(w http.ResponseWriter, r *http.Request) {
	store := s.genStore(w)
	if store == nil {
		return
	}
	data := genInstancesData{Rows: s.genInstanceRows(store), EvictOn: s.genEvictOn()}
	if !isHTMX(r) {
		respondJSON(w, data)
		return
	}
	respondHTML(w)
	s.renderPartial(w, "gen_instances_table", data)
}

// handleGenInstanceForm is the form for a new instance. With ?set=<id> it
// is for that set only (the Models page's New instance dialog): the set is
// fixed and only builds of its backend are offered.
func (s *Server) handleGenInstanceForm(w http.ResponseWriter, r *http.Request) {
	store := s.genStore(w)
	if store == nil {
		return
	}
	data := s.genFormChoices(store)
	if id := r.URL.Query().Get("set"); id != "" {
		set, err := store.ModelSet(id)
		if err != nil {
			http.Error(w, err.Error(), http.StatusNotFound)
			return
		}
		data = s.genFormChoicesFor(data, set)
	}
	respondHTML(w)
	s.renderPartial(w, "gen_instance_form", data)
}

// handleGenInstanceEditForm is the form for changing a stopped instance,
// filled with its current values. It is shown in a dialog, so the
// instance table can keep refreshing underneath.
func (s *Server) handleGenInstanceEditForm(w http.ResponseWriter, r *http.Request) {
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
	data := s.genFormChoices(store)
	data.Editing, data.EditID, data.EditName = true, inst.ID, inst.Name
	data.EditSet, data.EditBuild, data.EditGPU = inst.ModelSetID, inst.BuildID, inst.GPU
	data.EditArgs, data.EditAutoStart = strings.Join(inst.ExtraArgs, " "), inst.AutoStart
	respondHTML(w)
	s.renderPartial(w, "gen_instance_form", data)
}

// genFormChoices is what the create and edit forms offer: the downloaded
// sets, the successful builds that can run on this machine's GPUs, and the
// GPUs.
func (s *Server) genFormChoices(store *generation.Store) genFormData {
	data := genFormData{Builds: map[generation.BackendKind][]builder.BuildResult{}}
	for _, m := range store.ModelSets() {
		if m.Complete {
			data.Sets = append(data.Sets, m)
		}
	}
	// Video sets run on sd builds, so the sd list covers them too.
	for _, kind := range []generation.BackendKind{generation.KindSD, generation.KindAce} {
		data.Builds[kind] = s.genBuildsFor(kind)
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
	return data
}

// genFormChoicesFor narrows the new-instance choices to one set: the set
// itself and the builds of its backend.
func (s *Server) genFormChoicesFor(data genFormData, set generation.ModelSet) genFormData {
	data.ForSet = &set
	data.Sets, data.Missing = []generation.ModelSet{set}, nil
	kind := set.Kind
	if kind == generation.KindVideo {
		kind = generation.KindSD
	}
	data.Builds = map[generation.BackendKind][]builder.BuildResult{kind: data.Builds[kind]}
	if !set.Complete {
		data.Missing = append(data.Missing, "all of "+set.Name+"'s files (the download is not finished)")
	}
	if len(data.Builds[kind]) == 0 {
		data.Missing = append(data.Missing, "a successful "+kindProject(set.Kind)+" build for this machine's GPUs (Builds page)")
	}
	return data
}

// genBuildsFor lists the successful builds of the backend that runs kind,
// newest first, leaving out builds this machine's GPUs cannot run (a CUDA
// build on AMD GPUs, or a ROCm build on NVIDIA GPUs).
func (s *Server) genBuildsFor(kind generation.BackendKind) []builder.BuildResult {
	p, ok := builder.ProjectForKind(string(kind))
	if !ok {
		return nil
	}
	vendor := s.monitor.GPUVendor()
	var out []builder.BuildResult
	for _, b := range s.builder.SuccessfulBuildsRankedFor(p.ID) {
		if backend, err := generation.BuildBackend(b); err == nil && generation.CheckGPUVendor(backend, vendor) != nil {
			continue
		}
		out = append(out, b)
	}
	return out
}

// genInstanceFromForm reads the instance fields the create and edit forms
// share, checking them against the store, builds and GPUs.
func (s *Server) genInstanceFromForm(r *http.Request, store *generation.Store, inst *generation.Instance) error {
	if err := r.ParseForm(); err != nil {
		return fmt.Errorf("invalid form: %w", err)
	}
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
	// resolveBackendBuild falls back to the newest build when the chosen
	// one cannot run the set; refuse that here instead of saving a build
	// that is then quietly not used.
	if buildID != "" && build.ID != buildID {
		return fmt.Errorf("build %s is not a successful %s build, so it cannot run %s; choose another build or Automatic", buildID, kindProject(set.Kind), set.Name)
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
	if err := generation.CheckLoraArgs(set, args); err != nil {
		return err
	}
	// Running it differently invalidates any VRAM measurement, and the
	// last measurement's note or error with it.
	if inst.ModelSetID != set.ID || !slicesEqual(inst.ExtraArgs, args) || inst.BuildID != buildID || inst.GPU != gpu {
		inst.ReservedMiB, inst.MeasuredPeakMiB, inst.MeasuredAt, inst.MeasureEstimated = 0, 0, time.Time{}, false
		inst.MeasuredSuggestedMiB, inst.MeasuredFrames, inst.MeasuredWidth, inst.MeasuredHeight = 0, 0, 0, 0
		s.genMeasure.clear(inst.ID)
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
	if p, ok := builder.ProjectForKind(string(kind)); ok {
		return p.ID
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
	cfg := s.cfgCopy()
	start, count := cfg.GenPorts(s.inContainer())
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
	if err := s.genMove.busy(); err != nil {
		return err
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
		// Its reservation starts to count now, so it has to fit; with
		// eviction on it counts only once the instance runs, unless the
		// instance is kept running.
		if s.genEvictOn() && !inst.KeepRunning {
			inst.Enabled = true
			err = store.UpdateInstance(inst)
		} else if err = s.genFitCheck(inst, inst.ReservedMiB); err == nil {
			inst.Enabled = true
			err = store.UpdateInstance(inst)
		}
	case "disable":
		if err = s.genMgr.Stop(inst.ID); err == nil {
			inst.Enabled = false
			err = store.UpdateInstance(inst)
		}
	case "keep-running":
		// Its reservation now counts even while it is stopped, so an
		// enabled instance has to fit. A running one already counts.
		if inst.Enabled && !s.genMgr.IsActive(inst.ID) {
			err = s.genFitCheck(inst, inst.ReservedMiB)
		}
		if err == nil {
			inst.KeepRunning = true
			err = store.UpdateInstance(inst)
		}
	case "allow-stop":
		inst.KeepRunning = false
		err = store.UpdateInstance(inst)
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
	// Under cfgMu: this runs in the background, also right after Settings
	// switched generation on, while a further save can change it.
	s.cfgMu.Lock()
	on := s.cfg.GenerationEnabled
	s.cfgMu.Unlock()
	if !on || s.gen == nil {
		return
	}
	// Model files moved at startup must be in place before anything runs.
	s.genMove.wait()
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
			slog.Warn("failed to auto-start a generation instance", "instance", inst.ID, "error", err)
		} else {
			slog.Info("auto-started generation instance", "instance", inst.ID)
		}
	}
}
