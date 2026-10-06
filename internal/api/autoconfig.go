package api

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/go-chi/chi/v5"

	"github.com/tmac1973/llama-toolchest/internal/autoconfig"
	"github.com/tmac1973/llama-toolchest/internal/benchmark"
	"github.com/tmac1973/llama-toolchest/internal/models"
)

// autoconfigProfileName is the profile autoconfigure saves to.
const autoconfigProfileName = "Autoconfig"

// autoconfigTimeout bounds one run: loading the helper, reading the card
// and answering, then loading the proposal to check that it runs.
// Generous, because a first load reads the model from disk, and a
// proposal that does not fit is adjusted and loaded again, a few minutes
// each time.
const autoconfigTimeout = 30 * time.Minute

// autoconfigCheckPreset is the benchmark preset a test load runs.
const autoconfigCheckPreset = "autoconfig-check"

// autoconfigCheckProfile names the settings a test load runs under, in
// the benchmark job that records it. They are not a saved profile yet.
const autoconfigCheckProfile = "Autoconfig (proposed)"

// autoconfigState tracks the one autoconfigure run the server allows at a
// time, and keeps its result until the user saves or discards it.
type autoconfigState struct {
	mu  sync.Mutex
	run *autoconfigRun
}

type autoconfigRun struct {
	modelID  string
	progress string
	done     bool
	result   *autoconfig.Result
	err      error
}

// autoconfigSnapshot returns a copy of the current run, if any.
func (s *Server) autoconfigSnapshot() (autoconfigRun, bool) {
	s.autoconf.mu.Lock()
	defer s.autoconf.mu.Unlock()
	if s.autoconf.run == nil {
		return autoconfigRun{}, false
	}
	return *s.autoconf.run, true
}

// hardware describes this machine for the fit planner.
func (s *Server) hardware() models.Hardware {
	// The estimate's coefficients follow the build that will run: CUDA
	// and ROCm allocate differently enough to need their own. Read here
	// because every plan starts by asking for the machine, so a change of
	// active build is picked up by the next one.
	if s.builder != nil {
		models.SetVRAMBackend(buildBackend(s.resolveActiveBuild()))
	}
	if s.testHardware != nil {
		return *s.testHardware
	}
	if s.monitor == nil {
		return models.Hardware{}
	}
	m := s.monitor.Current()
	hw := models.Hardware{LogicalCores: m.CPU.Cores, RAMTotalMiB: m.Memory.TotalMB}
	other := s.otherVRAMMiB()
	for i, g := range m.GPU {
		spec := models.GPUSpec{Index: g.Index, Name: g.Name, VRAMTotalMiB: g.VRAMTotalMB, IsIGPU: g.IsIGPU}
		if i < len(other) {
			spec.OtherUsedMiB = other[i]
		}
		hw.GPUs = append(hw.GPUs, spec)
	}
	return hw
}

// otherLoadedModels lists models the router has loaded, other than skip,
// which loading the helper may unload.
func (s *Server) otherLoadedModels(skipRouterName string) []string {
	if !s.process.IsRunning() {
		return nil
	}
	list, err := s.process.ListModels()
	if err != nil {
		return nil
	}
	var out []string
	for _, lm := range list {
		if lm.Status.Value == "loaded" && lm.ID != skipRouterName {
			out = append(out, lm.ID)
		}
	}
	return out
}

// autoconfigDialogData is what the autoconfig_dialog partial renders.
type autoconfigDialogData struct {
	ModelID     string
	ModelName   string
	Helper      string // helper model name, "" when none is installed
	BusyReason  string
	OtherLoaded []string
	Classes     []contextClassOption
}

type contextClassOption struct {
	Value, Label, Help string
	Checked            bool
}

func contextClassOptions(maxCtx int) []contextClassOption {
	maxLabel := "Maximum — the largest this model supports that fits"
	if maxCtx > 0 {
		maxLabel = fmt.Sprintf("Maximum — up to %s tokens, as much as fits", models.GroupDigits(maxCtx))
	}
	return []contextClassOption{
		{Value: "short", Label: "Short — about 8,000 tokens", Help: "A few pages of text. Uses the least memory, leaving the most for speed."},
		{Value: "medium", Label: "Medium — about 32,000 tokens", Help: "A long document or a long conversation. A good default.", Checked: true},
		{Value: "long", Label: "Long — about 128,000 tokens", Help: "A small codebase or a book chapter. Needs much more memory."},
		{Value: "max", Label: maxLabel, Help: "The model's own limit, reduced only as far as needed to fit."},
	}
}

// handleAutoconfigDialog shows the start dialog, or the result of a
// finished run for this model that has not been saved or discarded.
func (s *Server) handleAutoconfigDialog(w http.ResponseWriter, r *http.Request) {
	id := s.registry.ResolveID(chi.URLParam(r, "id"))
	if run, ok := s.autoconfigSnapshot(); ok && run.modelID == id {
		s.renderAutoconfigStatus(w, id, run)
		return
	}
	m, err := s.registry.Get(id)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	d := autoconfigDialogData{ModelID: id, ModelName: m.PublicName(), Classes: contextClassOptions(m.ContextLength)}
	helperRouter := ""
	if h := s.helperModel(); h != nil {
		d.Helper = h.PublicName()
		helperRouter = s.registry.RouterName(h.ID)
	}
	if s.routerBusyWithJob() {
		d.BusyReason = "A benchmark is running and using the GPU. Start Autoconfigure when it finishes."
	} else if run, ok := s.autoconfigSnapshot(); ok && !run.done {
		d.BusyReason = "Autoconfigure is already running for another model. Wait for it to finish."
	}
	// Listed with or without a helper: the test load restarts the server
	// either way.
	d.OtherLoaded = s.otherLoadedModels(helperRouter)
	respondHTML(w)
	s.renderPartial(w, "autoconfig_dialog", d)
}

// handleAutoconfigStart starts a run (form: context_class).
func (s *Server) handleAutoconfigStart(w http.ResponseWriter, r *http.Request) {
	id := s.registry.ResolveID(chi.URLParam(r, "id"))
	r.ParseForm()
	class := models.ContextClass(r.FormValue("context_class"))
	switch class {
	case models.ContextShort, models.ContextMedium, models.ContextLong, models.ContextMax:
	default:
		class = models.ContextMedium
	}
	if _, err := s.registry.Get(id); err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	// Refusals render as a message rather than an error status: htmx does
	// not swap a non-2xx response, and a refusal nobody sees looks like a
	// button that does nothing.
	if s.routerBusyWithJob() {
		s.renderAutoconfigMessage(w, id, panelBanner{"error", "A benchmark is running and using the GPU. Start Autoconfigure when it finishes."})
		return
	}

	s.autoconf.mu.Lock()
	if cur := s.autoconf.run; cur != nil && !cur.done {
		s.autoconf.mu.Unlock()
		s.renderAutoconfigMessage(w, id, panelBanner{"error", "Autoconfigure is already running for " + cur.modelID + ". Wait for it to finish."})
		return
	}
	run := &autoconfigRun{modelID: id, progress: "Starting"}
	s.autoconf.run = run
	s.autoconf.mu.Unlock()

	deps := autoconfig.Deps{
		Registry: s.registry,
		Hardware: s.hardware(),
		Fetcher:  s.presets,
		HFBase:   s.hfSiteBase(),
		Hub:      s.hfClient,
		LLM:      s.llm,
		Progress: func(p string) {
			s.autoconf.mu.Lock()
			run.progress = p
			s.autoconf.mu.Unlock()
		},
		Verify: s.checkProposedConfig,
	}
	helper := s.helperModel()
	if helper != nil {
		// The helper is put on its fixed settings before it is used, so
		// the card budget is sized from those, not from what is stored
		// now.
		deps.HelperID = helper.ID
		deps.HelperContext = models.HelperConfig(helper, s.helperVRAMBudget()).ContextSize
	}

	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), autoconfigTimeout)
		defer cancel()
		res, err := autoconfig.Run(ctx, deps, id, class)
		if helper != nil {
			s.unloadHelper(helper.ID)
		}
		if errors.Is(err, context.DeadlineExceeded) {
			err = fmt.Errorf("Autoconfigure took longer than %s and was stopped", autoconfigTimeout)
		}
		s.autoconf.mu.Lock()
		run.done, run.result, run.err = true, res, err
		s.autoconf.mu.Unlock()
		if err != nil {
			slog.Warn("autoconfigure failed", "model", id, "error", err)
		}
	}()

	snap, _ := s.autoconfigSnapshot()
	s.renderAutoconfigStatus(w, id, snap)
}

// deviceInErrorRE finds the GPU a llama.cpp memory error names: "on
// device 2" when a buffer could not be allocated, "current device: 2"
// when a running graph could not.
var deviceInErrorRE = regexp.MustCompile(`device:? (\d+)`)

// checkProposedConfig loads cfg for the model and sends it one request,
// to find out whether settings that fit on paper run on this machine.
//
// It runs as an ordinary one-cell benchmark job. That is what takes the
// router over and hands it back, refuses to start while another job has
// it, and records what llama-server said when the load fails — and it
// leaves the test in the benchmark history, where a failed one can be
// read afterwards.
func (s *Server) checkProposedConfig(ctx context.Context, modelID string, cfg models.ModelConfig) (autoconfig.Check, error) {
	if s.jobs == nil {
		return autoconfig.Check{}, errors.New("benchmarks are not available on this server")
	}
	build := s.resolveActiveBuild()
	if build == nil {
		return autoconfig.Check{}, errors.New("no llama.cpp build is available to load the model with")
	}
	name := modelID
	if m, err := s.registry.Get(modelID); err == nil {
		name = m.PublicName()
	}
	job := benchmark.BenchmarkJob{
		ID:          newJobID(),
		Name:        "Autoconfigure: test load of the proposed settings — " + name,
		Description: "Loads the settings Autoconfigure proposes and sends one request, to check that they run on this machine before they are saved.",
		Kind:        benchmark.JobKindBatch,
		Status:      benchmark.JobStatusPending,
		CreatedAt:   time.Now(),
		ModelIDs:    []string{modelID},
		BuildIDs:    []string{build.ID},
		Presets:     []string{autoconfigCheckPreset},
		BaseProfile: &benchmark.BaseProfile{Name: autoconfigCheckProfile, Config: cfg},
		Cells:       benchmark.ExpandCells([]string{modelID}, []string{build.ID}, []string{autoconfigCheckPreset}),
	}
	if err := s.jobs.Submit(job); err != nil {
		if errors.Is(err, benchmark.ErrJobAlreadyRunning) {
			return autoconfig.Check{}, errors.New("a benchmark started and is using the GPU")
		}
		return autoconfig.Check{}, err
	}
	done, err := s.jobs.Wait(ctx, job.ID)
	if err != nil {
		if ctx.Err() != nil {
			// Out of time, or stopped: the job must not go on holding
			// the router after the run that asked for it has gone.
			_ = s.jobs.Cancel(job.ID)
		}
		return autoconfig.Check{}, err
	}
	return checkFromJob(done, s.bench.Get), nil
}

// checkFromJob reads a finished test-load job. getRun looks up the run
// behind the cell, which carries what llama-server printed.
func checkFromJob(job *benchmark.BenchmarkJob, getRun func(id string) (*benchmark.BenchmarkRun, error)) autoconfig.Check {
	if job == nil || len(job.Cells) == 0 {
		return autoconfig.Check{Device: -1, Reason: "the test load did not run"}
	}
	cell := job.Cells[0]
	if cell.Status == benchmark.CellStatusCompleted {
		// Loaded and answered, but with a card nearly full: normal use
		// would run out of memory on it, so it is treated like a load
		// that did, and the settings are adjusted for that card.
		if cell.BenchmarkRunID != "" && getRun != nil {
			if run, err := getRun(cell.BenchmarkRunID); err == nil && run != nil {
				if msg := run.MemoryShortfall(); msg != "" {
					card, _ := run.TightestCard()
					return autoconfig.Check{OutOfMemory: true, LowMemory: true, Device: card.Index, Reason: msg}
				}
			}
		}
		return autoconfig.Check{OK: true, Device: -1}
	}
	chk := autoconfig.Check{Device: -1, Reason: cell.Error}
	if chk.Reason == "" {
		chk.Reason = "the test load ended with status " + cell.Status
	}
	// The cell's error quotes one line; the run keeps every error line,
	// and the one naming the GPU is often not the one quoted.
	lines := []string{cell.Error}
	if cell.BenchmarkRunID != "" {
		if run, err := getRun(cell.BenchmarkRunID); err == nil && run != nil {
			lines = append(lines, run.FailureLog...)
		}
	}
	chk.OutOfMemory = benchmark.OutOfMemory(lines)
	if chk.OutOfMemory {
		for _, l := range lines {
			if m := deviceInErrorRE.FindStringSubmatch(l); m != nil {
				chk.Device, _ = strconv.Atoi(m[1])
				break
			}
		}
	}
	return chk
}

// hfSiteBase is the Hugging Face site model cards are read from.
func (s *Server) hfSiteBase() string {
	if s.presets != nil && s.presets.HFBase != "" {
		return s.presets.HFBase
	}
	return "https://huggingface.co"
}

// handleAutoconfigStatus is polled while a run is going; once it is done
// it returns the review.
func (s *Server) handleAutoconfigStatus(w http.ResponseWriter, r *http.Request) {
	id := s.registry.ResolveID(chi.URLParam(r, "id"))
	run, ok := s.autoconfigSnapshot()
	if !ok || run.modelID != id {
		respondHTML(w) // nothing to show: the container empties
		return
	}
	s.renderAutoconfigStatus(w, id, run)
}

func (s *Server) renderAutoconfigStatus(w http.ResponseWriter, id string, run autoconfigRun) {
	respondHTML(w)
	name := id
	if m, err := s.registry.Get(id); err == nil {
		name = m.PublicName()
	}
	if !run.done {
		s.renderPartial(w, "autoconfig_progress", struct {
			ModelID, ModelName, Progress string
		}{id, name, run.progress})
		return
	}
	s.renderPartial(w, "autoconfig_review", s.autoconfigReviewData(id, name, run))
}

// handleAutoconfigSave saves the proposal as the Autoconfig profile, and
// with apply=1 also makes it the live config.
func (s *Server) handleAutoconfigSave(w http.ResponseWriter, r *http.Request) {
	id := s.registry.ResolveID(chi.URLParam(r, "id"))
	r.ParseForm()
	apply := r.FormValue("apply") == "1"

	run, ok := s.autoconfigSnapshot()
	if !ok || run.modelID != id || !run.done || run.result == nil {
		s.renderAutoconfigMessage(w, id, panelBanner{"error", "There is no finished Autoconfigure result to save. Run it again."})
		return
	}
	replaced, err := s.registry.SaveProfileFrom(id, autoconfigProfileName, run.result.Profile(s.activeBuild()))
	if err != nil {
		s.renderAutoconfigMessage(w, id, panelBanner{"error", "Not saved: " + err.Error()})
		return
	}
	msg := fmt.Sprintf("Saved as profile %q.", autoconfigProfileName)
	if replaced {
		msg = fmt.Sprintf("Saved, replacing the earlier %q profile.", autoconfigProfileName)
	}
	if apply {
		if err := s.registry.ApplyProfile(id, autoconfigProfileName); err != nil {
			s.renderAutoconfigMessage(w, id, panelBanner{"error", msg + " It could not be applied: " + err.Error()})
			return
		}
		if cfg, err := s.registry.GetConfig(id); err == nil {
			s.afterConfigChange(w, r, id, cfg)
		}
		msg += " The settings are now in use and take effect the next time this model loads."
	} else {
		msg += " Restore it from Configure whenever you want to use it."
	}
	// Either way the Configure panel below is now out of date: the new
	// profile is missing from its picker, and after an apply the form
	// shows the settings from before it.
	s.markConfigPanelStale(w, r, id)
	s.clearAutoconfigRun(id)
	s.renderAutoconfigMessage(w, id, panelBanner{"ok", msg})
}

// handleAutoconfigDiscard drops a finished result.
func (s *Server) handleAutoconfigDiscard(w http.ResponseWriter, r *http.Request) {
	id := s.registry.ResolveID(chi.URLParam(r, "id"))
	s.clearAutoconfigRun(id)
	respondHTML(w)
}

func (s *Server) clearAutoconfigRun(id string) {
	s.autoconf.mu.Lock()
	defer s.autoconf.mu.Unlock()
	if s.autoconf.run != nil && s.autoconf.run.modelID == id && s.autoconf.run.done {
		s.autoconf.run = nil
	}
}

func (s *Server) renderAutoconfigMessage(w http.ResponseWriter, id string, b panelBanner) {
	respondHTML(w)
	s.renderPartial(w, "autoconfig_message", struct {
		ModelID string
		Banner  panelBanner
	}{id, b})
}

// reviewRow is one setting in the review table. Kept marks a setting
// Autoconfigure looked at and left as it is — worth showing when there is
// a reason for it, so "already right" does not read as "overlooked".
type reviewRow struct {
	Label    string
	Help     string
	Current  string
	Proposed string
	Why      []string
	Source   string
	Kept     bool
}

// autoconfigReviewData is what the autoconfig_review partial renders.
type autoconfigReviewData struct {
	ModelID     string
	ModelName   string
	Error       string
	Fits        bool
	EstimateGiB float64
	BudgetGiB   float64
	CPURAMGiB   float64
	Changed     []reviewRow
	Unchanged   []reviewRow
	General     []string
	Suggestions []autoconfig.DraftSuggestion
	Sources     []string
	// Check is the outcome of the test load, in the words the review
	// shows; CheckOK picks the style. Both empty when no check ran.
	Check     string
	CheckOK   bool
	CheckHelp string
}

// autoconfigCheckHelp is the tooltip on the test-load line: what was
// done, and how to read each outcome.
const autoconfigCheckHelp = "Autoconfigure loads the proposed settings once and sends the model one request with a 2,048-token prompt. " +
	"The memory estimate above is a calculation; this is the real thing. " +
	"\"Checked\" means the model loaded and answered. " +
	"If the first plan ran out of memory, Autoconfigure adjusted it and loaded it again, and each adjusted setting is marked \"test load\" in the Source column. " +
	"The test is kept in the benchmark history, where a failed one shows what llama-server reported."

// checkSummary says what the test load found, in one or two sentences.
func checkSummary(c autoconfig.Verification) (text string, ok bool) {
	switch c.Status {
	case autoconfig.VerifyPassed:
		if c.Adjusted == 0 {
			return "Checked: these settings were loaded on this machine and answered a test request.", true
		}
		settings := "settings were"
		if c.Adjusted == 1 {
			settings = "setting was"
		}
		return fmt.Sprintf("Checked: the first plan ran out of GPU memory when it was loaded, so %d %s adjusted. The settings below were loaded on this machine and answered a test request.",
			c.Adjusted, settings), true
	case autoconfig.VerifyFailed:
		return fmt.Sprintf("These settings did not run when they were loaded (%d test %s): %s. They are shown as estimated, and have not been made to work.",
			c.Attempts, plural(c.Attempts, "load", "loads"), c.Reason), false
	case autoconfig.VerifySkipped:
		return "Not checked: " + c.Reason + ". These settings are an estimate and have not been loaded.", false
	}
	return "", false
}

func (s *Server) autoconfigReviewData(id, name string, run autoconfigRun) autoconfigReviewData {
	d := autoconfigReviewData{ModelID: id, ModelName: name}
	if run.err != nil {
		d.Error = run.err.Error()
		return d
	}
	res := run.result
	d.Fits, d.EstimateGiB, d.BudgetGiB, d.CPURAMGiB = res.Fit.Fits, res.Fit.EstimateGiB, res.Fit.BudgetGiB, res.Fit.CPURAMGiB
	d.Suggestions, d.Sources = res.Suggestions, res.CardSources
	d.Check, d.CheckOK = checkSummary(res.Check)
	d.CheckHelp = autoconfigCheckHelp

	why := map[string][]string{}
	origin := map[string]string{}
	for _, n := range res.Notes {
		if n.Field == "" {
			d.General = append(d.General, n.Reason)
			continue
		}
		key := reviewKey(n.Field)
		// The same reason can arrive twice — a preset that sets several
		// values carries its sentence on each of them — and reads as a
		// mistake when repeated in one cell.
		if !slices.Contains(why[key], n.Reason) {
			why[key] = append(why[key], n.Reason)
		}
		if origin[key] == "" {
			origin[key] = n.Origin
		}
	}
	for _, f := range reviewFields {
		cur, prop := f.show(&res.Base), f.show(&res.Proposed)
		row := reviewRow{Label: f.label, Help: fieldHelp[f.key], Current: cur, Proposed: prop, Why: why[f.key], Source: origin[f.key]}
		switch {
		case cur != prop:
			d.Changed = append(d.Changed, row)
		case len(row.Why) > 0:
			// Already what Autoconfigure would choose, or carrying advice
			// that could not be used: shown with the changes, marked as
			// kept, because saying nothing reads as overlooking it.
			row.Kept = true
			d.Changed = append(d.Changed, row)
		default:
			row.Why = nil
			d.Unchanged = append(d.Unchanged, row)
		}
	}
	return d
}

// reviewKey folds note fields onto the review row that shows them.
func reviewKey(field string) string {
	switch field {
	case "batch_size":
		return "ubatch_size"
	case "sampling_preset":
		return "temperature"
	case "draft_model_path", "spec_assist":
		return "spec_type"
	}
	return field
}

// reviewField is one row of the review table.
type reviewField struct {
	key   string
	label string
	show  func(c *models.ModelConfig) string
}

func showInt(zero string, v int) string {
	if v == 0 {
		return zero
	}
	return models.GroupDigits(v)
}

func showFloat(p *float64) string {
	if p == nil {
		return "server default"
	}
	return strconv.FormatFloat(*p, 'f', -1, 64)
}

var reviewFields = []reviewField{
	{"context_size", "Context size", func(c *models.ModelConfig) string { return showInt("model default", c.ContextSize) + " tokens" }},
	{"kv_cache_quant", "KV cache", func(c *models.ModelConfig) string {
		if c.KVCacheQuant == "" {
			return "f16 (full precision)"
		}
		return c.KVCacheQuant
	}},
	{"gpu_layers", "GPU layers", func(c *models.ModelConfig) string {
		if c.GPULayers >= 999 {
			return "all"
		}
		return strconv.Itoa(c.GPULayers)
	}},
	{"cpu_moe", "CPU expert layers", func(c *models.ModelConfig) string { return showInt("none", c.CPUMoE) }},
	{"gpu_assign", "GPU assignment", func(c *models.ModelConfig) string {
		if c.GPUAssign == "custom" && c.TensorSplit != "" {
			// The split is the whole content of a custom assignment.
			return "a custom split (" + c.TensorSplit + ")"
		}
		return gpuAssignText(c.GPUAssign)
	}},
	{"flash_attention", "Flash attention", func(c *models.ModelConfig) string {
		if c.FlashAttention {
			return "on"
		}
		return "off"
	}},
	{"ubatch_size", "Batch / micro-batch", func(c *models.ModelConfig) string {
		return showInt("default", c.BatchSize) + " / " + showInt("default", c.UBatchSize)
	}},
	{"parallel", "Parallel conversations", func(c *models.ModelConfig) string {
		if c.Parallel <= 0 {
			return "auto (4, shared)"
		}
		return strconv.Itoa(c.Parallel)
	}},
	{"threads", "CPU threads", func(c *models.ModelConfig) string { return strconv.Itoa(c.Threads) }},
	{"spec_type", "Speculative decoding", func(c *models.ModelConfig) string {
		if c.SpecType == "" {
			return "off"
		}
		s := c.SpecType
		if c.DraftModelPath != "" && c.SpecType != "draft-mtp" {
			s += " with " + c.DraftModelPath[strings.LastIndex(c.DraftModelPath, "/")+1:]
		}
		return s
	}},
	{"temperature", "Temperature", func(c *models.ModelConfig) string { return showFloat(c.Temperature) }},
	{"top_p", "Top-p", func(c *models.ModelConfig) string { return showFloat(c.TopP) }},
	{"top_k", "Top-k", func(c *models.ModelConfig) string {
		if c.TopK == nil {
			return "server default"
		}
		return strconv.Itoa(*c.TopK)
	}},
	{"min_p", "Min-p", func(c *models.ModelConfig) string { return showFloat(c.MinP) }},
	{"presence_penalty", "Presence penalty", func(c *models.ModelConfig) string { return showFloat(c.PresencePenalty) }},
	{"repeat_penalty", "Repeat penalty", func(c *models.ModelConfig) string { return showFloat(c.RepeatPenalty) }},
}

// gpuAssignText spells out a GPU assignment for the review table, where
// the model card's compact "gpu:0" tag would not read as a sentence.
func gpuAssignText(assign string) string {
	switch {
	case assign == "" || assign == "all":
		return "all GPUs"
	case assign == "custom":
		return "a custom split"
	case strings.HasPrefix(assign, "tensor-"), strings.HasPrefix(assign, "tensor:"):
		return "tensor parallelism over GPUs " + strings.TrimPrefix(strings.TrimPrefix(assign, "tensor-"), "tensor:")
	case strings.ContainsAny(assign, ",-"):
		return "GPUs " + assign
	default:
		return "GPU " + assign
	}
}

// gpuBusyReason says in plain language what is using the GPU now, or "".
// Autoconfigure and autotune both ask before they start: they restart the
// router and load models, which would interrupt whatever else is running.
func (s *Server) gpuBusyReason() string {
	if s.routerBusyWithJob() {
		return "a benchmark is running"
	}
	if run, ok := s.autoconfigSnapshot(); ok && !run.done {
		return "Autoconfigure is running"
	}
	return ""
}
