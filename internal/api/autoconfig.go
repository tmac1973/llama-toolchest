package api

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/go-chi/chi/v5"

	"github.com/tmac1973/llama-toolchest/internal/autoconfig"
	"github.com/tmac1973/llama-toolchest/internal/models"
)

// autoconfigProfileName is the profile autoconfigure saves to.
const autoconfigProfileName = "Autoconfig"

// autoconfigTimeout bounds one run: loading the helper, reading the card
// and answering. Generous, because a first load reads the model from disk.
const autoconfigTimeout = 10 * time.Minute

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
	m := s.monitor.Current()
	hw := models.Hardware{LogicalCores: m.CPU.Cores, RAMTotalMiB: m.Memory.TotalMB}
	for _, g := range m.GPU {
		hw.GPUs = append(hw.GPUs, models.GPUSpec{Index: g.Index, Name: g.Name, VRAMTotalMiB: g.VRAMTotalMB, IsIGPU: g.IsIGPU})
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
	ModelID      string
	ModelName    string
	Helper       string // helper model name, "" when none is installed
	HelperIsDflt bool
	BusyReason   string
	OtherLoaded  []string
	Classes      []contextClassOption
}

type contextClassOption struct {
	Value, Label, Help string
	Checked            bool
}

func contextClassOptions(maxCtx int) []contextClassOption {
	maxLabel := "Maximum — the largest this model supports that fits"
	if maxCtx > 0 {
		maxLabel = fmt.Sprintf("Maximum — up to %s tokens, as much as fits", groupThousands(maxCtx))
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
	if h, chosen := s.helperModel(); h != nil {
		d.Helper, d.HelperIsDflt = h.PublicName(), !chosen
		helperRouter = s.registry.RouterName(h.ID)
	}
	if s.routerBusyWithJob() {
		d.BusyReason = "A benchmark is running and using the GPU. Start Autoconfigure when it finishes."
	} else if run, ok := s.autoconfigSnapshot(); ok && !run.done {
		d.BusyReason = "Autoconfigure is already running for another model. Wait for it to finish."
	}
	if d.Helper != "" {
		d.OtherLoaded = s.otherLoadedModels(helperRouter)
	}
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
	}
	helper, _ := s.helperModel()
	if helper != nil {
		deps.HelperID = helper.ID
		deps.HelperContext = helperMinContext
		if cfg, err := s.registry.GetConfig(helper.ID); err == nil && cfg.ContextSize > helperMinContext {
			deps.HelperContext = cfg.ContextSize
		}
	}

	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), autoconfigTimeout)
		defer cancel()
		res, err := autoconfig.Run(ctx, deps, id, class)
		if helper != nil {
			s.unloadHelper(helper.ID)
		}
		if err := s.registry.SetAutoconfigHint(id, false); err != nil {
			slog.Debug("clear autoconfig hint", "model", id, "error", err)
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
	s.clearAutoconfigRun(id)
	s.renderAutoconfigMessage(w, id, panelBanner{"ok", msg})
}

// handleAutoconfigDiscard drops a finished result.
func (s *Server) handleAutoconfigDiscard(w http.ResponseWriter, r *http.Request) {
	id := s.registry.ResolveID(chi.URLParam(r, "id"))
	s.clearAutoconfigRun(id)
	respondHTML(w)
}

// handleAutoconfigDismissHint hides the model card's suggestion.
func (s *Server) handleAutoconfigDismissHint(w http.ResponseWriter, r *http.Request) {
	id := s.registry.ResolveID(chi.URLParam(r, "id"))
	if err := s.registry.SetAutoconfigHint(id, false); err != nil {
		http.Error(w, err.Error(), registryErrorStatus(err, http.StatusNotFound))
		return
	}
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

// reviewRow is one setting in the review table.
type reviewRow struct {
	Label    string
	Help     string
	Current  string
	Proposed string
	Why      []string
	Source   string
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

	why := map[string][]string{}
	origin := map[string]string{}
	for _, n := range res.Notes {
		if n.Field == "" {
			d.General = append(d.General, n.Reason)
			continue
		}
		key := reviewKey(n.Field)
		why[key] = append(why[key], n.Reason)
		if origin[key] == "" {
			origin[key] = n.Origin
		}
	}
	for _, f := range reviewFields {
		cur, prop := f.show(&res.Base), f.show(&res.Proposed)
		row := reviewRow{Label: f.label, Help: fieldHelp[f.key], Current: cur, Proposed: prop, Why: why[f.key], Source: origin[f.key]}
		if cur != prop {
			d.Changed = append(d.Changed, row)
		} else {
			row.Why = nil
			d.Unchanged = append(d.Unchanged, row)
			// A note about a field the proposal left alone (the card's
			// advice that could not be used) still belongs on screen.
			d.General = append(d.General, why[f.key]...)
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
	return groupThousands(v)
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
	{"gpu_assign", "GPU assignment", func(c *models.ModelConfig) string { return models.GPUAssignLabel(c.GPUAssign) }},
	{"flash_attention", "Flash attention", func(c *models.ModelConfig) string {
		if c.FlashAttention {
			return "on"
		}
		return "off"
	}},
	{"ubatch_size", "Batch / micro-batch", func(c *models.ModelConfig) string {
		return showInt("default", c.BatchSize) + " / " + showInt("default", c.UBatchSize)
	}},
	{"parallel", "Parallel conversations", func(c *models.ModelConfig) string { return strconv.Itoa(max(1, c.Parallel)) }},
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

// groupThousands writes n with thousands separators.
func groupThousands(n int) string {
	s := strconv.Itoa(n)
	for i := len(s) - 3; i > 0; i -= 3 {
		s = s[:i] + "," + s[i:]
	}
	return s
}
