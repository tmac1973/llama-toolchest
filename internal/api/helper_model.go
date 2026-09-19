package api

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"sync"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/llmcall"
	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/modelsource"
	"github.com/tmac1973/llama-toolchest/internal/process"
)

// The recommended helper model: small enough to load next to nothing
// else on most GPUs (about 3 GB), good enough at reading a model card and
// filling in a form. Chosen by quant rather than file name, because
// publishers rename files when they re-upload a repository.
const (
	defaultHelperRepo  = "unsloth/Qwen3.5-4B-GGUF"
	defaultHelperQuant = "Q4_K_M"
	defaultHelperLabel = "Qwen3.5-4B"
)

// helperModel returns the app's helper model, or nil when none is
// installed. It is not a user choice: a model becomes the helper by being
// downloaded as one.
func (s *Server) helperModel() *models.Model {
	if h := s.registry.ListHelpers(); len(h) > 0 {
		return h[0]
	}
	return nil
}

// adoptExistingHelper marks an already-installed recommended model as the
// helper at startup. It covers the installs that downloaded one before
// helper models had a role of their own.
func (s *Server) adoptExistingHelper() {
	if len(s.registry.ListHelpers()) > 0 {
		return
	}
	m := s.installedDefaultHelper()
	if m == nil {
		return
	}
	if err := s.registry.SetHelperRole(m.ID, true); err != nil {
		slog.Debug("adopt helper model", "model", m.ID, "error", err)
		return
	}
	// Put it on the fixed settings straight away, so what is stored
	// matches what the Models page says about it.
	if _, err := s.ensureHelperConfig(m.ID); err != nil {
		slog.Warn("could not set the helper model's settings", "model", m.ID, "error", err)
	}
	slog.Info("marked the installed recommended model as the helper model", "model", m.ID)
}

// installedDefaultHelper returns the recommended helper model if it is
// installed.
func (s *Server) installedDefaultHelper() *models.Model {
	for _, m := range s.registry.List() {
		if m.ModelID == defaultHelperRepo && m.Quant == defaultHelperQuant {
			return m
		}
	}
	return nil
}

// helperBackend connects llmcall to this server: the job queue decides
// whether the GPU is free, and the router serves the helper.
type helperBackend struct {
	s  *Server
	mu sync.Mutex // one Prepare at a time: it may restart the router
}

func (b *helperBackend) Busy() string {
	if b.s.routerBusyWithJob() {
		return "A benchmark is running and using the GPU. Try again when it finishes."
	}
	return ""
}

func (b *helperBackend) RouterURL() string {
	return fmt.Sprintf("http://localhost:%d", b.s.cfg.LlamaPort)
}

// Prepare makes the helper model loadable and loads it. The router reads
// its model list and settings when it starts, so a helper downloaded
// since, or one whose context had to be raised, needs a restart first;
// a stopped router is started. Other loaded models may be unloaded to
// make room — the caller warns about that before asking.
func (b *helperBackend) Prepare(ctx context.Context, id string) (llmcall.Target, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	s := b.s

	m, err := s.registry.Get(id)
	if err != nil {
		return llmcall.Target{}, fmt.Errorf("the helper model is not installed: %w", err)
	}
	cfg, err := s.registry.GetConfig(id)
	if err != nil {
		return llmcall.Target{}, err
	}

	needRestart := false
	if changed, err := s.ensureHelperConfig(id); err != nil {
		return llmcall.Target{}, err
	} else if changed {
		needRestart = true
		cfg, _ = s.registry.GetConfig(id)
	}

	routerName := s.registry.RouterName(id)
	switch {
	case !s.process.IsRunning():
		slog.Info("starting the router for the helper model", "model", id)
		if err := s.startRouter(); err != nil {
			return llmcall.Target{}, fmt.Errorf("starting the server for the helper model: %w", err)
		}
	case needRestart || !s.routerKnows(routerName, m):
		slog.Info("restarting the router so it can load the helper model", "model", id)
		if err := s.process.Stop(); err != nil {
			slog.Debug("stop before helper restart", "error", err)
		}
		if err := s.startRouter(); err != nil {
			return llmcall.Target{}, fmt.Errorf("restarting the server for the helper model: %w", err)
		}
	}

	// A router that has just been started answers /models only once it
	// has read its preset, which takes a moment. Asking to load before
	// then reads as "this model is not in the preset" and fails a run
	// that would have worked a second later.
	if err := s.waitForRouterModel(ctx, routerName, m); err != nil {
		return llmcall.Target{}, err
	}

	body, _ := json.Marshal(map[string]string{"model": routerName})
	if err := s.ensureModelLoadedForRequest(ctx, body); err != nil {
		return llmcall.Target{}, fmt.Errorf("loading the helper model: %w", err)
	}
	rc := reasoningControl(m, cfg)
	return llmcall.Target{RouterName: routerName, Thinking: llmcall.Thinking{Toggle: rc.Toggle, Kwarg: rc.Kwarg}}, nil
}

// routerReadyTimeout bounds the wait for a just-started router to list
// its models. Generous: the router reads every model's metadata first.
const routerReadyTimeout = 90 * time.Second

// routerSettle is how long a router must have been answering before a
// model missing from its list counts as really missing rather than not
// listed yet.
const routerSettle = 5 * time.Second

// routerWait is what waitForRouterModel needs, as functions, so the loop
// can be tested without a live router.
type routerWait struct {
	// alive reports that the router is running or still starting up.
	alive    func() bool
	answers  func() (int, error) // how many models the router lists
	knows    func() bool
	restart  func() error
	name     string
	settle   time.Duration
	timeout  time.Duration
	interval time.Duration
}

// waitForRouterModel waits until the router lists the model.
func (s *Server) waitForRouterModel(ctx context.Context, routerName string, m *models.Model) error {
	return waitForRouterModel(ctx, routerWait{
		// Not IsRunning: that is false while the router is starting,
		// which is exactly the state this wait exists for.
		alive: func() bool { return routerAlive(s.process.GetStatus().State) },
		answers: func() (int, error) {
			list, err := s.process.ListModels()
			return len(list), err
		},
		knows: func() bool { return s.routerKnows(routerName, m) },
		restart: func() error {
			if err := s.process.Stop(); err != nil {
				slog.Debug("stop before helper restart", "error", err)
			}
			return s.startRouter()
		},
		name:     m.PublicName(),
		settle:   routerSettle,
		timeout:  routerReadyTimeout,
		interval: time.Second,
	})
}

// routerAlive reports whether the router is up or on its way up. A
// router that has just been started is "starting" until its first health
// check passes, which is most of the window this wait covers.
func routerAlive(state string) bool {
	return state == process.StateRunning || state == process.StateStarting
}

// waitForRouterModel waits for a router to list a model, restarting it
// once if it is answering and the model is not there — which is what
// happens when the preset was written after the router started.
func waitForRouterModel(ctx context.Context, w routerWait) error {
	deadline := time.Now().Add(w.timeout)
	restarted := false
	// A router answering /models may still be filling its list, so a
	// missing model only counts after it has been answering for a while.
	var answeringSince time.Time
	for {
		if !w.alive() {
			return fmt.Errorf("the server stopped while the helper model %s was being prepared. Its log is on the Server page", w.name)
		}
		listed, err := w.answers()
		if err == nil && answeringSince.IsZero() {
			answeringSince = time.Now()
		}
		settled := !answeringSince.IsZero() && time.Since(answeringSince) > w.settle
		switch {
		case err != nil:
			// Not answering yet; keep waiting.
		case w.knows():
			return nil
		case !settled:
			// Answering, but perhaps not with everything yet.
		case !restarted:
			slog.Info("restarting the router: the helper model is not in its list", "model", w.name, "listed", listed)
			restarted = true
			if err := w.restart(); err != nil {
				return fmt.Errorf("restarting the server for the helper model: %w", err)
			}
			deadline, answeringSince = time.Now().Add(w.timeout), time.Time{}
		default:
			return fmt.Errorf("the server was restarted, but the helper model %s is still not in its list. Check that the model is enabled on the Models page", w.name)
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(w.interval):
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("the server did not become ready within %s while preparing the helper model %s", w.timeout, w.name)
		}
	}
}

// routerKnows reports whether the running router lists the model, which
// it does only for models registered before it started.
func (s *Server) routerKnows(routerName string, m *models.Model) bool {
	_, known := s.lookupRouterState(routerName, m)
	return known
}

// ensureHelperConfig puts the helper model on its fixed settings, sized
// to the GPU, and reports whether anything changed. It runs before every
// use, so a helper registered with the ordinary defaults — or edited by
// hand — is corrected rather than failing in a way nobody can explain.
func (s *Server) ensureHelperConfig(id string) (bool, error) {
	cfg, err := s.registry.GetConfig(id)
	if err != nil {
		return false, err
	}
	m, err := s.registry.Get(id)
	if err != nil {
		return false, err
	}
	want := models.HelperConfig(m, s.helperVRAMBudget())
	if models.ProfileEqual(*cfg, want) && cfg.Enabled == want.Enabled {
		return false, nil
	}
	want.Aliases = cfg.Aliases
	if err := s.registry.SetConfig(id, &want); err != nil {
		return false, err
	}
	if _, err := s.registry.WritePresetINI(s.activeBackend()); err != nil {
		slog.Warn("failed to regenerate preset INI", "error", err)
	}
	slog.Info("helper model set to its fixed settings", "model", id, "context", want.ContextSize)
	return true, nil
}

// helperVRAMBudget is the GPU memory a helper model may use: the largest
// dedicated GPU, less the same safety margin the fit planner leaves.
func (s *Server) helperVRAMBudget() float64 {
	best := 0.0
	for _, g := range s.hardware().GPUs {
		if g.IsIGPU {
			continue
		}
		if gib := float64(g.VRAMTotalMiB) / 1024; gib > best {
			best = gib
		}
	}
	if best == 0 {
		return 0
	}
	return best * 0.92
}

// unloadHelper frees the GPU after autoconfigure. Failure is logged, not
// returned: the answer is already in hand.
func (s *Server) unloadHelper(id string) {
	if !s.process.IsRunning() {
		return
	}
	if err := s.process.UnloadModel(s.registry.RouterName(id)); err != nil {
		slog.Warn("could not unload the helper model", "model", id, "error", err)
	}
}

// claimDownloadedHelper marks a finished download as the app's helper
// model when it is the one the Settings panel asked for.
func (s *Server) claimDownloadedHelper(m *models.Model) {
	s.cfgMu.Lock()
	pending := s.cfg.PendingHelper
	s.cfgMu.Unlock()
	if pending == "" || pending != m.ModelID+"|"+m.Filename {
		return
	}
	if err := s.registry.SetHelperRole(m.ID, true); err != nil {
		slog.Warn("could not mark the downloaded helper model", "model", m.ID, "error", err)
		return
	}
	s.cfgMu.Lock()
	s.cfg.PendingHelper = ""
	s.saveConfigLocked()
	s.cfgMu.Unlock()
	if _, err := s.ensureHelperConfig(m.ID); err != nil {
		slog.Warn("could not set the helper model's settings", "model", m.ID, "error", err)
	}
	slog.Info("helper model installed", "model", m.ID)
}

// pickHelperFile chooses the recommended quant from a repository listing.
func pickHelperFile(d *modelsource.Detail) (modelsource.File, bool) {
	if d == nil {
		return modelsource.File{}, false
	}
	for _, f := range d.Files {
		if f.Quant == defaultHelperQuant && !models.IsMMProjFile(f.Filename) {
			return f, true
		}
	}
	return modelsource.File{}, false
}

// helperPanelData is what the helper_model_panel partial renders. The
// helper model is not a choice: it is installed, or it is not.
type helperPanelData struct {
	Installed    bool
	Name         string
	SizeGiB      float64
	ContextSize  int
	Downloading  bool
	DefaultLabel string
	Banner       *panelBanner
}

func (s *Server) helperPanelData() helperPanelData {
	d := helperPanelData{DefaultLabel: defaultHelperLabel + " (" + defaultHelperQuant + ")"}
	if m := s.helperModel(); m != nil {
		d.Installed = true
		d.Name = m.PublicName()
		d.SizeGiB = models.BytesToGiB(m.SizeBytes)
		d.ContextSize = models.HelperConfig(m, s.helperVRAMBudget()).ContextSize
	}
	for _, st := range s.downloader.ListActive() {
		if st.ModelID == defaultHelperRepo && st.Status == "downloading" {
			d.Downloading = true
		}
	}
	return d
}

func (s *Server) renderHelperPanel(w http.ResponseWriter, banner *panelBanner) {
	d := s.helperPanelData()
	d.Banner = banner
	respondHTML(w)
	s.renderPartial(w, "helper_model_panel", d)
}

// handleHelperPanel renders the Settings section for the helper model.
func (s *Server) handleHelperPanel(w http.ResponseWriter, r *http.Request) {
	s.renderHelperPanel(w, nil)
}

// removeHelper deletes the helper model and its files, and reports what
// was freed. Removing it is the only thing there is to manage about a
// helper model: it is downloaded when Autoconfigure needs one, and
// removed to free the disk.
func (s *Server) removeHelper() (name string, freedGiB float64, err error) {
	m := s.helperModel()
	if m == nil {
		return "", 0, errors.New("no helper model is installed")
	}
	name, freedGiB = m.PublicName(), models.BytesToGiB(m.SizeBytes)
	if err := s.registry.Delete(m.ID); err != nil {
		return "", 0, err
	}
	if _, err := s.registry.WritePresetINI(s.activeBackend()); err != nil {
		slog.Warn("failed to regenerate preset INI after removing the helper model", "error", err)
	}
	return name, freedGiB, nil
}

// handleRemoveHelperModel removes the helper model from the Settings
// panel and re-renders that panel.
func (s *Server) handleRemoveHelperModel(w http.ResponseWriter, r *http.Request) {
	name, gib, err := s.removeHelper()
	if err != nil {
		s.renderHelperPanel(w, &panelBanner{"error", "Not removed: " + err.Error()})
		return
	}
	w.Header().Set("HX-Trigger", "modelsChanged")
	s.renderHelperPanel(w, &panelBanner{"ok", fmt.Sprintf(
		"Removed %s and freed %.1f GiB. Autoconfigure will offer to download it again when it needs it.", name, gib)})
}

// handleDownloadHelperModel starts the download of the recommended helper
// model through the normal download queue.
func (s *Server) handleDownloadHelperModel(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()
	detail, err := s.hfClient.GetModel(ctx, defaultHelperRepo)
	if err != nil {
		s.renderHelperPanel(w, &panelBanner{"error", "Could not reach Hugging Face to find the recommended model: " + err.Error()})
		return
	}
	f, ok := pickHelperFile(detail)
	if !ok {
		s.renderHelperPanel(w, &panelBanner{"error", "The recommended helper model is not available right now. Choose an installed model instead."})
		return
	}
	if avail := s.downloader.AvailableForDownload(); avail >= 0 && f.Size > avail {
		s.renderHelperPanel(w, &panelBanner{"error", fmt.Sprintf("Not enough disk space: the model needs %.1f GiB.", models.BytesToGiB(f.Size))})
		return
	}
	s.cfgMu.Lock()
	s.cfg.PendingHelper = defaultHelperRepo + "|" + f.Filename
	s.saveConfigLocked()
	s.cfgMu.Unlock()
	if _, err := s.downloader.Start(context.Background(), modelsource.SourceHuggingFace, defaultHelperRepo, f.Filename, f.Size); err != nil {
		if errors.Is(err, context.Canceled) {
			return
		}
		s.renderHelperPanel(w, &panelBanner{"error", "Download not started: " + err.Error()})
		return
	}
	s.renderHelperPanel(w, &panelBanner{"ok", fmt.Sprintf("Downloading %s (%.1f GiB). Progress is in the Downloads panel on the Models page.",
		f.Filename, models.BytesToGiB(f.Size))})
}
