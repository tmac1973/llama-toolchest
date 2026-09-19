package api

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"sort"
	"sync"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/llmcall"
	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/modelsource"
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

// helperMinContext is the context the helper model needs: a trimmed model
// card, the instructions, and a 2,048-token answer.
const helperMinContext = 16384

// helperModel returns the model autoconfigure uses: the one chosen in
// Settings when it is installed, otherwise the recommended default when
// that is installed, otherwise nil. chosen reports which.
func (s *Server) helperModel() (m *models.Model, chosen bool) {
	s.cfgMu.Lock()
	id := s.cfg.HelperModelID
	s.cfgMu.Unlock()
	if id != "" {
		if m, err := s.registry.Get(id); err == nil {
			return m, true
		}
	}
	return s.installedDefaultHelper(), false
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

	body, _ := json.Marshal(map[string]string{"model": routerName})
	if err := s.ensureModelLoadedForRequest(ctx, body); err != nil {
		return llmcall.Target{}, fmt.Errorf("loading the helper model: %w", err)
	}
	rc := reasoningControl(m, cfg)
	return llmcall.Target{RouterName: routerName, Thinking: llmcall.Thinking{Toggle: rc.Toggle, Kwarg: rc.Kwarg}}, nil
}

// routerKnows reports whether the running router lists the model, which
// it does only for models registered before it started.
func (s *Server) routerKnows(routerName string, m *models.Model) bool {
	_, known := s.lookupRouterState(routerName, m)
	return known
}

// ensureHelperConfig makes sure the helper model is enabled (the router
// only serves enabled models) and has room for a model card. It reports
// whether it changed anything.
func (s *Server) ensureHelperConfig(id string) (bool, error) {
	cfg, err := s.registry.GetConfig(id)
	if err != nil {
		return false, err
	}
	m, _ := s.registry.Get(id)
	next := *cfg
	changed := false
	if !next.Enabled {
		next.Enabled = true
		changed = true
	}
	ctx := next.ContextSize
	if ctx == 0 && m != nil {
		ctx = m.ContextLength // "model default"
	}
	if ctx < helperMinContext {
		next.ContextSize = helperMinContext
		changed = true
	}
	if !changed {
		return false, nil
	}
	if err := s.registry.SetConfig(id, &next); err != nil {
		return false, err
	}
	if _, err := s.registry.WritePresetINI(s.activeBackend()); err != nil {
		slog.Warn("failed to regenerate preset INI", "error", err)
	}
	return true, nil
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

// helperOption is one entry in the helper model picker.
type helperOption struct {
	ID       string
	Label    string
	Selected bool
}

// helperPanelData is what the helper_model_panel partial renders.
type helperPanelData struct {
	Options          []helperOption
	Current          string // label of the model in use, "" when none
	CurrentIsDefault bool   // the recommended model is in use because nothing else was chosen
	DefaultInstalled bool
	Downloading      bool
	DefaultLabel     string
	Banner           *panelBanner
}

func (s *Server) helperPanelData() helperPanelData {
	d := helperPanelData{DefaultLabel: defaultHelperLabel + " (" + defaultHelperQuant + ")"}
	s.cfgMu.Lock()
	chosenID := s.cfg.HelperModelID
	s.cfgMu.Unlock()

	var list []*models.Model
	for _, m := range s.registry.List() {
		if !m.IsEmbedding() && !m.MTPHead {
			list = append(list, m)
		}
	}
	sort.Slice(list, func(i, j int) bool { return list[i].PublicName() < list[j].PublicName() })
	d.Options = append(d.Options, helperOption{ID: "", Label: "Recommended: " + d.DefaultLabel, Selected: chosenID == ""})
	for _, m := range list {
		d.Options = append(d.Options, helperOption{ID: m.ID, Label: m.PublicName(), Selected: m.ID == chosenID})
	}

	if m, chosen := s.helperModel(); m != nil {
		d.Current = m.PublicName()
		d.CurrentIsDefault = !chosen
	}
	d.DefaultInstalled = s.installedDefaultHelper() != nil
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

// handleSetHelperModel saves the helper choice (form: helper_model_id; ""
// is the recommended default) and gives the chosen model room for a model
// card.
func (s *Server) handleSetHelperModel(w http.ResponseWriter, r *http.Request) {
	r.ParseForm()
	id := r.FormValue("helper_model_id")
	if id != "" {
		if _, err := s.registry.Get(id); err != nil {
			s.renderHelperPanel(w, &panelBanner{"error", "That model is not installed."})
			return
		}
	}
	s.cfgMu.Lock()
	s.cfg.HelperModelID = id
	s.saveConfigLocked()
	s.cfgMu.Unlock()

	msg := "Autoconfigure will use the recommended model when it is installed."
	if m, _ := s.helperModel(); m != nil {
		msg = "Autoconfigure will use " + m.PublicName() + "."
		if changed, err := s.ensureHelperConfig(m.ID); err != nil {
			s.renderHelperPanel(w, &panelBanner{"error", "Saved, but its settings could not be updated: " + err.Error()})
			return
		} else if changed {
			msg += fmt.Sprintf(" Its context was raised to %d tokens so model cards fit.", helperMinContext)
		}
	}
	s.renderHelperPanel(w, &panelBanner{"ok", msg})
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
