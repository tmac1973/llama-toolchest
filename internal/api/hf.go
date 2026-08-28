package api

import (
	"encoding/json"
	"fmt"
	"html"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/tmac1973/llama-toolchest/internal/huggingface"
	"github.com/tmac1973/llama-toolchest/internal/models"
)

// hfFileView decorates a HuggingFace ModelFile with local-state flags used
// by the templates: whether we already have it on disk, and whether it would
// fit given current free space minus the safety margin and in-flight downloads.
type hfFileView struct {
	huggingface.ModelFile
	AlreadyDownloaded bool
	FitsOnDisk        bool
}

// hfModelView is the template payload for the HF file-list partial.
type hfModelView struct {
	ID             string
	Source         string // which host these files came from
	Files          []hfFileView
	AvailableBytes int64 // free - margin - in-flight
	FreeBytes      int64
	SafetyMargin   int64
}

func (s *Server) handleHFSearch(w http.ResponseWriter, r *http.Request) {
	query := r.URL.Query().Get("q")
	if query == "" {
		http.Error(w, "missing q parameter", http.StatusBadRequest)
		return
	}

	source := s.requestSource(r.URL.Query().Get("source"))
	results, err := s.sourceClient(source).Search(r.Context(), query)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}

	// htmx: return HTML partial. The source travels with the results so
	// each row can link to the right host and start a download against
	// the source it was found on.
	if isHTMX(r) {
		respondHTML(w)
		s.renderPartial(w, "hf_results", struct {
			Results any
			Source  string
		}{Results: results, Source: source})
		return
	}

	respondJSON(w, results)
}

func (s *Server) handleHFModel(w http.ResponseWriter, r *http.Request) {
	modelID := r.URL.Query().Get("id")
	if modelID == "" {
		http.Error(w, "missing id parameter", http.StatusBadRequest)
		return
	}

	source := s.requestSource(r.URL.Query().Get("source"))
	detail, err := s.sourceClient(source).GetModel(r.Context(), modelID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}

	available := s.downloader.AvailableForDownload()
	view := hfModelView{
		ID:             detail.ID,
		Source:         source,
		Files:          make([]hfFileView, 0, len(detail.Files)),
		AvailableBytes: available,
		FreeBytes:      s.downloader.FreeBytes(),
		SafetyMargin:   huggingface.DiskSafetyMarginBytes,
	}
	for _, f := range detail.Files {
		fv := hfFileView{ModelFile: f}
		if _, ok := s.registry.HasFile(detail.ID, f.Filename); ok {
			fv.AlreadyDownloaded = true
		} else {
			// available < 0 means "free space unknown" (statfs failed) — don't
			// ghost, let the user try. f.Size <= 0 means "size unknown" (HF
			// tree API didn't return it) — don't ghost for the same reason.
			fv.FitsOnDisk = available < 0 || f.Size <= 0 || f.Size <= available
		}
		view.Files = append(view.Files, fv)
	}

	if isHTMX(r) {
		respondHTML(w)
		s.renderPartial(w, "hf_files", view)
		return
	}

	respondJSON(w, view)
}

func (s *Server) handleHFDownload(w http.ResponseWriter, r *http.Request) {
	var req struct {
		ModelID  string `json:"model_id"`
		Filename string `json:"filename"`
		Size     int64  `json:"size"`
		Source   string `json:"source"`
	}

	if r.Header.Get("Content-Type") == "application/json" {
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
	} else {
		r.ParseForm()
		req.ModelID = r.FormValue("model_id")
		req.Filename = r.FormValue("filename")
		req.Size, _ = strconv.ParseInt(r.FormValue("size"), 10, 64)
		req.Source = r.FormValue("source")
	}
	req.Source = s.requestSource(req.Source)

	// Inline mode: callers whose swap target can't host the table-shaped
	// download_progress partial (the restore report rows, the pending
	// ghost cards) get a minimal text fragment instead — and ALWAYS 200,
	// success or error, because htmx doesn't swap non-2xx responses and
	// an invisible failure is worse than an inline error line. Live
	// progress lives in the Downloads panel.
	inline := r.FormValue("inline") == "1"
	inlineRespond := func(msg string, isErr bool) {
		respondHTML(w)
		if isErr {
			fmt.Fprintf(w, `<small><mark>%s</mark></small>`, html.EscapeString(msg))
			return
		}
		fmt.Fprintf(w, `<small>%s</small>`, html.EscapeString(msg))
	}

	// Defense-in-depth disk-space guard. The browse UI also disables the
	// button when a file won't fit, but a stale page or direct API call
	// could still POST here.
	if req.Size > 0 {
		avail := s.downloader.AvailableForDownload()
		if avail >= 0 && req.Size > avail {
			needGB := float64(req.Size) / (1024 * 1024 * 1024)
			haveGB := float64(avail) / (1024 * 1024 * 1024)
			msg := fmt.Sprintf("insufficient disk space: need %.1f GiB, only %.1f GiB available after reserving the 2 GiB safety margin and any in-flight downloads", needGB, haveGB)
			if inline && isHTMX(r) {
				inlineRespond(msg, true)
				return
			}
			http.Error(w, msg, http.StatusInsufficientStorage)
			return
		}
	}

	downloadID, err := s.downloader.Start(r.Context(), req.Source, req.ModelID, req.Filename, req.Size)
	if err != nil {
		if inline && isHTMX(r) {
			inlineRespond(err.Error(), true)
			return
		}
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	if inline && isHTMX(r) {
		inlineRespond("queued — progress in the Downloads panel", false)
		return
	}

	if isHTMX(r) {
		respondHTML(w)
		s.renderPartial(w, "download_progress", struct {
			DownloadID string
			Filename   string
		}{DownloadID: downloadID, Filename: req.Filename})
		return
	}

	w.WriteHeader(http.StatusAccepted)
	respondJSON(w, map[string]string{"download_id": downloadID})
}

// downloadProgressHTML renders the progress-bar + stats fragment shared by
// the SSE progress stream and the active-downloads panel. progressStyle is
// an optional style attribute for the <progress> element.
func downloadProgressHTML(status huggingface.DownloadStatus, progressStyle string) string {
	pct := float64(0)
	if status.TotalBytes > 0 {
		pct = float64(status.BytesDownloaded) / float64(status.TotalBytes) * 100
	}
	speedMB := float64(status.SpeedBPS) / (1024 * 1024)
	downloadedGB := models.BytesToGiB(status.BytesDownloaded)
	totalGB := models.BytesToGiB(status.TotalBytes)
	return fmt.Sprintf(
		`<progress value="%.0f" max="100"%s></progress><small>%.1f / %.1f GiB (%.1f MiB/s) — %.0f%%</small>`,
		pct, progressStyle, downloadedGB, totalGB, speedMB, pct)
}

func (s *Server) handleHFDownloadProgress(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	ch, ok := s.downloader.Subscribe(id)
	if !ok {
		http.NotFound(w, r)
		return
	}
	defer s.downloader.Unsubscribe(id, ch)

	sse, err := NewSSEWriter(w)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	for {
		select {
		case status := <-ch:
			data, _ := json.Marshal(status)
			// Send HTML progress update
			var html string
			switch status.Status {
			case "downloading":
				html = downloadProgressHTML(status, "")
			case "complete":
				html = `<p>Download complete!</p>`
			case "failed":
				html = fmt.Sprintf(`<p>Download failed: %s</p>`, status.Error)
			case "cancelled":
				html = `<p>Download paused — resume it from the Models page.</p>`
			default:
				html = string(data)
			}
			sse.SendEvent("progress", html)
			// Terminal states — stop streaming
			if status.Status == "complete" || status.Status == "failed" || status.Status == "cancelled" {
				return
			}
		case <-r.Context().Done():
			return
		}
	}
}

func (s *Server) handleHFActiveDownloads(w http.ResponseWriter, r *http.Request) {
	// Compat shim: a Models page cached from before the merged Downloads card
	// polls this endpoint with htmx and would otherwise swap raw JSON —
	// literally the text "null" for an empty list — into the page. Serve such
	// pages the merged card; its old sibling panel gets empty HTML below.
	if isHTMX(r) {
		s.handleDownloadsPanel(w, r)
		return
	}
	active := s.downloader.ListActive()
	if active == nil {
		active = []huggingface.DownloadStatus{}
	}
	respondJSON(w, active)
}

// downloadRow is one entry in the merged Downloads panel: either an active
// download (live progress + Pause) or paused/incomplete on-disk partials
// (Resume + Discard). A row leaves the panel only when the download completes
// (the model registers, so the partial scan stops reporting it) or the user
// discards the files.
type downloadRow struct {
	ID       string // download ID, also used as a stable DOM handle
	ModelID  string
	Filename string
	Active   bool

	// Active rows
	Pct     float64
	DownGB  float64
	TotalGB float64
	SpeedMB float64

	// Paused/incomplete rows
	OnDiskGB  float64
	PartCount int
}

// handleDownloadsPanel renders the single Downloads card on the Models page:
// all in-flight downloads plus all resumable partials on disk, deduped by
// download ID (an in-flight download's own .part files must not surface as a
// second "incomplete" row — the pre-merge UI had that bug, offering Resume on
// a download that was actively running).
func (s *Server) handleDownloadsPanel(w http.ResponseWriter, r *http.Request) {
	var rows []downloadRow
	activeIDs := make(map[string]bool)
	for _, dl := range s.downloader.ListActive() {
		activeIDs[dl.ID] = true
		row := downloadRow{
			ID: dl.ID, ModelID: dl.ModelID, Filename: dl.Filename, Active: true,
			DownGB:  models.BytesToGiB(dl.BytesDownloaded),
			TotalGB: models.BytesToGiB(dl.TotalBytes),
			SpeedMB: float64(dl.SpeedBPS) / (1024 * 1024),
		}
		if dl.TotalBytes > 0 {
			row.Pct = float64(dl.BytesDownloaded) / float64(dl.TotalBytes) * 100
		}
		rows = append(rows, row)
	}
	for _, p := range s.registry.OrphanParts() {
		id := huggingface.SafeModelID(p.ModelID) + "--" + huggingface.SafeFileID(p.Filename)
		if activeIDs[id] {
			continue
		}
		rows = append(rows, downloadRow{
			ID: id, ModelID: p.ModelID, Filename: p.Filename,
			OnDiskGB: models.BytesToGiB(p.BytesOnDisk), PartCount: p.PartCount,
		})
	}
	// Stable order — ListActive iterates a map, and the client skips the swap
	// when the rendered HTML is byte-identical to what it already shows.
	sort.Slice(rows, func(i, j int) bool { return rows[i].ID < rows[j].ID })

	respondHTML(w)
	s.renderPartial(w, "downloads_panel", struct{ Rows []downloadRow }{rows})
}

// domID sanitizes a string for use as an HTML id attribute or CSS selector
// fragment. Model IDs can contain '.' (e.g. "Qwen3.6"), which CSS parses as
// a class separator and errors on. Also exposed to templates as "cssID".
func domID(s string) string {
	return strings.Map(func(r rune) rune {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '-', r == '_':
			return r
		default:
			return '_'
		}
	}, s)
}

// handleIncompleteDownloads renders the "Incomplete downloads" panel: failed or
// partial downloads that left `.part` files on disk but never registered as
// models, each offering Resume (reuses the normal download path, which picks up
// from the existing partial) and Discard. Mirrors handleHFActiveDownloads.
func (s *Server) handleIncompleteDownloads(w http.ResponseWriter, r *http.Request) {
	// Compat shim, same as handleHFActiveDownloads: stale pages get empty
	// HTML here (their other panel already shows the merged card).
	if isHTMX(r) {
		respondHTML(w)
		return
	}
	parts := s.registry.OrphanParts()
	if parts == nil {
		parts = []models.OrphanPart{}
	}
	respondJSON(w, parts)
}

// handleIncompleteDiscard deletes the on-disk files (completed shards and
// `.part` fragments) for an orphan partial download identified by model_id +
// filename. Paths are constrained to the models directory.
func (s *Server) handleIncompleteDiscard(w http.ResponseWriter, r *http.Request) {
	r.ParseForm()
	modelID := r.FormValue("model_id")
	filename := r.FormValue("filename")
	if modelID == "" || filename == "" {
		http.Error(w, "model_id and filename required", http.StatusBadRequest)
		return
	}

	root := s.cfg.ModelsPath()
	safeName := huggingface.SafeModelID(modelID)
	modelDir := filepath.Join(root, safeName)

	removed := 0
	for _, fn := range huggingface.ExpandShards(filename) {
		final := filepath.Join(modelDir, fn)
		// Containment guard: never touch anything outside the models dir.
		if rel, err := filepath.Rel(root, final); err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
			continue
		}
		for _, path := range []string{final, final + ".part"} {
			if err := os.Remove(path); err == nil {
				removed++
			}
		}
	}
	slog.Info("discarded incomplete download", "model", modelID, "filename", filename, "files_removed", removed)

	// Empty response so the per-row target clears; the panel re-polls and drops
	// the row on its next refresh.
	respondHTML(w)
}

func (s *Server) handleHFDownloadCancel(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	if err := s.downloader.Cancel(id); err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

// onDownloadComplete is called by the downloader when a file finishes.
func (s *Server) onDownloadComplete(source, downloadID, modelID, filename string, sizeBytes int64) {
	safeName := huggingface.SafeModelID(modelID)
	filePath := filepath.Join(s.cfg.ModelsPath(), safeName, filename)

	// mmproj files are vision projectors — don't register as models.
	// Instead, auto-associate with sibling models in the same directory.
	if models.IsMMProjFile(filename) {
		slog.Info("mmproj downloaded, scanning for associated models", "file", filePath)
		s.registry.AutoDetectMMProj()
		return
	}

	meta, _ := models.ParseGGUFMeta(filePath)

	// MTP drafter heads (e.g. gemma-4's gemma4-assistant) aren't runnable
	// models — they load via --model-draft. Don't register; auto-associate
	// with sibling main models like we do for mmproj.
	if meta != nil && models.IsMTPHeadArch(meta.Architecture) {
		slog.Info("MTP drafter head downloaded, associating with sibling models", "file", filePath)
		s.registry.AutoDetectMTP()
		return
	}

	safeFilename := huggingface.SafeFileID(filename)
	m := &models.Model{
		ID:           fmt.Sprintf("%s--%s", safeName, safeFilename),
		ModelID:      modelID,
		Filename:     filename,
		Quant:        models.ParseQuant(filename),
		SizeBytes:    sizeBytes,
		Source:       source,
		FilePath:     filePath,
		VRAMEstGB:    models.EstimateVRAM(sizeBytes),
		DownloadedAt: time.Now(),
	}

	// Architecture-aware VRAM estimation from the GGUF header parsed above.
	if meta != nil {
		meta.ApplyTo(m)
	}

	s.registry.Add(m)

	// Check if an mmproj file already exists in the same directory
	if mmproj := models.FindMMProj(filePath); mmproj != "" {
		if cfg, err := s.registry.GetConfig(m.ID); err == nil && cfg.MmprojPath == "" {
			cfg.MmprojPath = mmproj
			s.registry.SetConfig(m.ID, cfg)
			slog.Info("auto-associated mmproj", "model", m.ID, "mmproj", mmproj)
		}
	}

	// Check if a separate MTP drafter head already exists nearby
	if mtp := models.FindMTP(filePath); mtp != "" {
		if cfg, err := s.registry.GetConfig(m.ID); err == nil && cfg.MtpPath == "" {
			cfg.MtpPath = mtp
			s.registry.SetConfig(m.ID, cfg)
			slog.Info("auto-associated MTP head", "model", m.ID, "mtp", mtp)
		}
	}

	// Fetch publisher sampling presets (Unsloth docs, generation_config.json)
	// in the background — the GGUF-embedded default was already attached via
	// meta.ApplyTo above.
	go s.enrichModelPresets(m.ID)
}
