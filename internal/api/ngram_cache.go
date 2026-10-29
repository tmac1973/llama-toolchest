package api

import (
	"fmt"
	"log/slog"
	"net/http"
	"os"

	"github.com/go-chi/chi/v5"

	"github.com/tmac1973/llama-toolchest/internal/models"
)

// prepareNgramCacheForLoad checks a model's learned n-gram file right
// before toolchest loads it, and deletes it when it is broken: a file cut
// short by a save that was interrupted makes llama-server stop on every
// load (see models.PrepareNgramCacheFile). Router starts check every
// model's file when the preset is written.
func (s *Server) prepareNgramCacheForLoad(id string) {
	cfg, err := s.registry.GetConfig(id)
	if err != nil {
		return
	}
	path := s.registry.NgramCacheFile(id, cfg, s.activeTarget())
	if path == "" {
		return
	}
	if err := models.PrepareNgramCacheFile(path); err != nil {
		slog.Warn("could not prepare the learned n-gram file", "model", id, "error", err)
	}
}

// ngramCacheInfo is what the Configure dialog says about a model's learned
// n-gram file: Text for the line under the method, and Note when the
// active build does not keep the file.
type ngramCacheInfo struct {
	Show bool
	Text string
	Note string
	Size int64
}

// ngramCacheView describes a model's learned n-gram file for Configure.
func (s *Server) ngramCacheView(id string, cfg *models.ModelConfig) ngramCacheInfo {
	if cfg == nil || cfg.SpecAssist != "ngram-cache" {
		return ngramCacheInfo{}
	}
	t := s.activeTarget()
	if !t.NgramCacheFileOption() {
		return ngramCacheInfo{Show: true, Note: "The active llama.cpp build is older than b11758 and does not keep what N-gram Cache learned; it starts empty at every load."}
	}
	path := s.registry.NgramCacheFile(id, cfg, t)
	if path == "" {
		return ngramCacheInfo{Show: true, Note: "The extra flags set lookup-cache-dynamic, so toolchest does not manage the file."}
	}
	info := ngramCacheInfo{Show: true, Text: "Nothing learned yet. The file is written when the model is unloaded."}
	if st, err := os.Stat(path); err == nil {
		info.Size = st.Size()
		info.Text = fmt.Sprintf("Learned n-grams: %s, kept between loads.", humanBytes(st.Size()))
	}
	return info
}

// humanBytes writes a file size in KB or MB.
func humanBytes(n int64) string {
	switch {
	case n >= 1<<20:
		return fmt.Sprintf("%.1f MB", float64(n)/(1<<20))
	case n >= 1<<10:
		return fmt.Sprintf("%.0f KB", float64(n)/(1<<10))
	default:
		return fmt.Sprintf("%d bytes", n)
	}
}

// handleClearNgramCache deletes a model's learned n-gram file. A loaded
// model still holds what it learned and writes it again when it is
// unloaded; the dialog's tooltip says so.
func (s *Server) handleClearNgramCache(w http.ResponseWriter, r *http.Request) {
	id := s.registry.ResolveID(chi.URLParam(r, "id"))
	if _, err := s.registry.Get(id); err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	if err := s.registry.RemoveNgramCacheFile(id); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	if isHTMX(r) {
		cfg, _ := s.registry.GetConfig(id)
		fmt.Fprintf(w, "%s", s.ngramCacheView(id, cfg).Text)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}
