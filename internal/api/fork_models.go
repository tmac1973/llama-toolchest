package api

import (
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/go-chi/chi/v5"
	"github.com/tmac1973/llama-toolchest/internal/forks"
	"github.com/tmac1973/llama-toolchest/internal/huggingface"
	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/modelsource"
)

// curatedForks is the fork list the API serves; a variable so tests can
// use small forks instead of downloading a real one.
var curatedForks = forks.Curated

// findFork returns the curated fork with the given ID.
func findFork(id string) (forks.Fork, bool) {
	for _, f := range curatedForks() {
		if f.ID == id {
			return f, true
		}
	}
	return forks.Fork{}, false
}

// forkStore returns the fork store, or writes the response and returns
// nil: 404 while forks are off, 500 when forks.json could not be read at
// startup.
func (s *Server) forkStore(w http.ResponseWriter) *forks.Store {
	if !s.forksOn() {
		http.NotFound(w, nil)
		return nil
	}
	if s.forks == nil {
		msg := "fork store not loaded"
		if s.forksErr != nil {
			msg = s.forksErr.Error()
		}
		http.Error(w, msg, http.StatusInternalServerError)
		return nil
	}
	return s.forks
}

// forkModelsDir is where the forks' model sets are downloaded: a folder
// of the models directory the LLM scan skips.
func (s *Server) forkModelsDir() string {
	cfg := s.cfgCopy()
	return filepath.Join(cfg.ModelsPath(), models.ForkModelsDirName)
}

// forkSetDir is the folder that holds a fork's model files, laid out as
// in the fork's ModelFiles.
func (s *Server) forkSetDir(f forks.Fork) string {
	return huggingface.FilePath(s.forkModelsDir(), f.ModelRepo, "")
}

// forkFileGroup is files one download fetches: all shards of a split
// GGUF (the downloader fetches every shard when given the first), or one
// other file.
type forkFileGroup struct {
	First string // the file the download is started with
	Files []forks.ModelFile
	Bytes int64
}

// forkFileGroups splits a fork's files into download groups. A shard
// must never get a download of its own: each would fetch all the shards,
// several at once, into the same files.
func forkFileGroups(f forks.Fork) []forkFileGroup {
	var out []forkFileGroup
	at := map[string]int{}
	for _, m := range f.ModelFiles {
		key := m.Path
		if base, _, ok := models.SplitShardName(m.Path); ok {
			key = "shards:" + base
		}
		i, ok := at[key]
		if !ok {
			i = len(out)
			at[key] = i
			out = append(out, forkFileGroup{First: m.Path})
		}
		g := &out[i]
		g.Files = append(g.Files, m)
		g.Bytes += m.Bytes
		if m.Path < g.First {
			g.First = m.Path
		}
	}
	return out
}

// forkFileOnDisk reports whether a model file is in place at its listed
// size.
func forkFileOnDisk(dir string, m forks.ModelFile) bool {
	info, err := os.Stat(filepath.Join(dir, filepath.FromSlash(m.Path)))
	return err == nil && info.Mode().IsRegular() && info.Size() == m.Bytes
}

// forkModelRow is one fork's line in the Models page section.
type forkModelRow struct {
	Fork        forks.Fork
	SizeGB      float64
	Complete    bool
	Downloading bool
	// Started is true when the set exists but is neither complete nor
	// downloading: Resume fetches the files still missing.
	Started     bool
	Percent     int
	DoneGB      float64
	FailedError string
	Instances   int
}

// forkModelsData feeds the fork_model_sets partial.
type forkModelsData struct {
	Rows []forkModelRow
	// FreeGB is the space a new download may use, or -1 when unknown.
	FreeGB float64
}

func (s *Server) forkModelsView(store *forks.Store) forkModelsData {
	data := forkModelsData{FreeGB: -1}
	if avail := s.downloader.AvailableForDownload(); avail >= 0 {
		data.FreeGB = float64(avail) / 1e9
	}
	instances := map[string]int{}
	for _, inst := range store.Instances() {
		instances[inst.ForkID]++
	}
	for _, f := range curatedForks() {
		row := forkModelRow{Fork: f, Instances: instances[f.ID]}
		var total, done int64
		for _, g := range forkFileGroups(f) {
			total += g.Bytes
			var onDisk int64
			for _, m := range g.Files {
				if forkFileOnDisk(s.forkSetDir(f), m) {
					onDisk += m.Bytes
				}
			}
			// A running download's count includes the files it found
			// already in place, so it replaces the group's count from disk
			// when it is larger rather than adding to it.
			for _, id := range s.forkDownloads.get(f.ID + "|" + g.First) {
				st, ok := s.downloader.Status(id)
				if !ok {
					continue
				}
				switch st.Status {
				case "downloading":
					row.Downloading = true
					onDisk = max(onDisk, st.BytesDownloaded)
				case "failed":
					row.FailedError = st.Error
				}
			}
			done += min(onDisk, g.Bytes)
		}
		row.SizeGB = float64(total) / 1e9
		row.DoneGB = float64(done) / 1e9
		if total > 0 {
			row.Percent = int(done * 100 / total)
		}
		if set, err := store.ModelSet(f.ID); err == nil {
			row.Complete = set.Complete
			row.Started = !set.Complete && !row.Downloading
		}
		if row.Complete {
			row.Percent = 100
		}
		data.Rows = append(data.Rows, row)
	}
	return data
}

func (s *Server) handleForkModels(w http.ResponseWriter, r *http.Request) {
	store := s.forkStore(w)
	if store == nil {
		return
	}
	data := s.forkModelsView(store)
	if !isHTMX(r) {
		respondJSON(w, data)
		return
	}
	respondHTML(w)
	s.renderPartial(w, "fork_model_sets", data)
}

// handleForkModelDownload starts, or resumes, downloading a fork's model
// set from the pinned revision of its repo. Files already in place at
// their size are kept.
func (s *Server) handleForkModelDownload(w http.ResponseWriter, r *http.Request) {
	store := s.forkStore(w)
	if store == nil {
		return
	}
	f, ok := findFork(chi.URLParam(r, "fork"))
	if !ok {
		http.Error(w, "unknown fork", http.StatusNotFound)
		return
	}
	dir := s.forkSetDir(f)

	s.forkSetMu.Lock()
	set, err := store.ModelSet(f.ID)
	switch {
	case err == nil && set.Complete:
		s.forkSetMu.Unlock()
		http.Error(w, fmt.Sprintf("%s is already downloaded", f.ModelName), http.StatusConflict)
		return
	case errors.Is(err, forks.ErrNotFound):
		if err := store.PutModelSet(forks.ModelSet{ForkID: f.ID, Dir: dir}); err != nil {
			s.forkSetMu.Unlock()
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
	case err != nil:
		s.forkSetMu.Unlock()
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	s.forkSetMu.Unlock()

	var todo []forkFileGroup
	var missing int64
	for _, g := range forkFileGroups(f) {
		need := int64(0)
		for _, m := range g.Files {
			if !forkFileOnDisk(dir, m) {
				need += m.Bytes
			}
		}
		if need > 0 {
			todo = append(todo, g)
			missing += need
		}
	}
	if avail := s.downloader.AvailableForDownload(); avail >= 0 && avail < missing {
		http.Error(w, fmt.Sprintf("Not enough free disk space: %.1f GB needed, %.1f GB free.",
			float64(missing)/1e9, float64(avail)/1e9), http.StatusInsufficientStorage)
		return
	}
	for _, g := range todo {
		// The downloader keeps any finished file it finds, whatever its
		// size, so a file of the wrong size (cut short, or from another
		// revision) would stop the set from ever completing. Remove it so
		// it is fetched again; a .part file is kept and resumed.
		for _, m := range g.Files {
			path := filepath.Join(dir, filepath.FromSlash(m.Path))
			if info, err := os.Stat(path); err == nil && info.Mode().IsRegular() && info.Size() != m.Bytes {
				slog.Warn("fork model file has the wrong size; downloading it again",
					"fork", f.ID, "file", m.Path, "size", info.Size(), "want", m.Bytes)
				if err := os.Remove(path); err != nil {
					http.Error(w, fmt.Sprintf("removing %s: %v", m.Path, err), http.StatusInternalServerError)
					return
				}
			}
		}
		id, err := s.downloader.StartToAt(modelsource.SourceHuggingFace, f.ModelRepo, f.ModelRevision, g.First,
			s.forkModelsDir(), g.Bytes, func(string, int64) { s.forkFileDone(f.ID) })
		if err != nil && id == "" {
			http.Error(w, fmt.Sprintf("starting %s: %v", g.First, err), http.StatusInternalServerError)
			return
		}
		// "already in progress" still returns the running download's ID.
		s.forkDownloads.set(f.ID+"|"+g.First, []string{id})
	}
	if len(todo) == 0 {
		// Every file was already there, for example after a restart.
		s.forkFileDone(f.ID)
	}
	slog.Info("fork model set download started", "fork", f.ID, "downloads", len(todo), "revision", f.ModelRevision)
	// The downloads list shows the new files without waiting for its poll.
	w.Header().Set("HX-Trigger", "dlPanelRefresh")
	s.handleForkModels(w, r)
}

// forkFileDone marks a fork's set complete once every file is in place at
// its listed size.
func (s *Server) forkFileDone(forkID string) {
	f, ok := findFork(forkID)
	if !ok || s.forks == nil {
		return
	}
	s.forkSetMu.Lock()
	defer s.forkSetMu.Unlock()
	set, err := s.forks.ModelSet(forkID)
	if err != nil {
		// Deleted while downloading.
		return
	}
	complete := true
	for _, m := range f.ModelFiles {
		if !forkFileOnDisk(set.Dir, m) {
			complete = false
			break
		}
	}
	if complete == set.Complete {
		return
	}
	set.Complete = complete
	if err := s.forks.PutModelSet(set); err != nil {
		slog.Error("failed to record a fork model set", "fork", forkID, "error", err)
		return
	}
	if complete {
		slog.Info("fork model set complete", "fork", forkID)
	}
}

// handleForkModelDelete stops a fork set's downloads and deletes its
// files. Refused while a fork instance is set up to use it.
func (s *Server) handleForkModelDelete(w http.ResponseWriter, r *http.Request) {
	store := s.forkStore(w)
	if store == nil {
		return
	}
	f, ok := findFork(chi.URLParam(r, "fork"))
	if !ok {
		http.Error(w, "unknown fork", http.StatusNotFound)
		return
	}
	for _, inst := range store.Instances() {
		if inst.ForkID == f.ID {
			http.Error(w, "Delete the fork instances that use this model set first.", http.StatusConflict)
			return
		}
	}
	for _, g := range forkFileGroups(f) {
		key := f.ID + "|" + g.First
		for _, id := range s.forkDownloads.get(key) {
			if st, ok := s.downloader.Status(id); ok && st.Status == "downloading" {
				s.downloader.Cancel(id)
			}
		}
		s.forkDownloads.drop(key)
	}

	if status, err := s.deleteForkSet(store, f); err != nil {
		http.Error(w, err.Error(), status)
		return
	}
	slog.Info("fork model set deleted", "fork", f.ID)
	s.handleForkModels(w, r)
}

// deleteForkSet removes a fork's set folder and store entry, and returns
// the HTTP status for an error.
func (s *Server) deleteForkSet(store *forks.Store, f forks.Fork) (int, error) {
	s.forkSetMu.Lock()
	defer s.forkSetMu.Unlock()
	set, err := store.ModelSet(f.ID)
	if errors.Is(err, forks.ErrNotFound) {
		return http.StatusNotFound, errors.New("nothing to delete")
	}
	if err != nil {
		return http.StatusInternalServerError, err
	}
	// Only a folder inside the fork models folder: a stored path is never
	// trusted to delete anything else.
	rel, err := filepath.Rel(s.forkModelsDir(), set.Dir)
	if err != nil || rel == "." || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) || filepath.IsAbs(rel) {
		return http.StatusInternalServerError, errors.New("the model set's folder is not in the fork models folder; delete it by hand")
	}
	if err := os.RemoveAll(set.Dir); err != nil {
		return http.StatusInternalServerError, err
	}
	if err := store.DeleteModelSet(f.ID); err != nil {
		return http.StatusInternalServerError, err
	}
	return 0, nil
}
