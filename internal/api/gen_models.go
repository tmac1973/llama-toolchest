package api

import (
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/tmac1973/llama-toolchest/internal/generation"
	"github.com/tmac1973/llama-toolchest/internal/huggingface"
	"github.com/tmac1973/llama-toolchest/internal/modelsource"
)

// genDownloads tracks the downloads started for each model set this run,
// so the Models page can show their progress. Lost on restart; a set left
// incomplete then shows a Resume button instead.
type genDownloads struct {
	mu    sync.Mutex
	bySet map[string][]string // set ID → download IDs
}

func (g *genDownloads) set(setID string, ids []string) {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.bySet == nil {
		g.bySet = map[string][]string{}
	}
	g.bySet[setID] = ids
}

func (g *genDownloads) get(setID string) []string {
	g.mu.Lock()
	defer g.mu.Unlock()
	return slices.Clone(g.bySet[setID])
}

func (g *genDownloads) drop(setID string) {
	g.mu.Lock()
	defer g.mu.Unlock()
	delete(g.bySet, setID)
}

// genStore returns the generation store, or writes the response and returns
// nil: 404 while generation backends are off, 500 when generation.json
// could not be read at startup.
func (s *Server) genStore(w http.ResponseWriter) *generation.Store {
	if !s.cfg.GenerationEnabled {
		http.NotFound(w, nil)
		return nil
	}
	if s.gen == nil {
		msg := "generation store not loaded"
		if s.genErr != nil {
			msg = s.genErr.Error()
		}
		http.Error(w, msg, http.StatusInternalServerError)
		return nil
	}
	return s.gen
}

// genModelsData feeds the gen_model_sets partial.
type genModelsData struct {
	Catalog []genCatalogRow `json:"catalog"`
	Sets    []genSetRow     `json:"sets"`
	Recipes []string        `json:"recipes"`
	// VideoRecipes are the recipes a custom video set may use.
	VideoRecipes []string `json:"video_recipes"`
	// FreeGB is the space a new download may use, or -1 when unknown.
	FreeGB float64 `json:"free_gb"`
}

type genCatalogRow struct {
	generation.CatalogEntry
	// Installed is true once the set is complete on this machine.
	Installed bool    `json:"installed"`
	SizeGB    float64 `json:"size_gb"`
	// Downloading and Incomplete describe a set that was started but is
	// not complete yet: its files are downloading now, or some are missing
	// and Resume fetches them.
	Downloading bool `json:"downloading"`
	Incomplete  bool `json:"incomplete"`
	Percent     int  `json:"percent"`
}

type genSetRow struct {
	generation.ModelSet
	// Downloading is true while any of its files is still downloading.
	Downloading bool `json:"downloading"`
	Percent     int  `json:"percent"`
	// DoneGB and TotalGB count the set's files; TotalGB is 0 when the
	// sizes are not known (custom sets).
	DoneGB  float64 `json:"done_gb"`
	TotalGB float64 `json:"total_gb"`
	// FailedError is set when a download of this set failed.
	FailedError string `json:"failed_error,omitempty"`
}

func (s *Server) genModelsView(store *generation.Store) genModelsData {
	data := genModelsData{Recipes: generation.Recipes(), VideoRecipes: generation.VideoRecipeIDs(), FreeGB: -1}
	if avail := s.downloader.AvailableForDownload(); avail >= 0 {
		data.FreeGB = float64(avail) / 1e9
	}
	for _, m := range store.ModelSets() {
		data.Sets = append(data.Sets, s.genSetRowFor(m))
	}
	for _, e := range generation.Catalog() {
		row := genCatalogRow{CatalogEntry: e, SizeGB: float64(e.TotalBytes()) / 1e9}
		if i := slices.IndexFunc(data.Sets, func(m genSetRow) bool { return m.ID == e.ID }); i >= 0 {
			set := data.Sets[i]
			row.Installed, row.Downloading, row.Percent = set.Complete, set.Downloading, set.Percent
			row.Incomplete = !set.Complete && !set.Downloading
		}
		data.Catalog = append(data.Catalog, row)
	}
	return data
}

// genSetRowFor reports a set's download progress from its files on disk
// and its downloads still in flight.
func (s *Server) genSetRowFor(m generation.ModelSet) genSetRow {
	row := genSetRow{ModelSet: m}
	var done, total int64
	for _, c := range m.Components {
		total += c.Bytes
		if c.Path != "" {
			done += c.Bytes
		}
	}
	for _, id := range s.genDownloads.get(m.ID) {
		st, ok := s.downloader.Status(id)
		if !ok {
			continue
		}
		switch st.Status {
		case "downloading":
			row.Downloading = true
			done += st.BytesDownloaded
		case "failed":
			row.FailedError = st.Error
		}
	}
	if total > 0 {
		row.TotalGB, row.DoneGB = float64(total)/1e9, float64(min(done, total))/1e9
		row.Percent = int(min(done, total) * 100 / total)
	}
	if m.Complete {
		row.Percent = 100
	}
	return row
}

func (s *Server) handleGenModels(w http.ResponseWriter, r *http.Request) {
	store := s.genStore(w)
	if store == nil {
		return
	}
	data := s.genModelsView(store)
	if !isHTMX(r) {
		respondJSON(w, data)
		return
	}
	respondHTML(w)
	s.renderPartial(w, "gen_model_sets", data)
}

// handleGenCatalog serves the catalog card on the Download Models tab.
// Non-htmx requests get the same JSON as GET /api/gen/models.
func (s *Server) handleGenCatalog(w http.ResponseWriter, r *http.Request) {
	store := s.genStore(w)
	if store == nil {
		return
	}
	data := s.genModelsView(store)
	if !isHTMX(r) {
		respondJSON(w, data)
		return
	}
	respondHTML(w)
	s.renderPartial(w, "gen_catalog", data)
}

// handleGenCatalogDownload starts, or resumes, downloading a catalog set.
// It answers with the section the button is in: the catalog card for
// view=catalog (Download Models tab), otherwise the Models tab's sets.
func (s *Server) handleGenCatalogDownload(w http.ResponseWriter, r *http.Request) {
	store := s.genStore(w)
	if store == nil {
		return
	}
	entry, ok := generation.LookupCatalog(chi.URLParam(r, "id"))
	if !ok {
		http.Error(w, "unknown catalog set", http.StatusNotFound)
		return
	}
	if err := s.genMove.busy(); err != nil {
		http.Error(w, err.Error(), http.StatusConflict)
		return
	}

	s.genSetMu.Lock()
	set, err := store.ModelSet(entry.ID)
	switch {
	case err == nil && set.Complete:
		s.genSetMu.Unlock()
		http.Error(w, fmt.Sprintf("%s is already downloaded", entry.Name), http.StatusConflict)
		return
	case errors.Is(err, generation.ErrNotFound):
		set = generation.ModelSet{
			ID: entry.ID, Name: entry.Name, Kind: entry.Kind, CatalogID: entry.ID, Recipe: entry.Recipe,
			Components: slices.Clone(entry.Components), LaunchArgs: entry.LaunchArgs,
			Capabilities: entry.Capabilities, License: entry.License,
			DefaultWidth: entry.DefaultWidth, DefaultHeight: entry.DefaultHeight,
			DefaultFrames: entry.DefaultFrames, DefaultFPS: entry.DefaultFPS,
			MaxFrames: entry.MaxFrames, MaxWidth: entry.MaxWidth, MaxHeight: entry.MaxHeight,
			Created: time.Now(),
		}
		if err := store.AddModelSet(set); err != nil {
			s.genSetMu.Unlock()
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
	case err != nil:
		s.genSetMu.Unlock()
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	s.genSetMu.Unlock()

	// A file already on disk (for example one ACE-Step set's shared files
	// when the other set is downloaded) is taken as it is; the rest are
	// downloaded.
	root := generation.ModelsDir(s.cfg)
	var ids []string
	for _, c := range set.Components {
		path := huggingface.FilePath(root, c.Repo, c.File)
		if info, err := os.Stat(path); err == nil && info.Mode().IsRegular() && (c.Bytes == 0 || info.Size() == c.Bytes) {
			s.genFileDone(set.ID, c.Slot, path, info.Size())
			continue
		}
		slot := c.Slot
		id, err := s.downloader.StartTo(modelsource.SourceHuggingFace, c.Repo, c.File, root, c.Bytes,
			func(path string, size int64) { s.genFileDone(set.ID, slot, path, size) })
		if err != nil && id == "" {
			http.Error(w, fmt.Sprintf("starting %s: %v", c.File, err), http.StatusInternalServerError)
			return
		}
		// "already in progress" still returns the running download's ID.
		ids = append(ids, id)
	}
	s.genDownloads.set(set.ID, ids)
	slog.Info("generation model set download started", "set", set.ID, "files", len(ids))
	// The downloads list shows the new files without waiting for its poll.
	w.Header().Set("HX-Trigger", "dlPanelRefresh")
	if r.URL.Query().Get("view") == "catalog" {
		s.handleGenCatalog(w, r)
		return
	}
	s.handleGenModels(w, r)
}

// genFileDone records a downloaded file in its set, and marks the set
// complete once every file is present.
func (s *Server) genFileDone(setID, slot, path string, size int64) {
	s.genSetMu.Lock()
	defer s.genSetMu.Unlock()
	set, err := s.gen.ModelSet(setID)
	if err != nil {
		// Deleted while downloading.
		return
	}
	complete := true
	for i := range set.Components {
		if set.Components[i].Slot == slot {
			set.Components[i].Path, set.Components[i].Bytes = path, size
		}
		if set.Components[i].Path == "" {
			complete = false
		}
	}
	set.Complete = complete
	if err := s.gen.UpdateModelSet(set); err != nil {
		slog.Error("recording a generation model file failed", "set", setID, "slot", slot, "error", err)
		return
	}
	if complete {
		slog.Info("generation model set complete", "set", setID)
	}
}

// handleGenCreateSet creates a custom set from files already on disk.
func (s *Server) handleGenCreateSet(w http.ResponseWriter, r *http.Request) {
	store := s.genStore(w)
	if store == nil {
		return
	}
	if !parseForm(w, r) {
		return
	}
	name := strings.TrimSpace(r.FormValue("name"))
	kind := generation.BackendKind(r.FormValue("kind"))
	recipe := r.FormValue("recipe")
	if kind == generation.KindAce {
		recipe = ""
	}
	set := generation.ModelSet{
		ID: generation.Slug(name), Name: name, Kind: kind, Recipe: recipe, Created: time.Now(), Complete: true,
	}
	if set.ID == "" {
		http.Error(w, "the set needs a name", http.StatusBadRequest)
		return
	}
	switch kind {
	case generation.KindSD:
		set.DefaultWidth, set.DefaultHeight = formInt(r, "width", 1024), formInt(r, "height", 1024)
	case generation.KindVideo:
		// A custom video set has no ceiling: its Max fields stay zero.
		set.DefaultWidth, set.DefaultHeight = formInt(r, "width", 832), formInt(r, "height", 480)
		set.DefaultFrames, set.DefaultFPS = formInt(r, "frames", 33), formInt(r, "fps", 24)
	}
	// The files field holds one "slot = /absolute/path" line per file;
	// blank lines are skipped.
	for _, line := range strings.Split(r.FormValue("files"), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		slot, path, ok := strings.Cut(line, "=")
		if !ok {
			http.Error(w, fmt.Sprintf("%q: write each file as slot = /path/to/file", line), http.StatusBadRequest)
			return
		}
		c := generation.Component{Slot: strings.TrimSpace(slot), Path: strings.TrimSpace(path)}
		if info, err := os.Stat(c.Path); err == nil {
			c.Bytes = info.Size()
		}
		set.Components = append(set.Components, c)
	}
	slices.SortFunc(set.Components, func(a, b generation.Component) int { return strings.Compare(a.Slot, b.Slot) })
	if err := generation.ValidateComponents(kind, recipe, set.Components); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if err := generation.CheckFiles(set.Components); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if _, err := generation.LoraDir(set); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	args, err := generation.ValidateExtraArgs(r.FormValue("launch_args"))
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if err := generation.CheckLoraArgs(set, args); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	set.LaunchArgs = args
	if kind == generation.KindSD || kind == generation.KindVideo {
		set.Capabilities = generation.RecipeCapabilities(recipe, set.Components)
	}

	s.genSetMu.Lock()
	err = store.AddModelSet(set)
	s.genSetMu.Unlock()
	if errors.Is(err, generation.ErrIDTaken) {
		http.Error(w, fmt.Sprintf("a set named %q already exists", set.ID), http.StatusConflict)
		return
	}
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	s.handleGenModels(w, r)
}

// handleGenDeleteSet removes a set. A catalog set's files are deleted too,
// except files another set still uses; a custom set's files are the
// user's own and are never touched.
func (s *Server) handleGenDeleteSet(w http.ResponseWriter, r *http.Request) {
	store := s.genStore(w)
	if store == nil {
		return
	}
	if err := s.genMove.busy(); err != nil {
		http.Error(w, err.Error(), http.StatusConflict)
		return
	}
	id := chi.URLParam(r, "id")
	for _, dl := range s.genDownloads.get(id) {
		if st, ok := s.downloader.Status(dl); ok && st.Status == "downloading" {
			s.downloader.Cancel(dl)
		}
	}

	s.genSetMu.Lock()
	set, err := store.ModelSet(id)
	if err == nil {
		err = store.DeleteModelSet(id)
	}
	others := store.ModelSets()
	s.genSetMu.Unlock()
	switch {
	case errors.Is(err, generation.ErrNotFound):
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	case errors.Is(err, generation.ErrInUse):
		http.Error(w, err.Error(), http.StatusConflict)
		return
	case err != nil:
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	s.genDownloads.drop(id)

	if set.CatalogID != "" {
		root := generation.ModelsDir(s.cfg)
		for _, c := range set.Components {
			// The path the file was recorded at, which may be outside
			// root (a file that could not be moved there).
			path := c.Path
			if path == "" {
				path = huggingface.FilePath(root, c.Repo, c.File)
			}
			shared := slices.ContainsFunc(others, func(m generation.ModelSet) bool {
				return slices.ContainsFunc(m.Components, func(o generation.Component) bool {
					return o.Path == path || huggingface.FilePath(root, o.Repo, o.File) == path
				})
			})
			if shared {
				continue
			}
			for _, p := range []string{path, path + ".part"} {
				if err := os.Remove(p); err != nil && !errors.Is(err, os.ErrNotExist) {
					slog.Warn("removing a generation model file failed", "path", p, "error", err)
				}
			}
		}
	}
	s.handleGenModels(w, r)
}

func formInt(r *http.Request, name string, def int) int {
	if v, err := strconv.Atoi(strings.TrimSpace(r.FormValue(name))); err == nil && v > 0 {
		return v
	}
	return def
}
