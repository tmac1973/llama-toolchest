package api

import (
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
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
	if !s.generationOn() {
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
	// Tags are the badges next to the set's name (catalogTags).
	Tags []genTag `json:"tags"`
}

// genTag is one badge in the catalog: a short label and its tooltip.
type genTag struct {
	Label string `json:"label"`
	Title string `json:"title"`
}

// catalogTags lists what a catalog set takes and gives, for display only:
// the inputs first, then the extras. Every image and music set takes a
// prompt, but only video sets store CapText (it drives job refusals and
// Measure), so the text tag comes from the kind here. The upscaler takes
// an image and no prompt.
func catalogTags(kind generation.BackendKind, caps []string) []genTag {
	var tags []genTag
	add := func(label, title string) { tags = append(tags, genTag{label, title}) }
	has := func(c string) bool { return slices.Contains(caps, c) }
	switch kind {
	case generation.KindSD:
		add("text", "Makes an image from a prompt.")
	case generation.KindAce:
		add("text", "Makes music from a prompt and optional lyrics.")
	case generation.KindVideo:
		if has(generation.CapText) {
			add("text", "Makes video from a prompt alone.")
		}
	case generation.KindUpscale:
		add("image", "Makes an image you send larger. It takes no prompt.")
	}
	if has(generation.CapStartImage) {
		add("start image", "Can start from an image you give.")
	}
	if has(generation.CapEndImage) {
		add("end image", "Can end on an image you give.")
	}
	if has(generation.CapEdit) {
		add("edit", "Can edit an image you send.")
	}
	if has("alpha") {
		add("alpha", "Outputs an alpha channel (transparent backgrounds).")
	}
	return tags
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
	// Instances counts the backend instances that run this set. The
	// Models page offers New instance only for a set with none.
	Instances int `json:"instances"`
}

func (s *Server) genModelsView(store *generation.Store) genModelsData {
	data := genModelsData{Recipes: generation.Recipes(), VideoRecipes: generation.VideoRecipeIDs(), FreeGB: -1}
	if avail := s.downloader.AvailableForDownload(); avail >= 0 {
		data.FreeGB = float64(avail) / 1e9
	}
	instances := map[string]int{}
	for _, inst := range store.Instances() {
		instances[inst.ModelSetID]++
	}
	for _, m := range store.ModelSets() {
		row := s.genSetRowFor(m)
		row.Instances = instances[m.ID]
		data.Sets = append(data.Sets, row)
	}
	for _, e := range generation.Catalog() {
		row := genCatalogRow{CatalogEntry: e, SizeGB: float64(e.TotalBytes()) / 1e9, Tags: catalogTags(e.Kind, e.Capabilities)}
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

// genDetailsView is what the gen_set_details partial renders. Tab is the
// tab whose dialog shows it ("models" or "catalog"), so its buttons open
// that tab's dialogs.
type genDetailsView struct {
	generation.SetDetails
	Tab string
}

// handleGenSetDetails describes how a set on this machine runs: the
// Details dialog for htmx, the same data as JSON otherwise.
func (s *Server) handleGenSetDetails(w http.ResponseWriter, r *http.Request) {
	store := s.genStore(w)
	if store == nil {
		return
	}
	set, err := store.ModelSet(chi.URLParam(r, "id"))
	if errors.Is(err, generation.ErrNotFound) {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	d, err := generation.DetailsForSet(set, generation.ModelsDir(s.cfg))
	s.respondGenDetails(w, r, d, err, "models")
}

// handleGenCatalogDetails is handleGenSetDetails for a catalog entry,
// downloaded or not.
func (s *Server) handleGenCatalogDetails(w http.ResponseWriter, r *http.Request) {
	if s.genStore(w) == nil {
		return
	}
	entry, ok := generation.LookupCatalog(chi.URLParam(r, "id"))
	if !ok {
		http.Error(w, "unknown catalog set", http.StatusNotFound)
		return
	}
	d, err := generation.DetailsForCatalog(entry, generation.ModelsDir(s.cfg))
	s.respondGenDetails(w, r, d, err, "catalog")
}

func (s *Server) respondGenDetails(w http.ResponseWriter, r *http.Request, d generation.SetDetails, err error, defaultTab string) {
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	if !isHTMX(r) {
		respondJSON(w, d)
		return
	}
	tab := r.URL.Query().Get("tab")
	if tab != "models" && tab != "catalog" {
		tab = defaultTab
	}
	respondHTML(w)
	s.renderPartial(w, "gen_set_details", genDetailsView{SetDetails: d, Tab: tab})
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
// complete once every file is present. A file whose size is not the
// size the set gives for it is not recorded, so the set stays
// incomplete rather than loading a wrong file.
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
		c := &set.Components[i]
		if c.Slot == slot {
			if c.Bytes > 0 && size != c.Bytes {
				slog.Warn("generation model file has the wrong size; not recorded",
					"set", setID, "slot", slot, "path", path, "size", size, "expected", c.Bytes)
				c.Path = ""
			} else {
				c.Path, c.Bytes = path, size
			}
		}
		if c.Path == "" {
			complete = false
		}
	}
	set.Complete = complete
	if err := s.gen.UpdateModelSet(set); err != nil {
		slog.Error("failed to record a generation model file", "set", setID, "slot", slot, "error", err)
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
	if kind == generation.KindAce || kind == generation.KindUpscale {
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
		// The ceiling is optional: without it, Max fields stay zero and
		// only the measured size holds requests back.
		set.DefaultWidth, set.DefaultHeight = formInt(r, "width", 832), formInt(r, "height", 480)
		set.DefaultFrames, set.DefaultFPS = formInt(r, "frames", 33), formInt(r, "fps", 24)
		set.MaxFrames = formInt(r, "max_frames", 0)
		set.MaxWidth, set.MaxHeight = formInt(r, "max_width", 0), formInt(r, "max_height", 0)
		if err := checkVideoCeiling(set); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
	case generation.KindUpscale:
		set.Scale = formInt(r, "scale", 4)
		if set.Scale > 8 {
			http.Error(w, "the upscale factor must be between 1 and 8", http.StatusBadRequest)
			return
		}
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
	if kind == generation.KindAce {
		stems := r.FormValue("stem_tasks") == "1"
		set.StemTasks = &stems
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
		shared := newSharedFiles(others, root)
		for _, c := range set.Components {
			// The path the file was recorded at, which may be outside
			// root (a file that could not be moved there).
			path := c.Path
			if path == "" {
				path = huggingface.FilePath(root, c.Repo, c.File)
			}
			if shared.has(path) {
				continue
			}
			for _, p := range []string{path, path + ".part"} {
				if err := os.Remove(p); err != nil && !errors.Is(err, os.ErrNotExist) {
					slog.Warn("failed to remove a generation model file", "path", p, "error", err)
				}
			}
		}
	}
	s.handleGenModels(w, r)
}

// checkVideoCeiling refuses a custom video set's largest size when it is
// half given, not a multiple of 8, or smaller than the set's test video.
// Width and height are compared as long and short sides, as
// videoMeasureSize does, so a portrait test fits a landscape ceiling.
func checkVideoCeiling(set generation.ModelSet) error {
	if (set.MaxWidth > 0) != (set.MaxHeight > 0) {
		return errors.New("give both a max width and a max height, or neither")
	}
	if set.MaxWidth%8 != 0 || set.MaxHeight%8 != 0 {
		return errors.New("max width and max height must be multiples of 8")
	}
	if set.MaxFrames > 0 && set.DefaultFrames > set.MaxFrames {
		return fmt.Errorf("the test video (%d frames) is longer than the max frames (%d)", set.DefaultFrames, set.MaxFrames)
	}
	long, short := max(set.DefaultWidth, set.DefaultHeight), min(set.DefaultWidth, set.DefaultHeight)
	maxLong, maxShort := max(set.MaxWidth, set.MaxHeight), min(set.MaxWidth, set.MaxHeight)
	if maxShort > 0 && (long > maxLong || short > maxShort) {
		return fmt.Errorf("the test size %dx%d is larger than the max size %dx%d", set.DefaultWidth, set.DefaultHeight, set.MaxWidth, set.MaxHeight)
	}
	return nil
}

// sharedFiles holds the files of the sets that stay, so a deleted catalog
// set keeps them. A file matches by its cleaned path, or by being the same
// file on disk: a custom set may reach a catalog file through a link or a
// path with "..".
type sharedFiles struct {
	paths map[string]bool
	infos []os.FileInfo
}

func newSharedFiles(sets []generation.ModelSet, root string) sharedFiles {
	sf := sharedFiles{paths: map[string]bool{}}
	for _, m := range sets {
		for _, c := range m.Components {
			paths := []string{c.Path}
			if c.Repo != "" {
				paths = append(paths, huggingface.FilePath(root, c.Repo, c.File))
			}
			for _, p := range paths {
				if p == "" {
					continue
				}
				sf.paths[filepath.Clean(p)] = true
				if info, err := os.Stat(p); err == nil {
					sf.infos = append(sf.infos, info)
				}
			}
		}
	}
	return sf
}

func (sf sharedFiles) has(path string) bool {
	if sf.paths[filepath.Clean(path)] {
		return true
	}
	info, err := os.Stat(path)
	if err != nil {
		return false
	}
	return slices.ContainsFunc(sf.infos, func(o os.FileInfo) bool { return os.SameFile(info, o) })
}

func formInt(r *http.Request, name string, def int) int {
	if v, err := strconv.Atoi(strings.TrimSpace(r.FormValue(name))); err == nil && v > 0 {
		return v
	}
	return def
}
