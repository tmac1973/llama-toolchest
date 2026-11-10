package api

import (
	"context"
	"fmt"
	"html"
	"io/fs"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/go-chi/chi/v5"

	"github.com/tmac1973/llama-toolchest/internal/huggingface"
	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/modelsource"
)

// Some publishers keep a model's vision projector (mmproj) in another
// repository: mradermacher's "X-i1-GGUF" has none, its "X-GGUF" has it.
// Configure offers to download it from there into the model's own folder,
// where the projector detection pairs it like any other.

// projectorLookupTTL is how long a finished lookup is kept, found or not;
// projectorRetryTTL is how long a failed one is, so a host that was down
// is asked again soon without being asked on every panel render.
const (
	projectorLookupTTL = 24 * time.Hour
	projectorRetryTTL  = 10 * time.Minute
	projectorTimeout   = 20 * time.Second
)

type projectorEntry struct {
	repo   *huggingface.ProjectorRepo // nil: none found, or the lookup failed
	failed bool
	at     time.Time
}

// projectorCache holds the projector lookups, keyed by repository.
type projectorCache struct {
	mu      sync.Mutex
	entries map[string]projectorEntry
}

func (c *projectorCache) get(repo string) (projectorEntry, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	e, ok := c.entries[repo]
	if !ok {
		return e, false
	}
	ttl := projectorLookupTTL
	if e.failed {
		ttl = projectorRetryTTL
	}
	return e, time.Since(e.at) < ttl
}

func (c *projectorCache) put(repo string, e projectorEntry) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.entries == nil {
		c.entries = map[string]projectorEntry{}
	}
	c.entries[repo] = e
}

// wantsProjectorLookup reports whether a model is one Configure looks up
// a projector for: a HuggingFace chat model without built-in vision that
// has no projector set or beside it.
func wantsProjectorLookup(m *models.Model, cfg *models.ModelConfig) bool {
	if m == nil || cfg == nil || m.HasBuiltinVision || m.Kind() != models.KindChat {
		return false
	}
	if modelsource.NormalizeSource(m.Source) != modelsource.SourceHuggingFace || !strings.Contains(m.ModelID, "/") {
		return false
	}
	return cfg.MmprojPath == "" && models.FindMMProj(m.FilePath) == ""
}

// projectorFor returns the repository holding a model's projector, from
// the cache or by asking HuggingFace. A failed lookup is logged and
// treated as none found.
func (s *Server) projectorFor(ctx context.Context, m *models.Model) *huggingface.ProjectorRepo {
	if e, ok := s.projectors.get(m.ModelID); ok {
		return e.repo
	}
	find := s.findProjector
	if find == nil {
		find = s.hfClient.FindProjectorRepo
	}
	ctx, cancel := context.WithTimeout(ctx, projectorTimeout)
	defer cancel()
	repo, err := find(ctx, m.ModelID, m.BaseModelRepo)
	if err != nil {
		slog.Debug("projector lookup failed", "repo", m.ModelID, "error", err)
	}
	s.projectors.put(m.ModelID, projectorEntry{repo: repo, failed: err != nil, at: time.Now()})
	return repo
}

// projectorOfferData is what the projector_offer partial renders.
type projectorOfferData struct {
	ModelID string
	Repo    string
	RepoURL string
	OwnRepo bool // the projector is in the model's own repository
	Files   []huggingface.ModelFile
}

// handleProjectorSources renders the offer to download a model's
// projector from another repository, or nothing when there is none.
func (s *Server) handleProjectorSources(w http.ResponseWriter, r *http.Request) {
	id := s.registry.ResolveID(chi.URLParam(r, "id"))
	m, err := s.registry.Get(id)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	cfg, _ := s.registry.GetConfig(id)
	respondHTML(w)
	if !wantsProjectorLookup(m, cfg) {
		return
	}
	repo := s.projectorFor(r.Context(), m)
	if repo == nil {
		return
	}
	s.renderPartial(w, "projector_offer", projectorOfferData{
		ModelID: id,
		Repo:    repo.Repo,
		RepoURL: huggingface.RepoURL(repo.Repo),
		OwnRepo: strings.EqualFold(repo.Repo, m.ModelID),
		Files:   repo.Files,
	})
}

// handleProjectorDownload downloads one of the projector files the lookup
// offered into the model's own folder, and pairs it with the model when
// it is done.
func (s *Server) handleProjectorDownload(w http.ResponseWriter, r *http.Request) {
	id := s.registry.ResolveID(chi.URLParam(r, "id"))
	m, err := s.registry.Get(id)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	if !parseForm(w, r) {
		return
	}
	// Only a file the lookup offered is downloaded, so the form cannot
	// name an arbitrary repository or path.
	repo := s.projectorFor(r.Context(), m)
	var file *huggingface.ModelFile
	if repo != nil && repo.Repo == r.FormValue("repo") {
		for i := range repo.Files {
			if repo.Files[i].Filename == r.FormValue("file") {
				file = &repo.Files[i]
			}
		}
	}
	reply := func(msg string, isErr bool) {
		respondHTML(w)
		if isErr {
			fmt.Fprintf(w, `<small><mark>%s</mark></small>`, html.EscapeString(msg))
			return
		}
		fmt.Fprintf(w, `<small>%s</small>`, html.EscapeString(msg))
	}
	if file == nil {
		reply("That projector is no longer offered. Close and open Configure again.", true)
		return
	}
	if avail := s.downloader.AvailableForDownload(); avail >= 0 && file.Size > avail {
		reply(fmt.Sprintf("Not enough disk space: the projector needs %s.", sizeText(file.Size)), true)
		return
	}

	dir := filepath.Dir(m.FilePath)
	_, err = s.downloader.StartInto(modelsource.SourceHuggingFace, repo.Repo, file.Filename, dir, file.Size,
		func(path string, _ int64) { s.pairDownloadedProjector(id, path) })
	if err != nil {
		reply(err.Error(), true)
		return
	}
	reply(fmt.Sprintf("Downloading %s. Progress is in the Downloads panel; the projector is set when it is done.", filepath.Base(file.Filename)), false)
}

// pairDownloadedProjector sets a downloaded projector on the model it was
// downloaded for, unless one was set meanwhile, and on any other model in
// the folder that has none.
func (s *Server) pairDownloadedProjector(id, path string) {
	if cfg, err := s.registry.GetConfig(id); err == nil && cfg.MmprojPath == "" {
		cfg.MmprojPath = path
		if err := s.registry.SetConfig(id, cfg); err != nil {
			slog.Error("failed to set the downloaded projector", "model", id, "error", err)
		} else {
			s.markDirty(id)
			slog.Info("paired the downloaded projector", "model", id, "mmproj", path)
		}
	}
	s.registry.AutoDetectMMProj()
}

// projectorChoice is one projector file on disk, for the Projector field's
// list.
type projectorChoice struct {
	Path  string
	Label string // the path inside the models folder
}

// maxProjectorDepth bounds the walk for projector files: models sit at
// <models>/<org--repo>/[<quant>/]<file>.
const maxProjectorDepth = 3

// projectorChoices lists the projector (mmproj) GGUF files under the
// models folder.
func projectorChoices(modelsDir string) []projectorChoice {
	var out []projectorChoice
	filepath.WalkDir(modelsDir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			if path == modelsDir {
				return err
			}
			return nil
		}
		rel, _ := filepath.Rel(modelsDir, path)
		if d.IsDir() {
			if rel != "." && strings.Count(rel, string(os.PathSeparator)) >= maxProjectorDepth-1 {
				return filepath.SkipDir
			}
			return nil
		}
		name := d.Name()
		if strings.HasSuffix(strings.ToLower(name), ".gguf") && models.IsMMProjFile(name) {
			out = append(out, projectorChoice{Path: path, Label: filepath.ToSlash(rel)})
		}
		return nil
	})
	return out
}
