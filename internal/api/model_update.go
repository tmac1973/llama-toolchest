package api

import (
	"context"
	"fmt"
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
)

// Updating a downloaded model.
//
// Repositories are sometimes changed in place: a file is uploaded again
// under the same name, for example with a fixed chat template. The
// Update button on a model card compares the model's files on disk with
// the repository's current ones by SHA-256, shows what changed, and on
// confirmation downloads only those files.
//
// Each new file is downloaded into a .part file beside the old one and
// replaces it only once complete and matching the repository's hash (see
// Downloader.ExpectSHA256), so the old version stays usable until then.
// A model that is loaded keeps the old file open; it is marked as
// needing a restart when the new one is in place.

// updateTarget is one file a model's update covers.
type updateTarget struct {
	Path  string // in the repository, which is also its path under the model's folder
	Local string // on disk
	Role  string // "model", "vision projector" or "MTP head"
}

// updateFile is a target the repository has a different version of.
type updateFile struct {
	updateTarget
	Size   int64  // in the repository
	SHA256 string // in the repository; empty when the host gives none
}

// modelUpdate is the state of one model's check or update.
type modelUpdate struct {
	mu     sync.Mutex
	state  string // "checking", "ready", "failed" or "started"
	cancel context.CancelFunc

	// While checking: the file being read to work out its hash, and the
	// bytes read of all the files that need it.
	hashing     string
	done, total int64

	changed []updateFile
	missing []string // files on disk that the repository no longer has
	err     string
}

// updateTargets lists the files an update of m covers: every shard of
// the model, and the vision projector and MTP head it uses when they
// came from the same repository. ok is false for a model that was not
// downloaded from a repository into the models folder, which cannot be
// updated.
func (s *Server) updateTargets(m *models.Model) (targets []updateTarget, ok bool) {
	if !strings.Contains(m.ModelID, "/") {
		return nil, false
	}
	modelDir := filepath.Join(s.cfg.ModelsPath(), huggingface.SafeModelID(m.ModelID))
	rel := func(p string) (string, bool) {
		r, err := filepath.Rel(modelDir, p)
		if err != nil || r == "." || strings.HasPrefix(r, "..") {
			return "", false
		}
		return filepath.ToSlash(r), true
	}
	main, ok := rel(m.FilePath)
	if !ok {
		return nil, false
	}
	for _, sh := range models.ExpandShards(main) {
		targets = append(targets, updateTarget{Path: sh, Local: filepath.Join(modelDir, filepath.FromSlash(sh)), Role: "model"})
	}
	if cfg, err := s.registry.GetConfig(m.ID); err == nil {
		if p, ok := rel(cfg.MmprojPath); ok && cfg.MmprojPath != "" {
			targets = append(targets, updateTarget{Path: p, Local: cfg.MmprojPath, Role: "vision projector"})
		}
		if p, ok := rel(cfg.MtpPath); ok && cfg.MtpPath != "" {
			targets = append(targets, updateTarget{Path: p, Local: cfg.MtpPath, Role: "MTP head"})
		}
	}
	return targets, true
}

func (s *Server) modelUpdateFor(id string) *modelUpdate {
	s.updatesMu.Lock()
	defer s.updatesMu.Unlock()
	return s.updates[id]
}

// handleModelUpdateCheck starts checking a model's files against its
// repository, unless a check is already running, and shows its state.
func (s *Server) handleModelUpdateCheck(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	m, err := s.registry.Get(id)
	if err != nil {
		http.Error(w, "model not found", http.StatusNotFound)
		return
	}
	targets, ok := s.updateTargets(m)
	if !ok {
		http.Error(w, "this model was not downloaded from a repository, so it cannot be updated", http.StatusBadRequest)
		return
	}

	s.updatesMu.Lock()
	u := s.updates[id]
	if u == nil || u.snapshotState() != "checking" {
		ctx, cancel := context.WithCancel(context.Background())
		u = &modelUpdate{state: "checking", cancel: cancel}
		s.updates[id] = u
		go s.checkModelUpdate(ctx, u, m, targets)
	}
	s.updatesMu.Unlock()

	s.renderModelUpdate(w, id, u)
}

// handleModelUpdateStatus shows the state of a model's check, which the
// panel asks for every second while the check runs.
func (s *Server) handleModelUpdateStatus(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	u := s.modelUpdateFor(id)
	if u == nil {
		respondHTML(w)
		return
	}
	s.renderModelUpdate(w, id, u)
}

// handleModelUpdateDismiss stops a running check and forgets the result.
func (s *Server) handleModelUpdateDismiss(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	s.updatesMu.Lock()
	if u := s.updates[id]; u != nil {
		u.cancel()
		delete(s.updates, id)
	}
	s.updatesMu.Unlock()
	respondHTML(w)
}

// handleModelUpdateApply downloads the files the last check found
// changed.
func (s *Server) handleModelUpdateApply(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	m, err := s.registry.Get(id)
	if err != nil {
		http.Error(w, "model not found", http.StatusNotFound)
		return
	}
	u := s.modelUpdateFor(id)
	if u == nil {
		http.Error(w, "check for an update first", http.StatusConflict)
		return
	}
	u.mu.Lock()
	if u.state != "ready" || len(u.changed) == 0 {
		u.mu.Unlock()
		http.Error(w, "there is no update to download", http.StatusConflict)
		return
	}
	changed := u.changed
	u.mu.Unlock()

	var need int64
	for _, f := range changed {
		need += f.Size
	}
	// The old files stay until each new one is complete, so the new ones
	// need the room in full.
	if avail := s.downloader.AvailableForDownload(); avail >= 0 && need > avail {
		u.fail(fmt.Sprintf("Not enough disk space: the update needs %.1f GiB and %.1f GiB is available, after keeping 2 GiB free and room for downloads in progress.",
			models.BytesToGiB(need), models.BytesToGiB(avail)))
		s.renderModelUpdate(w, id, u)
		return
	}

	if err := s.startModelUpdate(m, changed); err != nil {
		u.fail("The update could not start: " + err.Error())
		s.renderModelUpdate(w, id, u)
		return
	}
	u.mu.Lock()
	u.state = "started"
	u.mu.Unlock()
	s.renderModelUpdate(w, id, u)
}

// startModelUpdate starts the downloads that replace the changed files.
func (s *Server) startModelUpdate(m *models.Model, changed []updateFile) error {
	// A .part file beside the old file is what tells the downloader to
	// replace it rather than skip it as already downloaded. One left by
	// an earlier, interrupted update is resumed: the hash check catches
	// one that belongs to an older version.
	for _, f := range changed {
		part := f.Local + ".part"
		if _, err := os.Stat(part); err != nil {
			if err := os.WriteFile(part, nil, 0o644); err != nil {
				return err
			}
		}
		if f.SHA256 != "" {
			s.downloader.ExpectSHA256(f.Local, f.SHA256)
		}
		// The model's shards go in one download, which only gets their
		// total size, so each shard's own size is set here.
		s.downloader.ExpectSize(f.Local, f.Size)
	}

	// The model's shards go in one download, which works through every
	// shard and replaces those with a .part; the other files go in one
	// each.
	var main string
	var shardBytes int64
	var others []updateFile
	for _, f := range changed {
		if f.Role != "model" {
			others = append(others, f)
			continue
		}
		if main == "" {
			main = f.Path
		}
		shardBytes += f.Size
	}
	source := s.requestSource(m.Source)
	var errs []string
	if main != "" {
		if _, err := s.downloader.Start(context.Background(), source, m.ModelID, main, shardBytes); err != nil {
			errs = append(errs, err.Error())
		}
	}
	for _, f := range others {
		if _, err := s.downloader.Start(context.Background(), source, m.ModelID, f.Path, f.Size); err != nil {
			errs = append(errs, fmt.Sprintf("%s: %v", f.Path, err))
		}
	}
	if len(errs) > 0 {
		return fmt.Errorf("%s", strings.Join(errs, "; "))
	}
	return nil
}

// checkModelUpdate compares a model's files with its repository.
func (s *Server) checkModelUpdate(ctx context.Context, u *modelUpdate, m *models.Model, targets []updateTarget) {
	listCtx, cancel := context.WithTimeout(ctx, time.Minute)
	repo, err := s.sourceClient(m.Source).RepoFiles(listCtx, m.ModelID)
	cancel()
	if err != nil {
		if ctx.Err() == nil {
			u.fail("Could not read the repository's file list: " + err.Error())
		}
		return
	}
	byPath := make(map[string]int, len(repo))
	for i, f := range repo {
		byPath[f.Path] = i
	}

	// Sort the targets: those the size settles, and those that need the
	// file on disk read to work out its hash.
	var changed []updateFile
	var missing []string
	var toHash []updateFile
	var total int64
	for _, t := range targets {
		i, ok := byPath[t.Path]
		if !ok {
			missing = append(missing, t.Path)
			continue
		}
		rf := updateFile{updateTarget: t, Size: repo[i].Size, SHA256: repo[i].SHA256}
		info, err := os.Stat(t.Local)
		switch {
		case err != nil || info.Size() != rf.Size:
			changed = append(changed, rf)
		case rf.SHA256 == "":
			// Same size and no hash to compare with: nothing shows a
			// change. Every model file has one on both hosts.
		default:
			toHash = append(toHash, rf)
			if _, known := s.hashes.Lookup(t.Local); !known {
				total += info.Size()
			}
		}
	}

	u.mu.Lock()
	u.total = total
	u.mu.Unlock()
	var base int64
	for _, f := range toHash {
		u.mu.Lock()
		u.hashing = f.Path
		u.mu.Unlock()
		_, known := s.hashes.Lookup(f.Local)
		sha, err := s.hashes.Hash(ctx, f.Local, func(done int64) {
			u.mu.Lock()
			u.done = base + done
			u.mu.Unlock()
		})
		if err != nil {
			if ctx.Err() == nil {
				u.fail(fmt.Sprintf("Could not read %s: %v", f.Path, err))
			}
			return
		}
		if !known {
			if info, err := os.Stat(f.Local); err == nil {
				base += info.Size()
			}
		}
		if !strings.EqualFold(sha, f.SHA256) {
			changed = append(changed, f)
		}
	}

	u.mu.Lock()
	u.state = "ready"
	u.changed = changed
	u.missing = missing
	u.mu.Unlock()
	slog.Info("model update check", "model", m.ID, "changed", len(changed), "missing", len(missing))
}

func (u *modelUpdate) fail(msg string) {
	u.mu.Lock()
	defer u.mu.Unlock()
	u.state = "failed"
	u.err = msg
}

func (u *modelUpdate) snapshotState() string {
	u.mu.Lock()
	defer u.mu.Unlock()
	return u.state
}

// sizeText writes a file size in GiB, or in MiB when it is below 1 GiB,
// where GiB to one decimal place would read as 0.0.
func sizeText(b int64) string {
	if b < 1<<30 {
		return fmt.Sprintf("%.1f MiB", float64(b)/(1<<20))
	}
	return fmt.Sprintf("%.1f GiB", models.BytesToGiB(b))
}

// modelUpdateView is what the model_update partial renders.
type modelUpdateView struct {
	ID        string
	ModelName string
	State     string
	Hashing   string
	Pct       float64
	Changed   []updateFile
	Missing   []string
	Bytes     int64
	Err       string
	// Loaded is set when the model is loaded now, so it keeps using the
	// old files until it is next loaded.
	Loaded bool
}

func (s *Server) renderModelUpdate(w http.ResponseWriter, id string, u *modelUpdate) {
	u.mu.Lock()
	v := modelUpdateView{
		ID:      id,
		State:   u.state,
		Hashing: u.hashing,
		Changed: u.changed,
		Missing: u.missing,
		Err:     u.err,
	}
	if u.total > 0 {
		v.Pct = float64(u.done) / float64(u.total) * 100
	}
	u.mu.Unlock()
	for _, f := range v.Changed {
		v.Bytes += f.Size
	}
	if m, err := s.registry.Get(id); err == nil {
		v.ModelName = m.PublicName()
		if v.State == "ready" && len(v.Changed) > 0 && s.process.IsRunning() {
			v.Loaded = s.loadedIn(s.routerKnownStates(), m)
		}
	}
	respondHTML(w)
	s.renderPartial(w, "model_update", v)
}

// loadedIn reports whether a routerKnownStates map has m loaded, or
// loading.
func (s *Server) loadedIn(states map[string]string, m *models.Model) bool {
	st, _ := routerStateFor(states, s.registry.RouterName(m.ID), m.ID, m.PublicName())
	return st == "loaded" || st == "loading"
}

// markUpdatedModelsDirty marks every loaded model that uses the file at
// path as needing a restart. Called when a download finishes: when it
// replaced a file, a loaded model still has the old one open.
func (s *Server) markUpdatedModelsDirty(path string) {
	if !s.process.IsRunning() {
		return
	}
	states := s.routerKnownStates()
	for _, m := range s.registry.List() {
		uses := false
		dir := filepath.Dir(m.FilePath)
		for _, sh := range models.ExpandShards(filepath.Base(m.FilePath)) {
			if filepath.Join(dir, sh) == path {
				uses = true
			}
		}
		if cfg, err := s.registry.GetConfig(m.ID); err == nil && (cfg.MmprojPath == path || cfg.MtpPath == path) {
			uses = true
		}
		if uses && s.loadedIn(states, m) {
			s.markDirty(m.ID)
		}
	}
}
