package api

import (
	"errors"
	"log/slog"
	"os"
	"path/filepath"
	"sync"
	"syscall"

	"github.com/tmac1973/llama-toolchest/internal/atomicfile"
	"github.com/tmac1973/llama-toolchest/internal/generation"
	"github.com/tmac1973/llama-toolchest/internal/huggingface"
	"github.com/tmac1973/llama-toolchest/internal/models"
)

// genMove tracks the startup move of catalog model files into
// generation.ModelsDir. Older versions kept them next to the LLM models
// directory (in a container, inside the data volume instead of on the
// models mount), and changing the models directory in Settings leaves
// them behind too.
type genMove struct {
	mu     sync.Mutex
	active bool
	done   chan struct{}
}

// begin marks the move as running. done is closed by finish.
func (m *genMove) begin() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.active, m.done = true, make(chan struct{})
}

func (m *genMove) finish() {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.active {
		m.active = false
		close(m.done)
	}
}

// busy returns an error while the move runs, for actions that must not
// touch model files until it is done.
func (m *genMove) busy() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.active {
		return errors.New("generation model files are being moved to the models directory; try again when that is done (the log shows its progress)")
	}
	return nil
}

// wait blocks until a running move is done.
func (m *genMove) wait() {
	m.mu.Lock()
	done := m.done
	m.mu.Unlock()
	if done != nil {
		<-done
	}
}

// genFileMove is one file to move.
type genFileMove struct {
	from, to string
	bytes    int64
}

// pendingGenMoves lists the catalog files that are not where
// generation.ModelsDir puts them. Custom sets keep the paths their user
// gave. A file shared by two sets is listed once.
func pendingGenMoves(sets []generation.ModelSet, root string) []genFileMove {
	var moves []genFileMove
	seen := map[string]bool{}
	for _, set := range sets {
		if set.CatalogID == "" {
			continue
		}
		for _, c := range set.Components {
			to := huggingface.FilePath(root, c.Repo, c.File)
			if c.Path == "" || c.Path == to || seen[c.Path] {
				continue
			}
			seen[c.Path] = true
			moves = append(moves, genFileMove{from: c.Path, to: to, bytes: c.Bytes})
		}
	}
	return moves
}

// startGenMove starts moving catalog files into generation.ModelsDir in
// the background, when any are elsewhere. It runs at startup, before
// instances auto-start; until the move is done, auto-start waits, and
// starting or measuring an instance, downloading or deleting a set are
// refused.
func (s *Server) startGenMove() {
	if s.gen == nil {
		return
	}
	root := generation.ModelsDir(s.cfg)
	moves := pendingGenMoves(s.gen.ModelSets(), root)
	if len(moves) == 0 {
		return
	}
	s.genMove.begin()
	go s.moveGenModels(moves, root)
}

// moveGenModels moves the files and points their sets at the new paths.
// A file that cannot be moved stays where it is, and its sets keep using
// it there.
func (s *Server) moveGenModels(moves []genFileMove, root string) {
	defer s.genMove.finish()
	slog.Info("moving generation model files into the models directory", "files", len(moves), "to", root)
	for i, m := range moves {
		slog.Info("moving generation model file", "n", i+1, "of", len(moves), "from", m.from, "to", m.to)
		if err := moveGenFile(m); err != nil {
			slog.Warn("generation model file not moved; its sets keep using it where it is", "from", m.from, "error", err)
			continue
		}
		s.repointGenFile(m.from, m.to)
		removeEmptyParents(filepath.Dir(m.from))
	}
	slog.Info("generation model files moved", "to", root)
}

// moveGenFile moves one file. A file already at the target with the
// expected size (one an earlier, interrupted move copied) is kept and the
// old one removed. Moving between mounts, as from a container's data
// volume to its models mount, needs a copy even on the same disk.
func moveGenFile(m genFileMove) error {
	if info, err := os.Stat(m.to); err == nil && info.Mode().IsRegular() && (m.bytes == 0 || info.Size() == m.bytes) {
		if err := os.Remove(m.from); err != nil && !errors.Is(err, os.ErrNotExist) {
			return err
		}
		return nil
	}
	if _, err := os.Stat(m.from); err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(m.to), 0o755); err != nil {
		return err
	}
	err := os.Rename(m.from, m.to)
	if err == nil || !errors.Is(err, syscall.EXDEV) {
		return err
	}
	if err := atomicfile.Copy(m.from, m.to); err != nil {
		return err
	}
	return os.Remove(m.from)
}

// repointGenFile points every set component that used from at to.
func (s *Server) repointGenFile(from, to string) {
	s.genSetMu.Lock()
	defer s.genSetMu.Unlock()
	for _, set := range s.gen.ModelSets() {
		changed := false
		for i := range set.Components {
			if set.Components[i].Path == from {
				set.Components[i].Path = to
				changed = true
			}
		}
		if !changed {
			continue
		}
		if err := s.gen.UpdateModelSet(set); err != nil {
			slog.Error("failed to record a moved generation model file", "set", set.ID, "error", err)
		}
	}
}

// removeEmptyParents removes dir and its parents while they are empty, up
// to and including the old gen-models folder they are in. Outside a
// gen-models folder it removes nothing.
func removeEmptyParents(dir string) {
	top := dir
	for filepath.Base(top) != models.GenModelsDirName {
		if parent := filepath.Dir(top); parent != top {
			top = parent
			continue
		}
		return
	}
	for {
		if err := os.Remove(dir); err != nil || dir == top {
			return
		}
		dir = filepath.Dir(dir)
	}
}
