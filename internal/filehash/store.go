// Package filehash keeps the SHA-256 of downloaded model files, so a
// file is read in full to hash it at most once.
//
// A model file is tens of gigabytes, and hashing one takes a minute or
// more. The downloader hashes each file as it writes it and records the
// result here; a file downloaded before that existed is hashed the first
// time something asks for its hash. A record holds the file's size and
// modification time as well, and is used only while both still match,
// so a file changed on disk by anything else is hashed again.
package filehash

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"sync"
)

type entry struct {
	Size    int64  `json:"size"`
	ModTime int64  `json:"mod_time"` // UnixNano
	SHA256  string `json:"sha256"`
}

// Store maps file paths to their SHA-256. Safe for concurrent use.
type Store struct {
	path string

	mu      sync.Mutex
	entries map[string]entry
}

// Open loads the store saved at path. A missing or unreadable file gives
// an empty store: every record can be rebuilt by hashing again.
func Open(path string) *Store {
	s := &Store{path: path, entries: map[string]entry{}}
	data, err := os.ReadFile(path)
	if err != nil {
		return s
	}
	if err := json.Unmarshal(data, &s.entries); err != nil {
		slog.Warn("file hash store unreadable, starting empty", "path", path, "error", err)
		s.entries = map[string]entry{}
	}
	return s
}

// Lookup returns the recorded hash of a file, if there is one and the
// file's size and modification time still match the record.
func (s *Store) Lookup(path string) (string, bool) {
	info, err := os.Stat(path)
	if err != nil {
		return "", false
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	e, ok := s.entries[path]
	if !ok || e.Size != info.Size() || e.ModTime != info.ModTime().UnixNano() {
		return "", false
	}
	return e.SHA256, true
}

// Record saves the hash of a file as it is on disk now.
func (s *Store) Record(path, sha string) error {
	info, err := os.Stat(path)
	if err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.entries[path] = entry{Size: info.Size(), ModTime: info.ModTime().UnixNano(), SHA256: sha}
	return s.saveLocked()
}

// Hash returns a file's SHA-256, from the record when it is still valid
// and otherwise by reading the file, which is then recorded. progress,
// when set, is called with the bytes read so far.
func (s *Store) Hash(ctx context.Context, path string, progress func(done int64)) (string, error) {
	if sha, ok := s.Lookup(path); ok {
		return sha, nil
	}
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	buf := make([]byte, 1<<20)
	var done int64
	for {
		if err := ctx.Err(); err != nil {
			return "", err
		}
		n, readErr := f.Read(buf)
		if n > 0 {
			h.Write(buf[:n])
			done += int64(n)
			if progress != nil {
				progress(done)
			}
		}
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			return "", readErr
		}
	}
	sha := hex.EncodeToString(h.Sum(nil))
	if err := s.Record(path, sha); err != nil {
		slog.Warn("file hash not saved", "path", path, "error", err)
	}
	return sha, nil
}

func (s *Store) saveLocked() error {
	// Records for files that are gone are dropped here, so the store
	// does not keep every file ever downloaded.
	for p := range s.entries {
		if _, err := os.Stat(p); os.IsNotExist(err) {
			delete(s.entries, p)
		}
	}
	data, err := json.MarshalIndent(s.entries, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(s.path), 0o755); err != nil {
		return err
	}
	tmp := s.path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, s.path)
}
