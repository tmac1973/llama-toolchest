package modelsource

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/tmac1973/llama-toolchest/internal/atomicfile"
	"github.com/tmac1973/llama-toolchest/internal/models"
)

// MetaCache keeps the model descriptions ProbeMeta reads, in memory and
// on disk, so a repository's header is read once rather than on every
// visit or every recommendation build.
//
// Entries never expire. A published file is identified by its content
// hash where the host gives one (HuggingFace's LFS oid), and otherwise by
// its name and size; a new upload changes the key, so a stale entry is
// never served for it, only left unused.
//
// The disk side is a convenience: a cache that cannot be read or written
// behaves as an empty one.
type MetaCache struct {
	dir string

	mu  sync.RWMutex
	mem map[string]*models.GGUFMeta
}

// NewMetaCache returns a cache stored under dir. An empty dir keeps it in
// memory only.
func NewMetaCache(dir string) *MetaCache {
	return &MetaCache{dir: dir, mem: map[string]*models.GGUFMeta{}}
}

// MetaKey identifies one published file for the cache.
func MetaKey(source, repo string, f File) string {
	id := f.OID
	if id == "" {
		id = fmt.Sprintf("%s|%d", f.Filename, f.Size)
	}
	sum := sha256.Sum256([]byte(id))
	return filepath.Join(NormalizeSource(source), SafeRepoDir(repo), hex.EncodeToString(sum[:16])+".json")
}

// SafeRepoDir is a repository ID as a single directory name for an on-disk
// cache: slashes become "--" (as in huggingface.SafeModelID) and anything
// other than letters, digits, "-", "_" and "." becomes "_".
func SafeRepoDir(repo string) string {
	return strings.Map(func(r rune) rune {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '-', r == '_', r == '.':
			return r
		}
		return '_'
	}, strings.ReplaceAll(repo, "/", "--"))
}

// Get returns a cached description. A nil cache holds nothing.
func (c *MetaCache) Get(key string) (*models.GGUFMeta, bool) {
	if c == nil {
		return nil, false
	}
	c.mu.RLock()
	m, ok := c.mem[key]
	c.mu.RUnlock()
	if ok || c.dir == "" {
		return m, ok
	}
	data, err := os.ReadFile(filepath.Join(c.dir, key))
	if err != nil {
		return nil, false
	}
	m = &models.GGUFMeta{}
	if json.Unmarshal(data, m) != nil || m.NLayers == 0 {
		return nil, false
	}
	c.mu.Lock()
	c.mem[key] = m
	c.mu.Unlock()
	return m, true
}

// Put stores a description. A nil cache stores nothing.
func (c *MetaCache) Put(key string, m *models.GGUFMeta) {
	if c == nil {
		return
	}
	c.mu.Lock()
	c.mem[key] = m
	c.mu.Unlock()
	if c.dir == "" {
		return
	}
	data, err := json.Marshal(m)
	if err != nil {
		return
	}
	path := filepath.Join(c.dir, key)
	if os.MkdirAll(filepath.Dir(path), 0o755) != nil {
		return
	}
	_ = atomicfile.Write(path, data)
}
