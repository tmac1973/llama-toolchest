package api

import (
	"context"
	"errors"
	"fmt"
	"os"
	"sync"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/generation"
	"github.com/tmac1973/llama-toolchest/internal/huggingface"
	"github.com/tmac1973/llama-toolchest/internal/modelsource"
)

// hfResolveTimeout bounds the Hugging Face requests that one add or edit
// of a custom set makes to pin and size its linked files.
const hfResolveTimeout = 20 * time.Second

// hfRepoCacheTTL is how long a repository's file list is kept.
const hfRepoCacheTTL = 10 * time.Minute

// hfRepoAPI is the part of the Hugging Face client that custom set
// downloads use. Tests replace it.
type hfRepoAPI interface {
	RepoCommit(ctx context.Context, modelID, rev string) (string, error)
	RepoFilesAt(ctx context.Context, modelID, rev string) ([]modelsource.RepoFile, error)
}

func (s *Server) hfRepos() hfRepoAPI {
	if s.hfRepoFake != nil {
		return s.hfRepoFake
	}
	return s.hfClient
}

// hfRepoCache keeps repository file lists by "repo@rev". A list at a
// commit never changes; one at a branch may be up to hfRepoCacheTTL old,
// which is fine for browsing, and an add always pins a commit again.
type hfRepoCache struct {
	mu      sync.Mutex
	entries map[string]hfRepoCacheEntry
}

type hfRepoCacheEntry struct {
	files []modelsource.RepoFile
	at    time.Time
}

// repoFiles returns the files of repo at rev, from the cache when the
// list is recent enough.
func (s *Server) repoFiles(ctx context.Context, repo, rev string) ([]modelsource.RepoFile, error) {
	key := repo + "@" + rev
	c := &s.hfRepoCache
	c.mu.Lock()
	if e, ok := c.entries[key]; ok && time.Since(e.at) < hfRepoCacheTTL {
		c.mu.Unlock()
		return e.files, nil
	}
	c.mu.Unlock()
	files, err := s.hfRepos().RepoFilesAt(ctx, repo, rev)
	if err != nil {
		return nil, err
	}
	c.mu.Lock()
	if c.entries == nil {
		c.entries = map[string]hfRepoCacheEntry{}
	}
	c.entries[key] = hfRepoCacheEntry{files: files, at: time.Now()}
	c.mu.Unlock()
	return files, nil
}

// resolveHFComponents pins each linked component (one with a Repo) to a
// commit, gives it the file's size, and takes a file already at its
// download path as present. Components without a Repo are returned as
// they are. known are the components of the set an edit changes: one
// with the same repo, file and pinned commit that is still on disk is
// taken as it is, without asking Hugging Face.
func (s *Server) resolveHFComponents(ctx context.Context, comps, known []generation.Component) ([]generation.Component, error) {
	ctx, cancel := context.WithTimeout(ctx, hfResolveTimeout)
	defer cancel()
	root := generation.ModelsDir(s.cfg)
	out := make([]generation.Component, len(comps))
	copy(out, comps)
	for i := range out {
		c := &out[i]
		if c.Repo == "" {
			continue
		}
		if k, ok := knownFile(known, *c); ok {
			*c = k
			continue
		}
		rev := c.Revision
		if rev == "" {
			rev = "main"
		}
		commit, err := s.hfRepos().RepoCommit(ctx, c.Repo, rev)
		if err != nil {
			return nil, hfResolveError(ctx, c.Slot, err)
		}
		files, err := s.repoFiles(ctx, c.Repo, commit)
		if err != nil {
			return nil, hfResolveError(ctx, c.Slot, err)
		}
		size := int64(-1)
		for _, f := range files {
			if f.Path == c.File {
				size = f.Size
				break
			}
		}
		if size < 0 {
			return nil, fmt.Errorf("%s: %s is not in %s at %s", c.Slot, c.File, c.Repo, rev)
		}
		c.Revision, c.Bytes, c.Path = commit, size, ""
		path := huggingface.FilePath(root, c.Repo, c.File)
		if info, err := os.Stat(path); err == nil && info.Mode().IsRegular() && info.Size() == size {
			c.Path = path
		}
	}
	return out, nil
}

// knownFile finds c among known: the same slot, repo, file and pinned
// commit, with its file still on disk at the recorded size.
func knownFile(known []generation.Component, c generation.Component) (generation.Component, bool) {
	for _, k := range known {
		if k.Slot != c.Slot || k.Repo != c.Repo || k.File != c.File || k.Revision != c.Revision || k.Path == "" {
			continue
		}
		if info, err := os.Stat(k.Path); err == nil && info.Mode().IsRegular() && info.Size() == k.Bytes {
			k.Error = ""
			return k, true
		}
	}
	return generation.Component{}, false
}

// hfResolveError names the slot, and says so plainly when Hugging Face
// was too slow.
func hfResolveError(ctx context.Context, slot string, err error) error {
	if errors.Is(ctx.Err(), context.DeadlineExceeded) || errors.Is(err, context.DeadlineExceeded) {
		return fmt.Errorf("%s: Hugging Face did not answer in time; try again", slot)
	}
	return fmt.Errorf("%s: %w", slot, err)
}

// downloadBytes is the size of the linked files that are not on disk yet.
func downloadBytes(comps []generation.Component) int64 {
	var n int64
	for _, c := range comps {
		if c.Repo != "" && c.Path == "" {
			n += c.Bytes
		}
	}
	return n
}

// checkDownloadSpace refuses files that do not fit in the space a new
// download may use. Files already on disk do not count. Its error is
// sent with http.StatusInsufficientStorage.
func (s *Server) checkDownloadSpace(comps []generation.Component) error {
	need := downloadBytes(comps)
	avail := s.downloader.AvailableForDownload()
	if need == 0 || avail < 0 || need <= avail {
		return nil
	}
	return fmt.Errorf("The files to download need %.1f GB, and %.1f GB is free in the models folder.",
		float64(need)/1e9, float64(avail)/1e9)
}
