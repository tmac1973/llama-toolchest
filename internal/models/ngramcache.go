package models

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
)

// The N-gram Cache speculative method learns which tokens follow which
// n-grams. From b11758 llama-server keeps what it learned in the file
// named by --lookup-cache-dynamic: read when the model loads, written when
// it unloads. Toolchest gives each model that uses the method its own file
// under NgramCacheDir.

// NgramCacheDir is the folder of the learned n-gram files under the data
// directory.
func NgramCacheDir(dataDir string) string {
	return filepath.Join(dataDir, "cache", "ngram")
}

// NgramCachePath is model id's learned n-gram file in dir.
func NgramCachePath(dir, id string) string {
	return filepath.Join(dir, id+".bin")
}

// usesNgramCacheFile reports whether a model's config keeps a learned
// n-gram file on target t.
func usesNgramCacheFile(cfg *ModelConfig, t Target) bool {
	return cfg != nil && cfg.SpecAssist == "ngram-cache" && t.NgramCacheDir != "" &&
		t.NgramCacheFileOption() && !strings.Contains(cfg.ExtraFlags, "lookup-cache-dynamic")
}

// ngramCacheParam is the lookup-cache-dynamic value for a model, or ""
// when it keeps no file (another method, a benchmark preset, an older
// build, or a path given in its extra flags).
func ngramCacheParam(id string, cfg *ModelConfig, t Target) string {
	if !usesNgramCacheFile(cfg, t) {
		return ""
	}
	return NgramCachePath(t.NgramCacheDir, id)
}

// ngramRecordHead is one record's n-gram (four token IDs) and its count
// of following tokens; each following token is a token ID and a count.
// All are little-endian int32, as common_ngram_cache_save writes them.
const (
	ngramKeyBytes   = 4 * 4
	ngramCountBytes = 4
	ngramEntryBytes = 4 + 4
)

// CheckNgramCacheFile reads a learned n-gram file the way llama.cpp's
// common_ngram_cache_load does, and returns an error where it would stop
// with an assertion: a record cut short (a save that was interrupted), or
// a count that is not positive. A missing or empty file is fine.
func CheckNgramCacheFile(path string) error {
	f, err := os.Open(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	defer f.Close()
	r := bufio.NewReaderSize(f, 1<<20)
	var offset int64
	key := make([]byte, ngramKeyBytes)
	four := make([]byte, 4)
	for {
		n, err := io.ReadFull(r, key)
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return fmt.Errorf("record at byte %d is cut short (%d of %d bytes)", offset, n, ngramKeyBytes)
		}
		offset += ngramKeyBytes
		if _, err := io.ReadFull(r, four); err != nil {
			return fmt.Errorf("record at byte %d is cut short", offset)
		}
		count := int32(binary.LittleEndian.Uint32(four))
		offset += ngramCountBytes
		if count <= 0 {
			return fmt.Errorf("record at byte %d has %d following tokens", offset-ngramCountBytes, count)
		}
		for i := int32(0); i < count; i++ {
			entry := make([]byte, ngramEntryBytes)
			if _, err := io.ReadFull(r, entry); err != nil {
				return fmt.Errorf("record at byte %d is cut short", offset)
			}
			if c := int32(binary.LittleEndian.Uint32(entry[4:])); c <= 0 {
				return fmt.Errorf("token at byte %d has count %d", offset, c)
			}
			offset += ngramEntryBytes
		}
	}
}

// PrepareNgramCacheFile makes the folder for a learned n-gram file, and
// deletes the file when it is broken: llama-server stops with an
// assertion on every load of a broken file, so the model could never load
// again. It starts with an empty cache instead and learns again.
func PrepareNgramCacheFile(path string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	if err := CheckNgramCacheFile(path); err != nil {
		slog.Warn("deleting a broken learned n-gram file; the model starts with an empty cache",
			"file", path, "problem", err)
		if rmErr := os.Remove(path); rmErr != nil && !errors.Is(rmErr, os.ErrNotExist) {
			return rmErr
		}
	}
	return nil
}

// prepareNgramCaches prepares the file of every enabled model that keeps
// one, when the user's preset is written (each router start, and each
// config change).
func prepareNgramCaches(t Target, models []*Model, configs map[string]*ModelConfig) {
	for _, m := range models {
		cfg := configs[m.ID]
		if cfg == nil || !cfg.Enabled {
			continue
		}
		if path := ngramCacheParam(m.ID, cfg, t); path != "" {
			if err := PrepareNgramCacheFile(path); err != nil {
				slog.Warn("could not prepare the learned n-gram file", "model", m.ID, "error", err)
			}
		}
	}
}

// NgramCacheFile returns a model's learned n-gram file on the user's
// preset, or "" when it keeps none.
func (r *Registry) NgramCacheFile(id string, cfg *ModelConfig, t Target) string {
	t.NgramCacheDir = NgramCacheDir(r.dataDir)
	return ngramCacheParam(id, cfg, t)
}

// RemoveNgramCacheFile deletes model id's learned n-gram file, if any. It
// is not an error when there is none.
func (r *Registry) RemoveNgramCacheFile(id string) error {
	err := os.Remove(NgramCachePath(NgramCacheDir(r.dataDir), id))
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	return err
}
