package filehash

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func sum(s string) string {
	h := sha256.Sum256([]byte(s))
	return hex.EncodeToString(h[:])
}

// Hash reads a file once, saves the result, and a store opened again
// from disk answers without reading the file.
func TestHashIsSavedAndReused(t *testing.T) {
	dir := t.TempDir()
	file := filepath.Join(dir, "m.gguf")
	if err := os.WriteFile(file, []byte("model bytes"), 0o644); err != nil {
		t.Fatal(err)
	}
	storePath := filepath.Join(dir, "cache", "hashes.json")
	s := Open(storePath)
	var read int64
	got, err := s.Hash(context.Background(), file, func(done int64) { read = done })
	if err != nil {
		t.Fatal(err)
	}
	if got != sum("model bytes") {
		t.Errorf("hash = %s, want %s", got, sum("model bytes"))
	}
	if read != int64(len("model bytes")) {
		t.Errorf("progress reported %d bytes, want %d", read, len("model bytes"))
	}

	again := Open(storePath)
	if sha, ok := again.Lookup(file); !ok || sha != got {
		t.Errorf("reopened store Lookup = %q, %v; want the saved hash", sha, ok)
	}
}

// A file changed after its hash was recorded is not given the old hash.
func TestLookupIgnoresAChangedFile(t *testing.T) {
	dir := t.TempDir()
	file := filepath.Join(dir, "m.gguf")
	os.WriteFile(file, []byte("old"), 0o644)
	s := Open(filepath.Join(dir, "hashes.json"))
	if err := s.Record(file, sum("old")); err != nil {
		t.Fatal(err)
	}
	os.WriteFile(file, []byte("new"), 0o644)
	later := time.Now().Add(time.Minute)
	os.Chtimes(file, later, later)
	if _, ok := s.Lookup(file); ok {
		t.Fatal("Lookup returned the hash of the old contents")
	}
	got, err := s.Hash(context.Background(), file, nil)
	if err != nil || got != sum("new") {
		t.Errorf("Hash = %s, %v; want the hash of the new contents", got, err)
	}
}
