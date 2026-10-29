package huggingface

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/broadcast"
	"github.com/tmac1973/llama-toolchest/internal/filehash"
)

func sha(s string) string {
	h := sha256.Sum256([]byte(s))
	return hex.EncodeToString(h[:])
}

func updateFixture(t *testing.T, body string) (*Downloader, string, *filehash.Store) {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, body)
	}))
	t.Cleanup(srv.Close)
	dir := t.TempDir()
	d := NewDownloader(dir, dir, "")
	d.RegisterProvider("fake", Provider{URL: func(string, string) string { return srv.URL }})
	store := filehash.Open(filepath.Join(dir, "hashes.json"))
	d.SetHashStore(store)
	if err := os.WriteFile(filepath.Join(dir, "m.gguf"), []byte("old"), 0o644); err != nil {
		t.Fatal(err)
	}
	return d, dir, store
}

// A finished file is skipped, but one with a .part beside it is an
// update: it is downloaded again, replaces the old file, and its hash
// is recorded.
func TestPartBesideFinishedFileReplacesIt(t *testing.T) {
	d, dir, store := updateFixture(t, "new contents")
	dl := &download{bc: broadcast.New[DownloadStatus](1, 16)}
	final := filepath.Join(dir, "m.gguf")

	if _, err := d.downloadFile(context.Background(), "fake", "id", "org/m", "m.gguf", "m.gguf", dir, 0, 0, dl); err != nil {
		t.Fatal(err)
	}
	if got, _ := os.ReadFile(final); string(got) != "old" {
		t.Fatalf("file without a .part was downloaded again: %q", got)
	}

	os.WriteFile(final+".part", nil, 0o644)
	d.ExpectSHA256(final, sha("new contents"))
	if _, err := d.downloadFile(context.Background(), "fake", "id", "org/m", "m.gguf", "m.gguf", dir, 0, 0, dl); err != nil {
		t.Fatal(err)
	}
	if got, _ := os.ReadFile(final); string(got) != "new contents" {
		t.Errorf("file = %q, want the new contents", got)
	}
	if _, err := os.Stat(final + ".part"); !os.IsNotExist(err) {
		t.Error(".part file left behind")
	}
	if got, ok := store.Lookup(final); !ok || got != sha("new contents") {
		t.Errorf("recorded hash = %q, %v; want the new contents' hash", got, ok)
	}
}

// A download that does not match the repository's hash is discarded and
// the old file kept.
func TestUpdateWithWrongHashKeepsOldFile(t *testing.T) {
	d, dir, _ := updateFixture(t, "corrupted")
	dl := &download{bc: broadcast.New[DownloadStatus](1, 16)}
	final := filepath.Join(dir, "m.gguf")
	os.WriteFile(final+".part", nil, 0o644)
	d.ExpectSHA256(final, sha("new contents"))

	if _, err := d.downloadFile(context.Background(), "fake", "id", "org/m", "m.gguf", "m.gguf", dir, 0, 0, dl); err == nil {
		t.Fatal("downloadFile accepted a file with the wrong hash")
	}
	if got, _ := os.ReadFile(final); string(got) != "old" {
		t.Errorf("file = %q, want the old file kept", got)
	}
	if _, err := os.Stat(final + ".part"); !os.IsNotExist(err) {
		t.Error("the mismatched .part file was kept")
	}
}

// A resumed download's hash covers the part already on disk too.
func TestResumedDownloadHashCoversWholeFile(t *testing.T) {
	const whole = "0123456789"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Range", "bytes 4-9/10")
		w.WriteHeader(http.StatusPartialContent)
		io.WriteString(w, whole[4:])
	}))
	defer srv.Close()
	dir := t.TempDir()
	d := NewDownloader(dir, dir, "")
	d.RegisterProvider("fake", Provider{URL: func(string, string) string { return srv.URL }})
	store := filehash.Open(filepath.Join(dir, "hashes.json"))
	d.SetHashStore(store)
	os.WriteFile(filepath.Join(dir, "m.gguf.part"), []byte(whole[:4]), 0o644)
	dl := &download{bc: broadcast.New[DownloadStatus](1, 16)}

	if _, err := d.downloadFile(context.Background(), "fake", "id", "org/m", "m.gguf", "m.gguf", dir, 0, 0, dl); err != nil {
		t.Fatal(err)
	}
	if got, _ := store.Lookup(filepath.Join(dir, "m.gguf")); got != sha(whole) {
		t.Errorf("recorded hash = %s, want the whole file's %s", got, sha(whole))
	}
}
