package huggingface

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/broadcast"
)

// A finished file of the size the caller expects is skipped.
func TestFinishedFileOfExpectedSizeIsSkipped(t *testing.T) {
	d, dir, _ := updateFixture(t, "new contents")
	d.RegisterProvider("fake", Provider{URL: func(string, string) string {
		t.Error("a file of the expected size was downloaded again")
		return "http://127.0.0.1:0"
	}})
	dl := &download{bc: broadcast.New[DownloadStatus](1, 16)}

	n, err := d.downloadFile(context.Background(), "fake", "id", "org/m", "m.gguf", "m.gguf", dir, 0, 0, int64(len("old")), dl)
	if err != nil {
		t.Fatal(err)
	}
	if n != int64(len("old")) {
		t.Errorf("size = %d, want %d", n, len("old"))
	}
}

// A finished file whose size is not the expected size is downloaded
// again and replaced.
func TestFinishedFileOfWrongSizeIsReplaced(t *testing.T) {
	d, dir, store := updateFixture(t, "new contents")
	dl := &download{bc: broadcast.New[DownloadStatus](1, 16)}
	final := filepath.Join(dir, "m.gguf")

	n, err := d.downloadFile(context.Background(), "fake", "id", "org/m", "m.gguf", "m.gguf", dir, 0, 0, int64(len("new contents")), dl)
	if err != nil {
		t.Fatal(err)
	}
	if n != int64(len("new contents")) {
		t.Errorf("size = %d, want %d", n, len("new contents"))
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

// A download that ends at a size other than the expected one fails, and
// the file already on disk is kept.
func TestDownloadOfWrongSizeIsDiscarded(t *testing.T) {
	d, dir, _ := updateFixture(t, "short")
	dl := &download{bc: broadcast.New[DownloadStatus](1, 16)}
	final := filepath.Join(dir, "m.gguf")

	if _, err := d.downloadFile(context.Background(), "fake", "id", "org/m", "m.gguf", "m.gguf", dir, 0, 0, 100, dl); err == nil {
		t.Fatal("downloadFile accepted a file of the wrong size")
	}
	if got, _ := os.ReadFile(final); string(got) != "old" {
		t.Errorf("file = %q, want the old file kept", got)
	}
	if _, err := os.Stat(final + ".part"); !os.IsNotExist(err) {
		t.Error("the wrong-sized .part file was kept")
	}
}

// startAndWait runs StartTo and returns the size its callback reports, or
// -1 when the download fails.
func startAndWait(t *testing.T, d *Downloader, file, root string, expected int64) int64 {
	t.Helper()
	got := make(chan int64, 1)
	id, err := d.StartTo("fake", "org/m", file, root, expected, func(_ string, size int64) { got <- size })
	if err != nil {
		t.Fatal(err)
	}
	deadline := time.After(10 * time.Second)
	for {
		select {
		case size := <-got:
			return size
		case <-deadline:
			t.Fatal("download did not finish")
		case <-time.After(10 * time.Millisecond):
			if s, ok := d.Status(id); ok && (s.Status == "failed" || s.Status == "cancelled") {
				return -1
			}
		}
	}
}

// Through StartTo, a single file of the wrong size is replaced and the
// callback hears the new size.
func TestStartToReplacesWrongSizedFile(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, "new contents")
	}))
	defer srv.Close()
	root := t.TempDir()
	d := NewDownloader(root, root, "")
	d.RegisterProvider("fake", Provider{URL: func(string, string) string { return srv.URL }})
	path := FilePath(root, "org/m", "m.gguf")
	os.MkdirAll(filepath.Dir(path), 0o755)
	os.WriteFile(path, []byte("old"), 0o644)

	if size := startAndWait(t, d, "m.gguf", root, int64(len("new contents"))); size != int64(len("new contents")) {
		t.Errorf("callback size = %d, want %d", size, len("new contents"))
	}
	if got, _ := os.ReadFile(path); string(got) != "new contents" {
		t.Errorf("file = %q, want the new contents", got)
	}
}

// For shards the caller's size is the total of all of them, so no shard
// is checked against it: finished shards are kept as they are.
func TestShardsAreNotCheckedAgainstCombinedSize(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet {
			t.Errorf("finished shard %s was downloaded again", r.URL.Path)
		}
	}))
	defer srv.Close()
	root := t.TempDir()
	d := NewDownloader(root, root, "")
	d.RegisterProvider("fake", Provider{URL: func(_, file string) string { return srv.URL + "/" + file }})
	for _, f := range []string{"m-00001-of-00002.gguf", "m-00002-of-00002.gguf"} {
		path := FilePath(root, "org/m", f)
		os.MkdirAll(filepath.Dir(path), 0o755)
		os.WriteFile(path, []byte("shard"), 0o644)
	}

	if size := startAndWait(t, d, "m-00001-of-00002.gguf", root, 10); size != 10 {
		t.Errorf("callback size = %d, want 10", size)
	}
}

// StartToAt checks a single file's size in the same way as StartTo.
func TestStartToAtReplacesWrongSizedFile(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/abc123" {
			t.Errorf("path = %q, want the revision's URL", r.URL.Path)
		}
		io.WriteString(w, "new contents")
	}))
	defer srv.Close()
	root := t.TempDir()
	d := NewDownloader(root, root, "")
	d.RegisterProvider("fake", Provider{
		URL:         func(string, string) string { return srv.URL },
		RevisionURL: func(_, rev, _ string) string { return srv.URL + "/" + rev },
	})
	path := FilePath(root, "org/m", "m.gguf")
	os.MkdirAll(filepath.Dir(path), 0o755)
	os.WriteFile(path, []byte("old"), 0o644)

	got := make(chan int64, 1)
	if _, err := d.StartToAt("fake", "org/m", "abc123", "m.gguf", root, int64(len("new contents")),
		func(_ string, size int64) { got <- size }); err != nil {
		t.Fatal(err)
	}
	select {
	case size := <-got:
		if size != int64(len("new contents")) {
			t.Errorf("callback size = %d, want %d", size, len("new contents"))
		}
	case <-time.After(10 * time.Second):
		t.Fatal("download did not finish")
	}
	if got, _ := os.ReadFile(path); string(got) != "new contents" {
		t.Errorf("file = %q, want the new contents", got)
	}
}
