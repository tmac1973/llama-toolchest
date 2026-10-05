package huggingface

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/broadcast"
)

// ModelScope answers a resumed (ranged) request with 200 plus
// Content-Range instead of 206. The downloader must append that body to
// the partial file, not treat it as the whole file and keep only the tail.
func TestResumeAcceptsRangedBodyLabelled200(t *testing.T) {
	const whole = "0123456789"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Range") != "bytes=4-" {
			t.Errorf("Range = %q, want bytes=4-", r.Header.Get("Range"))
		}
		w.Header().Set("Content-Range", fmt.Sprintf("bytes 4-%d/%d", len(whole)-1, len(whole)))
		w.WriteHeader(http.StatusOK)
		io.WriteString(w, whole[4:])
	}))
	defer srv.Close()

	dir := t.TempDir()
	d := NewDownloader(dir, dir, "")
	d.RegisterProvider("quirky", Provider{URL: func(string, string) string { return srv.URL }})
	if err := os.WriteFile(filepath.Join(dir, "m.gguf.part"), []byte(whole[:4]), 0o644); err != nil {
		t.Fatal(err)
	}
	dl := &download{bc: broadcast.New[DownloadStatus](1, 16)}

	if _, err := d.downloadFile(context.Background(), "quirky", "id", "org/m", "m.gguf", "m.gguf", dir, 0, 0, dl); err != nil {
		t.Fatalf("downloadFile: %v", err)
	}
	got, err := os.ReadFile(filepath.Join(dir, "m.gguf"))
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != whole {
		t.Errorf("file = %q, want %q", got, whole)
	}
}
