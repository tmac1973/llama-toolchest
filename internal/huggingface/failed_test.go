package huggingface

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// failFixture is a downloader whose "fake" source answers every request
// with handler.
func failFixture(t *testing.T, handler http.HandlerFunc) *Downloader {
	t.Helper()
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)
	dir := t.TempDir()
	d := NewDownloader(dir, dir, "")
	d.RegisterProvider("fake", Provider{URL: func(string, string) string { return srv.URL }})
	return d
}

// waitFailed waits for a download to fail and returns its status.
func waitFailed(t *testing.T, d *Downloader, id string) DownloadStatus {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		if s, ok := d.Status(id); ok && s.Status != "downloading" {
			if s.Status != "failed" {
				t.Fatalf("status = %q, want failed", s.Status)
			}
			return s
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("download did not end")
	return DownloadStatus{}
}

// expire does what the 30-second timer at the end of run does.
func expire(d *Downloader, id string) {
	d.mu.Lock()
	dl := d.active[id]
	delete(d.active, id)
	d.mu.Unlock()
	if dl != nil {
		<-dl.done
	}
}

// A download of the wrong size stays listed as failed, with its error,
// after it leaves the active map, and is gone once dismissed.
func TestSizeMismatchStaysVisibleUntilDismissed(t *testing.T) {
	d := failFixture(t, func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, "short")
	})
	id, err := d.Start(context.Background(), "fake", "org/m", "m.gguf", 7)
	if err != nil {
		t.Fatal(err)
	}
	waitFailed(t, d, id)
	expire(d, id)

	s, ok := d.Status(id)
	if !ok || s.Status != "failed" || !strings.Contains(s.Error, "7 bytes were expected") {
		t.Fatalf("status after expiry = %+v, %v; want failed with the size error", s, ok)
	}
	failed := d.ListFailed()
	if len(failed) != 1 || failed[0].ID != id || failed[0].Error != s.Error {
		t.Fatalf("ListFailed = %+v", failed)
	}
	if len(d.ListActive()) != 0 {
		t.Error("a failed download is listed as active")
	}

	if !d.DismissFailed(id) {
		t.Error("DismissFailed found nothing to dismiss")
	}
	if _, ok := d.Status(id); ok {
		t.Error("status still held after dismiss")
	}
	if got := d.ListFailed(); len(got) != 0 {
		t.Errorf("ListFailed after dismiss = %+v", got)
	}
	if d.DismissFailed(id) {
		t.Error("a second dismiss found something")
	}
}

// An HTTP error is kept the same way, and starting the download again
// clears it.
func TestHTTPErrorIsKeptUntilRestart(t *testing.T) {
	var fail atomic.Bool
	fail.Store(true)
	d := failFixture(t, func(w http.ResponseWriter, r *http.Request) {
		if fail.Load() {
			http.Error(w, "busy", http.StatusServiceUnavailable)
			return
		}
		io.WriteString(w, "contents")
	})
	id, err := d.Start(context.Background(), "fake", "org/m", "m.gguf", 0)
	if err != nil {
		t.Fatal(err)
	}
	s := waitFailed(t, d, id)
	if s.Error == "" {
		t.Error("failed status has no error")
	}
	expire(d, id)
	if len(d.ListFailed()) != 1 {
		t.Fatalf("ListFailed = %+v, want the HTTP error", d.ListFailed())
	}

	fail.Store(false)
	if _, err := d.Start(context.Background(), "fake", "org/m", "m.gguf", 0); err != nil {
		t.Fatal(err)
	}
	if got := d.ListFailed(); len(got) != 0 {
		t.Errorf("ListFailed after a new start = %+v", got)
	}
	d.mu.Lock()
	dl := d.active[id]
	d.mu.Unlock()
	<-dl.done
	if s, _ := d.Status(id); s.Status != "complete" {
		t.Errorf("second attempt status = %q, want complete", s.Status)
	}
}
