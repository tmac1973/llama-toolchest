package huggingface

import (
	"context"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/broadcast"
)

// seedActive injects a download entry with the given last status, simulating
// the 30s late-subscriber grace window during which terminal downloads stay
// in the active map.
func seedActive(d *Downloader, downloadID, status string) {
	dl := &download{
		cancel: func() {},
		bc:     broadcast.New[DownloadStatus](1, 16),
	}
	dl.broadcast(DownloadStatus{ID: downloadID, Status: status})
	d.mu.Lock()
	d.active[downloadID] = dl
	d.mu.Unlock()
}

func TestStartRefusesWhileDownloading(t *testing.T) {
	d := NewDownloader(t.TempDir(), t.TempDir(), "")
	seedActive(d, "org--Repo-GGUF--file", "downloading")

	if _, err := d.Start(context.Background(), "org/Repo-GGUF", "file.gguf", 0); err == nil {
		t.Fatal("Start should refuse while the same download is in progress")
	}
}

// TestStartReplacesTerminalEntry covers pause→resume inside the grace window:
// a cancelled entry lingers in the active map for 30s so late subscribers can
// read its final status, but that must not block an immediate resume.
func TestStartReplacesTerminalEntry(t *testing.T) {
	for _, status := range []string{"cancelled", "failed", "complete"} {
		d := NewDownloader(t.TempDir(), t.TempDir(), "")
		seedActive(d, "org--Repo-GGUF--file", status)

		id, err := d.Start(context.Background(), "org/Repo-GGUF", "file.gguf", 0)
		if err != nil {
			t.Fatalf("Start after %q entry: %v", status, err)
		}
		d.Cancel(id) // stop the spawned goroutine; we only care about admission
	}
}
