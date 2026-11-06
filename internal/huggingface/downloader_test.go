package huggingface

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/broadcast"
	"github.com/tmac1973/llama-toolchest/internal/testutil"

	"github.com/tmac1973/llama-toolchest/internal/modelsource"
)

// seedActive injects a download entry with the given last status, simulating
// the 30s late-subscriber grace window during which terminal downloads stay
// in the active map.
func seedActive(d *Downloader, downloadID, status string) {
	dl := &download{
		cancel: func() {},
		done:   make(chan struct{}),
		bc:     broadcast.New[DownloadStatus](1, 16),
	}
	close(dl.done) // a seeded terminal entry represents a finished goroutine
	dl.broadcast(DownloadStatus{ID: downloadID, Status: status})
	d.mu.Lock()
	d.active[downloadID] = dl
	d.mu.Unlock()
}

func TestStartRefusesWhileDownloading(t *testing.T) {
	d := NewDownloader(t.TempDir(), t.TempDir(), "")
	seedActive(d, "org--Repo-GGUF--file", "downloading")

	if _, err := d.Start(context.Background(), modelsource.SourceHuggingFace, "org/Repo-GGUF", "file.gguf", 0); err == nil {
		t.Fatal("Start should refuse while the same download is in progress")
	}
}

// TestStartReplacesTerminalEntry covers pause→resume inside the grace window:
// a cancelled entry lingers in the active map for 30s so late subscribers can
// read its final status, but that must not block an immediate resume.
func TestStartReplacesTerminalEntry(t *testing.T) {
	srv := stallingServer(t, 1<<20)
	for _, status := range []string{"cancelled", "failed", "complete"} {
		d := NewDownloader(t.TempDir(), t.TempDir(), "")
		// The run goroutine downloads from the local server, not from
		// huggingface.co: the test must not depend on the network.
		d.RegisterProvider(modelsource.SourceHuggingFace, Provider{URL: func(string, string) string { return srv.URL }})
		seedActive(d, "org--Repo-GGUF--file", status)

		id, err := d.Start(context.Background(), modelsource.SourceHuggingFace, "org/Repo-GGUF", "file.gguf", 0)
		if err != nil {
			t.Fatalf("Start after %q entry: %v", status, err)
		}
		// Cancel joins the goroutine: when it returns, run() has fully
		// exited (no more filesystem writes racing TempDir cleanup — the
		// flake this test used to have) and the entry's status is
		// terminal.
		d.Cancel(id)
		d.mu.Lock()
		dl := d.active[id]
		d.mu.Unlock()
		if dl != nil {
			select {
			case <-dl.done:
			default:
				t.Fatalf("Cancel returned before the %q run goroutine exited", status)
			}
			if s := dl.last(); s.Status == "downloading" {
				t.Fatalf("post-Cancel status still %q", s.Status)
			}
		}
	}
}

// StartTo puts a non-GGUF file under its own root, reports it to its own
// callback, and never calls the downloader's LLM completion callback.
func TestStartToOwnRootAndCallback(t *testing.T) {
	const body = "tokenizer contents"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(body))
	}))
	defer srv.Close()

	models, gen := t.TempDir(), t.TempDir()
	d := NewDownloader(t.TempDir(), models, "")
	d.RegisterProvider("fake", Provider{URL: func(string, string) string { return srv.URL }})
	d.SetOnComplete(func(string, string, string, string, int64) {
		t.Error("LLM completion callback called for a StartTo download")
	})

	done := make(chan string, 1)
	id, err := d.StartTo("fake", "inclusionAI/Ming-Image-0.1-Design", "mllm/tokenizer.json", gen, int64(len(body)),
		func(path string, size int64) { done <- path })
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(id, "gen--") {
		t.Errorf("download ID %q lacks the gen-- prefix", id)
	}
	select {
	case path := <-done:
		want := FilePath(gen, "inclusionAI/Ming-Image-0.1-Design", "mllm/tokenizer.json")
		if path != want {
			t.Errorf("path = %q, want %q", path, want)
		}
		if got, _ := os.ReadFile(path); string(got) != body {
			t.Errorf("file holds %q", got)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("StartTo never finished")
	}
	if entries, _ := os.ReadDir(models); len(entries) != 0 {
		t.Errorf("StartTo wrote into the LLM models dir: %v", entries)
	}
	if st, ok := d.Status(id); !ok || st.Status != "complete" {
		t.Errorf("Status = %+v, %v; want complete", st, ok)
	}
}

// Two generation sets can share a file. When the second asks for it while
// the first's download is running, both must hear when it is done.
func TestStartToSharedFileCallsEveryCallback(t *testing.T) {
	const body = "vae contents"
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet {
			<-release
		}
		w.Write([]byte(body))
	}))
	defer srv.Close()

	gen := t.TempDir()
	d := NewDownloader(t.TempDir(), t.TempDir(), "")
	d.RegisterProvider("fake", Provider{URL: func(string, string) string { return srv.URL }})

	first, second := make(chan string, 1), make(chan string, 1)
	id1, err := d.StartTo("fake", "Comfy-Org/z_image_turbo", "split_files/vae/ae.safetensors", gen, int64(len(body)),
		func(path string, _ int64) { first <- path })
	if err != nil {
		t.Fatal(err)
	}
	id2, err := d.StartTo("fake", "Comfy-Org/z_image_turbo", "split_files/vae/ae.safetensors", gen, int64(len(body)),
		func(path string, _ int64) { second <- path })
	if err == nil || id2 != id1 {
		t.Fatalf("second StartTo = %q, %v; want the running download %q and an error", id2, err, id1)
	}
	close(release)
	for name, ch := range map[string]chan string{"first": first, "second": second} {
		select {
		case <-ch:
		case <-time.After(5 * time.Second):
			t.Fatalf("%s callback never called", name)
		}
	}
}

// stallingServer answers a GET with a body of size bytes, of which it
// sends 1 KiB every 50 ms and never the last KiB: a download that stays
// in progress until it is cancelled. A HEAD gets the size at once.
func stallingServer(t *testing.T, size int) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", strconv.Itoa(size))
		if r.Method == http.MethodHead {
			return
		}
		// A trickle, not one write and a stop: the downloader reports
		// progress only after a read returns, at most every 500 ms.
		for sent := 0; sent+1024 < size; sent += 1024 {
			w.Write(make([]byte, 1024))
			w.(http.Flusher).Flush()
			select {
			case <-r.Context().Done():
				return
			case <-time.After(50 * time.Millisecond):
			}
		}
		<-r.Context().Done()
	}))
	// A failed test can leave a download running: drop its connection, or
	// Close waits for the handler for ever.
	t.Cleanup(func() {
		srv.CloseClientConnections()
		srv.Close()
	})
	return srv
}

func TestCancelUnknownDownload(t *testing.T) {
	d := NewDownloader(t.TempDir(), t.TempDir(), "")
	if err := d.Cancel("org--none--file"); err == nil {
		t.Error("Cancel of an unknown download: no error")
	}
}

// A download cancelled while bytes are still coming stops, reports
// "cancelled", keeps its .part file for a resume, and no longer counts
// toward PendingBytes.
func TestCancelMidTransfer(t *testing.T) {
	const size = 1 << 20
	srv := stallingServer(t, size)
	gen := t.TempDir()
	d := NewDownloader(t.TempDir(), t.TempDir(), "")
	d.RegisterProvider("fake", Provider{URL: func(string, string) string { return srv.URL }})

	id, err := d.StartTo("fake", "org/repo", "big.safetensors", gen, size, func(string, int64) {
		t.Error("onDone called for a cancelled download")
	})
	if err != nil {
		t.Fatal(err)
	}
	testutil.Eventually(t, func() bool {
		st, _ := d.Status(id)
		return st.Status == "downloading" && st.BytesDownloaded > 0
	}, "download never started")
	if p := d.PendingBytes(); p <= 0 || p > size {
		t.Errorf("PendingBytes mid-transfer = %d, want 1..%d", p, size)
	}

	if err := d.Cancel(id); err != nil {
		t.Fatal(err)
	}
	if st, _ := d.Status(id); st.Status != "cancelled" {
		t.Errorf("status after Cancel = %q, want cancelled", st.Status)
	}
	if p := d.PendingBytes(); p != 0 {
		t.Errorf("PendingBytes after Cancel = %d, want 0", p)
	}
	part := FilePath(gen, "org/repo", "big.safetensors") + ".part"
	if _, err := os.Stat(part); err != nil {
		t.Errorf("the .part file is gone after a cancel: %v", err)
	}
}

// FreeBytes reads the models folder's file system; AvailableForDownload
// takes the safety margin off it.
func TestFreeBytes(t *testing.T) {
	d := NewDownloader(t.TempDir(), t.TempDir(), "")
	free := d.FreeBytes()
	if free <= 0 {
		t.Fatalf("FreeBytes = %d, want > 0 on a real file system", free)
	}
	if avail := d.AvailableForDownload(); avail < 0 || avail > free {
		t.Errorf("AvailableForDownload = %d, want 0..%d", avail, free)
	}
}
