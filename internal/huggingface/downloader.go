package huggingface

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/broadcast"

	"github.com/tmac1973/llama-toolchest/internal/modelsource"
)

// DownloadStatus tracks progress of a model download.
type DownloadStatus struct {
	ID              string `json:"id"`
	ModelID         string `json:"model_id"`
	Filename        string `json:"filename"`
	BytesDownloaded int64  `json:"bytes_downloaded"`
	TotalBytes      int64  `json:"total_bytes"`
	SpeedBPS        int64  `json:"speed_bps"`
	Status          string `json:"status"` // "downloading", "complete", "failed", "cancelled"
	Error           string `json:"error,omitempty"`
}

type download struct {
	cancel context.CancelFunc

	// done is closed when the run goroutine has fully exited. Cancel and
	// Start's terminal-entry eviction wait on it (bounded), so nothing
	// races a goroutine that hasn't noticed its cancelled context yet —
	// neither a resume opening the same .part file, nor a test's TempDir
	// cleanup racing run's MkdirAll.
	done chan struct{}

	// Fan-out to multiple subscribers. History size 1 means the current
	// state is retained: replayed to new subscribers (so they see where we
	// are immediately) and queryable via last() for ListActive/PendingBytes.
	bc *broadcast.Broadcaster[DownloadStatus]
}

func (dl *download) broadcast(status DownloadStatus) {
	dl.bc.Broadcast(status)
}

func (dl *download) subscribe() chan DownloadStatus {
	return dl.bc.Subscribe()
}

func (dl *download) unsubscribe(ch chan DownloadStatus) {
	dl.bc.Unsubscribe(ch)
}

// last returns the most recent status (zero value before the first broadcast).
func (dl *download) last() DownloadStatus {
	s, _ := dl.bc.Last()
	return s
}

// CompletionFunc is called when a download finishes successfully.
type CompletionFunc func(source, downloadID, modelID, filename string, sizeBytes int64)

// Downloader manages resumable GGUF downloads from HuggingFace.
// Provider supplies the per-source parts of a download: where a file
// lives and how to authenticate for it. The rest of the downloader —
// resume, shard handling, disk accounting, progress — is the same
// whichever host the bytes come from.
type Provider struct {
	// URL returns the download URL for one file in a repository.
	URL func(modelID, filename string) string
	// Token is the bearer token for this source, empty for anonymous.
	Token string
}

type Downloader struct {
	dataDir    string
	modelsDir  string
	token      string
	onComplete CompletionFunc

	// providers is keyed by modelsource source id. A download names its
	// source; an unknown or empty one falls back to HuggingFace, which is
	// what every download recorded before this existed came from.
	providers map[string]Provider

	mu     sync.Mutex
	active map[string]*download
}

func NewDownloader(dataDir, modelsDir, token string) *Downloader {
	d := &Downloader{
		dataDir:   dataDir,
		modelsDir: modelsDir,
		token:     token,
		active:    make(map[string]*download),
		providers: make(map[string]Provider),
	}
	d.providers[modelsource.SourceHuggingFace] = Provider{
		URL:   func(modelID, filename string) string { return hfDownloadURL(modelID, filename) },
		Token: token,
	}
	return d
}

// RegisterProvider adds or replaces the provider for a source.
func (d *Downloader) RegisterProvider(source string, p Provider) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.providers[source] = p
}

// provider returns the provider for a source, falling back to
// HuggingFace so an empty source keeps behaving as it always did.
func (d *Downloader) provider(source string) Provider {
	d.mu.Lock()
	defer d.mu.Unlock()
	if p, ok := d.providers[source]; ok && p.URL != nil {
		return p
	}
	return d.providers[modelsource.SourceHuggingFace]
}

func hfDownloadURL(modelID, filename string) string {
	return fmt.Sprintf("https://huggingface.co/%s/resolve/main/%s", modelID, filename)
}

// SetOnComplete registers a callback invoked when a download finishes.
func (d *Downloader) SetOnComplete(fn CompletionFunc) {
	d.onComplete = fn
}

// Start begins a download in the background. Returns the download ID.
// expectedBytes is a caller-provided size hint (from the HF tree API) so the
// in-flight reservation in AvailableForDownload is accurate from the moment
// Start returns, before the download goroutine has done its own HEAD. Pass 0
// when the size is unknown.
func (d *Downloader) Start(ctx context.Context, source, modelID, filename string, expectedBytes int64) (string, error) {
	// Create a stable download ID — replace all slashes to keep it URL-safe
	safeName := SafeModelID(modelID)
	safeFilename := SafeFileID(filename)
	downloadID := fmt.Sprintf("%s--%s", safeName, safeFilename)

	d.mu.Lock()
	if existing, exists := d.active[downloadID]; exists {
		if s := existing.last(); s.Status == "downloading" {
			d.mu.Unlock()
			return downloadID, fmt.Errorf("download already in progress")
		}
		// Terminal entry still inside its 30s late-subscriber grace window —
		// evict it so a paused/failed download can be resumed immediately.
		delete(d.active, downloadID)
		d.mu.Unlock()
		// Wait (bounded) for the old goroutine to fully exit before
		// spawning a replacement: its terminal status means it's at the
		// end of run(), but until done closes it may still hold the
		// .part file the new goroutine is about to open. (nil guard:
		// entries deserialized or constructed without a goroutine.)
		if existing.done != nil {
			select {
			case <-existing.done:
			case <-time.After(5 * time.Second):
			}
		}
		d.mu.Lock()
	}

	dlCtx, cancel := context.WithCancel(context.Background())
	dl := &download{
		cancel: cancel,
		done:   make(chan struct{}),
		bc:     broadcast.New[DownloadStatus](1, 16),
	}
	// Seed the status so PendingBytes counts this download immediately,
	// before run() reports its first progress tick.
	dl.broadcast(DownloadStatus{
		ID:         downloadID,
		ModelID:    modelID,
		Filename:   filename,
		TotalBytes: expectedBytes,
		Status:     "downloading",
	})
	d.active[downloadID] = dl
	d.mu.Unlock()

	go d.run(dlCtx, source, downloadID, modelID, filename, dl)

	return downloadID, nil
}

// Subscribe returns a channel that receives progress updates for a download.
// The current status is sent immediately. Call Unsubscribe when done.
func (d *Downloader) Subscribe(downloadID string) (chan DownloadStatus, bool) {
	d.mu.Lock()
	dl, ok := d.active[downloadID]
	d.mu.Unlock()
	if !ok {
		return nil, false
	}
	return dl.subscribe(), true
}

// Unsubscribe removes a progress subscriber.
func (d *Downloader) Unsubscribe(downloadID string, ch chan DownloadStatus) {
	d.mu.Lock()
	dl, ok := d.active[downloadID]
	d.mu.Unlock()
	if ok {
		dl.unsubscribe(ch)
	}
}

// ListActive returns the latest status of all in-progress downloads.
func (d *Downloader) ListActive() []DownloadStatus {
	d.mu.Lock()
	defer d.mu.Unlock()
	var out []DownloadStatus
	for _, dl := range d.active {
		if s := dl.last(); s.Status == "downloading" {
			out = append(out, s)
		}
	}
	return out
}

// PendingBytes returns the sum of remaining bytes across all active downloads.
// Used by AvailableForDownload to reserve space for in-flight downloads so a
// second download started while a first is still running sees the correct
// post-completion free space, not the current free space.
func (d *Downloader) PendingBytes() int64 {
	d.mu.Lock()
	defer d.mu.Unlock()
	var pending int64
	for _, dl := range d.active {
		s := dl.last()
		if s.Status != "downloading" {
			continue
		}
		remaining := s.TotalBytes - s.BytesDownloaded
		if remaining > 0 {
			pending += remaining
		}
	}
	return pending
}

// FreeBytes returns the free space on the filesystem hosting modelsDir.
func (d *Downloader) FreeBytes() int64 {
	return freeBytesAt(d.modelsDir)
}

// AvailableForDownload returns how many bytes a new download is allowed to
// consume: free disk minus the safety margin minus the bytes already reserved
// by in-flight downloads. Returns -1 when free space can't be determined, so
// callers can distinguish "unknown" from "actually full" — we don't want a
// failed statfs to falsely ghost every download button. A non-negative result
// is clamped to zero (genuinely no room).
func (d *Downloader) AvailableForDownload() int64 {
	free := d.FreeBytes()
	if free < 0 {
		return -1
	}
	avail := free - DiskSafetyMarginBytes - d.PendingBytes()
	if avail < 0 {
		return 0
	}
	return avail
}

// Cancel stops an active download.
func (d *Downloader) Cancel(downloadID string) error {
	d.mu.Lock()
	dl, ok := d.active[downloadID]
	d.mu.Unlock()

	if !ok {
		return fmt.Errorf("no active download: %s", downloadID)
	}

	dl.cancel()
	// Join the goroutine (bounded): when Cancel returns, the download has
	// actually stopped touching the filesystem — not merely been asked
	// to. The chunk loop and the HTTP request both watch the context, so
	// exit is prompt; the timeout only guards a wedged transfer.
	if dl.done != nil {
		select {
		case <-dl.done:
		case <-time.After(5 * time.Second):
		}
	}
	return nil
}

func (d *Downloader) run(ctx context.Context, source, downloadID, modelID, filename string, dl *download) {
	defer close(dl.done)
	// Cancelled before we ever ran (pause immediately after resume, or a
	// test's admission-only Start): report the terminal status and exit
	// without side effects — in particular before the MkdirAll below,
	// which otherwise races whoever is cleaning the directory up.
	if ctx.Err() != nil {
		dl.broadcast(DownloadStatus{ID: downloadID, ModelID: modelID, Filename: filename, Status: "cancelled"})
		go func() {
			time.Sleep(30 * time.Second)
			d.mu.Lock()
			delete(d.active, downloadID)
			d.mu.Unlock()
		}()
		return
	}
	defer func() {
		// Keep in active map briefly so late subscribers can see final status
		go func() {
			time.Sleep(30 * time.Second)
			d.mu.Lock()
			delete(d.active, downloadID)
			d.mu.Unlock()
		}()
	}()

	sendProgress := func(status DownloadStatus) {
		dl.broadcast(status)
	}

	// Expand sharded files: "model-00001-of-00005.gguf" → all 5 parts
	filenames := ExpandShards(filename)

	// Setup directory under the configured models dir (which may live
	// outside dataDir if ModelsDir is set).
	safeName := SafeModelID(modelID)
	modelDir := filepath.Join(d.modelsDir, safeName)
	os.MkdirAll(modelDir, 0o755)

	// Get combined total size via HEAD requests for accurate progress
	var combinedTotal int64
	if len(filenames) > 1 {
		combinedTotal = d.fetchCombinedSize(ctx, source, modelID, filenames)
	}

	// Download each file (single file or all shards sequentially)
	var totalDownloaded int64
	for i, fn := range filenames {
		select {
		case <-ctx.Done():
			sendProgress(DownloadStatus{ID: downloadID, ModelID: modelID, Filename: fn, Status: "cancelled"})
			return
		default:
		}

		label := fn
		if len(filenames) > 1 {
			label = fmt.Sprintf("%s [%d/%d]", fn, i+1, len(filenames))
		}

		downloaded, err := d.downloadFile(ctx, source, downloadID, modelID, fn, label, modelDir, totalDownloaded, combinedTotal, dl)
		if err != nil {
			// A mid-file cancellation surfaces as ctx.Err() from downloadFile —
			// report it as "cancelled" like the between-files check above, not
			// as a failure.
			if errors.Is(err, context.Canceled) {
				sendProgress(DownloadStatus{ID: downloadID, ModelID: modelID, Filename: label, Status: "cancelled"})
				return
			}
			sendProgress(DownloadStatus{ID: downloadID, ModelID: modelID, Filename: label, Status: "failed", Error: err.Error()})
			return
		}
		totalDownloaded += downloaded
	}

	sendProgress(DownloadStatus{
		ID:              downloadID,
		ModelID:         modelID,
		Filename:        filename,
		BytesDownloaded: totalDownloaded,
		TotalBytes:      combinedTotal,
		Status:          "complete",
	})

	slog.Info("download complete", "model", modelID, "file", filename, "shards", len(filenames), "size", totalDownloaded)

	if d.onComplete != nil {
		d.onComplete(source, downloadID, modelID, filenames[0], totalDownloaded)
	}
}

// fetchCombinedSize does HEAD requests to get the total size of all shard files.
func (d *Downloader) fetchCombinedSize(ctx context.Context, source, modelID string, filenames []string) int64 {
	client := &http.Client{Timeout: 30 * time.Second}
	p := d.provider(source)
	var total int64
	for _, fn := range filenames {
		dlURL := p.URL(modelID, fn)
		req, err := http.NewRequestWithContext(ctx, "HEAD", dlURL, nil)
		if err != nil {
			continue
		}
		if p.Token != "" {
			req.Header.Set("Authorization", "Bearer "+p.Token)
		}
		resp, err := client.Do(req)
		if err != nil {
			continue
		}
		resp.Body.Close()
		if resp.ContentLength > 0 {
			total += resp.ContentLength
		}
	}
	return total
}

// downloadFile downloads a single file, reporting progress with a base offset for multi-shard tracking.
// Returns the number of bytes downloaded for this file.
func (d *Downloader) downloadFile(ctx context.Context, source, downloadID, modelID, filename, label, modelDir string,
	baseDownloaded, combinedTotal int64, dl *download) (int64, error) {

	sendProgress := func(status DownloadStatus) {
		dl.broadcast(status)
	}

	partPath := filepath.Join(modelDir, filename+".part")
	finalPath := filepath.Join(modelDir, filename)

	// Ensure subdirectories exist (for files like "Q8_0/model.gguf")
	os.MkdirAll(filepath.Dir(finalPath), 0o755)

	// Skip if already downloaded
	if info, err := os.Stat(finalPath); err == nil {
		return info.Size(), nil
	}

	// Check for existing partial download
	var existingSize int64
	if info, err := os.Stat(partPath); err == nil {
		existingSize = info.Size()
	}

	p := d.provider(source)
	req, err := http.NewRequestWithContext(ctx, "GET", p.URL(modelID, filename), nil)
	if err != nil {
		return 0, err
	}

	if p.Token != "" {
		req.Header.Set("Authorization", "Bearer "+p.Token)
	}
	if existingSize > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", existingSize))
	}

	client := &http.Client{Timeout: 0}
	resp, err := client.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusPartialContent {
		return 0, fmt.Errorf("HTTP %d", resp.StatusCode)
	}

	// Whether the body is the whole file or just the tail we asked for
	// cannot be decided from the status code alone. ModelScope answers a
	// ranged request with 200 plus a Content-Range header rather than the
	// 206 the RFC requires (see modelscope.DownloadURL), and reading that
	// as "the server ignored my Range" would truncate the partial file
	// below and write only the tail into it — a corrupt model that looks
	// complete. A Content-Range header means partial, whatever the status.
	fileTotal := resp.ContentLength
	if responseIsPartial(resp) {
		fileTotal += existingSize
	} else {
		existingSize = 0
	}

	// Use file-level total if we don't have a combined total
	reportTotal := combinedTotal
	if reportTotal == 0 {
		reportTotal = fileTotal
	}

	flags := os.O_CREATE | os.O_WRONLY
	if existingSize > 0 {
		flags |= os.O_APPEND
	} else {
		flags |= os.O_TRUNC
	}

	f, err := os.OpenFile(partPath, flags, 0o644)
	if err != nil {
		return 0, err
	}
	defer f.Close()

	buf := make([]byte, 256*1024)
	downloaded := existingSize
	lastReport := time.Now()
	lastBytes := baseDownloaded + downloaded

	for {
		select {
		case <-ctx.Done():
			return downloaded, ctx.Err()
		default:
		}

		n, readErr := resp.Body.Read(buf)
		if n > 0 {
			if _, werr := f.Write(buf[:n]); werr != nil {
				return downloaded, werr
			}
			downloaded += int64(n)
		}

		if time.Since(lastReport) >= 500*time.Millisecond {
			globalDownloaded := baseDownloaded + downloaded
			speed := int64(float64(globalDownloaded-lastBytes) / time.Since(lastReport).Seconds())
			lastReport = time.Now()
			lastBytes = globalDownloaded

			sendProgress(DownloadStatus{
				ID:              downloadID,
				ModelID:         modelID,
				Filename:        label,
				BytesDownloaded: globalDownloaded,
				TotalBytes:      reportTotal,
				SpeedBPS:        speed,
				Status:          "downloading",
			})
		}

		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			return downloaded, readErr
		}
	}

	f.Close()
	if err := os.Rename(partPath, finalPath); err != nil {
		return downloaded, err
	}

	return downloaded, nil
}

// responseIsPartial reports whether a response to a ranged request
// carries only part of the file. See the call site for why the status
// code is not sufficient on its own.
func responseIsPartial(resp *http.Response) bool {
	return resp.StatusCode == http.StatusPartialContent || resp.Header.Get("Content-Range") != ""
}
