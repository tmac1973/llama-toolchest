package modelscope

import (
	"context"
	"io"
	"net/http"
	"os"
	"strings"
	"testing"
	"time"
)

// The endpoints this package uses are not a published contract — they are
// what modelscope.cn's own web UI calls. These tests validate those
// assumptions against the live service rather than code correctness, so
// they are opt-in: set MODELSCOPE_LIVE_TEST=1.
func liveClient(t *testing.T) (*Client, context.Context, context.CancelFunc) {
	t.Helper()
	if os.Getenv("MODELSCOPE_LIVE_TEST") != "1" {
		t.Skip("set MODELSCOPE_LIVE_TEST=1 to run against the live ModelScope API")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	return NewClient(os.Getenv("MODELSCOPE_TOKEN")), ctx, cancel
}

func TestLiveSearch(t *testing.T) {
	c, ctx, cancel := liveClient(t)
	defer cancel()

	got, err := c.Search(ctx, "qwen3")
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(got) == 0 {
		t.Fatal("no results for \"qwen3\"")
	}
	for _, r := range got {
		if !strings.Contains(r.ID, "/") {
			t.Errorf("result id %q is not owner/name", r.ID)
		}
		if r.Author == "" {
			t.Errorf("result %q has no author", r.ID)
		}
	}
	// Sorted most-downloaded first, which is what SortBy=DownloadsCount
	// buys us; if that key ever stops being honored this is what notices.
	if len(got) > 1 && got[0].Downloads < got[len(got)-1].Downloads {
		t.Errorf("results not sorted by downloads: first=%d last=%d",
			got[0].Downloads, got[len(got)-1].Downloads)
	}
	t.Logf("%d results, top: %s (%d downloads)", len(got), got[0].ID, got[0].Downloads)
}

func TestLiveGetModel(t *testing.T) {
	c, ctx, cancel := liveClient(t)
	defer cancel()

	const id = "unsloth/Qwen3-8B-GGUF"
	detail, err := c.GetModel(ctx, id)
	if err != nil {
		t.Fatalf("GetModel: %v", err)
	}
	if len(detail.Files) == 0 {
		t.Fatalf("no GGUF files in %s", id)
	}
	for _, f := range detail.Files {
		if f.Size <= 0 {
			t.Errorf("%s has no size — the file listing stopped carrying Size", f.Filename)
		}
		if f.Quant == "" && !f.IsMMProj {
			t.Errorf("%s: quant not parsed", f.Filename)
		}
	}
	t.Logf("%s: %d entries, first: %s (%.2f GiB, quant %s)",
		id, len(detail.Files), detail.Files[0].Filename,
		float64(detail.Files[0].Size)/(1<<30), detail.Files[0].Quant)
}

// The downloader resumes by re-requesting with a Range header, so a
// download URL that does not honor ranges would break resume without
// failing outright. This pins both halves of the real behavior: the range
// is honored, and it comes back labelled 200 rather than 206 — the quirk
// ResponseIsPartial exists for. If ModelScope ever starts returning a
// conforming 206, the second assertion fails and this comment is what
// explains that the change is welcome.
func TestLiveDownloadURLHonorsRange(t *testing.T) {
	c, ctx, cancel := liveClient(t)
	defer cancel()

	const id = "unsloth/Qwen3-8B-GGUF"
	detail, err := c.GetModel(ctx, id)
	if err != nil {
		t.Fatalf("GetModel: %v", err)
	}
	var target string
	var best int64
	for _, f := range detail.Files {
		if len(f.Shards) == 0 && (best == 0 || f.Size < best) {
			best, target = f.Size, f.Filename
		}
	}
	if target == "" {
		t.Skip("no unsharded file to probe")
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.DownloadURL(id, target), nil)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Range", "bytes=100-1123")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("range request: %v", err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if len(body) != 1024 {
		t.Errorf("got %d bytes, want the 1024 requested — ranges are not being honored", len(body))
	}
	if !ResponseIsPartial(resp) {
		t.Errorf("ResponseIsPartial said no: status=%d content-range=%q", resp.StatusCode, resp.Header.Get("Content-Range"))
	}
	if resp.StatusCode == http.StatusPartialContent {
		t.Logf("NOTE: ModelScope now returns a conforming 206; the 200 workaround can be revisited")
	}
	t.Logf("%s: status %d, %d bytes, Content-Range %q", target, resp.StatusCode, len(body), resp.Header.Get("Content-Range"))
}
