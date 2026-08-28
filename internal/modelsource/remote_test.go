package modelsource

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

// countingHost serves a body over ranges while recording how many
// requests and bytes it took. Request count is what decides probe
// latency against a real host, so it is what these tests assert on.
type countingHost struct {
	requests atomic.Int64
	bytes    atomic.Int64
	srv      *httptest.Server
}

func newCountingHost(t *testing.T, body []byte) *countingHost {
	t.Helper()
	h := &countingHost{}
	h.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		h.requests.Add(1)
		var start, end int
		if n, _ := fmt.Sscanf(r.Header.Get("Range"), "bytes=%d-%d", &start, &end); n == 2 {
			if end >= len(body) {
				end = len(body) - 1
			}
			if start > end || start >= len(body) {
				w.WriteHeader(http.StatusRequestedRangeNotSatisfiable)
				return
			}
			chunk := body[start : end+1]
			h.bytes.Add(int64(len(chunk)))
			w.WriteHeader(http.StatusPartialContent)
			w.Write(chunk)
			return
		}
		h.bytes.Add(int64(len(body)))
		w.Write(body)
	}))
	t.Cleanup(h.srv.Close)
	return h
}

// The cheap case: a split model's tensor-carrying shard has no metadata,
// so its table sits in the first few kilobytes. One request should do it.
func TestProbeCostSplitShard(t *testing.T) {
	body := buildGGUF(t, 0, []tensorSpec{
		{"token_embd.weight", 0},
		{pleTensorNameForTest, 1 << 20},
		{"output.weight", 1<<20 + 6<<30},
	}, 4096)
	h := newCountingHost(t, body)

	got := ProbePLE(context.Background(), h.srv.Client(), "", []ShardURL{
		{URL: h.srv.URL, Size: 8 << 30},
		{URL: h.srv.URL, Size: 8 << 30},
	}, probeMinBytes+1)
	if got.StreamedBytes != 6<<30 {
		t.Fatalf("streamed = %d, want %d", got.StreamedBytes, int64(6)<<30)
	}
	if n := h.requests.Load(); n > 2 {
		t.Errorf("took %d requests; a split shard's table should be reachable in one", n)
	}
	t.Logf("split shard: %d request(s), %d bytes", h.requests.Load(), h.bytes.Load())
}

// The expensive case: an unsplit GGUF puts its tokenizer ahead of the
// tensor table. The window has to grow, or this becomes a long chain of
// tiny requests — which is exactly what made a listing take 20 seconds.
func TestProbeCostUnsplitWithLargeTokenizer(t *testing.T) {
	const tokenizer = 12 * 1024 * 1024
	body := buildGGUF(t, tokenizer, []tensorSpec{
		{"token_embd.weight", 0},
		{pleTensorNameForTest, 1 << 20},
		{"output.weight", 1<<20 + 6<<30},
	}, 4096)
	h := newCountingHost(t, body)

	got := ProbePLE(context.Background(), h.srv.Client(), "", []ShardURL{
		{URL: h.srv.URL, Size: 8 << 30},
		{URL: h.srv.URL, Size: 8 << 30},
	}, probeMinBytes+1)
	if got.StreamedBytes != 6<<30 {
		t.Fatalf("streamed = %d, want the table found behind the tokenizer", got.StreamedBytes)
	}
	// Fixed 64 KB windows would need ~190 requests for 12 MB. Growth
	// should bring that down by more than an order of magnitude.
	if n := h.requests.Load(); n > 15 {
		t.Errorf("took %d requests to walk %d MB; the window is not growing",
			n, tokenizer/(1024*1024))
	}
	t.Logf("unsplit + %d MB tokenizer: %d requests, %.1f MB fetched",
		tokenizer/(1024*1024), h.requests.Load(), float64(h.bytes.Load())/(1024*1024))
}

// Whatever a file looks like, a probe must never fetch more than its
// budget: a listing that quietly turned into a multi-gigabyte download
// would be far worse than an uncorrected estimate.
func TestProbeNeverExceedsBudget(t *testing.T) {
	body := buildGGUF(t, probeBudget*2, []tensorSpec{{pleTensorNameForTest, 0}}, 4096)
	h := newCountingHost(t, body)

	ProbePLE(context.Background(), h.srv.Client(), "", []ShardURL{
		{URL: h.srv.URL, Size: 8 << 30},
	}, probeMinBytes+1)

	// One in-flight window may straddle the limit, hence the allowance.
	if b := h.bytes.Load(); b > probeBudget+maxProbeChunk {
		t.Errorf("fetched %d bytes, over the %d budget", b, probeBudget)
	}
	t.Logf("%d requests, %.1f MB fetched (budget %.0f MB)",
		h.requests.Load(), float64(h.bytes.Load())/(1024*1024), float64(probeBudget)/(1024*1024))
}

// The budget is enforced by the reader itself, so a caller cannot walk
// past it by continuing to read.
func TestRangeReaderStopsAtBudget(t *testing.T) {
	body := make([]byte, 4*1024*1024)
	h := newCountingHost(t, body)
	const budget = 256 * 1024
	r := newRangeReader(context.Background(), h.srv.Client(), h.srv.URL, "",
		int64(len(body)), 64*1024, budget)

	buf := make([]byte, 32*1024)
	var err error
	for i := 0; i < 1000 && err == nil; i++ {
		_, err = r.Read(buf)
	}
	if !errors.Is(err, ErrProbeBudget) {
		t.Fatalf("read stopped with %v, want ErrProbeBudget", err)
	}
	if b := h.bytes.Load(); b > budget+maxProbeChunk {
		t.Errorf("fetched %d bytes against a %d budget", b, budget)
	}
}

func TestRangeReaderSeek(t *testing.T) {
	body := []byte(strings.Repeat("abcdefgh", 1024))
	h := newCountingHost(t, body)
	r := newRangeReader(context.Background(), h.srv.Client(), h.srv.URL, "", int64(len(body)), 1024, 1<<20)

	// SeekEnd uses the declared size, which is where a shard's last
	// tensor gets its length from.
	if got, err := r.Seek(0, 2); err != nil || got != int64(len(body)) {
		t.Fatalf("SeekEnd = %d, %v; want %d", got, err, len(body))
	}
	if _, err := r.Seek(8, 0); err != nil {
		t.Fatal(err)
	}
	buf := make([]byte, 8)
	if _, err := r.Read(buf); err != nil {
		t.Fatal(err)
	}
	if string(buf) != "abcdefgh" {
		t.Errorf("read %q after seek", buf)
	}
	// Re-reading inside the window already fetched must not refetch —
	// the parser does exactly this when it re-reads a header after
	// checking the tensor count.
	before := h.requests.Load()
	if _, err := r.Seek(8, 0); err != nil {
		t.Fatal(err)
	}
	if _, err := r.Read(buf); err != nil {
		t.Fatal(err)
	}
	if h.requests.Load() != before {
		t.Error("re-read inside the window issued another request")
	}
}
