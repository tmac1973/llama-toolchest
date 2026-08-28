package modelsource

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

// buildGGUF writes a minimal GGUF: one string KV of the requested padding
// size (standing in for a tokenizer), then the named tensors, then
// dataBytes of tensor data.
func buildGGUF(t *testing.T, kvPadding int, tensors []struct {
	name   string
	offset uint64
}, dataBytes int) []byte {
	t.Helper()
	var b bytes.Buffer
	ws := func(s string) {
		binary.Write(&b, binary.LittleEndian, uint64(len(s)))
		b.WriteString(s)
	}
	b.WriteString("GGUF")
	binary.Write(&b, binary.LittleEndian, uint32(3))
	binary.Write(&b, binary.LittleEndian, uint64(len(tensors)))
	kv := 1
	if kvPadding > 0 {
		kv = 2
	}
	binary.Write(&b, binary.LittleEndian, uint64(kv))

	ws("general.architecture")
	binary.Write(&b, binary.LittleEndian, uint32(8)) // string
	ws("testarch")
	if kvPadding > 0 {
		// A real tokenizer is an array of many short strings, not one
		// long one. The distinction decides the cost of reaching the
		// tensor table: a single string is one seek, while an array's
		// per-element length prefixes have to be read through.
		ws("tokenizer.ggml.tokens")
		binary.Write(&b, binary.LittleEndian, uint32(9)) // array
		binary.Write(&b, binary.LittleEndian, uint32(8)) // of strings
		const tokenLen = 8
		count := kvPadding / (tokenLen + 8)
		binary.Write(&b, binary.LittleEndian, uint64(count))
		for i := 0; i < count; i++ {
			ws(strings.Repeat("t", tokenLen))
		}
	}
	for _, tn := range tensors {
		ws(tn.name)
		binary.Write(&b, binary.LittleEndian, uint32(1))
		binary.Write(&b, binary.LittleEndian, uint64(8))
		binary.Write(&b, binary.LittleEndian, uint32(0))
		binary.Write(&b, binary.LittleEndian, tn.offset)
	}
	for b.Len()%32 != 0 {
		b.WriteByte(0)
	}
	b.Write(make([]byte, dataBytes))
	return b.Bytes()
}

type tensorSpec = struct {
	name   string
	offset uint64
}

// serveRanges hosts fixed bodies at /0, /1, ... and counts bytes served,
// so a test can assert on what a probe actually cost.
func serveRanges(t *testing.T, bodies [][]byte) (*httptest.Server, *int64) {
	t.Helper()
	var served int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		idx := strings.TrimPrefix(r.URL.Path, "/")
		var i int
		if _, err := fmtSscan(idx, &i); err != nil || i >= len(bodies) {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		body := bodies[i]
		rng := r.Header.Get("Range")
		var start, end int
		if n, _ := fmtSscanRange(rng, &start, &end); n == 2 {
			if end >= len(body) {
				end = len(body) - 1
			}
			if start > end {
				w.WriteHeader(http.StatusRequestedRangeNotSatisfiable)
				return
			}
			chunk := body[start : end+1]
			atomic.AddInt64(&served, int64(len(chunk)))
			w.Header().Set("Content-Range", "bytes")
			w.WriteHeader(http.StatusPartialContent)
			w.Write(chunk)
			return
		}
		atomic.AddInt64(&served, int64(len(body)))
		w.Write(body)
	}))
	return srv, &served
}

func fmtSscan(s string, i *int) (int, error)         { return fmt.Sscanf(s, "%d", i) }
func fmtSscanRange(s string, a, b *int) (int, error) { return fmt.Sscanf(s, "bytes=%d-%d", a, b) }

// A split model's metadata shard carries no tensors. Probing it would
// mean walking a tokenizer megabytes long to find nothing, so it must be
// skipped on the strength of its header alone.
func TestProbeSkipsTensorlessShard(t *testing.T) {
	// Several chunks worth of metadata, so "did it walk the whole
	// shard?" is actually measurable.
	meta := buildGGUF(t, 5*1024*1024, nil, 0)
	data := buildGGUF(t, 0, []tensorSpec{
		{"token_embd.weight", 0},
		{pleTensorNameForTest, 1 << 20},
		{"output.weight", 1<<20 + 6<<30},
	}, 1024)

	srv, served := serveRanges(t, [][]byte{meta, data})
	defer srv.Close()

	// The stub serves only the header bytes; Size is the shard's real
	// length, which is where the last tensor's size comes from and is
	// exactly what a file listing gives us in production.
	shards := []ShardURL{
		{URL: srv.URL + "/0", Size: int64(len(meta))},
		{URL: srv.URL + "/1", Size: 8 << 30},
	}
	got := ProbePLE(context.Background(), srv.Client(), "", shards, probeMinBytes+1)
	if !got.Probed {
		t.Fatal("probe did not complete")
	}
	if got.StreamedBytes != 6<<30 {
		t.Errorf("streamed = %d, want %d", got.StreamedBytes, int64(6)<<30)
	}
	// Walking the 5 MB metadata shard would show up here. One chunk to
	// read its header and recognize it as tensorless is the whole cost.
	if *served > 2*probeChunk {
		t.Errorf("probe fetched %d bytes, more than the %d it takes to read a header and move on",
			*served, 2*probeChunk)
	}
}

// Below the size threshold the answer cannot change, so nothing should be
// fetched at all.
func TestProbeSkippedBelowThreshold(t *testing.T) {
	srv, served := serveRanges(t, [][]byte{buildGGUF(t, 0, nil, 0)})
	defer srv.Close()
	got := ProbePLE(context.Background(), srv.Client(), "",
		[]ShardURL{{URL: srv.URL + "/0", Size: 10}}, probeMinBytes-1)
	if got.Probed || got.StreamedBytes != 0 {
		t.Errorf("small model was probed: %+v", got)
	}
	if *served != 0 {
		t.Errorf("fetched %d bytes for a model under the threshold", *served)
	}
}

// A table under llama.cpp's 4 GiB lazy threshold stays resident, so it
// must not be subtracted — the estimate is already right for it.
func TestProbeIgnoresSmallTable(t *testing.T) {
	data := buildGGUF(t, 0, []tensorSpec{
		{pleTensorNameForTest, 0},
		{"output.weight", 1 << 30}, // 1 GiB table
	}, 1024)
	srv, _ := serveRanges(t, [][]byte{data})
	defer srv.Close()
	got := ProbePLE(context.Background(), srv.Client(), "", []ShardURL{
		{URL: srv.URL + "/1", Size: 1 << 20}, // stand-in metadata shard
		{URL: srv.URL + "/0", Size: 3 << 30},
	}, probeMinBytes+1)
	if !got.Probed {
		t.Fatal("probe did not complete")
	}
	if got.StreamedBytes != 0 {
		t.Errorf("streamed = %d, want 0 for a table under the lazy threshold", got.StreamedBytes)
	}
}

// A model with no such table is a real answer, not a failure: Probed must
// be true so the caller can cache it and not ask again.
func TestProbeReportsNoTableAsAnswer(t *testing.T) {
	data := buildGGUF(t, 0, []tensorSpec{{"token_embd.weight", 0}, {"output.weight", 1 << 20}}, 1024)
	srv, _ := serveRanges(t, [][]byte{data})
	defer srv.Close()
	got := ProbePLE(context.Background(), srv.Client(), "", []ShardURL{
		{URL: srv.URL + "/1", Size: 1 << 20},
		{URL: srv.URL + "/0", Size: 8 << 30},
	}, probeMinBytes+1)
	if !got.Probed || got.StreamedBytes != 0 {
		t.Errorf("got %+v, want a completed probe reporting no table", got)
	}
}

// The probe improves an estimate; it must never turn a host problem into
// a user-visible failure.
func TestProbeDegradesQuietly(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
	}))
	defer srv.Close()
	got := ProbePLE(context.Background(), srv.Client(), "",
		[]ShardURL{{URL: srv.URL + "/0", Size: 1 << 30}}, probeMinBytes+1)
	if got.StreamedBytes != 0 {
		t.Errorf("streamed = %d, want 0 when the host refuses", got.StreamedBytes)
	}
}

const pleTensorNameForTest = "per_layer_token_embd.weight"
