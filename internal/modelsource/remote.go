package modelsource

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
)

// ErrProbeBudget is returned when reading a remote header would cost more
// than the caller allowed. It is not a failure of the file — just a
// refusal to keep spending — so callers fall back to their uncorrected
// estimate rather than reporting the model as broken.
var ErrProbeBudget = errors.New("remote header probe exceeded its byte budget")

// rangeReader is an io.ReadSeeker over an HTTP resource, fetching only
// the spans that are actually read. It exists so the GGUF header parser
// can run against a file still sitting on a model host: the parser seeks
// and reads its way through a header, and paying for a whole multi-GiB
// download to answer a question about its first few kilobytes would be
// absurd.
//
// Reads are served from a single sliding window rather than a general
// cache, because the parser walks forward. A backwards seek inside the
// window is free; one outside it refetches.
type rangeReader struct {
	ctx      context.Context
	client   *http.Client
	url      string
	token    string
	size     int64 // total resource length, known from the file listing
	chunk    int64 // bytes to request next; grows as reading continues
	maxChunk int64
	budget   int64 // total bytes this reader may fetch
	fetched  int64

	pos      int64
	buf      []byte
	bufStart int64
}

func newRangeReader(ctx context.Context, client *http.Client, url, token string, size, chunk, budget int64) *rangeReader {
	return &rangeReader{
		ctx: ctx, client: client, url: url, token: token,
		size: size, chunk: chunk, maxChunk: maxProbeChunk, budget: budget,
		bufStart: -1,
	}
}

func (r *rangeReader) Read(p []byte) (int, error) {
	if r.pos >= r.size {
		return 0, io.EOF
	}
	if r.bufStart < 0 || r.pos < r.bufStart || r.pos >= r.bufStart+int64(len(r.buf)) {
		if err := r.fetchAt(r.pos); err != nil {
			return 0, err
		}
	}
	off := int(r.pos - r.bufStart)
	n := copy(p, r.buf[off:])
	r.pos += int64(n)
	return n, nil
}

func (r *rangeReader) Seek(offset int64, whence int) (int64, error) {
	var abs int64
	switch whence {
	case io.SeekStart:
		abs = offset
	case io.SeekCurrent:
		abs = r.pos + offset
	case io.SeekEnd:
		// The length comes from the host's file listing, which is why a
		// probe needs the size passed in rather than discovering it.
		abs = r.size + offset
	default:
		return 0, fmt.Errorf("invalid whence %d", whence)
	}
	if abs < 0 {
		return 0, errors.New("negative seek position")
	}
	r.pos = abs
	return abs, nil
}

// fetchAt loads the chunk containing off into the buffer.
func (r *rangeReader) fetchAt(off int64) error {
	end := off + r.chunk - 1
	if end >= r.size {
		end = r.size - 1
	}
	want := end - off + 1
	if r.fetched+want > r.budget {
		return ErrProbeBudget
	}

	req, err := http.NewRequestWithContext(r.ctx, http.MethodGet, r.url, nil)
	if err != nil {
		return err
	}
	req.Header.Set("Range", fmt.Sprintf("bytes=%d-%d", off, end))
	if r.token != "" {
		req.Header.Set("Authorization", "Bearer "+r.token)
	}
	resp, err := r.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusPartialContent {
		return fmt.Errorf("range request returned %d", resp.StatusCode)
	}
	// A host that ignores Range answers 200 with the whole file. Reading
	// it would blow the budget on a file measured in gigabytes, so cap
	// the read and let the parser work with what arrives.
	body, err := io.ReadAll(io.LimitReader(resp.Body, want))
	if err != nil {
		return err
	}
	if len(body) == 0 {
		return io.ErrUnexpectedEOF
	}
	r.buf = body
	r.bufStart = off
	r.fetched += int64(len(body))

	// Grow the window as reading continues. The two access patterns want
	// opposite things: finding a tensor table in a split shard needs one
	// small request, while walking a tokenizer to reach the table in an
	// unsplit file is dominated by round trips. Starting small and
	// growing serves both — the cheap case never pays for a large fetch,
	// and the expensive one stops being a long chain of small ones.
	if r.chunk < r.maxChunk {
		r.chunk *= 4
		if r.chunk > r.maxChunk {
			r.chunk = r.maxChunk
		}
	}
	return nil
}
