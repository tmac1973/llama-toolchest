package modelsource

import (
	"context"
	"encoding/binary"
	"errors"
	"io"
	"net/http"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/models"
)

const (
	// probeChunk is how much of a header to fetch per request. Sized for
	// the case that matters: a split GGUF's tensor-carrying shards have
	// no metadata block, so their tensor table ends around 40 KB in and a
	// single request covers it.
	probeChunk = 64 * 1024

	// maxProbeChunk bounds how large the sliding window grows while
	// walking toward a tensor table that is a long way in.
	maxProbeChunk = 4 * 1024 * 1024

	// probeBudget caps what one file may cost, so a pathological file
	// cannot turn a listing into a download.
	//
	// In a split model every shard is cheap: the metadata shard declares
	// no tensors and is dropped on its 24-byte header, and the others
	// have no metadata to walk. An unsplit GGUF is the opposite — its
	// tensor table sits behind the tokenizer, which for a large model
	// runs to tens of megabytes, and paying that for every file in a
	// listing made an ordinary repository take nine seconds to expand.
	//
	// The budget is generous enough to reach a table behind a large
	// tokenizer; the sliding window grows as it goes so that walk costs a
	// handful of requests rather than hundreds. A file that still does
	// not yield within the budget falls back to the uncorrected estimate,
	// which is the status quo rather than a new problem.
	probeBudget = 16 * 1024 * 1024

	// probeMinBytes is the download size below which probing cannot
	// change the verdict. llama.cpp only streams tables over 4 GiB, and a
	// table is a minority of the file, so anything under this threshold
	// either has no table or has one that stays resident. Skipping these
	// keeps the probe off the path of every ordinary model.
	probeMinBytes = 8 * 1024 * 1024 * 1024
)

// ProbeResult is what a remote header probe learned about one file.
type ProbeResult struct {
	// StreamedBytes is the part of the download that never occupies VRAM:
	// the per-layer embedding table, which llama.cpp holds host-mapped at
	// every size. Zero when the file has no such table.
	StreamedBytes int64
	// Probed records that the header was actually read. False means the
	// probe was skipped or failed, and StreamedBytes says nothing.
	Probed bool
}

// ShardURL resolves one shard of a file to a download URL and its size.
type ShardURL struct {
	URL  string
	Size int64
}

// ProbePLE reads the GGUF headers of a file's shards to find the
// per-layer embedding table, and reports how much of the download will be
// streamed rather than loaded.
//
// Shards are probed in order and the first table found wins: the tensor
// lives in exactly one shard, and stopping there means the common case
// costs a single small ranged read.
//
// Every failure path returns Probed false rather than an error the caller
// must handle. The probe is an improvement on an estimate, not a
// precondition for anything, so a host that rate-limits or a network that
// drops should leave the UI showing its uncorrected number rather than an
// error.
func ProbePLE(ctx context.Context, client *http.Client, token string, shards []ShardURL, totalBytes int64) ProbeResult {
	if totalBytes < probeMinBytes || len(shards) == 0 {
		return ProbeResult{}
	}
	// Only split files are probed, and the reason is cost. A split
	// model's shards are nearly free to inspect: the metadata shard
	// declares no tensors and is dropped on its 24-byte header, and the
	// rest carry no metadata, so their tensor table sits in the first few
	// kilobytes — one request, about 4 KB. An unsplit GGUF puts its
	// tensor table behind the tokenizer, which for a large model means
	// roughly 12 MB and six requests per file; across a repository's
	// worth of quants that turned expanding a listing into a nine-second
	// wait for every ordinary model, to correct the rare one.
	//
	// The trade is sound because the two sets barely overlap: a table big
	// enough to be streamed at all is over 4 GiB, and a model carrying
	// one is large enough that publishers split it. Where it does not
	// hold, the estimate stays as it was — conservative, and corrected
	// exactly once the file is on disk.
	if len(shards) < 2 {
		return ProbeResult{}
	}
	if client == nil {
		client = &http.Client{Timeout: 30 * time.Second}
	}

	for _, sh := range shards {
		if sh.Size <= 0 || sh.URL == "" {
			continue
		}
		r := newRangeReader(ctx, client, sh.URL, token, sh.Size, probeChunk, probeBudget)

		// Skip a shard that declares no tensors before parsing it. This
		// is not a micro-optimization: a split model puts all of its
		// metadata in the first shard, and for a large model that is a
		// tokenizer megabytes long that the parser would walk in full to
		// reach a tensor table that isn't there. On Qwen3.8-Flash-Next
		// skipping it takes the probe from 33 seconds to under one.
		if n, err := tensorCount(r); err == nil && n == 0 {
			continue
		}
		if _, err := r.Seek(0, io.SeekStart); err != nil {
			continue
		}
		meta, err := models.ParseGGUFMetaFrom(r)
		if err != nil {
			// A shard that isn't parseable on its own — or a probe that
			// ran out of budget — tells us nothing about the others, so
			// keep going rather than giving up on the file.
			continue
		}
		if meta.PLEBytes > 0 {
			return ProbeResult{StreamedBytes: offDeviceBytes(meta.PLEBytes), Probed: true}
		}
	}
	// Every shard parsed and none held a table: that is a real answer,
	// not a failure. Saying so lets the caller skip re-probing.
	return ProbeResult{StreamedBytes: 0, Probed: true}
}

// offDeviceBytes is the part of a download that never occupies VRAM.
//
// This used to apply llama.cpp's 4 GiB streaming threshold, on the reading
// that only a streamed table stayed out of VRAM. Measurement says the table
// is held host-mapped at every size and in every mode, so the whole of it
// is off-device regardless — see plan/ple-vram-findings.md. The threshold
// governs host residency, which was never what this number is for.
//
// Kept the same rule as the post-download estimate, so the figure shown
// before downloading and the one shown after still agree.
func offDeviceBytes(pleBytes int64) int64 {
	return pleBytes
}

// tensorCount reads a GGUF file's tensor count from its 24-byte header,
// leaving the reader positioned after it.
func tensorCount(r io.ReadSeeker) (uint64, error) {
	if _, err := r.Seek(0, io.SeekStart); err != nil {
		return 0, err
	}
	var hdr struct {
		Magic   [4]byte
		Version uint32
		Tensors uint64
		KV      uint64
	}
	if err := binary.Read(r, binary.LittleEndian, &hdr); err != nil {
		return 0, err
	}
	if string(hdr.Magic[:]) != "GGUF" {
		return 0, errors.New("not a GGUF file")
	}
	return hdr.Tensors, nil
}
