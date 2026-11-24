// Package videofile reads the size, frame count, frame rate and audio of a
// video clip without decoding its frames, for the formats that
// stable-diffusion.cpp's DGL video::load reads: AVI (24-bit RGB or MJPEG,
// with PCM or float audio), WebP (still or animated) and WebM with
// independent VP8 frames and optional PCM16 audio (src/dgl/ops/video.cpp,
// video_webp.cpp, video_webm.cpp). The checks follow sd.cpp's own, so a
// clip Probe accepts is one video::load can read, and a clip it refuses
// gets a clear reason before any GPU work.
//
// It also writes an uncompressed RGB AVI (WriteRGBAVI), which Measure
// uses as its test clip.
package videofile

import (
	"errors"
	"fmt"
	"io"
)

// Formats is the list of accepted formats, for error messages.
const Formats = "AVI (MJPEG or RGB), WebM (VP8) or animated WebP"

var (
	// ErrMP4 is the error for an MP4 or MOV file.
	ErrMP4 = errors.New("MP4 is not supported: send " + Formats)
	// ErrUnknownFormat is the error for a file that is none of the
	// accepted formats.
	ErrUnknownFormat = errors.New("unknown video format: send " + Formats)
	// ErrCodec is wrapped by every error about a file of a known format
	// that video::load cannot read (codec, layout, damage).
	ErrCodec = errors.New("unsupported video")
)

// codecError is an ErrCodec with its own message.
type codecError struct{ msg string }

func (e *codecError) Error() string { return e.msg }
func (e *codecError) Unwrap() error { return ErrCodec }

func codecErr(format string, args ...any) error {
	return &codecError{fmt.Sprintf(format, args...)}
}

// Info describes a clip.
type Info struct {
	Format   string `json:"format"` // "avi", "webp" or "webm"
	Width    int    `json:"width"`
	Height   int    `json:"height"`
	Frames   int    `json:"frames"`
	FPSNum   int    `json:"fps_num"` // 0 when the file gives no rate
	FPSDen   int    `json:"fps_den"`
	HasAudio bool   `json:"has_audio"`
}

// DecodedBytes is what video::load holds for the clip's frames:
// 3 × width × height × frames.
func (i Info) DecodedBytes() int64 {
	return 3 * int64(i.Width) * int64(i.Height) * int64(i.Frames)
}

// FPS is the frame rate as a number, or 0 when the file gives none.
func (i Info) FPS() float64 {
	if i.FPSNum <= 0 || i.FPSDen <= 0 {
		return 0
	}
	return float64(i.FPSNum) / float64(i.FPSDen)
}

// Scan limits. Payloads are skipped, never read, and do not count. The
// limits only stop a hostile file: a clip at video::load's 256 MiB limit
// has at most about 40,000 frames, far below maxChunks.
const (
	maxReadBytes = 64 << 20
	maxChunks    = 2_000_000
)

// src reads parts of a file and counts what it reads.
type src struct {
	r      io.ReaderAt
	size   int64
	read   int64
	chunks int
}

// at reads n bytes at off. A read past the end is io.ErrUnexpectedEOF.
func (s *src) at(off int64, n int) ([]byte, error) {
	if off < 0 || n < 0 || off+int64(n) > s.size {
		return nil, io.ErrUnexpectedEOF
	}
	s.read += int64(n)
	if s.read > maxReadBytes {
		return nil, codecErr("file has too many chunks to check")
	}
	b := make([]byte, n)
	if _, err := s.r.ReadAt(b, off); err != nil {
		if errors.Is(err, io.EOF) {
			return nil, io.ErrUnexpectedEOF
		}
		return nil, err
	}
	return b, nil
}

// tick counts one chunk or element.
func (s *src) tick() error {
	s.chunks++
	if s.chunks > maxChunks {
		return codecErr("file has too many chunks to check")
	}
	return nil
}

// Probe reads a clip's Info. size is the file's length. Errors are
// ErrMP4, ErrUnknownFormat, an ErrCodec (unsupported or damaged file of a
// known format) or a read error.
func Probe(r io.ReaderAt, size int64) (Info, error) {
	s := &src{r: r, size: size}
	head, err := s.at(0, min(16, int(size)))
	if err != nil {
		return Info{}, err
	}
	var info Info
	switch {
	case len(head) >= 12 && string(head[0:4]) == "RIFF" && string(head[8:12]) == "AVI ":
		info, err = probeAVI(s)
	case len(head) >= 12 && string(head[0:4]) == "RIFF" && string(head[8:12]) == "WEBP":
		info, err = probeWebP(s)
	case len(head) >= 4 && head[0] == 0x1A && head[1] == 0x45 && head[2] == 0xDF && head[3] == 0xA3:
		info, err = probeWebM(s)
	case len(head) >= 8 && string(head[4:8]) == "ftyp":
		return Info{}, ErrMP4
	default:
		return Info{}, ErrUnknownFormat
	}
	if errors.Is(err, io.ErrUnexpectedEOF) {
		return Info{}, codecErr("the %s file is truncated or damaged", formatName(head))
	}
	if err != nil {
		return Info{}, err
	}
	if info.Frames < 1 {
		return Info{}, codecErr("the %s file has no video frames", info.Format)
	}
	return info, nil
}

func formatName(head []byte) string {
	switch {
	case len(head) >= 12 && string(head[8:12]) == "AVI ":
		return "AVI"
	case len(head) >= 12 && string(head[8:12]) == "WEBP":
		return "WebP"
	}
	return "WebM"
}

func gcd(a, b int) int {
	for b != 0 {
		a, b = b, a%b
	}
	return a
}

// reduce returns num/den in lowest terms, or 0/0 when either is not
// positive.
func reduce(num, den int) (int, int) {
	if num <= 0 || den <= 0 {
		return 0, 0
	}
	g := gcd(num, den)
	return num / g, den / g
}
