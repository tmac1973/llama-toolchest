package videofile

import (
	"bufio"
	"encoding/binary"
	"io"
)

// aviMaxChunks is sd.cpp's own chunk limit (AVIReader::chunk).
const aviMaxChunks = 1_000_000

func le16(b []byte) int { return int(binary.LittleEndian.Uint16(b)) }
func le32(b []byte) uint32 {
	return binary.LittleEndian.Uint32(b)
}

type aviChunk struct {
	id         string
	data, size int64
	next       int64
}

type aviReader struct {
	*src
	chunks int
}

func (a *aviReader) chunk(off, end int64) (aviChunk, error) {
	a.chunks++
	if a.chunks > aviMaxChunks {
		return aviChunk{}, codecErr("the AVI file has too many chunks")
	}
	if err := a.tick(); err != nil {
		return aviChunk{}, err
	}
	if end-off < 8 {
		return aviChunk{}, codecErr("the AVI file has a damaged chunk structure")
	}
	b, err := a.at(off, 8)
	if err != nil {
		return aviChunk{}, err
	}
	size := int64(le32(b[4:]))
	padded := size + size&1
	if padded > end-off-8 {
		return aviChunk{}, codecErr("an AVI chunk is larger than the list that holds it; the file is damaged")
	}
	return aviChunk{id: string(b[:4]), data: off + 8, size: size, next: off + 8 + padded}, nil
}

// listType reads a LIST chunk's 4-byte type.
func (a *aviReader) listType(c aviChunk) (string, error) {
	if c.size < 4 {
		return "", codecErr("the AVI file has a damaged list")
	}
	b, err := a.at(c.data, 4)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

type aviVideo struct {
	index         int
	width, height int
	fpsNum        int
	fpsDen        int
}

func probeAVI(s *src) (Info, error) {
	a := &aviReader{src: s}
	root, err := a.at(0, 12)
	if err != nil {
		return Info{}, err
	}
	end := int64(le32(root[4:])) + 8
	if end != s.size {
		if end < s.size {
			if b, err := s.at(end, min(12, int(s.size-end))); err == nil && len(b) == 12 && string(b[:4]) == "RIFF" && string(b[8:12]) == "AVIX" {
				return Info{}, codecErr("segmented OpenDML AVI is not supported; send an AVI under 1 GiB written as one RIFF part")
			}
		}
		return Info{}, codecErr("the AVI file size does not match its header; the file is damaged, or it is a segmented OpenDML AVI, which is not supported")
	}
	video := aviVideo{index: -1}
	hasAudio, headerFound := false, false
	var movies []aviChunk
	for pos := int64(12); pos < end; {
		c, err := a.chunk(pos, end)
		if err != nil {
			return Info{}, err
		}
		if c.id == "LIST" {
			typ, err := a.listType(c)
			if err != nil {
				return Info{}, err
			}
			switch typ {
			case "hdrl":
				if headerFound {
					return Info{}, codecErr("the AVI file has two headers")
				}
				headerFound = true
				if hasAudio, err = a.header(c.data+4, c.data+c.size, &video); err != nil {
					return Info{}, err
				}
			case "movi":
				movies = append(movies, c)
			}
		}
		pos = c.next
	}
	if video.index < 0 {
		return Info{}, codecErr("the AVI file has no video stream")
	}
	frames := 0
	for _, m := range movies {
		n, err := a.frames(m.data+4, m.data+m.size, video.index, 0)
		if err != nil {
			return Info{}, err
		}
		frames += n
	}
	return Info{
		Format: "avi", Width: video.width, Height: video.height, Frames: frames,
		FPSNum: video.fpsNum, FPSDen: video.fpsDen, HasAudio: hasAudio,
	}, nil
}

// header reads hdrl: each strl stream. It fills the first video stream
// and reports whether there is a usable audio stream.
func (a *aviReader) header(start, end int64, video *aviVideo) (bool, error) {
	index, hasAudio := 0, false
	for pos := start; pos < end; {
		c, err := a.chunk(pos, end)
		if err != nil {
			return false, err
		}
		if c.id == "LIST" {
			typ, err := a.listType(c)
			if err != nil {
				return false, err
			}
			if typ == "strl" {
				if index >= 100 {
					return false, codecErr("the AVI file has too many streams")
				}
				audio, err := a.stream(c.data+4, c.data+c.size, index, video, hasAudio)
				if err != nil {
					return false, err
				}
				hasAudio = hasAudio || audio
				index++
			}
		}
		pos = c.next
	}
	return hasAudio, nil
}

// stream reads one strl. It reports true for the first audio stream when
// its format is one sd.cpp reads, and fills video for the first video
// stream.
func (a *aviReader) stream(start, end int64, index int, video *aviVideo, haveAudio bool) (bool, error) {
	var header, format []byte
	for pos := start; pos < end; {
		c, err := a.chunk(pos, end)
		if err != nil {
			return false, err
		}
		switch c.id {
		case "strh":
			if c.size < 56 || header != nil {
				return false, codecErr("the AVI file has a damaged stream header")
			}
			if header, err = a.at(c.data, 56); err != nil {
				return false, err
			}
		case "strf":
			if c.size >= 16 {
				if format, err = a.at(c.data, int(min(c.size, 40))); err != nil {
					return false, err
				}
			}
		}
		pos = c.next
	}
	if header == nil {
		return false, nil
	}
	switch string(header[:4]) {
	case "auds":
		if haveAudio {
			return false, nil
		}
		if len(format) < 16 || le32(header[28:]) != 0 {
			return false, codecErr("the AVI audio stream has a damaged header or does not start at 0")
		}
		enc, ch, rate := le16(format), le16(format[2:]), int(le32(format[4:]))
		block, bits := le16(format[12:]), le16(format[14:])
		pcm := enc == 1 && (bits == 8 || bits == 16 || bits == 24 || bits == 32)
		float := enc == 3 && bits == 32
		if ch < 1 || ch > 32 || rate < 1000 || rate > 384000 || !(pcm || float) || block != ch*(bits/8) {
			return false, codecErr("the AVI audio is not PCM or 32-bit float (format 0x%04x); send PCM audio or no audio", enc)
		}
		return true, nil
	case "vids":
		if video.index >= 0 {
			return false, nil
		}
		if len(format) < 40 || le32(format) < 40 {
			return false, codecErr("the AVI video stream has no bitmap header")
		}
		width := int64(le32(format[4:]))
		h := le32(format[8:])
		height := int64(h)
		if h&0x80000000 != 0 {
			height = int64(0x100000000 - uint64(h))
		}
		comp := string(format[16:20])
		jpeg := comp == "MJPG" || comp == "mjpg"
		rgb := le32(format[16:]) == 0
		if width < 1 || width > 16384 || height < 1 || height > 16384 || width*height > 64<<20 ||
			le16(format[12:]) != 1 || le16(format[14:]) != 24 || (!jpeg && !rgb) {
			name := comp
			if rgb {
				name = "RGB"
			}
			return false, codecErr("the AVI video is %s with %d bits per pixel; send MJPEG or 24-bit RGB AVI", printable(name), le16(format[14:]))
		}
		num, den := int(le32(header[24:])), int(le32(header[20:]))
		if num == 0 || den == 0 {
			return false, codecErr("the AVI file has no frame rate")
		}
		num, den = reduce(num, den)
		if num > 1_000_000 || den > 1_000_000 || le32(header[28:]) != 0 {
			return false, codecErr("the AVI frame rate is not supported, or the video does not start at 0")
		}
		*video = aviVideo{index: index, width: int(width), height: int(height), fpsNum: num, fpsDen: den}
	}
	return false, nil
}

// frames counts the video stream's frame chunks in a movi list (and its
// rec lists), as sd.cpp's read_avi_frames does.
func (a *aviReader) frames(start, end int64, stream, depth int) (int, error) {
	if depth > 8 {
		return 0, codecErr("the AVI lists are nested too deeply")
	}
	prefix := string([]byte{byte('0' + stream/10), byte('0' + stream%10)})
	n := 0
	for pos := start; pos < end; {
		c, err := a.chunk(pos, end)
		if err != nil {
			return 0, err
		}
		switch {
		case c.id == prefix+"db" || c.id == prefix+"dc":
			if c.size == 0 {
				return 0, codecErr("the AVI file has dropped (empty) frames, which are not supported")
			}
			n++
		case c.id == "LIST":
			typ, err := a.listType(c)
			if err != nil {
				return 0, err
			}
			if typ == "rec " {
				m, err := a.frames(c.data+4, c.data+c.size, stream, depth+1)
				if err != nil {
					return 0, err
				}
				n += m
			}
		}
		pos = c.next
	}
	return n, nil
}

// printable keeps a FourCC readable in a message.
func printable(s string) string {
	b := []byte(s)
	for i, c := range b {
		if c < 0x20 || c > 0x7e {
			b[i] = '?'
		}
	}
	return string(b)
}

// WriteRGBAVI writes an AVI 1.0 file with one uncompressed 24-bit RGB
// video stream, no audio and an idx1 index. frame fills frame i as
// top-down RGB, width × height × 3 bytes; the file stores it bottom-up
// BGR, as AVI expects.
func WriteRGBAVI(w io.Writer, width, height, frames, fps int, frame func(i int, rgb []byte)) error {
	if width < 1 || height < 1 || frames < 1 || fps < 1 {
		return codecErr("WriteRGBAVI: width, height, frames and fps must be positive")
	}
	stride := (width*3 + 3) &^ 3
	frameSize := stride * height
	const (
		avihSize = 56
		strhSize = 56
		strfSize = 40
	)
	strl := 4 + (8 + strhSize) + (8 + strfSize)
	hdrl := 4 + (8 + avihSize) + (8 + strl)
	movi := 4 + frames*(8+frameSize)
	idx1 := frames * 16
	riff := 4 + (8 + hdrl) + (8 + movi) + (8 + idx1)
	if int64(riff) > 1<<32-9 {
		return codecErr("WriteRGBAVI: the clip is larger than 4 GiB")
	}

	bw := bufio.NewWriterSize(w, 1<<20)
	u32 := func(v int) { _ = binary.Write(bw, binary.LittleEndian, uint32(v)) }
	u16 := func(v int) { _ = binary.Write(bw, binary.LittleEndian, uint16(v)) }
	tag := func(s string) { _, _ = bw.WriteString(s) }

	tag("RIFF")
	u32(riff)
	tag("AVI ")

	tag("LIST")
	u32(hdrl)
	tag("hdrl")
	tag("avih")
	u32(avihSize)
	u32(1_000_000 / fps) // microseconds per frame
	u32(frameSize * fps) // max bytes per second
	u32(0)               // padding granularity
	u32(0x10)            // AVIF_HASINDEX
	u32(frames)
	u32(0) // initial frames
	u32(1) // streams
	u32(frameSize)
	u32(width)
	u32(height)
	for range 4 {
		u32(0)
	}

	tag("LIST")
	u32(strl)
	tag("strl")
	tag("strh")
	u32(strhSize)
	tag("vids")
	u32(0) // handler: uncompressed
	u32(0) // flags
	u16(0) // priority
	u16(0) // language
	u32(0) // initial frames
	u32(1) // scale
	u32(fps)
	u32(0) // start
	u32(frames)
	u32(frameSize)
	u32(0xFFFFFFFF) // quality: default
	u32(0)          // sample size
	u16(0)
	u16(0)
	u16(width)
	u16(height)
	tag("strf")
	u32(strfSize)
	u32(strfSize)
	u32(width)
	u32(height) // positive: bottom-up
	u16(1)      // planes
	u16(24)     // bits per pixel
	u32(0)      // BI_RGB
	u32(frameSize)
	for range 4 {
		u32(0)
	}

	tag("LIST")
	u32(movi)
	tag("movi")
	rgb := make([]byte, width*height*3)
	row := make([]byte, stride)
	for i := range frames {
		clear(rgb)
		frame(i, rgb)
		tag("00db")
		u32(frameSize)
		for y := height - 1; y >= 0; y-- {
			src := rgb[y*width*3 : (y+1)*width*3]
			for x := range width {
				row[x*3], row[x*3+1], row[x*3+2] = src[x*3+2], src[x*3+1], src[x*3]
			}
			if _, err := bw.Write(row); err != nil {
				return err
			}
		}
	}

	tag("idx1")
	u32(idx1)
	off := 4 // offsets count from the movi list's type field
	for range frames {
		tag("00db")
		u32(0x10) // AVIIF_KEYFRAME
		u32(off)
		u32(frameSize)
		off += 8 + frameSize
	}
	return bw.Flush()
}
