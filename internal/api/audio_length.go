package api

import (
	"bytes"
	"encoding/binary"
	"errors"
	"math"
)

// audioSeconds is the length of a WAV or MP3 file, rounded to the
// millisecond: what a music job reports as its duration, since
// ace-server's song can be shorter or longer than the one asked for.
func audioSeconds(b []byte) (float64, error) {
	var s float64
	var err error
	if len(b) >= 12 && string(b[:4]) == "RIFF" && string(b[8:12]) == "WAVE" {
		s, err = wavSeconds(b)
	} else {
		s, err = mp3Seconds(b)
	}
	if err != nil {
		return 0, err
	}
	return math.Round(s*1000) / 1000, nil
}

// wavSeconds is the size of the data chunk over the fmt chunk's byte rate.
func wavSeconds(b []byte) (float64, error) {
	var byteRate uint32
	for p := 12; p+8 <= len(b); {
		id, size := string(b[p:p+4]), int(binary.LittleEndian.Uint32(b[p+4:p+8]))
		body := p + 8
		switch id {
		case "fmt ":
			if size < 16 || body+16 > len(b) {
				return 0, errors.New("short WAV fmt chunk")
			}
			byteRate = binary.LittleEndian.Uint32(b[body+8 : body+12])
		case "data":
			if byteRate == 0 {
				return 0, errors.New("WAV data before a valid fmt chunk")
			}
			// A streamed WAV can leave the size unset (0 or 0xFFFFFFFF).
			if size == 0 || body+size > len(b) {
				size = len(b) - body
			}
			return float64(size) / float64(byteRate), nil
		}
		p = body + size + size&1
	}
	return 0, errors.New("no WAV data chunk")
}

// mp3Bitrates are the bitrate tables in kbit/s, by [MPEG-1][layer-1].
var mp3Bitrates = [2][3][16]int{
	{ // MPEG-2 and 2.5
		{0, 32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256, 0},
		{0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0},
		{0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0},
	},
	{ // MPEG-1
		{0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 0},
		{0, 32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 0},
		{0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0},
	},
}

// mp3Seconds walks the MP3 frames and adds up their samples. It skips an
// ID3v2 tag at the start and a Xing/Info frame, and stops at the first
// bytes that are not a frame, such as an ID3v1 tag at the end.
func mp3Seconds(b []byte) (float64, error) {
	p := 0
	if len(b) >= 10 && string(b[:3]) == "ID3" {
		p = 10 + (int(b[6]&0x7f)<<21 | int(b[7]&0x7f)<<14 | int(b[8]&0x7f)<<7 | int(b[9]&0x7f))
		if b[5]&0x10 != 0 {
			p += 10 // footer
		}
	}
	frames, seconds := 0, 0.0
	for p+4 <= len(b) {
		h := b[p : p+4]
		if h[0] != 0xff || h[1]&0xe0 != 0xe0 {
			break
		}
		version, layer := h[1]>>3&3, 4-int(h[1]>>1&3) // version 3 = MPEG-1, 2 = MPEG-2, 0 = MPEG-2.5
		brIdx, srIdx, pad := h[2]>>4, h[2]>>2&3, int(h[2]>>1&1)
		if version == 1 || layer == 4 || srIdx == 3 {
			break
		}
		mpeg1 := 0
		if version == 3 {
			mpeg1 = 1
		}
		kbps := mp3Bitrates[mpeg1][layer-1][brIdx]
		if kbps == 0 {
			break // free-format or bad bitrate
		}
		rate := [3]int{44100, 48000, 32000}[srIdx] >> map[byte]int{3: 0, 2: 1, 0: 2}[version]
		var size, samples int
		switch {
		case layer == 1:
			size, samples = (12*kbps*1000/rate+pad)*4, 384
		case layer == 3 && mpeg1 == 0:
			size, samples = 72*kbps*1000/rate+pad, 576
		default:
			size, samples = 144*kbps*1000/rate+pad, 1152
		}
		if frames == 0 && isXingFrame(b[p:min(p+size, len(b))], mpeg1 == 1, h[3]>>6 == 3) {
			p += size
			continue
		}
		frames++
		seconds += float64(samples) / float64(rate)
		p += size
	}
	if frames == 0 {
		return 0, errors.New("no MP3 frames")
	}
	return seconds, nil
}

// isXingFrame says if a first frame is a Xing or Info header, which holds
// no audio. Its tag sits after the side information.
func isXingFrame(frame []byte, mpeg1, mono bool) bool {
	off := 4 + 17
	switch {
	case mpeg1 && !mono:
		off = 4 + 32
	case !mpeg1 && mono:
		off = 4 + 9
	}
	if off+4 > len(frame) {
		return false
	}
	tag := frame[off : off+4]
	return bytes.Equal(tag, []byte("Xing")) || bytes.Equal(tag, []byte("Info"))
}
