package audiofile

import (
	"bytes"
	"encoding/binary"
)

// WAV format codes: PCM, IEEE float, and WAVE_FORMAT_EXTENSIBLE (whose
// real format is in its sub-format GUID).
const (
	wavPCM        = 1
	wavFloat      = 3
	wavExtensible = 0xFFFE
)

// wavSeconds checks a WAV the way sd-server's load_wav_from_memory reads
// one (stable-diffusion.cpp examples/common/media_io.cpp) and returns its
// length. It takes PCM at 8, 16, 24 or 32 bits and float at 32 or 64 bits.
func wavSeconds(b []byte) (float64, error) {
	if len(b) < 12 {
		return 0, formatErr("the WAV is too short")
	}
	var (
		format, channels, bits uint16
		rate                   uint32
		haveFmt                bool
		dataBytes              = -1
	)
	for pos := 12; pos+8 <= len(b); {
		id := string(b[pos : pos+4])
		size := int(binary.LittleEndian.Uint32(b[pos+4 : pos+8]))
		body := pos + 8
		if size < 0 || body+size > len(b) {
			if id == "data" {
				// Streamed WAVs can give a data size past the end; count
				// what is there.
				size = len(b) - body
			} else {
				return 0, formatErr("the WAV's %q chunk runs past the end", id)
			}
		}
		switch id {
		case "fmt ":
			if size < 16 {
				return 0, formatErr("the WAV's format chunk is too short")
			}
			format = binary.LittleEndian.Uint16(b[body:])
			channels = binary.LittleEndian.Uint16(b[body+2:])
			rate = binary.LittleEndian.Uint32(b[body+4:])
			bits = binary.LittleEndian.Uint16(b[body+14:])
			if format == wavExtensible && size >= 40 {
				format = binary.LittleEndian.Uint16(b[body+24:])
			}
			haveFmt = true
		case "data":
			dataBytes = size
		}
		pos = body + size + size%2
	}
	switch {
	case !haveFmt:
		return 0, formatErr("the WAV has no format chunk")
	case channels == 0:
		return 0, formatErr("the WAV has no channels")
	case rate == 0:
		return 0, formatErr("the WAV has no sample rate")
	case !(format == wavPCM && (bits == 8 || bits == 16 || bits == 24 || bits == 32)) &&
		!(format == wavFloat && (bits == 32 || bits == 64)):
		return 0, formatErr("the WAV's encoding (format %d, %d bits) is not 8/16/24/32-bit PCM or 32/64-bit float", format, bits)
	case dataBytes <= 0:
		return 0, formatErr("the WAV has no data")
	}
	frame := int(channels) * int(bits) / 8
	return float64(dataBytes/frame) / float64(rate), nil
}

// WriteWAV returns a 16-bit PCM WAV of interleaved samples.
func WriteWAV(samples []int16, rate, channels int) []byte {
	var buf bytes.Buffer
	dataBytes := len(samples) * 2
	buf.Grow(44 + dataBytes)
	w := func(v any) { _ = binary.Write(&buf, binary.LittleEndian, v) }
	buf.WriteString("RIFF")
	w(uint32(36 + dataBytes))
	buf.WriteString("WAVEfmt ")
	w(uint32(16))
	w(uint16(wavPCM))
	w(uint16(channels))
	w(uint32(rate))
	w(uint32(rate * channels * 2))
	w(uint16(channels * 2))
	w(uint16(16))
	buf.WriteString("data")
	w(uint32(dataBytes))
	w(samples)
	return buf.Bytes()
}
