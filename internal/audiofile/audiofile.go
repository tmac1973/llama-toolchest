// Package audiofile reads the audio a client uploads (WAV, MP3 or FLAC)
// and turns it into a WAV that stable-diffusion.cpp's sd-server can read:
// its ref_audios field takes WAV only (examples/common/common.cpp,
// load_wav_from_memory).
package audiofile

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"

	"github.com/hajimehoshi/go-mp3"
	"github.com/mewkiz/flac"
)

// Format is an audio file format Decode knows.
type Format string

const (
	WAV  Format = "wav"
	MP3  Format = "mp3"
	FLAC Format = "flac"
)

// ErrFormat is the error for audio that is not a WAV, MP3 or FLAC file, or
// one that cannot be read. Every Decode error other than ErrTooLong wraps it,
// so its text always starts "audio must be WAV, MP3 or FLAC".
var ErrFormat = errors.New("audio must be WAV, MP3 or FLAC")

// ErrTooLong is the error for audio longer than MaxSeconds.
var ErrTooLong = errors.New("audio is longer than 10 minutes")

// MaxSeconds is the longest audio Decode takes. It guards memory: a
// decoded track is kept whole. The video limits are far lower.
const MaxSeconds = 600

// Audio is a decoded upload.
type Audio struct {
	// WAV is the audio as a WAV file: the upload itself for a WAV, else a
	// 16-bit PCM WAV at the source's sample rate and channel count.
	WAV []byte
	// Format is the upload's format.
	Format Format
	// Seconds is the audio's length.
	Seconds float64
}

func formatErr(format string, args ...any) error {
	return fmt.Errorf("%w: "+format, append([]any{ErrFormat}, args...)...)
}

// Detect returns the format of b from its first bytes, or "" when it is
// none Decode knows.
func Detect(b []byte) Format {
	switch {
	case len(b) >= 12 && string(b[0:4]) == "RIFF" && string(b[8:12]) == "WAVE":
		return WAV
	case len(b) >= 4 && string(b[0:4]) == "fLaC":
		return FLAC
	case len(b) >= 3 && string(b[0:3]) == "ID3":
		return MP3
	case len(b) >= 2 && b[0] == 0xFF && b[1]&0xE0 == 0xE0:
		return MP3
	}
	return ""
}

// Decode reads an uploaded audio file. A WAV is checked and kept as it is;
// an MP3 or a FLAC is decoded to a 16-bit PCM WAV.
func Decode(b []byte) (Audio, error) {
	switch Detect(b) {
	case WAV:
		seconds, err := wavSeconds(b)
		if err != nil {
			return Audio{}, err
		}
		if seconds > MaxSeconds {
			return Audio{}, ErrTooLong
		}
		return Audio{WAV: b, Format: WAV, Seconds: seconds}, nil
	case MP3:
		return decodeMP3(b)
	case FLAC:
		return decodeFLAC(b)
	}
	if len(b) == 0 {
		return Audio{}, formatErr("the audio is empty")
	}
	return Audio{}, formatErr("the file is none of these")
}

// skipID3v2 returns b after a leading ID3v2 tag, if it has one.
func skipID3v2(b []byte) []byte {
	if len(b) < 10 || string(b[0:3]) != "ID3" {
		return b
	}
	size := int(b[6]&0x7F)<<21 | int(b[7]&0x7F)<<14 | int(b[8]&0x7F)<<7 | int(b[9]&0x7F)
	end := 10 + size
	if b[5]&0x10 != 0 { // footer present
		end += 10
	}
	if end > len(b) {
		return nil
	}
	return b[end:]
}

func decodeMP3(b []byte) (Audio, error) {
	d, err := mp3.NewDecoder(bytes.NewReader(skipID3v2(b)))
	if err != nil {
		return Audio{}, formatErr("the MP3 could not be decoded: %v", err)
	}
	rate := d.SampleRate()
	if rate <= 0 {
		return Audio{}, formatErr("the MP3 could not be decoded: no sample rate")
	}
	// go-mp3 gives 16-bit little-endian stereo.
	limit := (MaxSeconds*rate + 1) * 4
	var pcm bytes.Buffer
	n, err := io.Copy(&pcm, io.LimitReader(d, int64(limit)+1))
	if err != nil {
		return Audio{}, formatErr("the MP3 could not be decoded: %v", err)
	}
	if n > int64(limit) {
		return Audio{}, ErrTooLong
	}
	frames := pcm.Len() / 4
	if frames == 0 {
		return Audio{}, formatErr("the MP3 could not be decoded: no audio in it")
	}
	samples := make([]int16, frames*2)
	if err := binary.Read(bytes.NewReader(pcm.Bytes()[:frames*4]), binary.LittleEndian, samples); err != nil {
		return Audio{}, formatErr("the MP3 could not be decoded: %v", err)
	}
	return Audio{WAV: WriteWAV(samples, rate, 2), Format: MP3, Seconds: float64(frames) / float64(rate)}, nil
}

func decodeFLAC(b []byte) (Audio, error) {
	stream, err := flac.New(bytes.NewReader(b))
	if err != nil {
		return Audio{}, formatErr("the FLAC could not be decoded: %v", err)
	}
	defer stream.Close()
	rate, channels, bits := int(stream.Info.SampleRate), int(stream.Info.NChannels), int(stream.Info.BitsPerSample)
	if rate <= 0 || channels <= 0 || bits <= 0 || bits > 32 {
		return Audio{}, formatErr("the FLAC could not be decoded: bad stream header")
	}
	if stream.Info.NSamples > 0 && float64(stream.Info.NSamples)/float64(rate) > MaxSeconds {
		return Audio{}, ErrTooLong
	}
	shift := func(v int32) int16 {
		if bits > 16 {
			return int16(v >> (bits - 16))
		}
		return int16(v << (16 - bits))
	}
	var samples []int16
	for {
		f, err := stream.ParseNext()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return Audio{}, formatErr("the FLAC could not be decoded: %v", err)
		}
		if len(f.Subframes) != channels {
			return Audio{}, formatErr("the FLAC could not be decoded: a frame has %d channels, not %d", len(f.Subframes), channels)
		}
		n := f.Subframes[0].NSamples
		for i := range n {
			for ch := range channels {
				samples = append(samples, shift(f.Subframes[ch].Samples[i]))
			}
		}
		if len(samples)/channels > MaxSeconds*rate {
			return Audio{}, ErrTooLong
		}
	}
	frames := len(samples) / channels
	if frames == 0 {
		return Audio{}, formatErr("the FLAC could not be decoded: no audio in it")
	}
	return Audio{WAV: WriteWAV(samples, rate, channels), Format: FLAC, Seconds: float64(frames) / float64(rate)}, nil
}
