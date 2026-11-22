package audiofile

import (
	"bytes"
	"encoding/binary"
	"errors"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func readTestdata(t *testing.T, name string) []byte {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func sine(seconds float64, rate int) []int16 {
	n := int(seconds * float64(rate))
	s := make([]int16, n)
	for i := range s {
		s[i] = int16(0.3 * math.MaxInt16 * math.Sin(2*math.Pi*220*float64(i)/float64(rate)))
	}
	return s
}

func TestDecodeTestdata(t *testing.T) {
	for _, tc := range []struct {
		file   string
		format Format
	}{
		{"tone.wav", WAV},
		{"tone.mp3", MP3},
		{"tone.flac", FLAC},
	} {
		t.Run(tc.file, func(t *testing.T) {
			in := readTestdata(t, tc.file)
			a, err := Decode(in)
			if err != nil {
				t.Fatal(err)
			}
			if a.Format != tc.format {
				t.Errorf("format %q, want %q", a.Format, tc.format)
			}
			// MP3 encoders add a little delay and padding (about 0.07 s
			// for this file).
			if math.Abs(a.Seconds-1) > 0.1 {
				t.Errorf("%.3f s, want about 1", a.Seconds)
			}
			if tc.format == WAV && !bytes.Equal(a.WAV, in) {
				t.Error("a WAV must pass through unchanged")
			}
			again, err := Decode(a.WAV)
			if err != nil || again.Format != WAV {
				t.Fatalf("the output does not read back as a WAV: %v, %q", err, again.Format)
			}
			if math.Abs(again.Seconds-a.Seconds) > 0.001 {
				t.Errorf("WAV out is %.3f s, decode said %.3f s", again.Seconds, a.Seconds)
			}
		})
	}
}

func TestDecodeRefuses(t *testing.T) {
	for name, in := range map[string][]byte{
		"ogg":    readTestdata(t, "tone.ogg"),
		"empty":  nil,
		"random": []byte("this is not audio at all, just some text"),
		"broken": readTestdata(t, "broken.mp3"),
	} {
		t.Run(name, func(t *testing.T) {
			_, err := Decode(in)
			if !errors.Is(err, ErrFormat) {
				t.Fatalf("got %v, want ErrFormat", err)
			}
			if !strings.HasPrefix(err.Error(), "audio must be WAV, MP3 or FLAC") {
				t.Errorf("message %q", err)
			}
		})
	}
}

// wavHeader builds a WAV with the given format fields and data size.
func wavHeader(format, channels uint16, rate uint32, bits uint16, data int) []byte {
	var b bytes.Buffer
	w := func(v any) { _ = binary.Write(&b, binary.LittleEndian, v) }
	b.WriteString("RIFF")
	w(uint32(36 + data))
	b.WriteString("WAVEfmt ")
	w(uint32(16))
	w(format)
	w(channels)
	w(rate)
	w(rate * uint32(channels) * uint32(bits) / 8)
	w(channels * bits / 8)
	w(bits)
	if data >= 0 {
		b.WriteString("data")
		w(uint32(data))
		b.Write(make([]byte, data))
	}
	return b.Bytes()
}

func TestWAVHeaders(t *testing.T) {
	a, err := Decode(wavHeader(wavFloat, 1, 16000, 32, 16000*4))
	if err != nil || math.Abs(a.Seconds-1) > 1e-9 {
		t.Errorf("float32 WAV: %v, %.3f s", err, a.Seconds)
	}
	for name, in := range map[string][]byte{
		"12-bit":     wavHeader(wavPCM, 1, 16000, 12, 100),
		"no data":    wavHeader(wavPCM, 1, 16000, 16, -1),
		"0 channels": wavHeader(wavPCM, 0, 16000, 16, 100),
		"0 rate":     wavHeader(wavPCM, 1, 0, 16, 100),
	} {
		if _, err := Decode(in); !errors.Is(err, ErrFormat) {
			t.Errorf("%s: got %v, want ErrFormat", name, err)
		}
	}
}

func TestWriteWAVRoundTrip(t *testing.T) {
	a, err := Decode(WriteWAV(sine(1, 16000), 16000, 1))
	if err != nil || a.Format != WAV || math.Abs(a.Seconds-1) > 1e-9 {
		t.Fatalf("%v, %q, %.3f s", err, a.Format, a.Seconds)
	}
}

func TestDecodeTooLong(t *testing.T) {
	long := WriteWAV(make([]int16, 8000*(MaxSeconds+1)), 8000, 1)
	if _, err := Decode(long); !errors.Is(err, ErrTooLong) {
		t.Fatalf("got %v, want ErrTooLong", err)
	}
}

func TestDetect(t *testing.T) {
	for _, tc := range []struct {
		in   []byte
		want Format
	}{
		{readTestdata(t, "tone.wav"), WAV},
		{readTestdata(t, "tone.mp3"), MP3},
		{[]byte{0xFF, 0xFB, 0x90, 0x00}, MP3},
		{readTestdata(t, "tone.flac"), FLAC},
		{readTestdata(t, "tone.ogg"), ""},
	} {
		if got := Detect(tc.in); got != tc.want {
			t.Errorf("Detect(%q...) = %q, want %q", tc.in[:4], got, tc.want)
		}
	}
}
