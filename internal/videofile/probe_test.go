package videofile

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func probeFile(t *testing.T, name string) (Info, error) {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatal(err)
	}
	return Probe(bytes.NewReader(b), int64(len(b)))
}

// The fixtures were made with ffmpeg n9.0.2 (testsrc and sine sources, 64×48).
func TestProbeFixtures(t *testing.T) {
	tests := []struct {
		file string
		want Info
		err  error  // errors.Is target
		msg  string // part of the error text
	}{
		{file: "mjpeg-pcm-64x48-8f.avi", want: Info{Format: "avi", Width: 64, Height: 48, Frames: 8, FPSNum: 16, FPSDen: 1, HasAudio: true}},
		{file: "mjpeg-aac.avi", err: ErrCodec, msg: "not PCM or 32-bit float"},
		{file: "anim-64x48-6f.webp", want: Info{Format: "webp", Width: 64, Height: 48, Frames: 6, FPSNum: 10, FPSDen: 1}},
		{file: "still-64x48.webp", want: Info{Format: "webp", Width: 64, Height: 48, Frames: 1}},
		{file: "vp8-key-64x48-9f.webm", want: Info{Format: "webm", Width: 64, Height: 48, Frames: 9, FPSNum: 16, FPSDen: 1}},
		{file: "vp8-key-pcm.mkv", want: Info{Format: "webm", Width: 64, Height: 48, Frames: 9, FPSNum: 16, FPSDen: 1, HasAudio: true}},
		{file: "vp8-key-opus.webm", err: ErrCodec, msg: "must be PCM16"},
		{file: "vp8-interframe.webm", err: ErrCodec, msg: "independent VP8 frames"},
		{file: "vp9.webm", err: ErrCodec, msg: "V_VP9"},
		{file: "tiny.mp4", err: ErrMP4, msg: "MP4 is not supported"},
	}
	for _, tt := range tests {
		t.Run(tt.file, func(t *testing.T) {
			got, err := probeFile(t, tt.file)
			if tt.err != nil {
				if !errors.Is(err, tt.err) || !strings.Contains(err.Error(), tt.msg) {
					t.Fatalf("err = %v, want %v containing %q", err, tt.err, tt.msg)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if got != tt.want {
				t.Errorf("got %+v, want %+v", got, tt.want)
			}
		})
	}
}

func TestProbeUnknown(t *testing.T) {
	for _, b := range [][]byte{[]byte("hello, this is text"), {}, {0, 0, 0}} {
		if _, err := Probe(bytes.NewReader(b), int64(len(b))); err == nil {
			t.Errorf("Probe(%q) = nil error", b)
		}
	}
	b := []byte("GIF89a and more bytes")
	if _, err := Probe(bytes.NewReader(b), int64(len(b))); !errors.Is(err, ErrUnknownFormat) {
		t.Errorf("err = %v, want ErrUnknownFormat", err)
	}
}

func TestWriteRGBAVIRoundTrip(t *testing.T) {
	var buf bytes.Buffer
	if err := WriteRGBAVI(&buf, 64, 48, 5, 16, func(i int, rgb []byte) {
		for p := range rgb {
			rgb[p] = byte(i*40 + p)
		}
	}); err != nil {
		t.Fatal(err)
	}
	got, err := Probe(bytes.NewReader(buf.Bytes()), int64(buf.Len()))
	if err != nil {
		t.Fatal(err)
	}
	want := Info{Format: "avi", Width: 64, Height: 48, Frames: 5, FPSNum: 16, FPSDen: 1}
	if got != want {
		t.Errorf("got %+v, want %+v", got, want)
	}
	// An odd width needs row padding to 4 bytes.
	buf.Reset()
	if err := WriteRGBAVI(&buf, 5, 3, 2, 24, func(int, []byte) {}); err != nil {
		t.Fatal(err)
	}
	if got, err := Probe(bytes.NewReader(buf.Bytes()), int64(buf.Len())); err != nil || got.Width != 5 || got.Frames != 2 {
		t.Errorf("odd width: got %+v, %v", got, err)
	}
}

// TestWriteRGBAVIPixels checks that the file holds bottom-up BGR rows.
func TestWriteRGBAVIPixels(t *testing.T) {
	var buf bytes.Buffer
	if err := WriteRGBAVI(&buf, 2, 2, 1, 1, func(_ int, rgb []byte) {
		copy(rgb, []byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}) // top row, then bottom row
	}); err != nil {
		t.Fatal(err)
	}
	i := bytes.Index(buf.Bytes(), []byte("00db"))
	frame := buf.Bytes()[i+8 : i+8+16] // stride 8 (6 rounded up to 4)
	want := []byte{9, 8, 7, 12, 11, 10, 0, 0, 3, 2, 1, 6, 5, 4, 0, 0}
	if !bytes.Equal(frame, want) {
		t.Errorf("frame bytes = %v, want %v", frame, want)
	}
}

func TestProbeOpenDML(t *testing.T) {
	var buf bytes.Buffer
	if err := WriteRGBAVI(&buf, 16, 16, 2, 16, func(int, []byte) {}); err != nil {
		t.Fatal(err)
	}
	buf.WriteString("RIFF\x04\x00\x00\x00AVIX")
	_, err := Probe(bytes.NewReader(buf.Bytes()), int64(buf.Len()))
	if !errors.Is(err, ErrCodec) || !strings.Contains(err.Error(), "OpenDML") {
		t.Errorf("err = %v, want the OpenDML error", err)
	}
}

func TestDecodedBytes(t *testing.T) {
	if got := (Info{Width: 1280, Height: 720, Frames: 120}).DecodedBytes(); got != 331_776_000 {
		t.Errorf("DecodedBytes = %d", got)
	}
}

func TestRate(t *testing.T) {
	for _, tt := range []struct {
		fps      float64
		num, den int
	}{{16, 16, 1}, {24.0000001, 24, 1}, {29.97002997, 30000, 1001}, {12.5, 25, 2}, {0, 0, 0}} {
		if n, d := rate(tt.fps); n != tt.num || d != tt.den {
			t.Errorf("rate(%v) = %d/%d, want %d/%d", tt.fps, n, d, tt.num, tt.den)
		}
	}
}

// TestProbeTruncated cuts every fixture at several points: Probe returns
// an error and never panics.
func TestProbeTruncated(t *testing.T) {
	files := fixtures(t)
	for _, f := range files {
		b, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		for i := 1; i <= 10; i++ {
			n := len(b) * i / 11
			if _, err := Probe(bytes.NewReader(b[:n]), int64(n)); err == nil {
				t.Errorf("%s cut at %d: no error", f, n)
			}
		}
	}
}

func FuzzProbe(f *testing.F) {
	files := fixtures(f)
	for _, name := range files {
		b, err := os.ReadFile(name)
		if err != nil {
			f.Fatal(err)
		}
		f.Add(b)
	}
	var buf bytes.Buffer
	_ = WriteRGBAVI(&buf, 8, 8, 2, 16, func(int, []byte) {})
	f.Add(buf.Bytes())
	f.Fuzz(func(t *testing.T, b []byte) {
		_, _ = Probe(bytes.NewReader(b), int64(len(b)))
	})
}

// fixtures lists the test clips (every testdata file but the README).
func fixtures(tb testing.TB) []string {
	all, err := filepath.Glob("testdata/*")
	if err != nil {
		tb.Fatal(err)
	}
	var out []string
	for _, f := range all {
		if !strings.HasSuffix(f, ".md") {
			out = append(out, f)
		}
	}
	return out
}
