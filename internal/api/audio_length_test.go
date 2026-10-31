package api

import (
	"bytes"
	"encoding/binary"
	"testing"
)

// testWAV is a 16-bit stereo 48 kHz WAV of n frames, with an extra chunk
// before the data to check that chunks are walked.
func testWAV(n int) []byte {
	var b bytes.Buffer
	data := n * 4
	b.WriteString("RIFF")
	binary.Write(&b, binary.LittleEndian, uint32(4+8+16+8+2+8+data))
	b.WriteString("WAVEfmt ")
	for _, v := range []any{uint32(16), uint16(1), uint16(2), uint32(48000), uint32(48000 * 4), uint16(4), uint16(16)} {
		binary.Write(&b, binary.LittleEndian, v)
	}
	b.WriteString("LIST")
	binary.Write(&b, binary.LittleEndian, uint32(1))
	b.Write([]byte{0, 0}) // odd size and its pad byte
	b.WriteString("data")
	binary.Write(&b, binary.LittleEndian, uint32(data))
	b.Write(make([]byte, data))
	return b.Bytes()
}

// testMP3 is n frames of MPEG-1 Layer III, 128 kbit/s, 48 kHz: 384 bytes
// and 1152 samples (24 ms) each, as ace-server writes them.
func testMP3(n int) []byte {
	frame := make([]byte, 384)
	copy(frame, []byte{0xff, 0xfb, 0x94, 0x64})
	return bytes.Repeat(frame, n)
}

func TestAudioSeconds(t *testing.T) {
	xing := make([]byte, 384)
	copy(xing, []byte{0xff, 0xfb, 0x94, 0x64})
	copy(xing[4+32:], "Info")
	id3 := append([]byte("ID3\x04\x00\x00\x00\x00\x01\x00"), make([]byte, 128)...) // 128-byte tag body

	for _, tc := range []struct {
		name string
		in   []byte
		want float64
	}{
		{"wav", testWAV(48000 * 3 / 2), 1.5},
		// 927 frames is the 22.248 s song from issue #399.
		{"mp3", testMP3(927), 22.248},
		{"mp3 with ID3v2 and Info frame", append(append(id3, xing...), testMP3(50)...), 1.2},
		{"mp3 with ID3v1 at the end", append(testMP3(50), append([]byte("TAG"), make([]byte, 125)...)...), 1.2},
	} {
		got, err := audioSeconds(tc.in)
		if err != nil || got != tc.want {
			t.Errorf("%s: audioSeconds = %v, %v; want %v", tc.name, got, err, tc.want)
		}
	}
	for _, bad := range [][]byte{nil, []byte("MP3DATA"), []byte("RIFF\x00\x00\x00\x00WAVE")} {
		if got, err := audioSeconds(bad); err == nil {
			t.Errorf("audioSeconds(%q) = %v; want an error", bad, got)
		}
	}
}
