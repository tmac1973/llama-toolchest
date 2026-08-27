package models

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

// pleTensor describes one entry for writePLEGGUF.
type pleTensor struct {
	name   string
	offset uint64
}

// writePLEGGUF builds a minimal GGUF v3 file with a tensor-info block, so
// the size-by-offset arithmetic in scanPLETensorBytes can be exercised
// without a multi-gigabyte fixture. dataBytes is how many bytes of tensor
// data follow the (aligned) end of the tensor-info block.
func writePLEGGUF(t *testing.T, alignment uint32, tensors []pleTensor, dataBytes int) string {
	t.Helper()
	var buf bytes.Buffer
	writeStr := func(s string) {
		binary.Write(&buf, binary.LittleEndian, uint64(len(s)))
		buf.WriteString(s)
	}

	buf.WriteString("GGUF")
	binary.Write(&buf, binary.LittleEndian, uint32(3))
	binary.Write(&buf, binary.LittleEndian, uint64(len(tensors)))
	kvCount := 1
	if alignment != 0 {
		kvCount = 2
	}
	binary.Write(&buf, binary.LittleEndian, uint64(kvCount))

	writeStr("general.architecture")
	binary.Write(&buf, binary.LittleEndian, ggufTypeString)
	writeStr("testarch")
	if alignment != 0 {
		writeStr("general.alignment")
		binary.Write(&buf, binary.LittleEndian, ggufTypeUint32)
		binary.Write(&buf, binary.LittleEndian, alignment)
	}

	// Tensor info: name, n_dims, dims..., ggml type, offset.
	for _, tn := range tensors {
		writeStr(tn.name)
		binary.Write(&buf, binary.LittleEndian, uint32(2))
		binary.Write(&buf, binary.LittleEndian, uint64(8))
		binary.Write(&buf, binary.LittleEndian, uint64(16))
		binary.Write(&buf, binary.LittleEndian, uint32(0)) // F32
		binary.Write(&buf, binary.LittleEndian, tn.offset)
	}

	// Pad to the alignment, then the data region.
	align := int(alignment)
	if align == 0 {
		align = 32
	}
	for buf.Len()%align != 0 {
		buf.WriteByte(0)
	}
	buf.Write(make([]byte, dataBytes))

	path := filepath.Join(t.TempDir(), "model.gguf")
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestPLETensorSizeFromOffsets(t *testing.T) {
	tests := []struct {
		name    string
		tensors []pleTensor
		data    int
		want    int64
	}{
		{
			// Bounded by the tensor that follows it in the data region.
			name: "followed by another tensor",
			tensors: []pleTensor{
				{"token_embd.weight", 0},
				{pleTensorName, 1024},
				{"output.weight", 5120},
			},
			data: 8192,
			want: 4096,
		},
		{
			// Last in the data region: runs to the end of the file.
			name:    "last tensor in the file",
			tensors: []pleTensor{{"token_embd.weight", 0}, {pleTensorName, 2048}},
			data:    6144,
			want:    4096,
		},
		{
			// Declaration order must not matter — only the offsets do.
			name: "declared out of offset order",
			tensors: []pleTensor{
				{"output.weight", 8192},
				{pleTensorName, 1024},
				{"token_embd.weight", 0},
			},
			data: 16384,
			want: 7168,
		},
		{
			name:    "model without a PLE table",
			tensors: []pleTensor{{"token_embd.weight", 0}, {"output.weight", 1024}},
			data:    2048,
			want:    0,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			meta, err := ParseGGUFMeta(writePLEGGUF(t, 32, tt.tensors, tt.data))
			if err != nil {
				t.Fatalf("parse: %v", err)
			}
			if meta.PLEBytes != tt.want {
				t.Errorf("PLEBytes = %d, want %d", meta.PLEBytes, tt.want)
			}
		})
	}
}

// A non-default general.alignment shifts where the data region starts,
// which only changes the answer for a table that runs to end-of-file.
func TestPLETensorSizeHonorsAlignment(t *testing.T) {
	tensors := []pleTensor{{pleTensorName, 0}}
	for _, align := range []uint32{32, 64, 4096} {
		meta, err := ParseGGUFMeta(writePLEGGUF(t, align, tensors, 4096))
		if err != nil {
			t.Fatalf("align %d: parse: %v", align, err)
		}
		if meta.PLEBytes != 4096 {
			t.Errorf("align %d: PLEBytes = %d, want 4096", align, meta.PLEBytes)
		}
	}
}

// A file that ends inside the tensor-info block must not fail the parse:
// the metadata gathered before it is still usable.
func TestPLETensorScanToleratesTruncation(t *testing.T) {
	path := writePLEGGUF(t, 32, []pleTensor{{pleTensorName, 0}}, 4096)
	full, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	truncated := filepath.Join(t.TempDir(), "truncated.gguf")
	// Keep the header and metadata, cut into the tensor-info block.
	if err := os.WriteFile(truncated, full[:len(full)-4200], 0o644); err != nil {
		t.Fatal(err)
	}
	meta, err := ParseGGUFMeta(truncated)
	if err != nil {
		t.Fatalf("parse of truncated file failed: %v", err)
	}
	if meta.Architecture != "testarch" {
		t.Errorf("Architecture = %q, want testarch (metadata should survive)", meta.Architecture)
	}
	if meta.PLEBytes != 0 {
		t.Errorf("PLEBytes = %d, want 0 on a truncated tensor block", meta.PLEBytes)
	}
}
