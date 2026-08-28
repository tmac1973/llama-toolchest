package models

import (
	"bytes"
	"encoding/binary"
	"fmt"
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

// writeSplitGGUF writes a shard set into dir: shard 1 carries the
// metadata and no tensors at all, later shards carry tensors and no
// metadata. That is the shape publishers actually ship, and the shape
// that made the per-layer embedding control never appear.
func writeSplitGGUF(t *testing.T, dir, base string, total int, pleShard int, pleSize int64) string {
	t.Helper()
	for i := 1; i <= total; i++ {
		var buf bytes.Buffer
		writeStr := func(s string) {
			binary.Write(&buf, binary.LittleEndian, uint64(len(s)))
			buf.WriteString(s)
		}
		var tensors []pleTensor
		kv := 0
		if i == 1 {
			kv = 1 // metadata shard: architecture only, zero tensors
		} else if i == pleShard {
			tensors = []pleTensor{
				{"token_embd.weight", 0},
				{pleTensorName, 4096},
				{"output.weight", uint64(4096 + pleSize)},
			}
		} else {
			tensors = []pleTensor{{fmt.Sprintf("blk.%d.attn_q.weight", i), 0}}
		}

		buf.WriteString("GGUF")
		binary.Write(&buf, binary.LittleEndian, uint32(3))
		binary.Write(&buf, binary.LittleEndian, uint64(len(tensors)))
		binary.Write(&buf, binary.LittleEndian, uint64(kv))
		if kv > 0 {
			writeStr("general.architecture")
			binary.Write(&buf, binary.LittleEndian, ggufTypeString)
			writeStr("qwen4exp")
		}
		for _, tn := range tensors {
			writeStr(tn.name)
			binary.Write(&buf, binary.LittleEndian, uint32(1))
			binary.Write(&buf, binary.LittleEndian, uint64(8))
			binary.Write(&buf, binary.LittleEndian, uint32(0))
			binary.Write(&buf, binary.LittleEndian, tn.offset)
		}
		for buf.Len()%32 != 0 {
			buf.WriteByte(0)
		}
		// Enough trailing data for the last tensor in the shard to have a
		// length; the PLE table's size comes from the gap to the tensor
		// after it, so this only has to be non-empty.
		buf.Write(make([]byte, 8192))

		name := fmt.Sprintf("%s-%05d-of-%05d.gguf", base, i, total)
		path := filepath.Join(dir, name)
		if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
			t.Fatal(err)
		}
		// The table's size is the gap between its offset and the next
		// tensor's, so the shard has to actually be that long or the
		// arithmetic clamps to the file end. Extend it sparsely: the
		// bytes are never read, and the file costs nothing on disk.
		if i == pleShard && pleSize > 0 {
			if err := os.Truncate(path, 4096+pleSize+8192); err != nil {
				t.Fatal(err)
			}
		}
	}
	return filepath.Join(dir, fmt.Sprintf("%s-%05d-of-%05d.gguf", base, 1, total))
}

// Parsing shard 1 of a split model must still find the table, which lives
// in a later shard. This is the case that shipped broken: architecture
// parsed, table missed, control hidden.
func TestPLEFoundInLaterShard(t *testing.T) {
	dir := t.TempDir()
	const pleSize = int64(6) << 30
	first := writeSplitGGUF(t, dir, "Qwen3.8-Flash-Next-UD-Q4_K_XL", 4, 2, pleSize)

	meta, err := ParseGGUFMeta(first)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if meta.Architecture != "qwen4exp" {
		t.Errorf("Architecture = %q; the metadata shard should still be read", meta.Architecture)
	}
	if meta.PLEBytes != pleSize {
		t.Errorf("PLEBytes = %d, want %d from the shard that carries the table", meta.PLEBytes, pleSize)
	}
	if !meta.PLEChecked {
		t.Error("PLEChecked should be set once the shards have been inspected")
	}
}

// The table can be in any shard, not just the second.
func TestPLEFoundInAnyShard(t *testing.T) {
	const pleSize = int64(5) << 30
	for _, shard := range []int{2, 3, 4} {
		dir := t.TempDir()
		first := writeSplitGGUF(t, dir, "m-UD-Q4_K_XL", 4, shard, pleSize)
		meta, err := ParseGGUFMeta(first)
		if err != nil {
			t.Fatalf("shard %d: %v", shard, err)
		}
		if meta.PLEBytes != pleSize {
			t.Errorf("table in shard %d: PLEBytes = %d, want %d", shard, meta.PLEBytes, pleSize)
		}
	}
}

// A split model without such a table reports zero, and says it looked.
func TestSplitModelWithoutPLE(t *testing.T) {
	dir := t.TempDir()
	first := writeSplitGGUF(t, dir, "plain-Q4_K_M", 3, 0, 0) // no shard carries one
	meta, err := ParseGGUFMeta(first)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if meta.PLEBytes != 0 {
		t.Errorf("PLEBytes = %d, want 0", meta.PLEBytes)
	}
	if !meta.PLEChecked {
		t.Error("PLEChecked should be set even when nothing was found")
	}
}

// A partial download leaves siblings missing; that must not fail the
// parse, since the metadata read from shard 1 is still good.
func TestSplitModelWithMissingShards(t *testing.T) {
	dir := t.TempDir()
	first := writeSplitGGUF(t, dir, "partial-UD-Q4_K_XL", 4, 2, 6<<30)
	if err := os.Remove(filepath.Join(dir, "partial-UD-Q4_K_XL-00002-of-00004.gguf")); err != nil {
		t.Fatal(err)
	}
	meta, err := ParseGGUFMeta(first)
	if err != nil {
		t.Fatalf("parse should survive a missing shard: %v", err)
	}
	if meta.Architecture != "qwen4exp" {
		t.Errorf("Architecture = %q, want the metadata to survive", meta.Architecture)
	}
	if meta.PLEBytes != 0 {
		t.Errorf("PLEBytes = %d, want 0 when the shard holding it is absent", meta.PLEBytes)
	}
}
