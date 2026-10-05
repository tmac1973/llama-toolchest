package models

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// rawGGUF writes GGUF bytes field by field, so a test can stop at any
// point or write a value the format does not allow. ggufBuilder (in
// gguf_metaonly_test.go) only writes well-formed key/value pairs and no
// tensor table, which is not enough to build damaged files.
type rawGGUF struct{ buf bytes.Buffer }

func (g *rawGGUF) u32(v uint32) *rawGGUF {
	binary.Write(&g.buf, binary.LittleEndian, v)
	return g
}

func (g *rawGGUF) u64(v uint64) *rawGGUF {
	binary.Write(&g.buf, binary.LittleEndian, v)
	return g
}

func (g *rawGGUF) str(s string) *rawGGUF {
	g.u64(uint64(len(s)))
	g.buf.WriteString(s)
	return g
}

// header writes the magic, the version and the two counts.
func (g *rawGGUF) header(version uint32, tensors, kvs uint64) *rawGGUF {
	g.buf.WriteString("GGUF")
	return g.u32(version).u64(tensors).u64(kvs)
}

func (g *rawGGUF) kvString(key, val string) *rawGGUF {
	return g.str(key).u32(ggufTypeString).str(val)
}

func (g *rawGGUF) kvU32(key string, v uint32) *rawGGUF {
	return g.str(key).u32(ggufTypeUint32).u32(v)
}

// tensor writes one tensor-info entry: name, two dimensions, type F32,
// and the offset into the data region.
func (g *rawGGUF) tensor(name string, offset uint64) *rawGGUF {
	return g.str(name).u32(2).u64(8).u64(16).u32(0).u64(offset)
}

// pad fills to the next 32-byte boundary, the default alignment, and
// appends n bytes of tensor data.
func (g *rawGGUF) data(n int) *rawGGUF {
	for g.buf.Len()%32 != 0 {
		g.buf.WriteByte(0)
	}
	g.buf.Write(make([]byte, n))
	return g
}

func (g *rawGGUF) bytes() []byte { return g.buf.Bytes() }

// parseBounded runs the parser on data and fails the test if it panics
// or allocates far more than the input could justify. A length field is
// read straight from the file, so a damaged file must not be able to make
// the parser reserve gigabytes.
func parseBounded(t *testing.T, data []byte) (*GGUFMeta, error) {
	t.Helper()
	const maxAlloc = 64 << 20

	var before, after runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&before)

	var meta *GGUFMeta
	var err error
	func() {
		defer func() {
			if p := recover(); p != nil {
				t.Fatalf("parser panicked: %v", p)
			}
		}()
		meta, err = ParseGGUFMetaFrom(bytes.NewReader(data))
	}()

	runtime.ReadMemStats(&after)
	if got := after.TotalAlloc - before.TotalAlloc; got > maxAlloc {
		t.Errorf("parser allocated %d bytes for a %d-byte input", got, len(data))
	}
	return meta, err
}

// A file whose fixed header is wrong or cut short is not a GGUF file, and
// the parse must say so. Callers (the disk scan, the remote probe) rely on
// the error to skip the file rather than register a model with no
// metadata.
func TestGGUFHeaderErrors(t *testing.T) {
	full := new(rawGGUF).header(3, 0, 0).bytes()

	tests := []struct {
		name string
		data []byte
	}{
		{"empty file", nil},
		{"magic cut short", []byte("GGU")},
		{"wrong magic", append([]byte("GGML"), full[4:]...)},
		{"lower-case magic", append([]byte("gguf"), full[4:]...)},
		{"version missing", full[:4]},
		{"version cut short", full[:6]},
		{"version 1", new(rawGGUF).header(1, 0, 0).bytes()},
		{"version 4", new(rawGGUF).header(4, 0, 0).bytes()},
		{"version 0", new(rawGGUF).header(0, 0, 0).bytes()},
		{"tensor count missing", full[:8]},
		{"tensor count cut short", full[:12]},
		{"kv count missing", full[:16]},
		{"kv count cut short", full[:20]},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			meta, err := parseBounded(t, tt.data)
			if err == nil {
				t.Fatalf("no error; meta = %+v", meta)
			}
			if meta != nil {
				t.Errorf("meta = %+v alongside error %v", meta, err)
			}
			if _, err := ParseGGUFMetaOnly(bytes.NewReader(tt.data)); err == nil {
				t.Error("metadata-only parse: no error")
			}
		})
	}
}

// The file-path entry point reports the same header errors as the
// reader one, and a missing file is an error rather than an empty result.
func TestParseGGUFMetaFileErrors(t *testing.T) {
	dir := t.TempDir()
	bad := filepath.Join(dir, "bad.gguf")
	if err := os.WriteFile(bad, []byte("not a gguf file at all"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := ParseGGUFMeta(bad); err == nil {
		t.Error("wrong magic: no error")
	}
	if _, err := ParseGGUFMeta(filepath.Join(dir, "missing.gguf")); err == nil {
		t.Error("missing file: no error")
	}
}

// A small but complete file parses, and every part of it is read: the
// header keys, the model keys and the tensor table. This is the control
// for the damaged-file cases below, which start from the same layout.
func TestGGUFMinimalWellFormed(t *testing.T) {
	g := new(rawGGUF).header(3, 2, 6).
		kvString("general.architecture", "llama").
		kvU32("llama.block_count", 2).
		kvU32("llama.embedding_length", 64).
		kvU32("llama.attention.head_count", 4).
		kvU32("llama.attention.head_count_kv", 2).
		kvU32("llama.context_length", 4096).
		tensor("token_embd.weight", 0).
		tensor("blk.0.attn_q.weight", 512).
		data(1024)

	path := filepath.Join(t.TempDir(), "tiny.gguf")
	if err := os.WriteFile(path, g.bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	meta, err := ParseGGUFMeta(path)
	if err != nil {
		t.Fatal(err)
	}
	if meta.Architecture != "llama" || meta.NLayers != 2 || meta.NEmbd != 64 ||
		meta.NHead != 4 || meta.NKVHead != 2 || meta.ContextLength != 4096 {
		t.Errorf("model keys not read: %+v", meta)
	}
	if meta.TokenEmbdBytes != 512 {
		t.Errorf("TokenEmbdBytes = %d, want 512", meta.TokenEmbdBytes)
	}
	if !meta.HasBlockTensors || !meta.HasTrunkBlock0 || meta.HasOutputTensor {
		t.Errorf("tensor layout: blocks %v, block 0 %v, output %v",
			meta.HasBlockTensors, meta.HasTrunkBlock0, meta.HasOutputTensor)
	}
	if !meta.PLEChecked || !meta.ReasoningChecked || !meta.SamplingChecked {
		t.Error("a complete parse should mark every section as checked")
	}
	// 2 layers × 2 KV heads × (16 + 16) head dims.
	if meta.KVFullPerTok != 2*2*32 {
		t.Errorf("KVFullPerTok = %d, want %d", meta.KVFullPerTok, 2*2*32)
	}
}

// Damage after the fixed header does not fail the parse. This is on
// purpose (see the comment above scanTensorBlock in parseGGUFMeta): the
// keys read before the damage are still correct and still useful, and a
// file over ranged HTTP may be cut off on purpose. What must hold is that
// the parser stops cleanly — no panic, no large allocation from a length
// field it cannot trust — keeps what it read before the damage, and
// reports nothing it did not actually read from the tensor table.
func TestGGUFDamagedBody(t *testing.T) {
	// archOnly is a header plus general.architecture, with room for the
	// test to append one more key or a tensor table.
	archOnly := func(tensors, kvs uint64) *rawGGUF {
		return new(rawGGUF).header(3, tensors, kvs).kvString("general.architecture", "llama")
	}

	tests := []struct {
		name   string
		data   []byte
		noArch bool // the damage is in general.architecture itself
	}{
		{
			name: "more keys promised than written",
			data: archOnly(0, 1<<40).bytes(),
		},
		{
			name: "key name longer than the rest of the file",
			data: archOnly(0, 2).u64(1000).bytes(),
		},
		{
			name: "key name length beyond the 1 MB limit",
			data: archOnly(0, 2).u64(1 << 62).bytes(),
		},
		{
			name: "string value longer than the rest of the file",
			data: new(rawGGUF).header(3, 0, 1).
				str("general.architecture").u32(ggufTypeString).u64(500).bytes(),
			noArch: true,
		},
		{
			name: "string value length beyond the 1 MB limit",
			data: new(rawGGUF).header(3, 0, 1).
				str("general.architecture").u32(ggufTypeString).u64(1 << 40).bytes(),
			noArch: true,
		},
		{
			name: "number array longer than the file",
			data: archOnly(0, 3).
				str("llama.attention.head_count_kv").u32(ggufTypeArray).u32(ggufTypeInt32).u64(1<<40).
				kvU32("llama.block_count", 2).bytes(),
		},
		{
			name: "number array longer than the 1M sanity bound",
			data: archOnly(0, 2).
				str("llama.attention.sliding_window_pattern").u32(ggufTypeArray).u32(ggufTypeBool).u64(1<<20 + 1).bytes(),
		},
		{
			name: "string array longer than the file",
			data: archOnly(0, 2).
				str("general.tags").u32(ggufTypeArray).u32(ggufTypeString).u64(1 << 40).str("one").bytes(),
		},
		{
			name: "array of unknown element type",
			data: archOnly(0, 2).
				str("general.tags").u32(ggufTypeArray).u32(99).u64(1 << 40).bytes(),
		},
		{
			name: "unknown value type",
			data: archOnly(0, 3).
				str("general.oddity").u32(99).u64(0xdeadbeef).
				kvU32("llama.block_count", 2).bytes(),
		},
		{
			name: "array header cut short",
			data: archOnly(0, 2).str("general.tags").u32(ggufTypeArray).u32(ggufTypeString).bytes(),
		},
		{
			name: "tensor table cut off mid-entry",
			data: func() []byte {
				b := archOnly(3, 1).tensor("token_embd.weight", 0).tensor("blk.0.attn_q.weight", 512).bytes()
				return b[:len(b)-6] // inside the second entry's offset
			}(),
		},
		{
			name: "tensor table with fewer entries than promised",
			data: archOnly(3, 1).tensor("token_embd.weight", 0).bytes(),
		},
		{
			name: "tensor name longer than the rest of the file",
			data: archOnly(1, 1).u64(4096).bytes(),
		},
		{
			name: "tensor with more than four dimensions",
			data: archOnly(1, 1).str("token_embd.weight").u32(1 << 30).bytes(),
		},
		{
			name: "tensor count beyond the 1M sanity bound",
			data: archOnly(1<<40, 1).tensor("token_embd.weight", 0).data(64).bytes(),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			want := "llama"
			if tt.noArch {
				want = ""
			}
			meta, err := parseBounded(t, tt.data)
			if err != nil {
				t.Fatalf("damage after the header should not fail the parse: %v", err)
			}
			if meta.Architecture != want {
				t.Errorf("Architecture = %q, want %q", meta.Architecture, want)
			}
			// None of these files has a readable tensor table, so nothing
			// measured from it may be set.
			if meta.TokenEmbdBytes != 0 || meta.PLEBytes != 0 || meta.ExpertBytes != 0 ||
				meta.HasBlockTensors || meta.HasTrunkBlock0 || meta.HasOutputTensor {
				t.Errorf("tensor-table fields set from a damaged table: %+v", meta)
			}
			if _, err := ParseGGUFMetaOnly(bytes.NewReader(tt.data)); err != nil {
				t.Errorf("metadata-only parse: %v", err)
			}
		})
	}
}

// A key whose value is damaged loses that value only. The keys read
// before it keep their values, and a scalar key of the wrong type is
// skipped by its own size, so the keys after it are still read.
func TestGGUFWrongTypeKeyIsSkipped(t *testing.T) {
	g := new(rawGGUF).header(3, 0, 4).
		kvString("general.architecture", "llama").
		kvString("llama.block_count", "thirty-two"). // a string where a number belongs
		kvU32("llama.embedding_length", 64).
		kvU32("llama.context_length", 4096)

	meta, err := parseBounded(t, g.bytes())
	if err != nil {
		t.Fatal(err)
	}
	if meta.NLayers != 0 {
		t.Errorf("NLayers = %d from a string value", meta.NLayers)
	}
	if meta.NEmbd != 64 || meta.ContextLength != 4096 {
		t.Errorf("keys after the wrong-type key were lost: %+v", meta)
	}
}

// The vocabulary size is the token list's length field, read straight
// from the file. A damaged file claiming 2^40 tokens (or 2^63, which is
// negative as an int) used to set VocabSize to that, and it feeds tensor
// size and KL-logits estimates. An implausible count now reads as unknown.
func TestGGUFImplausibleVocabIsUnknown(t *testing.T) {
	for _, count := range []uint64{1 << 40, 1 << 63} {
		g := new(rawGGUF).header(3, 0, 2).kvString("general.architecture", "llama")
		g.str("tokenizer.ggml.tokens").u32(ggufTypeArray).u32(ggufTypeString).u64(count)
		meta, _ := parseBounded(t, g.bytes())
		if meta != nil && meta.VocabSize != 0 {
			t.Errorf("token count %d: VocabSize = %d, want 0 (unknown)", count, meta.VocabSize)
		}
	}

	g := new(rawGGUF).header(3, 0, 2).kvString("general.architecture", "llama")
	g.str("tokenizer.ggml.tokens").u32(ggufTypeArray).u32(ggufTypeString).u64(2)
	g.str("a").str("b")
	meta, err := parseBounded(t, g.bytes())
	if err != nil || meta.VocabSize != 2 {
		t.Errorf("a real token list: VocabSize = %v (err %v), want 2", meta, err)
	}
}

// tinyGGUF is a small well-formed model file: two tensors and their data.
func tinyGGUF() []byte {
	return new(rawGGUF).header(3, 2, 6).
		kvString("general.architecture", "llama").
		kvU32("llama.block_count", 2).
		kvU32("llama.embedding_length", 64).
		kvU32("llama.attention.head_count", 4).
		kvU32("llama.attention.head_count_kv", 2).
		kvU32("llama.context_length", 4096).
		tensor("token_embd.weight", 0).
		tensor("blk.0.attn_q.weight", 512).
		data(1024).bytes()
}

// A file whose header or data stops short — damaged, or scanned while it
// was still being copied into the models folder — is read as far as it
// goes but is not marked checked, so it is read again later. A whole file
// is marked checked.
func TestGGUFPartialFileIsNotMarkedChecked(t *testing.T) {
	whole := tinyGGUF()
	cases := []struct {
		name    string
		data    []byte
		partial bool
	}{
		{"whole file", whole, false},
		{"data cut short", whole[:len(whole)-600], true},
		{"tensor table cut short", whole[:len(whole)-1024-40], true},
	}
	for _, c := range cases {
		meta, err := parseBounded(t, c.data)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		if meta.partial != c.partial || meta.PLEChecked == c.partial {
			t.Errorf("%s: partial = %v, PLEChecked = %v; want partial %v", c.name, meta.partial, meta.PLEChecked, c.partial)
		}
		if meta.Architecture != "llama" {
			t.Errorf("%s: the keys read before the cut were lost", c.name)
		}
	}
}

// Backfill used to record the current parser version after any parse, so
// a file read while still being copied in kept its partial metadata for
// good. It now keeps what it read and tries again at the next start, and
// records the version once the file is whole.
func TestBackfillRereadsAPartialFile(t *testing.T) {
	r, _, modelsDir := newTestRegistry(t)
	path := filepath.Join(modelsDir, "org--m", "m.gguf")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	whole := tinyGGUF()
	if err := os.WriteFile(path, whole[:len(whole)-600], 0o644); err != nil {
		t.Fatal(err)
	}
	if err := r.Add(&Model{ID: "m", ModelID: "org/m", Filename: "m.gguf", FilePath: path}); err != nil {
		t.Fatal(err)
	}

	r.BackfillGGUFMeta()
	m, _ := r.Get("m")
	if m.GGUFMetaVersion != 0 || m.Arch != "llama" {
		t.Fatalf("after a partial read: version %d, arch %q; want version 0 and the keys kept", m.GGUFMetaVersion, m.Arch)
	}

	if err := os.WriteFile(path, whole, 0o644); err != nil { // the copy finishes
		t.Fatal(err)
	}
	r.BackfillGGUFMeta()
	m, _ = r.Get("m")
	if m.GGUFMetaVersion != GGUFMetaVersion || !m.PLEChecked {
		t.Errorf("after the file was whole: version %d, PLEChecked %v; want %d and true", m.GGUFMetaVersion, m.PLEChecked, GGUFMetaVersion)
	}
}
