package modelsource

import (
	"bytes"
	"context"
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/models"
)

// metaGGUF is a GGUF whose model keys come before a tokenizer of
// tokenBytes, the layout every published file checked has.
func metaGGUF(tokenBytes int) []byte {
	var b bytes.Buffer
	ws := func(s string) {
		binary.Write(&b, binary.LittleEndian, uint64(len(s)))
		b.WriteString(s)
	}
	u32 := func(k string, v uint32) {
		ws(k)
		binary.Write(&b, binary.LittleEndian, uint32(4))
		binary.Write(&b, binary.LittleEndian, v)
	}
	b.WriteString("GGUF")
	binary.Write(&b, binary.LittleEndian, uint32(3))
	binary.Write(&b, binary.LittleEndian, uint64(0))
	binary.Write(&b, binary.LittleEndian, uint64(6))
	ws("general.architecture")
	binary.Write(&b, binary.LittleEndian, uint32(8))
	ws("llama")
	u32("llama.block_count", 32)
	u32("llama.embedding_length", 4096)
	u32("llama.attention.head_count", 32)
	u32("llama.context_length", 8192)
	ws("tokenizer.ggml.tokens")
	binary.Write(&b, binary.LittleEndian, uint32(9))
	binary.Write(&b, binary.LittleEndian, uint32(8))
	count := tokenBytes / 16
	binary.Write(&b, binary.LittleEndian, uint64(count))
	for range count {
		ws("tokentok")
	}
	return b.Bytes()
}

// The whole point of the probe: one small request, however long the
// tokenizer behind the model keys.
func TestProbeMetaReadsOnlyTheStart(t *testing.T) {
	body := metaGGUF(8 << 20)
	srv, served := serveRanges(t, [][]byte{body})
	defer srv.Close()

	meta, err := ProbeMeta(context.Background(), srv.Client(), "", srv.URL+"/0", int64(len(body)))
	if err != nil {
		t.Fatal(err)
	}
	if meta.Architecture != "llama" || meta.NLayers != 32 || !meta.MetaOnly {
		t.Errorf("meta = %+v", meta)
	}
	if meta.VocabSize != (8<<20)/16 {
		t.Errorf("VocabSize = %d", meta.VocabSize)
	}
	if *served > probeChunk {
		t.Errorf("served %d bytes, want at most one %d-byte request", *served, probeChunk)
	}
}

func TestProbeMetaRejectsAFileWithNoModel(t *testing.T) {
	body := []byte("GGUF\x03\x00\x00\x00" + string(make([]byte, 16)))
	srv, _ := serveRanges(t, [][]byte{body})
	defer srv.Close()
	if _, err := ProbeMeta(context.Background(), srv.Client(), "", srv.URL+"/0", int64(len(body))); err == nil {
		t.Error("a header without a model was accepted")
	}
}

func TestMetaCache(t *testing.T) {
	dir := t.TempDir()
	f := File{Filename: "m-Q4_K_M.gguf", Size: 5 << 30, OID: "abc"}
	key := MetaKey(SourceHuggingFace, "org/Model-GGUF", f)
	meta := &models.GGUFMeta{Architecture: "llama", NLayers: 32, MetaOnly: true}

	NewMetaCache(dir).Put(key, meta)
	// A fresh cache over the same folder finds it on disk.
	got, ok := NewMetaCache(dir).Get(key)
	if !ok || got.NLayers != 32 || !got.MetaOnly {
		t.Fatalf("Get = %+v, %v", got, ok)
	}

	// A new upload of the file is a new key.
	f.OID = "def"
	if _, ok := NewMetaCache(dir).Get(MetaKey(SourceHuggingFace, "org/Model-GGUF", f)); ok {
		t.Error("a changed file was served from the cache")
	}

	// A folder that cannot be written is ignored; memory still works.
	blocked := filepath.Join(dir, "blocked")
	if err := os.WriteFile(blocked, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	c := NewMetaCache(blocked)
	c.Put(key, meta)
	if _, ok := c.Get(key); !ok {
		t.Error("an unwritable folder broke the in-memory cache")
	}

	// A nil cache holds nothing and does not fail.
	var none *MetaCache
	none.Put(key, meta)
	if _, ok := none.Get(key); ok {
		t.Error("a nil cache returned something")
	}
}
