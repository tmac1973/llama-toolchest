package models

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"testing"
)

// ggufBuilder writes GGUF v3 metadata for tests that need arrays and an
// exact key order.
type ggufBuilder struct {
	kvs   bytes.Buffer
	count int
}

func (b *ggufBuilder) str(s string) {
	binary.Write(&b.kvs, binary.LittleEndian, uint64(len(s)))
	b.kvs.WriteString(s)
}

func (b *ggufBuilder) u32(key string, v uint32) {
	b.str(key)
	binary.Write(&b.kvs, binary.LittleEndian, ggufTypeUint32)
	binary.Write(&b.kvs, binary.LittleEndian, v)
	b.count++
}

func (b *ggufBuilder) s(key, v string) {
	b.str(key)
	binary.Write(&b.kvs, binary.LittleEndian, ggufTypeString)
	b.str(v)
	b.count++
}

func (b *ggufBuilder) i32s(key string, vs []int32) {
	b.str(key)
	binary.Write(&b.kvs, binary.LittleEndian, ggufTypeArray)
	binary.Write(&b.kvs, binary.LittleEndian, ggufTypeInt32)
	binary.Write(&b.kvs, binary.LittleEndian, uint64(len(vs)))
	binary.Write(&b.kvs, binary.LittleEndian, vs)
	b.count++
}

func (b *ggufBuilder) strs(key string, vs []string) {
	b.str(key)
	binary.Write(&b.kvs, binary.LittleEndian, ggufTypeArray)
	binary.Write(&b.kvs, binary.LittleEndian, ggufTypeString)
	binary.Write(&b.kvs, binary.LittleEndian, uint64(len(vs)))
	for _, v := range vs {
		b.str(v)
	}
	b.count++
}

func (b *ggufBuilder) bytes() []byte {
	var out bytes.Buffer
	out.WriteString("GGUF")
	binary.Write(&out, binary.LittleEndian, uint32(3))
	binary.Write(&out, binary.LittleEndian, uint64(0))
	binary.Write(&out, binary.LittleEndian, uint64(b.count))
	out.Write(b.kvs.Bytes())
	return out.Bytes()
}

// farthestReader records the farthest byte read, which is what a ranged
// HTTP read pays for. Seeks cost nothing.
type farthestReader struct {
	r        *bytes.Reader
	farthest int64
}

func (f *farthestReader) Read(p []byte) (int, error) {
	n, err := f.r.Read(p)
	if pos, _ := f.r.Seek(0, io.SeekCurrent); pos > f.farthest {
		f.farthest = pos
	}
	return n, err
}

func (f *farthestReader) Seek(off int64, whence int) (int64, error) { return f.r.Seek(off, whence) }

func tokens(n int) []string {
	out := make([]string, n)
	for i := range out {
		out[i] = fmt.Sprintf("token-%06d", i)
	}
	return out
}

// The usual layout: every model key first, then the tokenizer. The parse
// passes a small number array, takes the vocabulary size from the token
// list's header and reads none of the list itself.
func TestMetaOnlyStopsAtTheTokenList(t *testing.T) {
	var b ggufBuilder
	b.s("general.architecture", "llama")
	b.u32("llama.block_count", 32)
	b.u32("llama.embedding_length", 4096)
	b.u32("llama.attention.head_count", 32)
	b.u32("llama.attention.head_count_kv", 8)
	b.u32("llama.context_length", 131072)
	b.s("tokenizer.ggml.model", "gpt2")
	b.i32s("tokenizer.ggml.suppress_tokens", []int32{1, 2})
	head := len(b.bytes())
	b.strs("tokenizer.ggml.tokens", tokens(5000))
	b.s("tokenizer.chat_template", "{{ tools }}")
	data := b.bytes()

	r := &farthestReader{r: bytes.NewReader(data)}
	meta, err := ParseGGUFMetaOnly(r)
	if err != nil {
		t.Fatal(err)
	}
	if !meta.MetaOnly || meta.NLayers != 32 || meta.NKVHead != 8 || meta.ContextLength != 131072 {
		t.Errorf("meta = %+v", meta)
	}
	if meta.VocabSize != 5000 {
		t.Errorf("VocabSize = %d, want 5000", meta.VocabSize)
	}
	// Key, type, element type and count of the token list: under 64 bytes.
	if r.farthest > int64(head)+64 {
		t.Errorf("read %d bytes; the token list starts at %d of %d", r.farthest, head, len(data))
	}
	if meta.SupportsTools {
		t.Error("the chat template was read")
	}

	full, err := ParseGGUFMetaFrom(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	if full.MetaOnly || full.KVFullPerTok != meta.KVFullPerTok || full.VocabSize != meta.VocabSize {
		t.Errorf("full parse %+v differs from metadata-only %+v", full, meta)
	}
}

// A file that puts a model key after the tokenizer is read on, so the
// key is not lost.
func TestMetaOnlyReadsOnWhenKeysComeLate(t *testing.T) {
	var b ggufBuilder
	b.s("general.architecture", "llama")
	b.strs("tokenizer.ggml.tokens", tokens(100))
	b.u32("llama.block_count", 32)
	b.u32("llama.context_length", 8192)

	meta, err := ParseGGUFMetaOnly(bytes.NewReader(b.bytes()))
	if err != nil {
		t.Fatal(err)
	}
	if meta.NLayers != 32 || meta.ContextLength != 8192 || meta.VocabSize != 100 {
		t.Errorf("meta = %+v", meta)
	}
}

func TestMetaOnlyReadsTheDerivationKeys(t *testing.T) {
	var b ggufBuilder
	b.s("general.architecture", "deepseek2")
	b.u32("deepseek2.block_count", 61)
	b.u32("deepseek2.vocab_size", 129280)
	b.u32("deepseek2.leading_dense_block_count", 3)
	b.u32("deepseek2.expert_feed_forward_length", 2048)
	b.u32("deepseek2.embedding_length_per_layer_input", 256)

	meta, err := ParseGGUFMetaOnly(bytes.NewReader(b.bytes()))
	if err != nil {
		t.Fatal(err)
	}
	if meta.LeadingDenseBlocks != 3 || meta.ExpertFFLength != 2048 || meta.PLEInputDim != 256 || meta.VocabSize != 129280 {
		t.Errorf("meta = %+v", meta)
	}
}

func TestDerivedFor(t *testing.T) {
	const gib = int64(1) << 30
	// shareOf is the derivation's own arithmetic: the tensor's share of
	// the parameters, applied to the file size.
	shareOf := func(file int64, params, total float64) int64 {
		return int64(float64(file) * (params / total))
	}
	t.Run("dense", func(t *testing.T) {
		meta := &GGUFMeta{MetaOnly: true, NLayers: 32, NEmbd: 4096, VocabSize: 128000}
		d := meta.DerivedFor(8*gib, 8_000_000_000)
		// 128000 × 4096 of 8B parameters, applied to 8 GiB.
		want := shareOf(8*gib, 128000*4096, 8e9)
		if d.TokenEmbdBytes != want || d.ExpertBytes != 0 || d.PLEBytes != 0 {
			t.Errorf("derived %+v, want token embedding %d", d, want)
		}
		if meta.TokenEmbdBytes != 0 {
			t.Error("DerivedFor changed the shared description")
		}
	})
	t.Run("no parameter count", func(t *testing.T) {
		d := (&GGUFMeta{MetaOnly: true, NLayers: 32, NEmbd: 4096, VocabSize: 128000}).DerivedFor(8*gib, 0)
		if d.TokenEmbdBytes != 0 {
			t.Errorf("TokenEmbdBytes = %d without a parameter count; want 0", d.TokenEmbdBytes)
		}
	})
	t.Run("moe with leading dense layers", func(t *testing.T) {
		meta := &GGUFMeta{MetaOnly: true, NLayers: 61, NEmbd: 7168, ExpertCount: 256, ExpertFFLength: 2048, LeadingDenseBlocks: 3}
		d := meta.DerivedFor(400*gib, 671_000_000_000)
		if d.ExpertLayerFirst != 3 || d.ExpertLayers != 58 {
			t.Errorf("expert layers %d+%d, want 3+58", d.ExpertLayerFirst, d.ExpertLayers)
		}
		if want := shareOf(400*gib, 58.0*256*3*7168*2048, 671e9); d.ExpertBytes != want {
			t.Errorf("ExpertBytes = %d, want %d", d.ExpertBytes, want)
		}
	})
	t.Run("moe without the expert size", func(t *testing.T) {
		d := (&GGUFMeta{MetaOnly: true, NLayers: 48, NEmbd: 2048, ExpertCount: 128}).DerivedFor(10*gib, 0)
		if d.ExpertLayers != 48 || d.ExpertBytes != int64(float64(10*gib)*unknownExpertShare) {
			t.Errorf("derived %+v", d)
		}
	})
	t.Run("per-layer embeddings", func(t *testing.T) {
		d := (&GGUFMeta{MetaOnly: true, NLayers: 42, NEmbd: 2560, VocabSize: 262144, PLEInputDim: 256}).DerivedFor(5*gib, 8_000_000_000)
		if want := shareOf(5*gib, 262144*42*256, 8e9); d.PLEBytes != want {
			t.Errorf("PLEBytes = %d, want %d", d.PLEBytes, want)
		}
	})
	t.Run("a full parse is left alone", func(t *testing.T) {
		meta := &GGUFMeta{NLayers: 32, NEmbd: 4096, VocabSize: 128000, TokenEmbdBytes: 123}
		if d := meta.DerivedFor(8*gib, 8_000_000_000); d.TokenEmbdBytes != 123 {
			t.Errorf("TokenEmbdBytes = %d, want the measured 123", d.TokenEmbdBytes)
		}
	})
}

// A writer that scatters model keys through the tokenizer: once a model
// key has followed a tokenizer key, no later point is known to be past
// every model key, so the parse reads to the end of the metadata even
// though the core keys are already known.
func TestMetaOnlyReadsScatteredKeysToTheEnd(t *testing.T) {
	var b ggufBuilder
	b.s("general.architecture", "qwen35")
	b.strs("tokenizer.ggml.tokens", tokens(100))
	b.u32("qwen35.block_count", 64)
	b.u32("qwen35.embedding_length", 5120)
	b.u32("qwen35.attention.head_count", 24)
	b.u32("qwen35.context_length", 262144)
	b.strs("tokenizer.ggml.merges", tokens(100))
	b.u32("qwen35.attention.head_count_kv", 4)

	meta, err := ParseGGUFMetaOnly(bytes.NewReader(b.bytes()))
	if err != nil {
		t.Fatal(err)
	}
	if meta.NKVHead != 4 {
		t.Errorf("NKVHead = %d: the parse stopped before a model key", meta.NKVHead)
	}
}
