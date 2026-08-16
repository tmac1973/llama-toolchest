package models

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

// writeVocabGGUF builds a minimal GGUF v3 file whose metadata is:
// general.architecture = "testarch", then the given string array under
// key `key` (element type ggufTypeString), then testarch.block_count =
// 42, so parser alignment after the (potentially large) string array is
// checked by the second read.
func writeVocabGGUF(t *testing.T, key string, tokens []string) string {
	t.Helper()
	var buf bytes.Buffer
	buf.WriteString("GGUF")
	binary.Write(&buf, binary.LittleEndian, uint32(3)) // version
	binary.Write(&buf, binary.LittleEndian, uint64(0)) // tensor count
	binary.Write(&buf, binary.LittleEndian, uint64(3)) // kv count
	writeStr := func(s string) {
		binary.Write(&buf, binary.LittleEndian, uint64(len(s)))
		buf.WriteString(s)
	}
	// kv 1: general.architecture
	writeStr("general.architecture")
	binary.Write(&buf, binary.LittleEndian, ggufTypeString)
	writeStr("testarch")
	// kv 2: key, type=array, elem=string, count, strings...
	writeStr(key)
	binary.Write(&buf, binary.LittleEndian, ggufTypeArray)
	binary.Write(&buf, binary.LittleEndian, ggufTypeString)
	binary.Write(&buf, binary.LittleEndian, uint64(len(tokens)))
	for _, s := range tokens {
		writeStr(s)
	}
	// kv 3: testarch.block_count = 42
	writeStr("testarch.block_count")
	binary.Write(&buf, binary.LittleEndian, ggufTypeUint32)
	binary.Write(&buf, binary.LittleEndian, uint32(42))

	path := filepath.Join(t.TempDir(), "vocab.gguf")
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestParseGGUFTokenizerVocabSize(t *testing.T) {
	// A realistic-ish vocab: 500 tokens (the parser must seek past all
	// 500 strings without reading their contents).
	tokens := make([]string, 500)
	for i := range tokens {
		tokens[i] = "token" + string(rune('a'+i%26)) + string(rune('0'+i%10))
	}
	path := writeVocabGGUF(t, "tokenizer.ggml.tokens", tokens)

	meta, err := ParseGGUFMeta(path)
	if err != nil {
		t.Fatal(err)
	}
	if meta.VocabSize != 500 {
		t.Errorf("VocabSize = %d, want 500 (tokenizer.ggml.tokens length)", meta.VocabSize)
	}
	// Alignment: the key after the array must still parse.
	if meta.NLayers != 42 {
		t.Errorf("NLayers = %d, want 42 (parser misaligned after the string array?)", meta.NLayers)
	}
}

func TestParseGGUFNoVocabKey(t *testing.T) {
	path := writeVocabGGUF(t, "tokenizer.ggml.tok_type", nil) // a different tokenizer key
	meta, err := ParseGGUFMeta(path)
	if err != nil {
		t.Fatal(err)
	}
	if meta.VocabSize != 0 {
		t.Errorf("VocabSize = %d, want 0 when tokenizer.ggml.tokens is absent", meta.VocabSize)
	}
	if meta.NLayers != 42 {
		t.Errorf("NLayers = %d, want 42 (parser misaligned?)", meta.NLayers)
	}
}

func TestGGUFMetaApplyToVocabSize(t *testing.T) {
	var m Model
	m.VocabSize = 128256            // a previously backfilled value...
	meta := &GGUFMeta{VocabSize: 0} // ...must not be clobbered by a
	meta.ApplyTo(&m)                // parse that found nothing.
	if m.VocabSize != 128256 {
		t.Errorf("ApplyTo clobbered VocabSize: %d", m.VocabSize)
	}
	meta.VocabSize = 32000
	meta.ApplyTo(&m)
	if m.VocabSize != 32000 {
		t.Errorf("ApplyTo did not copy VocabSize: %d", m.VocabSize)
	}
}
