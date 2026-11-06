package gguftest

import (
	"bytes"
	"encoding/binary"
	"testing"
)

// The layout is the one the GGUF v3 spec gives: magic, version, tensor
// count, KV count, the KVs, the tensor infos, then the padded data.
func TestBytesLayout(t *testing.T) {
	var b Builder
	b.String("general.architecture", "x").Tensor("t", []uint64{4}, F32, 0).Data(16)

	var want bytes.Buffer
	le := func(v any) { binary.Write(&want, binary.LittleEndian, v) }
	want.WriteString("GGUF")
	le(uint32(3))
	le(uint64(1))
	le(uint64(1))
	le(uint64(len("general.architecture")))
	want.WriteString("general.architecture")
	le(TypeString)
	le(uint64(1))
	want.WriteString("x")
	le(uint64(1))
	want.WriteString("t")
	le(uint32(1))
	le(uint64(4))
	le(F32)
	le(uint64(0))
	for want.Len()%32 != 0 {
		want.WriteByte(0)
	}
	want.Write(make([]byte, 16))

	if got := b.Bytes(); !bytes.Equal(got, want.Bytes()) {
		t.Errorf("Bytes() =\n%x\nwant\n%x", got, want.Bytes())
	}
}

// Without Data the file ends after the metadata, with no padding.
func TestNoDataNoPadding(t *testing.T) {
	var b Builder
	b.Uint32("k", 7)
	if got, want := len(b.Bytes()), 4+4+8+8+8+1+4+4; got != want {
		t.Errorf("len = %d, want %d", got, want)
	}
}
