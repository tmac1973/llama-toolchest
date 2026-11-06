// Package gguftest builds small GGUF v3 files for tests. It writes
// well-formed key/value pairs and tensor-info entries in the order they
// are added; a test that needs damaged bytes writes them by hand.
//
// It imports nothing from this module, so the tests of any package,
// internal/models included, can use it without an import cycle.
package gguftest

import (
	"bytes"
	"encoding/binary"
	"os"
	"testing"
)

// The GGUF value types, numbered as in the format.
const (
	TypeUint8   uint32 = 0
	TypeInt8    uint32 = 1
	TypeUint16  uint32 = 2
	TypeInt16   uint32 = 3
	TypeUint32  uint32 = 4
	TypeInt32   uint32 = 5
	TypeFloat32 uint32 = 6
	TypeBool    uint32 = 7
	TypeString  uint32 = 8
	TypeArray   uint32 = 9
	TypeUint64  uint32 = 10
	TypeInt64   uint32 = 11
	TypeFloat64 uint32 = 12
)

// F32 is the ggml tensor type the fixtures use. The data is filler, so
// the type only has to be a valid one.
const F32 uint32 = 0

// Builder collects metadata and tensor infos. The zero value is ready to
// use. Every method returns the Builder, so calls can be chained.
type Builder struct {
	kvs, tensors bytes.Buffer
	nkv, ntensor int
	align        int
	data         int
	hasData      bool
}

func writeStr(w *bytes.Buffer, s string) {
	binary.Write(w, binary.LittleEndian, uint64(len(s)))
	w.WriteString(s)
}

func (b *Builder) key(k string, typ uint32) {
	writeStr(&b.kvs, k)
	binary.Write(&b.kvs, binary.LittleEndian, typ)
	b.nkv++
}

// Value adds a key of any type: a string value is written as a GGUF
// string, any other value as its fixed-size little-endian bytes.
func (b *Builder) Value(k string, typ uint32, v any) *Builder {
	b.key(k, typ)
	if s, ok := v.(string); ok {
		writeStr(&b.kvs, s)
	} else {
		binary.Write(&b.kvs, binary.LittleEndian, v)
	}
	return b
}

// String adds a string key.
func (b *Builder) String(k, v string) *Builder { return b.Value(k, TypeString, v) }

// Uint32 adds a uint32 key.
func (b *Builder) Uint32(k string, v uint32) *Builder { return b.Value(k, TypeUint32, v) }

// Float32 adds a float32 key.
func (b *Builder) Float32(k string, v float32) *Builder { return b.Value(k, TypeFloat32, v) }

// Bool adds a bool key.
func (b *Builder) Bool(k string, v bool) *Builder { return b.Value(k, TypeBool, v) }

// Int32Array adds an array of int32.
func (b *Builder) Int32Array(k string, vs []int32) *Builder {
	b.key(k, TypeArray)
	binary.Write(&b.kvs, binary.LittleEndian, TypeInt32)
	binary.Write(&b.kvs, binary.LittleEndian, uint64(len(vs)))
	binary.Write(&b.kvs, binary.LittleEndian, vs)
	return b
}

// StringArray adds an array of strings.
func (b *Builder) StringArray(k string, vs []string) *Builder {
	b.key(k, TypeArray)
	binary.Write(&b.kvs, binary.LittleEndian, TypeString)
	binary.Write(&b.kvs, binary.LittleEndian, uint64(len(vs)))
	for _, v := range vs {
		writeStr(&b.kvs, v)
	}
	return b
}

// Alignment adds the general.alignment key and pads the data region to
// it. Without it the data region is padded to 32 bytes, the default.
func (b *Builder) Alignment(a uint32) *Builder {
	b.align = int(a)
	return b.Uint32("general.alignment", a)
}

// Tensor adds one tensor-info entry: name, dimensions, ggml type and the
// offset into the data region.
func (b *Builder) Tensor(name string, dims []uint64, ggmlType uint32, offset uint64) *Builder {
	writeStr(&b.tensors, name)
	binary.Write(&b.tensors, binary.LittleEndian, uint32(len(dims)))
	binary.Write(&b.tensors, binary.LittleEndian, dims)
	binary.Write(&b.tensors, binary.LittleEndian, ggmlType)
	binary.Write(&b.tensors, binary.LittleEndian, offset)
	b.ntensor++
	return b
}

// Data makes the file end in a data region: padding to the alignment,
// then n zero bytes. Without it the file ends after the last tensor info.
func (b *Builder) Data(n int) *Builder {
	b.hasData = true
	b.data = n
	return b
}

// Bytes returns the file: header, metadata, tensor infos and, when Data
// was called, the padded data region.
func (b *Builder) Bytes() []byte {
	var out bytes.Buffer
	out.WriteString("GGUF")
	binary.Write(&out, binary.LittleEndian, uint32(3))
	binary.Write(&out, binary.LittleEndian, uint64(b.ntensor))
	binary.Write(&out, binary.LittleEndian, uint64(b.nkv))
	out.Write(b.kvs.Bytes())
	out.Write(b.tensors.Bytes())
	if b.hasData {
		align := b.align
		if align == 0 {
			align = 32
		}
		for out.Len()%align != 0 {
			out.WriteByte(0)
		}
		out.Write(make([]byte, b.data))
	}
	return out.Bytes()
}

// WriteFile writes the file to path and returns path. It fails the test
// when the write fails.
func (b *Builder) WriteFile(t testing.TB, path string) string {
	t.Helper()
	if err := os.WriteFile(path, b.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}
