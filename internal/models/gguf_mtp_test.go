package models

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

// writeMTPGGUF builds a minimal GGUF v3 carrying an architecture, an
// optional {arch}.nextn_predict_layers key, and a tensor-info block with
// the given tensor names. It is the tensor NAMES that matter here — the
// dimensions and offsets are filler — because IsMTPHead reads the block
// layout, not the data.
func writeMTPGGUF(t *testing.T, arch string, nextn int, tensorNames []string) string {
	t.Helper()
	var buf bytes.Buffer
	writeStr := func(s string) {
		binary.Write(&buf, binary.LittleEndian, uint64(len(s)))
		buf.WriteString(s)
	}

	buf.WriteString("GGUF")
	binary.Write(&buf, binary.LittleEndian, uint32(3))
	binary.Write(&buf, binary.LittleEndian, uint64(len(tensorNames)))
	kvCount := 1
	if nextn > 0 {
		kvCount = 2
	}
	binary.Write(&buf, binary.LittleEndian, uint64(kvCount))

	writeStr("general.architecture")
	binary.Write(&buf, binary.LittleEndian, ggufTypeString)
	writeStr(arch)
	if nextn > 0 {
		writeStr(arch + ".nextn_predict_layers")
		binary.Write(&buf, binary.LittleEndian, ggufTypeUint32)
		binary.Write(&buf, binary.LittleEndian, uint32(nextn))
	}

	for i, name := range tensorNames {
		writeStr(name)
		binary.Write(&buf, binary.LittleEndian, uint32(2))
		binary.Write(&buf, binary.LittleEndian, uint64(8))
		binary.Write(&buf, binary.LittleEndian, uint64(16))
		binary.Write(&buf, binary.LittleEndian, uint32(0))     // F32
		binary.Write(&buf, binary.LittleEndian, uint64(i*512)) // offset
	}

	for buf.Len()%32 != 0 {
		buf.WriteByte(0)
	}
	buf.Write(make([]byte, 512*len(tensorNames)+512))

	path := filepath.Join(t.TempDir(), "model.gguf")
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

// TestIsMTPHead covers the shapes that actually ship. The two Qwen3.8
// cases are the file layouts read off unsloth/Qwen3.8-Flash-Next-GGUF:
// both declare block_count 49 with nextn_predict_layers 1 and carry only
// blk.48, differing solely in whether they bring their own token_embd.
func TestIsMTPHead(t *testing.T) {
	tests := []struct {
		name    string
		arch    string
		nextn   int
		tensors []string
		want    bool
	}{
		{
			name:  "qwen3.8 shared head borrows token_embd from the target",
			arch:  "qwen4exp",
			nextn: 1,
			tensors: []string{
				"blk.48.attn_q.weight", "blk.48.attn_k.weight",
				"blk.48.nextn.eh_proj.weight", "blk.48.nextn.enorm.weight",
			},
			want: true,
		},
		{
			name:  "qwen3.8 self-contained head carries its own token_embd",
			arch:  "qwen4exp",
			nextn: 1,
			tensors: []string{
				"token_embd.weight", "output.weight",
				"blk.48.attn_q.weight", "blk.48.nextn.eh_proj.weight",
			},
			want: true,
		},
		{
			name:  "gemma-4 head announces itself by architecture",
			arch:  "gemma4-assistant",
			nextn: 0,
			tensors: []string{
				"token_embd.weight", "blk.0.attn_q.weight",
			},
			want: true,
		},
		{
			// The head is baked into a runnable model: nextn is set, but
			// the whole trunk is present. Must stay a servable model.
			name:  "qwen self-speculation model with a baked-in head",
			arch:  "qwen3moe",
			nextn: 1,
			tensors: []string{
				"token_embd.weight", "blk.0.attn_q.weight",
				"blk.47.attn_q.weight", "blk.47.nextn.eh_proj.weight",
			},
			want: false,
		},
		{
			name:  "ordinary model",
			arch:  "qwen4exp",
			nextn: 0,
			tensors: []string{
				"token_embd.weight", "blk.0.attn_q.weight", "blk.1.attn_q.weight",
			},
			want: false,
		},
		{
			// A split model's first shard holds the metadata and, as
			// publishers ship them, no tensors at all. Without the
			// "carries blocks" requirement this would read as a head.
			name:    "first shard of a split model has metadata but no tensors",
			arch:    "qwen4exp",
			nextn:   1,
			tensors: nil,
			want:    false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			path := writeMTPGGUF(t, tc.arch, tc.nextn, tc.tensors)
			meta, err := ParseGGUFMeta(path)
			if err != nil {
				t.Fatalf("ParseGGUFMeta: %v", err)
			}
			if got := meta.IsMTPHead(); got != tc.want {
				t.Errorf("IsMTPHead() = %v, want %v (arch=%q nextn=%d blocks=%v blk0=%v)",
					got, tc.want, meta.Architecture, meta.NextNPredictLayers,
					meta.HasBlockTensors, meta.HasTrunkBlock0)
			}
		})
	}
}

// TestFindDraftCandidatesSkipsMTPHead is the bug that started this: an
// MTP head matches its target's architecture exactly and is far under the
// size bar, so it was offered as a plain draft model — a pairing that
// fails at load with "tensor 'token_embd.weight' not found".
func TestFindDraftCandidatesSkipsMTPHead(t *testing.T) {
	r := &Registry{
		dataDir:   t.TempDir(),
		modelsDir: t.TempDir(),
		data: registryData{
			Models: map[string]*Model{
				"target": {
					ID: "target", Arch: "qwen4exp", SizeBytes: 111 << 30,
				},
				"head": {
					ID: "head", Arch: "qwen4exp", SizeBytes: 3 << 30, MTPHead: true,
				},
				"real-draft": {
					ID: "real-draft", Arch: "qwen4exp", SizeBytes: 2 << 30,
				},
			},
			Configs: map[string]*ModelConfig{},
		},
	}

	var got []string
	for _, c := range r.FindDraftCandidates("target") {
		got = append(got, c.ID)
	}
	if len(got) != 1 || got[0] != "real-draft" {
		t.Errorf("FindDraftCandidates = %v, want [real-draft] only", got)
	}
}
