package models

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

// writeHeadGGUF builds a minimal GGUF v3 with an architecture, optional
// uint32 keys and a tensor-info block with the given tensor names: enough
// for DraftHeadKind, which reads the architecture, the lilicorr.* keys
// and whether the Markov head tensor is present.
func writeHeadGGUF(t *testing.T, arch string, keys map[string]uint32, tensorNames []string) string {
	t.Helper()
	var buf bytes.Buffer
	writeStr := func(s string) {
		binary.Write(&buf, binary.LittleEndian, uint64(len(s)))
		buf.WriteString(s)
	}

	buf.WriteString("GGUF")
	binary.Write(&buf, binary.LittleEndian, uint32(3))
	binary.Write(&buf, binary.LittleEndian, uint64(len(tensorNames)))
	binary.Write(&buf, binary.LittleEndian, uint64(1+len(keys)))

	writeStr("general.architecture")
	binary.Write(&buf, binary.LittleEndian, ggufTypeString)
	writeStr(arch)
	for k, v := range keys {
		writeStr(k)
		binary.Write(&buf, binary.LittleEndian, ggufTypeUint32)
		binary.Write(&buf, binary.LittleEndian, v)
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

	path := filepath.Join(t.TempDir(), "head.gguf")
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

// The kinds are told apart the way llama.cpp tells them apart: by the
// architecture, then by the Markov head (DSpark) or the lilicorr.* keys
// (LiliCorr) on a "dflash" file.
func TestDraftHeadKind(t *testing.T) {
	tests := []struct {
		name    string
		arch    string
		keys    map[string]uint32
		tensors []string
		want    string
	}{
		{"eagle3", "eagle3", nil, []string{"fc.weight"}, "eagle3"},
		{"plain dflash", "dflash", nil, []string{"fc.weight"}, "dflash"},
		{"dspark", "dflash", nil, []string{"fc.weight", "markov_w1.weight"}, "dspark"},
		{"lilicorr", "dflash", map[string]uint32{"lilicorr.candidate_topk": 8}, []string{"fc.weight", "lilicorr.token_proj.weight"}, "lilicorr"},
		{"chat model", "qwen3", nil, []string{"blk.0.attn_q.weight"}, ""},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			meta, err := ParseGGUFMeta(writeHeadGGUF(t, tc.arch, tc.keys, tc.tensors))
			if err != nil {
				t.Fatalf("ParseGGUFMeta: %v", err)
			}
			if got := meta.DraftHeadKind(); got != tc.want {
				t.Errorf("DraftHeadKind() = %q, want %q", got, tc.want)
			}
		})
	}
}

// A head whose kind is known is matched on it, whatever its name says; a
// head without one falls back to its name. LiliCorr runs as draft-dflash.
func TestHeadsForMode(t *testing.T) {
	cands := []DraftCandidate{
		{ID: "lili", Filename: "Nemotron-LiliCorr-Q8_0.gguf", DraftHead: "lilicorr"},
		{ID: "spark", Filename: "Model-DFlash-Q8_0.gguf", DraftHead: "dspark"}, // misnamed: metadata wins
		{ID: "unread", Filename: "Model-LiliCorr-F16.gguf"},                    // no kind recorded: name decides
		{ID: "chat", Filename: "granite-4.2-3b-IQ4_NL.gguf"},
	}
	ids := func(mode string) []string {
		var out []string
		for _, c := range HeadsForMode(mode, cands) {
			out = append(out, c.ID)
		}
		return out
	}
	if got := ids("draft-dflash"); len(got) != 2 || got[0] != "lili" || got[1] != "unread" {
		t.Errorf("draft-dflash heads = %v, want [lili unread]", got)
	}
	if got := ids("draft-dspark"); len(got) != 1 || got[0] != "spark" {
		t.Errorf("draft-dspark heads = %v, want [spark]", got)
	}
	if got := ids("draft-eagle3"); len(got) != 0 {
		t.Errorf("draft-eagle3 heads = %v, want none", got)
	}
}

// A converted head carries no trunk, so it is never offered as a plain
// draft model, even when its recorded architecture happens to match.
func TestFindDraftCandidatesSkipsHeadsForDraft(t *testing.T) {
	r := &Registry{
		dataDir:   t.TempDir(),
		modelsDir: t.TempDir(),
		data: registryData{
			Models: map[string]*Model{
				"target": {ID: "target", Arch: "qwen3", SizeBytes: 30 << 30},
				"head":   {ID: "head", Arch: "qwen3", SizeBytes: 1 << 30, DraftHead: "dflash"},
				"small":  {ID: "small", Arch: "qwen3", SizeBytes: 1 << 30},
			},
			Configs: map[string]*ModelConfig{},
		},
	}
	got := r.FindDraftCandidates("target", "draft")
	if len(got) != 1 || got[0].ID != "small" {
		t.Errorf("draft candidates = %+v, want [small]", got)
	}
	if got := r.FindDraftCandidates("target", "draft-dflash"); len(got) != 2 {
		t.Errorf("draft-dflash candidates = %+v, want both for the picker", got)
	}
}
