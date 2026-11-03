package models

import (
	"os"
	"testing"
)

// A reranker needs a classification head, cls.weight or cls.output.weight
// (bge has both, Qwen3 only the second). The parse records whether the
// tensor table has one, and only a parse that read the table counts.
func TestParseGGUFClassifierHead(t *testing.T) {
	tests := []struct {
		name    string
		tensors []string
		want    bool
	}{
		{"bge: hidden projection and score", []string{"token_embd.weight", "cls.weight", "cls.bias", "cls.output.weight", "cls.output.bias"}, true},
		{"qwen3: score only", []string{"token_embd.weight", "cls.output.weight"}, true},
		{"no head", []string{"token_embd.weight", "output.weight"}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			path := writeHeadGGUF(t, "bert", map[string]uint32{"bert.pooling_type": PoolingRank}, tt.tensors)
			meta, err := ParseGGUFMeta(path)
			if err != nil {
				t.Fatal(err)
			}
			if meta.HasClassifierHead != tt.want {
				t.Errorf("HasClassifierHead = %v, want %v", meta.HasClassifierHead, tt.want)
			}
			m := &Model{ID: "r"}
			meta.ApplyTo(m)
			if m.HasClassifierHead != tt.want || !m.ClassifierChecked {
				t.Errorf("after ApplyTo: HasClassifierHead %v, ClassifierChecked %v; want %v, true",
					m.HasClassifierHead, m.ClassifierChecked, tt.want)
			}
		})
	}

	// A header-only read never reaches the tensor table, so nothing is
	// known about the head.
	path := writeHeadGGUF(t, "bert", nil, []string{"cls.output.weight"})
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	meta, err := ParseGGUFMetaOnly(f)
	if err != nil {
		t.Fatal(err)
	}
	m := &Model{ID: "r"}
	meta.ApplyTo(m)
	if m.ClassifierChecked {
		t.Error("a metadata-only parse set ClassifierChecked")
	}
}

func TestRerankHeadMissing(t *testing.T) {
	tests := []struct {
		name string
		m    Model
		want bool
	}{
		{"reranker without a head", Model{ID: "Qwen--Qwen3-Reranker-0.6B", ClassifierChecked: true}, true},
		{"reranker with a head", Model{ID: "Qwen--Qwen3-Reranker-0.6B", ClassifierChecked: true, HasClassifierHead: true}, false},
		{"not read yet", Model{ID: "Qwen--Qwen3-Reranker-0.6B"}, false},
		{"embedding model", Model{ID: "e", Embedding: true, ClassifierChecked: true}, false},
		{"chat model", Model{ID: "c", ClassifierChecked: true}, false},
	}
	for _, tt := range tests {
		if got := tt.m.RerankHeadMissing(); got != tt.want {
			t.Errorf("%s: RerankHeadMissing() = %v, want %v", tt.name, got, tt.want)
		}
	}
}

func TestCapabilitiesRerank(t *testing.T) {
	// Qwen3-Reranker's file has Qwen3's chat template, which handles tools.
	r := &Model{ID: "r", PoolingType: PoolingRank, Arch: "qwen3", HasChatTemplate: true, SupportsTools: true}
	if got := r.Capabilities(nil); len(got) != 1 || got[0] != "rerank" {
		t.Errorf("reranker capabilities = %v, want [rerank]", got)
	}
}
