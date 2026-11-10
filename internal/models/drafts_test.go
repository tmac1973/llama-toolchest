package models

import (
	"slices"
	"testing"
)

func draftIDs(cands []DraftCandidate) []string {
	var out []string
	for _, c := range cands {
		out = append(out, c.ID)
	}
	return out
}

// The case found on a real machine: a reasoning finetune of Qwen3.8-27B,
// with the publisher's Qwen3.5-4B and a 3D-code finetune of it installed.
// Only the publisher's model is a drafter worth measuring.
func TestRelatedDraftsDropsAnUnrelatedFinetune(t *testing.T) {
	m := &Model{ModelID: "mradermacher/ThinkingCap-Qwen3.8-27B-i1-GGUF", BaseModelRepo: "Qwen/Qwen3.8-27B"}
	got := RelatedDrafts(m, []DraftCandidate{
		{ID: "coder", SizeGB: 4.2, ModelID: "ccore/Qwen3.5-4B-3D-Coder"},
		{ID: "glb-coder", SizeGB: 4.2, ModelID: "ccore/Qwen3.5-4B-GLB-3D-Coder"},
		{ID: "qwen", SizeGB: 2.6, ModelID: "unsloth/Qwen3.5-4B-GGUF", BaseModelRepo: "Qwen/Qwen3.5-4B"},
	})
	if want := []string{"qwen"}; !slices.Equal(draftIDs(got), want) {
		t.Errorf("drafts = %v, want %v", draftIDs(got), want)
	}
}

// Same publisher and same family beats the publisher alone, and a
// finetune of the same family that does not name its base still loses.
func TestRelatedDraftsPrefersTheSameFamily(t *testing.T) {
	m := &Model{ModelID: "unsloth/Qwen3.6-27B-GGUF", BaseModelRepo: "Qwen/Qwen3.6-27B"}
	got := RelatedDrafts(m, []DraftCandidate{
		{ID: "older", SizeGB: 1, ModelID: "unsloth/Qwen3.5-0.8B-GGUF", BaseModelRepo: "Qwen/Qwen3.5-0.8B"},
		{ID: "big", SizeGB: 3, ModelID: "unsloth/Qwen3.6-4B-GGUF", BaseModelRepo: "Qwen/Qwen3.6-4B"},
		{ID: "small", SizeGB: 1.5, ModelID: "bartowski/Qwen_Qwen3.6-1.7B-GGUF", BaseModelRepo: "Qwen/Qwen3.6-1.7B"},
		{ID: "tune", SizeGB: 0.5, ModelID: "someone/Qwen3.6-0.6B-Roleplay"},
	})
	if want := []string{"small", "big"}; !slices.Equal(draftIDs(got), want) {
		t.Errorf("drafts = %v, want %v", draftIDs(got), want)
	}
}

// With no base recorded anywhere, the family in the repository names is
// what links them.
func TestRelatedDraftsMatchesOnFamilyAlone(t *testing.T) {
	m := &Model{ModelID: "unsloth/gemma-4-27b-it-GGUF"}
	got := RelatedDrafts(m, []DraftCandidate{
		{ID: "other", SizeGB: 1, ModelID: "unsloth/Qwen3.5-0.8B-GGUF"},
		{ID: "gemma", SizeGB: 2, ModelID: "unsloth/gemma-4-1b-it-GGUF"},
	})
	if want := []string{"gemma"}; !slices.Equal(draftIDs(got), want) {
		t.Errorf("drafts = %v, want %v", draftIDs(got), want)
	}
}

// Nothing related: every candidate stays, smallest first.
func TestRelatedDraftsKeepsAllWhenNothingMatches(t *testing.T) {
	m := &Model{ModelID: "someone/Mystery-Model-GGUF"}
	got := RelatedDrafts(m, []DraftCandidate{
		{ID: "b", SizeGB: 2, ModelID: "x/Foo-1B"},
		{ID: "a", SizeGB: 1, ModelID: "y/Bar-1B"},
	})
	if want := []string{"a", "b"}; !slices.Equal(draftIDs(got), want) {
		t.Errorf("drafts = %v, want %v", draftIDs(got), want)
	}
}
