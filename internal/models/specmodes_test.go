package models

import "testing"

// A head is matched to its model by name, whatever the separators; the
// MiniCPM-V-4.7 head is not taken for Qwen3.6, whose shape it shares.
func TestHeadsMadeFor(t *testing.T) {
	minicpm := DraftCandidate{ID: "openbmb--MiniCPM-V-4.7-35B-A3B-DSpark-gguf--dspark-MiniCPM-V-4_7-35B-A3B-Q8_0",
		Filename: "dspark-MiniCPM-V-4_7-35B-A3B-Q8_0.gguf"}
	qwen := DraftCandidate{ID: "z-lab--Qwen3.6-35B-A3B-DSpark-GGUF--Qwen3.6-35B-A3B-DSpark-Q8_0",
		Filename: "Qwen3.6-35B-A3B-DSpark-Q8_0.gguf"}
	heads := []DraftCandidate{minicpm, qwen}

	// The repository adds "-MTP"; the weights file's name gives the model.
	q := &Model{ModelID: "unsloth/Qwen3.6-35B-A3B-MTP-GGUF", Filename: "Qwen3.6-35B-A3B-UD-IQ4_NL.gguf"}
	if got := HeadsMadeFor(q, heads); len(got) != 1 || got[0].ID != qwen.ID {
		t.Errorf("Qwen3.6 heads = %+v, want only the Qwen3.6 head", got)
	}
	m := &Model{ModelID: "mradermacher/MiniCPM-V-4.7-35B-A3B-GGUF", Filename: "MiniCPM-V-4.7-35B-A3B.Q6_K.gguf"}
	if got := HeadsMadeFor(m, heads); len(got) != 1 || got[0].ID != minicpm.ID {
		t.Errorf("MiniCPM-V heads = %+v, want only the MiniCPM-V head", got)
	}
	// The upstream repository's name is used when the GGUF's is unlike it.
	b := &Model{ModelID: "someone/q36-gguf", Filename: "q36.gguf", BaseModelRepo: "Qwen/Qwen3.6-35B-A3B"}
	if got := HeadsMadeFor(b, heads); len(got) != 1 || got[0].ID != qwen.ID {
		t.Errorf("heads by base repository = %+v, want the Qwen3.6 head", got)
	}
	if got := HeadsMadeFor(nil, heads); got != nil {
		t.Errorf("no model: %+v, want none", got)
	}
}
