package autoconfig

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/llmcall"
	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/modelsource"
)

const runModelID = "unsloth--Qwen3.5-9B-GGUF--Qwen3.5-9B-Q4_K_M"

func runRegistry(t *testing.T) *models.Registry { return runRegistryWithMTP(t, 1) }

// runRegistryWithMTP registers the test model; nextN is how many built-in
// MTP draft layers it has.
func runRegistryWithMTP(t *testing.T, nextN int) *models.Registry {
	t.Helper()
	dir := t.TempDir()
	reg := models.NewRegistry(dir, filepath.Join(dir, "models"))
	m := &models.Model{ID: runModelID, ModelID: "testorg/Test-9B-GGUF", Filename: "Qwen3.5-9B-Q4_K_M.gguf",
		NLayers: 36, NEmbd: 4096, NHead: 32, NKVHead: 8, ContextLength: 262144, SizeBytes: 5 << 30, NextNLayers: nextN}
	if err := reg.Add(m); err != nil {
		t.Fatal(err)
	}
	return reg
}

func helperServer(t *testing.T, answer string) *llmcall.Client {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{"choices": []map[string]any{{"message": map[string]any{"content": answer}, "finish_reason": "stop"}}})
	}))
	t.Cleanup(srv.Close)
	return &llmcall.Client{Backend: backend{srv.URL}}
}

const cardAnswer = `{"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0, "presence_penalty": null,
	"repeat_penalty": null, "sampling_quote": "use temperature=0.6", "thinking": null, "thinking_quote": "",
	"draft_method": "draft-mtp", "draft_repo": null, "assist_mode": null, "speculative_quote": "supports MTP",
	"recommended_context": 16384, "context_quote": "16K is enough", "other_notes": ["Use the chat template."]}`

func gpu24() models.Hardware {
	return models.Hardware{GPUs: []models.GPUSpec{{Index: 0, VRAMTotalMiB: 24 * 1024}}, LogicalCores: 16, RAMTotalMiB: 64 * 1024}
}

func TestRunMergesFitAndCard(t *testing.T) {
	reg := runRegistry(t)
	var steps []string
	d := Deps{
		Registry: reg, Hardware: gpu24(), HFBase: "https://hf.test",
		Fetcher: &fakeFetcher{pages: map[string]string{
			"https://hf.test/testorg/Test-9B-GGUF/raw/main/README.md": readTestdata(t, "unsloth_card.md"),
		}},
		LLM: helperServer(t, cardAnswer), HelperID: "helper", HelperContext: 16384,
		Progress: func(s string) { steps = append(steps, s) },
	}
	res, err := Run(context.Background(), d, runModelID, models.ContextMedium)
	if err != nil {
		t.Fatal(err)
	}
	p := res.Proposed
	if p.ContextSize != 32768 {
		t.Errorf("context = %d; the card's 16K must not override the medium class", p.ContextSize)
	}
	if p.SpecType != "draft-mtp" || p.DraftMax == 0 {
		t.Errorf("built-in MTP not turned on with its defaults: %q/%d", p.SpecType, p.DraftMax)
	}
	if p.Temperature == nil || *p.Temperature != 0.6 || p.TopK == nil || *p.TopK != 20 {
		t.Errorf("card sampling not applied: %+v", p)
	}
	if err := models.ValidateProfileConfig(p); err != nil {
		t.Errorf("proposal fails validation: %v", err)
	}
	var mtpNotes, contextNotes int
	var sawOther bool
	for _, n := range res.Notes {
		switch {
		case n.Field == "spec_type":
			mtpNotes++
		case n.Field == "context_size":
			contextNotes++
		case strings.Contains(n.Reason, "Use the chat template"):
			sawOther = true
		}
	}
	if mtpNotes != 1 {
		t.Errorf("speculative decoding has %d notes, want 1 (the model file; the card agreed)", mtpNotes)
	}
	if contextNotes != 2 {
		t.Errorf("context has %d notes, want the fit's and the card's recommendation", contextNotes)
	}
	if !sawOther {
		t.Error("the card's other advice is missing")
	}
	if len(steps) < 4 || !strings.Contains(steps[0], "fits") {
		t.Errorf("progress steps = %v", steps)
	}
	if len(res.CardSources) != 1 {
		t.Errorf("card sources = %v", res.CardSources)
	}
}

// Maximum means maximum. A 48 GiB machine has room for this model's whole
// 262,144-token context, and the run used to hand back 131,072 because the
// helper model read the card's "262,144 tokens by default, extensible to
// 1,010,000 with YaRN" as a recommendation of 131,072. The card is worth
// recording and is not worth overruling an explicit choice with.
func TestRunKeepsTheMaximumContextTheCardDisagreesWith(t *testing.T) {
	const answer = `{"temperature": null, "top_p": null, "top_k": null, "min_p": null, "presence_penalty": null,
		"repeat_penalty": null, "sampling_quote": "", "thinking": null, "thinking_quote": "",
		"draft_method": null, "draft_repo": null, "assist_mode": null, "speculative_quote": "",
		"recommended_context": 131072, "context_quote": "262,144 tokens by default", "other_notes": []}`
	hw := models.Hardware{GPUs: []models.GPUSpec{{Index: 0, VRAMTotalMiB: 48 * 1024}}, LogicalCores: 16, RAMTotalMiB: 64 * 1024}
	d := Deps{
		Registry: runRegistry(t), Hardware: hw, HFBase: "https://hf.test",
		Fetcher: &fakeFetcher{pages: map[string]string{
			"https://hf.test/testorg/Test-9B-GGUF/raw/main/README.md": readTestdata(t, "unsloth_card.md"),
		}},
		LLM: helperServer(t, answer), HelperID: "helper",
	}
	res, err := Run(context.Background(), d, runModelID, models.ContextMax)
	if err != nil {
		t.Fatal(err)
	}
	if res.Proposed.ContextSize != 262144 {
		t.Errorf("context = %d, want the model's full 262144", res.Proposed.ContextSize)
	}
	if res.Proposed.ContextSize != res.Fit.Config.ContextSize {
		t.Errorf("context %d differs from what the fit decided (%d)", res.Proposed.ContextSize, res.Fit.Config.ContextSize)
	}
	var kept bool
	for _, n := range res.Notes {
		if n.Field == "context_size" && strings.Contains(n.Reason, "size you chose is kept") {
			kept = true
		}
	}
	if !kept {
		t.Errorf("the card's recommendation is not recorded: %+v", res.Notes)
	}
}

// A context that does not fit is reduced by the planner, and the KV cache
// drops to 8 bits before the context is halved — never to 4 bits, which
// the planner does not offer at all.
func TestRunTradesKVPrecisionForContextBeforeHalvingIt(t *testing.T) {
	reg := runRegistry(t)
	hw := models.Hardware{GPUs: []models.GPUSpec{{Index: 0, VRAMTotalMiB: 32 * 1024}}, LogicalCores: 16, RAMTotalMiB: 64 * 1024}
	res, err := Run(context.Background(), Deps{Registry: reg, Hardware: hw}, runModelID, models.ContextMax)
	if err != nil {
		t.Fatal(err)
	}
	if res.Proposed.KVCacheQuant != "q8_0" {
		t.Errorf("kv_cache_quant = %q, want q8_0", res.Proposed.KVCacheQuant)
	}
	if res.Proposed.ContextSize != 262144 {
		t.Errorf("context = %d; an 8-bit KV cache fits the full 262144 here", res.Proposed.ContextSize)
	}
}

func TestRunWithoutHelperUsesTheFitAlone(t *testing.T) {
	reg := runRegistry(t)
	res, err := Run(context.Background(), Deps{Registry: reg, Hardware: gpu24()}, runModelID, models.ContextShort)
	if err != nil {
		t.Fatal(err)
	}
	if res.Proposed.ContextSize != 8192 || res.Proposed.Temperature != nil {
		t.Errorf("proposal = %+v", res.Proposed)
	}
	found := false
	for _, n := range res.Notes {
		if strings.Contains(n.Reason, "No helper model is installed") {
			found = true
		}
	}
	if !found {
		t.Error("no note says the card was not read")
	}
}

func TestRunSurvivesAHelperFailure(t *testing.T) {
	reg := runRegistry(t)
	d := Deps{Registry: reg, Hardware: gpu24(), HFBase: "https://hf.test",
		Fetcher: &fakeFetcher{pages: map[string]string{
			"https://hf.test/testorg/Test-9B-GGUF/raw/main/README.md": readTestdata(t, "unsloth_card.md"),
		}},
		LLM: helperServer(t, `not json at all`), HelperID: "helper"}
	res, err := Run(context.Background(), d, runModelID, models.ContextMedium)
	if err != nil {
		t.Fatalf("a helper failure should not fail the run: %v", err)
	}
	found := false
	for _, n := range res.Notes {
		if strings.Contains(n.Reason, "could not read the model card") {
			found = true
		}
	}
	if !found || res.Proposed.ContextSize != 32768 {
		t.Errorf("fallback result = %+v", res)
	}
}

// When the card recommends a draft method this machine cannot serve, the
// note says what can be done about it, and never points at an empty list
// of downloads.
func TestRunFinishesTheSpeculativeNote(t *testing.T) {
	const answer = `{"temperature": null, "top_p": null, "top_k": null, "min_p": null, "presence_penalty": null,
		"repeat_penalty": null, "sampling_quote": "", "thinking": null, "thinking_quote": "",
		"draft_method": "draft-mtp", "draft_repo": null, "assist_mode": null, "speculative_quote": "use MTP",
		"recommended_context": null, "context_quote": "", "other_notes": []}`
	deps := func(hub Hub) Deps {
		return Deps{
			Registry: runRegistryWithMTP(t, 0), Hardware: gpu24(), HFBase: "https://hf.test", Hub: hub,
			Fetcher: &fakeFetcher{pages: map[string]string{
				"https://hf.test/testorg/Test-9B-GGUF/raw/main/README.md": readTestdata(t, "unsloth_card.md"),
			}},
			LLM: helperServer(t, answer), HelperID: "helper",
		}
	}
	find := func(res *Result) string {
		for _, n := range res.Notes {
			if n.Field == "spec_type" && n.Origin == "model card" {
				return n.Reason
			}
		}
		return ""
	}

	// Nothing to download: say so.
	res, err := Run(context.Background(), deps(fakeHub{}), runModelID, models.ContextMedium)
	if err != nil {
		t.Fatal(err)
	}
	reason := find(res)
	if !strings.Contains(reason, "no MTP layers") || !strings.Contains(reason, "stays off") {
		t.Errorf("note = %q", reason)
	}
	if strings.Contains(reason, "suggested downloads") {
		t.Error("the note points at downloads that do not exist")
	}

	// A head published next to the model: point at it.
	hub := fakeHub{repos: map[string][]modelsource.File{
		"testorg/Test-9B-GGUF": {{Filename: "Test-9B-MTP-Q8_0.gguf", Quant: "Q8_0", Size: 1 << 20}},
	}}
	res, err = Run(context.Background(), deps(hub), runModelID, models.ContextMedium)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Suggestions) != 1 || !strings.Contains(find(res), "suggested downloads below") {
		t.Errorf("note = %q, suggestions = %+v", find(res), res.Suggestions)
	}
}

// Built-in MTP: turned on when it is off, and said to be kept when it is
// already on. Saying nothing about it reads as having missed it.
func TestRunSpeaksAboutBuiltInMTP(t *testing.T) {
	note := func(res *Result) string {
		for _, n := range res.Notes {
			if n.Field == "spec_type" && n.Origin == "model file" {
				return n.Reason
			}
		}
		return ""
	}

	reg := runRegistry(t) // the test model has one MTP layer
	res, err := Run(context.Background(), Deps{Registry: reg, Hardware: gpu24()}, runModelID, models.ContextShort)
	if err != nil {
		t.Fatal(err)
	}
	if res.Proposed.SpecType != "draft-mtp" || !strings.Contains(note(res), "own draft layers") {
		t.Errorf("MTP not turned on: %q / %q", res.Proposed.SpecType, note(res))
	}

	// Already on: kept, and said so.
	cfg, _ := reg.GetConfig(runModelID)
	on := *cfg
	on.SpecType, on.DraftMax = "draft-mtp", 6
	if err := reg.SetConfig(runModelID, &on); err != nil {
		t.Fatal(err)
	}
	res, err = Run(context.Background(), Deps{Registry: reg, Hardware: gpu24()}, runModelID, models.ContextShort)
	if err != nil {
		t.Fatal(err)
	}
	if res.Proposed.SpecType != "draft-mtp" || res.Proposed.DraftMax != 6 {
		t.Errorf("an existing MTP setting was changed: %+v", res.Proposed)
	}
	if !strings.Contains(note(res), "Already on and kept") {
		t.Errorf("nothing said about the MTP already on: %q", note(res))
	}

	// Another draft method: kept, with MTP mentioned as an alternative.
	on.SpecType, on.DraftModelPath = "draft", "/models/small.gguf"
	if err := reg.SetConfig(runModelID, &on); err != nil {
		t.Fatal(err)
	}
	res, err = Run(context.Background(), Deps{Registry: reg, Hardware: gpu24()}, runModelID, models.ContextShort)
	if err != nil {
		t.Fatal(err)
	}
	if res.Proposed.SpecType != "draft" || !strings.Contains(note(res), "Autotune can compare") {
		t.Errorf("draft method not kept, or MTP not mentioned: %q / %q", res.Proposed.SpecType, note(res))
	}
}

// The n-gram assist is never turned on unmeasured, and the note names the
// combination with a draft method, which is the part nobody guesses.
func TestRunLeavesTheNgramAssistToAutotune(t *testing.T) {
	reg := runRegistry(t) // has built-in MTP
	res, err := Run(context.Background(), Deps{Registry: reg, Hardware: gpu24()}, runModelID, models.ContextShort)
	if err != nil {
		t.Fatal(err)
	}
	if res.Proposed.SpecAssist != "" {
		t.Errorf("an n-gram assist was turned on without measuring it: %q", res.Proposed.SpecAssist)
	}
	var note string
	for _, n := range res.Notes {
		if strings.Contains(n.Reason, "n-gram assist") {
			note = n.Reason
		}
	}
	if !strings.Contains(note, "alongside draft-mtp") {
		t.Errorf("the note does not name running it alongside the draft method: %q", note)
	}
}
