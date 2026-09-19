# Phase 08 — Model card reading and advice extraction

**Depends on:** 02 (`ProfileNote` origins), 05 (`Model.NextNLayers`), 06
(`ContextClass`, `FitResult`), 07 · **Enables:** 09

## Goal
Fetch a model's documentation, ask the helper model for model-specific advice
as JSON, check every value against what the app allows, and find draft files
that the card or the model family suggests. The output is a set of proposed
settings with reasons. This phase saves and displays nothing.

## Files touched
- `internal/autoconfig/card.go` (new): fetch and trim the model card.
- `internal/autoconfig/advice.go` (new): the prompt, the JSON schema, the
  call, and validation.
- `internal/autoconfig/drafts.go` (new): finding draft files.
- `internal/autoconfig/*_test.go` (new), with testdata READMEs.
- `internal/presets/fetcher.go`: export `CachedGet(ctx, url, withAuth bool)`
  so the autoconfig package reuses the cache and the HF token. Today's
  `cachedGet` never sends the token; the new parameter keeps existing callers
  on `false`.
- `internal/presets/hf.go`: export `ResolveBaseRepo`.

## Steps
1. **Fetching the card.** `FetchCard(ctx, m *models.Model) (Card, error)`:
   - GET `https://huggingface.co/{m.ModelID}/raw/main/README.md` (the GGUF
     repo), and the same for `ResolveBaseRepo(m.ModelID)` when it is
     different.
   - Use `m.BaseModelRepo` when it is already set.
   - Use the cached fetcher with auth, because gated repos need the token.
   - A missing README is not an error; the card is just empty.
2. **Trimming.**
   - Strip YAML front matter, HTML tags, image links and badge lines.
   - Keep sections whose heading or body mentions any of: recommend, sampling,
     temperature, top_p, top_k, min_p, llama.cpp, llama-server, gguf, context,
     thinking, reasoning, speculative, mtp, draft, eagle, dflash, jinja,
     chat template, tool.
   - Cap the result at 24,000 characters (about 6,000 tokens), keeping the GGUF
     repo's sections before the base repo's.
   - The helper needs a context of at least 16,384 tokens so the prompt and
     the answer fit. Phase 07's settings handler and download completion
     raise the helper model's `ContextSize` to 16,384 when it is lower, and
     say so in the Settings page note.
3. **The advice schema** (strict JSON schema; every field is optional and
   carries a `reason` string):
   - `sampling`: temperature, top_p, top_k, min_p, presence_penalty,
     repeat_penalty;
   - `thinking`: `"on" | "off" | "model default"`;
   - `jinja`: bool;
   - `speculative`: `draft_method` (enum from `models.DraftModes()` plus
     `"none"`), `draft_repo` (string), `assist_mode` (enum from
     `models.AssistModes()` plus `"none"`);
   - `recommended_context`: int;
   - `other_notes`: string[], for anything else the card says about running
     the model.
4. **The prompt** (system message) tells the model to:
   - use only the card text;
   - leave out any field the card does not state;
   - quote the card in each `reason`.
   Temperature is 0.
5. **Validation** (`Validate(adv Advice, m *models.Model, class models.ContextClass, fit models.FitResult) []Proposal`). Each `Proposal` is
   `{Field, Value, Reason, Origin: "model card"}`.
   - Sampling values are range-checked (temperature 0–2, top_p 0–1,
     top_k 0–1000, min_p 0–1, penalties 0–2). Out-of-range values are
     dropped, with a note.
   - `draft_method: "draft-mtp"` is kept only if `m.NextNLayers > 0`
     or a separate MTP head is attached (`MtpPath`). Otherwise it becomes a
     draft-file suggestion (step 6).
   - Other draft methods are kept only if `FindDraftCandidates(id, mode)`
     returns an installed file. Otherwise they become a suggestion.
   - `recommended_context` becomes a `context_size` proposal only when
     `class == "max"` and it is lower than `fit.Config.ContextSize`. This is
     the one case where advice may change a fit field, and only downwards.
     Otherwise it is shown as a note.
   - `other_notes` are never turned into flags. They are passed through as
     "The model card also mentions: …".
   - **Structured sources win over the LLM.** Where `m.SamplingPresets`
     already has a preset (origin "publisher preset") from generation_config.json, GGUF metadata or
     Unsloth docs, that preset's values are used, with origin "publisher
     preset", and the LLM's sampling values are shown only when they differ,
     as a note.
6. **Draft-file suggestions** (`FindDraftSuggestions(ctx, m, adv) []DraftSuggestion{Repo, File, SizeBytes, Mode, Why}`):
   - The `draft_repo` named by the card, if its file list has a GGUF.
   - For models with built-in MTP (`m.NextNLayers > 0`), nothing, because
     the draft layers are part of the model file.
   - For models with a separate MTP head in the same repo (a file that
     `IsMTPHead` recognizes, with a name containing `mtp`), that file.
   - Otherwise, a family search:
     - **Family name:** take the base repo's name (the part after `/`) and
       cut it at the first match of `-\d+(\.\d+)?[BbMm]\b`, e.g.
       `Qwen3.5-9B-Instruct` → `Qwen3.5`.
     - **Parameter count:** parse it from the same regex match, e.g. 9B.
     - **Query:** `"<family> GGUF"` with the existing
       `huggingface.Client.Search`, keeping only results whose owner equals
       the GGUF repo's owner.
     - **Choice:** the result whose parsed parameter count is the smallest,
       below 2B and at most a quarter of the main model's. From its files,
       take Q8_0 if present, else Q4_K_M.
     - Nothing is suggested when no result qualifies, or when the name has no
       size token.
     - Whether the vocabularies match is checked after download, by the
       existing `FindDraftCandidates` rules.
   - Only candidates not already installed are returned.
7. **No model card at all.** Return an empty advice set with a single note,
   "No model card was found; settings come from the hardware fit and
   defaults only." Autoconfigure still works in this case.

## Build gate
`go build ./... && go vet ./... && go test ./...`

## Test plan
- Card trimming on three testdata READMEs (an Unsloth GGUF card, a Qwen base
  card, a card with no llama.cpp section): expected sections are kept and the
  cap is respected.
- Validation table tests:
  - `recommended_context` lowers the context only under "max";
  - an out-of-range temperature is dropped;
  - draft-mtp on a model without NextN becomes a suggestion;
  - a publisher preset overrides LLM sampling;
  - `other_notes` produce no flag.
- The advice call against a fake router returning canned JSON.
- The family-name and parameter-count parsing on `Qwen3.5-9B-Instruct`,
  `gemma-4-27b-it` and `Llama-3.3-70B-Instruct`, plus a name with no size.
- Draft suggestions with a fake HF client (named repo, same-repo MTP head,
  family search).
- Manual: run `go test ./internal/autoconfig -run TestLiveAdvice -tags live`
  (a test file behind the build tag, pointed at a running dev server with
  the helper installed) on one real model, and read the proposals.

## Commit
`feat(autoconfig): read a model card and extract checked settings advice`

## Rollback
Revert the commit. Nothing outside `internal/autoconfig` calls it until
Phase 09. The exported fetcher helpers can stay.
