# Phase 03 — Recommendation engine

**Depends on:** 01 (remote metadata, derived sizes), 02 (supported
architectures) · **Enables:** 04 (the feed), 05 (seeding)

## Goal
A package that, given this machine, builds a list of GGUF models from
HuggingFace that should run well here. For each model and each context
class, it picks a quant and plans it with `models.PlanFit`. It then
computes four orders for each class, so the UI can switch category or
context with no further requests. Two endpoints serve the result. There
is no UI in this phase beyond JSON.

## Files touched
- `internal/recommend/` (new package):
  - `profile.go`: the hardware profile and its cache key.
  - `publishers.conf` (embedded): the trusted publishers, in order of
    preference.
  - `candidates.go`: HuggingFace queries, the servable filter, the size
    filter.
  - `group.go`: grouping repos by base model and choosing a publisher.
  - `finalist.go`: file tree and metadata for each finalist, cached on disk.
  - `quant.go`: the quant picker.
  - `score.go`: the four orders.
  - `engine.go`: the pool, its cache and its age.
  - Tests for each file, with a fake hub.
- `internal/huggingface/client.go`:
  - `Candidates(ctx, q ListQuery)` for list queries with `expand[]`;
  - the tree listing follows `Link: rel="next"` pages;
  - one retry on HTTP 429.
- `internal/api/recommend.go` (new): the handlers and the profile adapter.
- `internal/api/server.go`: routes and the engine field.

## Steps

### 1. Hardware profile
- `Profile` holds:
  - the discrete GPUs: name and total MiB each. The integrated-GPU rule is
    `spanAllOption`'s, so the feed and Autoconfigure use the same cards;
  - system RAM in MiB and logical cores;
  - the active build ID, whether its architecture list is known, and a hash
    of that list.
- Built from `Server.hardware()` (`internal/api/autoconfig.go:66`) and
  `supportedArchs()` (phase 02).
- `Key()` is a hash of all of the above. Any change means a new pool on the
  next click. A machine with no GPU reading gives an empty pool with
  `Unavailable: "No GPU was detected, so there is nothing to plan for.
  Search below still works."`

### 2. Candidate pool
**HuggingFace list queries.** Every query uses `filter=gguf`, `direction=-1`
and these `expand[]` fields: `author`, `downloads`, `likes`, `tags`,
`gated`, `private`, `createdAt`, `lastModified`, `pipeline_tag`, `cardData`,
`gguf`, `sha`. Once any `expand[]` is present, the Hub returns only the
fields named, so every field used must be listed. Add a test that checks
the list against the struct fields, like vllm-toolchest's
`TestSearchExpandsEveryFieldItUses`.

The queries:
- `pipeline_tag=text-generation` and `pipeline_tag=image-text-to-text`,
  each sorted by `downloads` and by `trendingScore`, `limit=100`. That is
  4 queries.
- For each trusted publisher, `author=<name>&sort=createdAt&limit=50`.
  Trusted publishers release GGUFs of new models within days, so this is
  where "Newest" gets its candidates. That is about 10 queries.

At most 4 queries run at once. Results are merged by repo ID. A query that
fails is skipped. The build fails only if every query fails, with
`Unavailable: "Could not reach Hugging Face: <reason>. Search below still
works."`

**Trusted publishers** (`publishers.conf`, one per line, in preference
order): unsloth, bartowski, ggml-org, lmstudio-community, Qwen, google,
mistralai, microsoft, ibm-granite, nvidia. Being on the list raises a repo's
coarse rank and decides which publisher a card names (step 3). It never
lowers a repo's rank.

**Servable filter.** Drop silently:
- `pipeline_tag` other than text generation or image-text-to-text (for
  example embedding, reranking, speech, object detection). An empty tag is
  kept.
- No `gguf` object or no `gguf.architecture`.
- `gguf.total` under 1 billion parameters. Small models are real models,
  but they would fill Fastest on any machine with a GPU. Users who want one
  can search for it.
- MTP heads and draft models: the architecture is a known MTP head
  architecture (the list `IsMTPHead` uses), or the repo name contains
  `draft`, `-mtp`, `eagle` or `dflash` (case-insensitive).
- Fewer than 50 downloads, unless the author is a trusted publisher.
- Private repos.

Mark as **unverified** (listed, with a reason):
- The architecture is not in the active build's list (only when the list is
  known): "This llama.cpp build does not know the `<arch>` architecture. A
  newer build may support it."

### 3. Group by base model
- The group key is the first entry of `cardData.base_model`, in lower case.
  If it is missing, use the repo name without a trailing `-GGUF`, `_GGUF` or
  `.gguf`, in lower case.
- The repos in a group are ordered as follows. The first is the card's
  publisher, and the rest go under Details.
  1. a repo by the base model's own author (an official quant);
  2. trusted publishers, in list order;
  3. everyone else, by downloads.
- Group values:
  - parameter count and architecture from the chosen repo's `gguf`;
  - trained context from `gguf.context_length`;
  - downloads and likes summed over the group;
  - `Released` is the earliest `createdAt` in the group, which is close to
    when the model came out. `lastModified` is not used, because publishers
    re-upload often;
  - `Gated` from the chosen repo.

### 4. Size filter and finalists
- The smallest possible weights are `params × 3.0 / 8` bytes (about IQ3_XXS).
- The budgets are the ones `PlanFit` uses: VRAM = Σ(card − max(1 GiB, 8%)),
  RAM = total − max(4 GiB, 10%).
- Drop a group whose smallest possible weights exceed VRAM + RAM. Whether a
  model is MoE is not known until its header is read, so the RAM is allowed
  for every model here. Phase 03 step 6 then rejects dense models that would
  need it.
- **Coarse score:**
  `0.45·pct(downloads) + 0.30·pct(Released) + 0.15·trusted + 0.10·pct(likes)`.
  `pct` is the percentile rank within the pool, with ties sharing their
  average rank, as in vllm-toolchest's `percentiles`.
- **Finalists: 40 groups.** Some places are reserved by size, measured as
  the weights at about 4.8 bits per weight (Q4_K_M) against the VRAM budget:
  - 10 places for small (under 25% of the budget);
  - 14 places for medium (25–90%);
  - 10 places for large (over 90%, which needs a low-bit quant or expert
    offload).
  - The remaining places go to the best remaining coarse scores.
  - Without reserved places, the first live build in vllm-toolchest had
    700 GB models filling half the list.

### 5. Finalist detail
For each finalist group, with up to 8 running at once:
- Get the chosen repo's file tree (`GetFiles`, with paging). Group shards
  (`modelsource.GroupShards`). Split the files into:
  - main model files (quants);
  - mmproj files: the group is marked "vision", and the smallest mmproj
    size is recorded;
  - MTP heads, which are ignored.
- Read the metadata of the largest main file with `ProbeMeta` (phase 01,
  cached on disk). Use the same `metaProbeFile` as the browse tab: the
  largest file is certainly the main model, and a header costs the same
  whatever the file size.
- Cache the file list at `<DataDir>/recommend/repos/<safe id>@<sha>.json`.
  The `sha` comes from the list query, so a new commit gives a new entry.
  Entries never expire. A warm build makes no tree or header requests.
- **Unverified** reasons, each a plain sentence:
  - "The file list could not be read from Hugging Face."
  - "The repo has no model files."
  - "The model's description could not be read from its file."
  - "The model's description is missing values the estimate needs."
    (`NLayers` or `NEmbd` is 0.)

### 6. Quant picker (`quant.go`)
For each finalist, build one `models.Model` per main file. Use the shared
metadata, the file size and `DeriveSizes(meta, size, params)`, then
`meta.ApplyTo(m)`. Work out the bits per weight:
`bpw = size × 8 / params`.

Leave out files under 3.0 bpw. The quant label is only shown, never used for
decisions, so unsloth's "UD" quants and other mixed quants are judged by
their real size.

For each class (8K, 32K, 128K, max):
- `requested = min(class tokens, trained context)`. Under "max" it is the
  trained context. A model trained for less than the class has **no pick**
  at that class. The feed hides it there and counts it (phase 04).
- Run `models.PlanFit(m, models.DefaultConfig(), hw, class)` for each file,
  largest first. Read the result as:
  - **on GPU**: `Fits`, `ContextSize == requested`, `GPULayers` covers
    every layer, `CPUMoE == 0`;
  - **experts in RAM**: `Fits`, `ContextSize == requested`, `CPUMoE > 0`;
  - anything else (a shorter context, layers on the CPU) does not count.
- Choose the first match in this order:
  1. the largest file of at least 4.2 bpw (about IQ4_XS and up) on GPU;
  2. the largest file of 3.0–4.2 bpw on GPU;
  3. MoE only: the largest file of at least 4.2 bpw with experts in RAM;
  4. MoE only: the largest file of 3.0–4.2 bpw with experts in RAM.
- The live build (step 10) should check whether 3 ought to come before 2
  for MoE models, where a few expert layers in RAM may cost less than a
  low-bit quant.
- **Fewest GPUs.** For the chosen file, find the smallest number of cards
  (largest first) on which `PlanFit` still gives an "on GPU" result at the
  same context. This is shown on the card ("fits on 1 of 3 GPUs"). The
  plan itself, and the seeding in phase 05, use the same placement as
  Autoconfigure, which is all discrete GPUs.
- The pick records: file name (first shard), quant label, size, bpw,
  context, KV cache type, estimate and budget in GiB, `CPUMoE`,
  `CPURAMGiB`, fewest GPUs, and whether it is "on GPU" or "experts in RAM".

### 7. Scoring (`score.go`)
For each class, score only the groups with a pick at that class. The
weights below are starting values; step 10 records any changes.
- **headroom** = `(budget − estimate) / budget`.
- **Best quality:**
  `0.75·pct(params × q(bpw)) + 0.15·headroom + 0.10·onGPU`.
  - `q` = 1.0 at 5.5 bpw or more, 0.97 at 4.2–5.5, 0.85 at 3.5–4.2, and
    0.75 at 3.0–3.5.
  - This makes a much larger model at a 4-bit quant beat a smaller one at
    8 bits, while a 3-bit quant has to be clearly larger to win.
- **Fastest:** token generation in llama.cpp is mostly limited by how many
  bytes are read per token.
  - `readGB = (weights − experts) + experts × expert_used / expert_count`;
    for a dense model, `readGB = weights`.
  - `fastest = clamp01(0.60·pct(1/readGB) + 0.25·pct(params) + 0.15·headroom − 0.40·expertsInRAM)`.
  - The parameter term keeps the list from being only the smallest models.
- **Longest context:** order by the largest class that has a pick, then by
  trained context, then by the quality score. This order does not depend on
  the selected class.
- **Newest:** `Released`, newest first.
- **Ties:** more downloads first, then the repo ID alphabetically.

### 8. The pool (`engine.go`)
- One pool per profile key, held in memory. It contains the groups, their
  picks per class, the 16 orders (4 categories × 4 classes) and the
  unverified list, and records when it was built.
- **Concurrency:** one build at a time. A second request waits for the
  running build rather than starting another.
- The whole build is capped at 2 minutes. On timeout, finalists not yet
  fetched are marked unverified ("Hugging Face did not answer in time.
  Refresh to try again.").
- **Age:** after 6 hours the pool is still served, but marked `Stale`. Only
  Refresh rebuilds it. Nothing runs in the background.
- `Suggested(repo, file string) (class ContextClass, ok bool)` reports
  whether a file is the pick for some class in the current pool, and for
  which class. Phase 05 uses it.

### 9. Endpoints (`internal/api/recommend.go`)
- `GET /api/recommend?intent=quality|fastest|context|newest&ctx=8k|32k|128k|max`
  - The defaults are `quality` and `32k`.
  - It builds the pool on first use, then serves the requested order.
  - It always returns 200. A failure is a sentence in `unavailable`.
  - Phase 04 adds the htmx partial. In this phase every request gets JSON.
- `POST /api/recommend/refresh` rebuilds the pool and returns JSON.
- **The HuggingFace client:**
  - On HTTP 429, wait for `Retry-After` (at most 10 seconds) and retry
    once. Otherwise fail that request.
  - The token is sent when set. The engine looks up the client on each
    build (as vllm-toolchest's `serverHub` does), so a token set later in
    Settings is used without a restart.

### 10. First live build
- Deploy to compute2 (3× A4000, 48 GB) and build the feed.
- Record in an "As built" section below:
  - the request count and time, cold and warm;
  - how many repos were verified, unverified and dropped;
  - the top 8 of each order at 32K;
  - every scoring or filter change needed to make those lists sensible.
- If possible, also check a machine with one small card, using compute2
  with only one GPU visible, so the feed is not tuned only for large
  machines.

## Build gate
`go build ./... && go vet ./... && go test ./...`

## Test plan
A fake hub with a small sample market, as in vllm-toolchest's
`recommend_test.go`.
- **Filter:** an embedding model, a `locateanything` repo, a 0.5B model, a
  draft repo and a 10-download fine-tune are dropped. A trusted publisher's
  10-download repo is kept.
- **Unsupported architecture:** listed as unverified with the reason. When
  the architecture list is unknown, it is not checked.
- **Grouping:** three publishers of one base model make one group; the base
  author's own repo is chosen first; the rest are listed in order.
- **Size filter:** a 1T-parameter model on a 48 GB / 64 GB machine is
  dropped before any tree or header request.
- **Reserved places:** a pool of 60 large models and 10 small ones still
  gives 10 small finalists.
- **Quant picker:**
  - a dense 32B on 48 GB gets a ≥4.2 bpw file at 32K;
  - the same model at 128K moves to a smaller file or 8-bit KV;
  - a dense 70B gets a 3–4.2 bpw pick or none, and never "layers on CPU";
  - a 30B-A3B MoE on 16 GB gets an "experts in RAM" pick;
  - a model trained for 32K has no pick at 128K;
  - a file under 3.0 bpw is never picked;
  - a UD quant is judged by its measured bpw, not its label.
- **Scoring:** a weight-only, very small model never leads Fastest; an
  "experts in RAM" pick ranks below an "on GPU" pick of similar size under
  Fastest; Newest follows `Released`, not `lastModified`.
- **Cache:** a second build with the same repo SHAs makes no tree or header
  requests; a new SHA refetches that repo only.
- **Pool:** a different profile key rebuilds; a 7-hour-old pool is served as
  stale; two requests during one build start only one build.
- **Unavailable:** no GPU; every query failing.
- **Expand list:** every field the code reads is named in `expand[]`.
- **429:** honours `Retry-After` once, then gives up.

## Commit
`feat: recommend GGUF models and quants that run well on this machine`

## Rollback
Revert the commit. The disk cache under `<DataDir>/recommend` can be
deleted. Nothing outside the new endpoints reads the engine.
