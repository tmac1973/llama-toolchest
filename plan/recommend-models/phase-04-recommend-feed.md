# Phase 04 — The feed on Download Models

**Depends on:** 03 · **Enables:** 05 (downloads started from the feed carry
the chosen context)

## Goal
Show the engine's results above the search on the Download Models tab. The
look and flow match vllm-toolchest's feed (its phase 18), with a context
selector added and cards that name a suggested quant.

## Files touched
- `web/templates/models_browse.html`: the collapsed section, the feed
  container, a small script to collapse it again, and CSS.
- `web/templates/partials/recommend_feed.html` (new): the feed and the
  `recommend_card` template.
- `internal/api/recommend_view.go` (new): builds every sentence the template
  shows, so the template has no logic beyond loops.
- `internal/api/recommend.go`: return the partial to htmx requests.
- `internal/api/hf.go` and `web/templates/partials/hf_files.html`: the
  `suggest` and `ctx` parameters on the Details panel.
- `web/templates/help.html`: a "Recommended models" section.
- Render tests and golden files.

## Steps
1. **Collapsed state** (the page loads like this):
   - a **Find recommended models** button;
   - one sentence: "Models from Hugging Face that should run on this
     machine, with the quant and context each would get."
   - Clicking it sends `hx-get="/api/recommend?intent=quality&ctx=32k"` into
     `#recommend-feed`, with `aria-busy` on the button while it waits. The
     text under the button says "The first search can take up to a
     minute."
2. **Header**:
   - "Recommended for this machine".
   - The profile line, e.g. "3× RTX A4000 · 48 GB VRAM · 64 GB RAM · build
     b6500". Tooltip: "What these recommendations were worked out for. If
     one looks wrong, check this first."
   - The age: "updated 3 minutes ago".
   - **Refresh** (`hx-post /api/recommend/refresh`, then re-requests the
     current category and context). Tooltip: "Ask Hugging Face again. The
     list is otherwise kept for six hours."
   - **Hide**: collapses the section again.
3. **Context selector.** A row of four buttons: 8K, 32K, 128K, Model
   maximum. Each uses `aria-pressed` and `hx-get` with the current
   category. Tooltip on the row: "How much text the model can keep in
   mind at once: the conversation, files and its own answers. More context
   needs more memory, so a smaller quant may be suggested."
4. **Category buttons:** Best quality, Fastest, Longest context, Newest.
   Each has a tooltip:
   - **Best quality:** "Larger models first, counting the quant: a much
     larger model at 4 bits usually beats a smaller one at 8 bits."
   - **Fastest:** "Models that read the least data per generated word
     first. Mixture-of-experts models read only part of their weights, so
     they are often fast for their size."
   - **Longest context:** "Models that can hold the most text on this
     machine first."
   - **Newest:** "Models first published most recently."
5. **Cards** (`newRecommendCard` in `recommend_view.go`):
   - **Line 1:** the base model name (linked ↗ to HuggingFace), "from
     <publisher>", a `[gated]` mark (tooltip: "You must accept the model's
     license on Hugging Face and set your token in Settings."), and a
     "vision" mark when the repo has an mmproj file (tooltip: "Can read
     images. The image reader needs about N GB more memory when it is
     turned on.").
   - **Line 2:** the quant label, its size in GB and its bits per weight.
     Tooltip on the bits: "Average bits stored per weight. Around 4.5 or
     more keeps nearly all of the model's quality; below 4 the loss starts
     to show."
   - **Line 3:** how it runs:
     - on GPU: "All on GPU · 32K context · f16 KV cache · 26.1 GB of 44 GB
       · fits on 1 of 3 GPUs". "fits on N of M" is shown only when N < M.
     - experts in RAM: "Experts in system memory · 32K context · 8-bit KV
       cache · 18 GB in RAM — generation is slower".
     - Tooltip on the memory figure: "Estimated memory use against what is
       available after a safety margin on each card."
     - Tooltip on the KV cache: "The KV cache holds the conversation. 8-bit
       uses half the memory of f16, with a very small effect on quality."
   - **Line 4** (when it adds anything): the pick at the next larger class,
     e.g. "At 128K: Q4_K_M with 8-bit KV cache", or "At 128K: does not
     fit".
   - **Details** button: loads
     `/api/hf/model?id=<repo>&source=hf&suggest=<file>&ctx=<class>` inline
     under the card. Tooltip: "Files, sizes, disk space and the download
     button."
6. **Details panel changes** (`hf.go` `handleHFModel`, `hf_files.html`):
   - With `suggest`, that file's row is highlighted and labelled
     "Suggested for 32K".
   - With `ctx`, the class is passed on to the download request
     (`ctx=<class>&from=recommend`), for phase 05.
   - Under the file table: "Also published by: bartowski, lmstudio-community"
     with links that open each publisher's own Details panel. The feed
     passes the group's other repos as a `alts` parameter.
7. **Lists:**
   - Show the first 8 cards, with the rest under "Show N more".
   - Under the cards, when some models were hidden at this context: "N more
     models fit only with a shorter context."
   - "N could not be fully checked": a folded list of repo, `[gated]` mark
     and reason.
8. **Empty, stale and unavailable states:**
   - Stale (over 6 hours): "This list is from N hours ago. Refresh to check
     for newer models."
   - Empty at a class: "Nothing among the models looked at runs fully on
     this machine at 128K. Try a shorter context."
   - Unavailable: the engine's sentence, e.g. "No recommendations right
     now: could not reach Hugging Face. Search below still works."
9. **Help page.** A "Recommended models" section that explains the four
   orders, the context selector, bits per weight, "experts in system
   memory", "could not be fully checked", that every figure is an estimate,
   and that the list is kept for six hours.

## Build gate
`go build ./... && go vet ./... && go test ./...`

## Test plan
- Golden files for the feed: normal, with hidden models, empty at a class,
  stale, unavailable, architecture list unknown (no architecture clause).
- Render tests:
  - an "experts in RAM" card shows the slower-generation sentence;
  - "fits on 1 of 3 GPUs" appears only when fewer cards are needed;
  - the feed folds after 8 cards;
  - the browse page still renders its search with the feed collapsed.
- The Details panel with `suggest` highlights one row, and passes `ctx` on
  to the download button.
- Every metric on a card has a tooltip. A test walks the rendered card and
  checks that each figure's element has a `data-tooltip`.

## Commit
`feat: Find recommended models on the Download Models tab`

## Rollback
Revert the commit. The engine and its endpoints from phase 03 keep working
as JSON.
