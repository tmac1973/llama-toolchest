# To do

Open work, in one place. Plans that are finished are in `plan/archive/`;
plans with work left are in `plan/`.

## Waiting on llama.cpp

- **`--spec-draft-sampling greedy|probabilistic`** (llama.cpp #27694).
  Chooses how draft tokens are picked for draft-simple and MTP
  speculative decoding. It landed on llama.cpp master after b11364 but is
  not in a tagged release yet. When it is:
  1. Find the first release tag that has it (for example with
     `gh api repos/ggml-org/llama.cpp/compare/<tag>...<commit> --jq .status`;
     "identical" or "behind" means the tag includes the commit).
  2. Add a version constant next to the others in
     `internal/models/launch.go` and write the option only for builds at
     or past it. An option the router does not know stops it, so older
     builds must get nothing.
  3. Add a per-model setting (Default / Greedy / Probabilistic) shown
     only for the draft methods it applies to.
- **`draft-mtp-adaptive`**: held back until builds from branches and PRs
  can be told apart from releases (see `plan/builder-refs.md`), so the
  mode can be offered only to builds that have it.

## Plans with work left

- **Build from branches, PRs and commits** (`plan/builder-refs.md`): not
  started.
- **Download-time sampling presets, phase 4**
  (`plan/download-time-sampling-presets.md`): remove the old scraped
  snapshot (`internal/models/sampling_presets_data.go`,
  `LookupSamplingPresets`, the fallback in `EffectiveSamplingPresets`),
  the `scripts/scrape-sampling-presets/` tool and its Makefile target, and
  move `pickRepoWithPreset` in `internal/api/capabilities_test.go` to test
  data.
- **Refactor** (`plan/refactor.md`): only the testing section is done.
  Left: one helper for download progress figures, `requireModel` /
  `requireBuild` lookups for handlers, and one error response type for
  JSON and HTML. The plan's testing section still says there are no
  tests and needs rewriting.
- **VRAM estimate findings** (`plan/ple-vram-findings.md`): turn a
  benchmark sweep's measured cells into estimate test points directly,
  and test the findings on more model architectures. Update the doc's
  Status line, which is out of date.

## Small fixes

- `web/templates/help.html` (around line 233) says a streamed per-layer
  embedding table is not counted toward the VRAM estimate. The table is
  never counted, whatever the mode; match the model config tooltip.
- Add `presence_penalty` to the benchmark overrides and sweep list, so
  sampling can be swept (from `plan/archive/speculative-multi-mode/results.md`).
- Host install: `setup.sh` prints how to install `uv` when it is missing.
  The batch-benchmarks plan wanted it to offer to install it. Decide
  whether the hint is enough.

## Ideas from the llama.cpp audit (2026-10-02)

- **`stop-timeout` and `load-on-startup`**: options that only exist in
  router presets. `load-on-startup` could load chosen models when the
  router starts.
- **"sleeping" model status**: the router's `/models` can report it when
  `--sleep-idle-seconds` is set. The app only expects loaded, loading and
  unloaded, so this only matters if someone sets it through Extra flags.
- **rocWMMA**: llama.cpp #28102 (b10905) made the gfx1201 flash-attention
  kernel about 6% faster. Issue #26220 is still open. Measure a current
  build against an older ref with rocWMMA on our own cards before
  changing the toggle's advice.

## Checks to do by hand

- Model profiles: the profile bar in a real browser; the helper model's
  JSON output on the active build; an Autoconfigure run on a GPU (helper
  loads, card advice shows in the review, helper unloads). Later fixes
  suggest these have been done, but nothing records it.
- ROCm 10: compare generated text between the 7.2.4 and 10 containers.
- Graphite theme: the design recommended it as the default dark look; it
  is opt-in today. Decide whether to make it the default.

## Other projects

- **haruspex client** (its own repository): read the `capabilities`
  object, drop the context-size fallbacks, use `context_per_request` and
  the new `context_shared`, and set parallel inference from `parallel`.
  This repo's side is done (`plan/archive/haruspex-capability-discovery.md`).
