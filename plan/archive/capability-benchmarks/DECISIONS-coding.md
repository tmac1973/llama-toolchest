# Coding decisions

Decisions from the preflight interview (2026-08-15) plus defaults settled
by the preflight. The unattended run may treat everything here as binding.

## Verification command

Ran at preflight against the clean tree: passes in ~1s (cached test
results; `go test ./...` includes the node-backed `TestParameterControlsJS`
in `internal/api`, and node v26 is installed). Chosen because it covers
every stack in the repo (single Go module; `web/` has no test files) and
adds vet over the per-phase gates. Not enforced: `gofmt` (5 files are
already unformatted at baseline, so it would fail on untouched code) and
`make js-test` (a subset of `go test ./...`).

```
go build ./... && go vet ./internal/... && go test ./...
```

## Live end-to-end validation scope

**User decision: full live validation on this dev box** (RX 9070, ROCm +
vulkan builds, app currently stopped, ports 3000/8080 free). After all four
phases are code-complete and the verification command passes, the run
performs the manual E2E from the phase plans on the real app.

**User decision on E2E scope: "trim to the minimum — you decide."**
Settled minimum that exercises every success criterion exactly once:

- **Build:** rebuild the vulkan build at ref `b10067` — `POST /api/builds`
  with `{"profile":"vulkan","git_ref":"b10067"}`. If the API answers the
  duplicate-build prompt (existing `b10067-vulkan` has identical flags),
  re-POST with `"force":1` to rebuild in place. This is what installs
  `llama-perplexity` into `builds/b10067-vulkan/` and is already the
  configured `active_build`. (Phase 01's "rebuild the vulkan build"
  manual step.)
- **Models:** use the two installed quants of
  `unsloth/Qwen3.5-4B-GGUF`: the existing `Qwen3.5-4B-UD-Q4_K_XL.gguf`
  plus one download, `Qwen3.5-4B-Q8_0.gguf` (~3.3 GB; obtain through the
  app's model flow or by placing it in
  `models/unsloth--Qwen3.5-4B-GGUF/` and scanning). The 9B model is not
  used. No further downloads.
- **Job A (flagship):** `perplexity-quick` + `hellaswag-quick` +
  `winogrande-quick` × the 2 models → proves all four parser paths except
  KL, score rendering, and quality-degradation-per-cell.
- **Job B (KL, run twice):** `kl-divergence-quick` × the 2 models,
  automatic reference. First run: Q8_0's own cell skips (reference
  model), Q4_K_XL's cell generates the base logits as a visible step then
  scores; second run: cache hit, completes in a fraction of the time.
  `kl-divergence-full` is SKIPPED (tens of GiB of logits, hours of GPU
  time) — left to the user.
- **Job C (mixed):** `internal-quick` + `hellaswag-quick` × the 2 models
  → proves t/s + accuracy side by side, capability→performance router
  revival, and router restore at job end (router is not running before
  the run; it must be left not running after).
- Also verify via the E2E: the Evaluation Data card lists the three
  datasets as verified and the generated logits entry with size, and the
  logits Delete control works (deleting it before a re-run proves
  regeneration).
- **Post-E2E state:** leave the app stopped; leave the new build, the
  downloaded Q8_0 model, `eval-data/` datasets, and the KL logits cache
  in place; remove nothing else.

Jobs are created via `POST /api/benchmark-jobs` JSON
(`name`, `model_ids`, `build_ids`, `presets`, optional `params`) and
polled via `GET /api/benchmark-jobs/{id}` + `/progress`; the app is
started with `make build` then `./bin/llama-toolchest --config
config.yaml` (background; log to a file).

## Git: branch, commits, no push

**User decision: commit on a feature branch, no push.**

- Branch `feat/capability-benchmarks`, created from `main` (currently
  `d9bcf2c`) before any code changes.
- One commit per phase, using the plan's exact commit messages
  (`feat(evaluate): ...` ×2, `feat(benchmarks): ...` ×2).
- Do NOT push to origin, do NOT open a PR, do NOT touch `main`.
- The untracked `audit/` and `audits/` directories belong to other work —
  never stage, commit, or delete them.

## Dataset sources for the pinned table

The plan says the implementer verifies URLs and pins SHA-256 at
implementation time by downloading once. Verified reachable from this box
during preflight; the run downloads each file and records the actual
SHA-256 (do NOT trust precomputed hashes):

- **wikitext-2 test** (CC BY-SA 3.0): download
  `https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip`
  (4,721,645 bytes) and extract `wikitext-2-raw/wiki.test.raw`
  (1,290,590 bytes) to `eval-data/datasets/wikitext-2-raw-test.txt`. This
  is llama.cpp's own `scripts/get-wikitext-2.sh` source.
- **HellaSwag validation** (MIT):
  `https://raw.githubusercontent.com/klosax/hellaswag_text_data/main/hellaswag_val_full.txt`
  (7,770,677 bytes) to `eval-data/datasets/hellaswag_val_full.txt`. This
  is llama.cpp's own `scripts/get-hellaswag.sh` source (verified 200 OK;
  a 404 can occur transiently on raw.githubusercontent — retry before
  failing).
- **Winogrande debiased eval** (Apache-2.0):
  `https://huggingface.co/datasets/ikawrakow/winogrande-eval-for-llama.cpp/resolve/main/winogrande-debiased-eval.csv`
  (1,268 lines) to `eval-data/datasets/winogrande-debiased-eval.csv`. The
  dataset named in `perplexity.cpp`'s comment; format matches
  `load_winogrande_from_csv`.

## Environment facts the run can rely on (checked at preflight)

- Go 1.26.5 on PATH (module requires 1.25.7 — fine, do not pin or
  retoolchain).
- Working directory is `/home/tim/Projects/llama-toolchest`; `git status`
  is clean apart from the untracked `audit/` dirs.
- App data dir `~/.local/share/llama-toolchest` holds the managed
  llama.cpp clone (currently at `b10442`; a `b10067` rebuild will
  force-checkout that ref — expected, it is the app's clone), builds
  (`b10067-vulkan`, `b10442-rocm`, `b10442-rocm-optimized`), and models.
  `config.yaml`: `active_build: b10067-vulkan`, `models_max: 1`,
  `auto_start: false`, port 3000, llama port 8080.
- All plan source references were verified against the b10067 checkout
  (`tools/perplexity/perplexity.cpp` lines 654, 1006, 1300, 1717-1722,
  1949-2005, 2015, 493, 461-470; `common/arg.cpp:2656` `--direct-io`);
  they hold. `common/arg.cpp` marks `--direct-io` DEPRECATED (warning
  only) — still valid, keep the mapping.
- Build install step is at `internal/builder/builder.go:506-512` as the
  plan states; existing builds contain `llama-server` + `llama-bench`
  only (no `llama-perplexity`), confirming the "old builds lack the
  binary" path.
- `jobCreateRequest` JSON fields confirmed: `name`, `model_ids`,
  `build_ids`, `presets`, `overrides`, `sweeps`, `params`. Phase 03 adds
  `kl_reference`.
- Route registration is `internal/api/server.go` `buildRouter()` as the
  plan says; results/compare partials are
  `web/templates/partials/benchmark_detail.html` /
  `benchmark_compare.html`, job form is `partials/job_form.html`.
- The toolchain for live validation is present: cmake, ninja, gcc/g++,
  Vulkan headers + `vulkaninfo`, ROCm under `/opt/rocm`, 60 GB RAM,
  ~1.3 TB free disk. Network egress to huggingface.co and
  raw.githubusercontent.com works.
- If a live E2E step fails in a way that looks like a code bug, fix it
  (with a test), re-run the verification command, commit the fix on the
  branch, and retry the step — rather than stopping. If it fails for an
  environment reason (GPU driver crash, network), record the exact
  symptom in a `## E2E report` section appended to this file and finish
  the remaining E2E steps that do not depend on it.
