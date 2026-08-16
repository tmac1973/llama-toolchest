# Coding TODO

Owned by the autonomous-coding run — do not edit while a run is active.
Marks: [ ] todo · [x] done · [!] blocked.


## Phase 01 — Evaluation engine and binary install (verify: passed, repairs: 0)

- [x] 01. Mode registry in `internal/evaluate`: (attempts: 0)
  Step 1 of 6 in Phase 01 ("Evaluation engine and binary install"). Implement EXACTLY this step as specified — read it in full under "## Steps" in plan/capability-benchmarks/phase-01-eval-engine.md before writing code.
- [x] 02. Flag mapping — `MapConfigFlags(snap SnapshotSubset) []string`, the (attempts: 0)
  Step 2 of 6 in Phase 01 ("Evaluation engine and binary install"). Implement EXACTLY this step as specified — read it in full under "## Steps" in plan/capability-benchmarks/phase-01-eval-engine.md before writing code.
- [x] 03. Command assembly per mode (documented against (attempts: 0)
  Step 3 of 6 in Phase 01 ("Evaluation engine and binary install"). Implement EXACTLY this step as specified — read it in full under "## Steps" in plan/capability-benchmarks/phase-01-eval-engine.md before writing code.
- [x] 04. Execution: `Run(ctx, Spec) (Result, error)` — `exec.CommandContext` (attempts: 0)
  Step 4 of 6 in Phase 01 ("Evaluation engine and binary install"). Implement EXACTLY this step as specified — read it in full under "## Steps" in plan/capability-benchmarks/phase-01-eval-engine.md before writing code.
- [x] 05. Parsers (`parse.go`), one per mode, anchored to the source's exact (attempts: 0)
  Step 5 of 6 in Phase 01 ("Evaluation engine and binary install"). Implement EXACTLY this step as specified — read it in full under "## Steps" in plan/capability-benchmarks/phase-01-eval-engine.md before writing code.
- [x] 06. `Result` struct mirrors `benchmark.EvalScores` fields: mode, dataset (attempts: 0)
  Step 6 of 6 in Phase 01 ("Evaluation engine and binary install"). Implement EXACTLY this step as specified — read it in full under "## Steps" in plan/capability-benchmarks/phase-01-eval-engine.md before writing code.

## Phase 02 — Dataset download, verification, and the logits cache store (verify: pending, repairs: 0)

- [x] 07. Layout under the data dir (constant root passed in by callers — (attempts: 0)
  Step 1 of 5 in Phase 02 ("Dataset download, verification, and the logits cache store"). Implement EXACTLY this step as specified — read it in full under "## Steps" in plan/capability-benchmarks/phase-02-datasets.md before writing code.
- [x] 08. Pinned dataset table — one struct per dataset: name, download URL, (attempts: 0)
  Step 2 of 5 in Phase 02 ("Dataset download, verification, and the logits cache store"). Implement EXACTLY this step as specified — read it in full under "## Steps" in plan/capability-benchmarks/phase-02-datasets.md before writing code.
- [x] 09. `EnsureDataset(ctx, root, name) (path, error)`: present + hash (attempts: 0)
  Step 3 of 5 in Phase 02 ("Dataset download, verification, and the logits cache store"). Implement EXACTLY this step as specified — read it in full under "## Steps" in plan/capability-benchmarks/phase-02-datasets.md before writing code.
- [x] 10. Logits cache (`klcache.go`): (attempts: 0)
  Step 4 of 5 in Phase 02 ("Dataset download, verification, and the logits cache store"). Implement EXACTLY this step as specified — read it in full under "## Steps" in plan/capability-benchmarks/phase-02-datasets.md before writing code.
- [x] 11. License strings live in the dataset table and render in phase 04's UI (attempts: 0)
  Step 5 of 5 in Phase 02 ("Dataset download, verification, and the logits cache store"). Implement EXACTLY this step as specified — read it in full under "## Steps" in plan/capability-benchmarks/phase-02-datasets.md before writing code.

## Phase 03 — Capability presets, job runner integration, and the job form (verify: pending, repairs: 0)

- [ ] 12. Presets (labels follow the existing duration-hint style; all counts (attempts: 0)
  Step 1 of 7 in Phase 03 ("Capability presets, job runner integration, and the job form"). Implement EXACTLY this step as specified — read it in full under "## Steps" in plan/capability-benchmarks/phase-03-runner-integration.md before writing code.
- [ ] 13. `JobEnv` additions (implemented in `jobs_env.go` with the api server's (attempts: 0)
  Step 2 of 7 in Phase 03 ("Capability presets, job runner integration, and the job form"). Implement EXACTLY this step as specified — read it in full under "## Steps" in plan/capability-benchmarks/phase-03-runner-integration.md before writing code.
- [ ] 14. Cell expansion — the collapse rule: capability cells run only what (attempts: 0)
  Step 3 of 7 in Phase 03 ("Capability presets, job runner integration, and the job form"). Implement EXACTLY this step as specified — read it in full under "## Steps" in plan/capability-benchmarks/phase-03-runner-integration.md before writing code.
- [ ] 15. Capability cell execution (the branch in the cell loop — entered (attempts: 0)
  Step 4 of 7 in Phase 03 ("Capability presets, job runner integration, and the job form"). Implement EXACTLY this step as specified — read it in full under "## Steps" in plan/capability-benchmarks/phase-03-runner-integration.md before writing code.
- [ ] 16. Router lifecycle: a capability cell following a performance cell (attempts: 0)
  Step 5 of 7 in Phase 03 ("Capability presets, job runner integration, and the job form"). Implement EXACTLY this step as specified — read it in full under "## Steps" in plan/capability-benchmarks/phase-03-runner-integration.md before writing code.
- [ ] 17. Single-run quick-benchmark form: `bench.go:467` feeds (attempts: 0)
  Step 6 of 7 in Phase 03 ("Capability presets, job runner integration, and the job form"). Implement EXACTLY this step as specified — read it in full under "## Steps" in plan/capability-benchmarks/phase-03-runner-integration.md before writing code.
- [ ] 18. Job form: capability presets render in the existing preset checkbox list (attempts: 0)
  Step 7 of 7 in Phase 03 ("Capability presets, job runner integration, and the job form"). Implement EXACTLY this step as specified — read it in full under "## Steps" in plan/capability-benchmarks/phase-03-runner-integration.md before writing code.

## Phase 04 — Results display, export, evaluation-data card, and docs (verify: pending, repairs: 0)

- [ ] 19. Results columns, conditional: a job or run list containing any run (attempts: 0)
  Step 1 of 7 in Phase 04 ("Results display, export, evaluation-data card, and docs"). Implement EXACTLY this step as specified — read it in full under "## Steps" in plan/capability-benchmarks/phase-04-results-and-management.md before writing code.
- [ ] 20. Compare view: the same conditional columns; comparing runs of (attempts: 0)
  Step 2 of 7 in Phase 04 ("Results display, export, evaluation-data card, and docs"). Implement EXACTLY this step as specified — read it in full under "## Steps" in plan/capability-benchmarks/phase-04-results-and-management.md before writing code.
- [ ] 21. Export: the CSV/JSON export gains the eval fields (mode, dataset, (attempts: 0)
  Step 3 of 7 in Phase 04 ("Results display, export, evaluation-data card, and docs"). Implement EXACTLY this step as specified — read it in full under "## Steps" in plan/capability-benchmarks/phase-04-results-and-management.md before writing code.
- [ ] 22. Evaluation Data card (Benchmarks tab, beneath the jobs list): (attempts: 0)
  Step 4 of 7 in Phase 04 ("Results display, export, evaluation-data card, and docs"). Implement EXACTLY this step as specified — read it in full under "## Steps" in plan/capability-benchmarks/phase-04-results-and-management.md before writing code.
- [ ] 23. About-benchmarks modal (`bench_about.html`): its preset table's (attempts: 0)
  Step 5 of 7 in Phase 04 ("Results display, export, evaluation-data card, and docs"). Implement EXACTLY this step as specified — read it in full under "## Steps" in plan/capability-benchmarks/phase-04-results-and-management.md before writing code.
- [ ] 24. Help page: a "Capability benchmarks" subsection under Benchmarks — (attempts: 0)
  Step 6 of 7 in Phase 04 ("Results display, export, evaluation-data card, and docs"). Implement EXACTLY this step as specified — read it in full under "## Steps" in plan/capability-benchmarks/phase-04-results-and-management.md before writing code.
- [ ] 25. Render tests: run row with `Eval` set shows the score and em-dash (attempts: 0)
  Step 7 of 7 in Phase 04 ("Results display, export, evaluation-data card, and docs"). Implement EXACTLY this step as specified — read it in full under "## Steps" in plan/capability-benchmarks/phase-04-results-and-management.md before writing code.
