# Phase 02 — Architectures each build supports

**Depends on:** — · **Enables:** 03 (repos whose architecture the active
build cannot load are marked "could not be fully checked")

## Goal
Know which `general.architecture` names the active llama.cpp build can
load, so the feed never recommends a model that will fail to start. This
plays the same part as vllm-toolchest's registry probe (its phase 16).

HuggingFace has GGUF repos for programs other than llama.cpp. For example,
`mudler/locate-anything.cpp-gguf` is in the top downloads with architecture
`locateanything`. There are also models newer than the build.

## Background
- llama.cpp lists every architecture name in `src/llama-arch.cpp`, in
  `LLM_ARCH_NAMES`:
  ```cpp
  static const std::map<llm_arch, const char *> LLM_ARCH_NAMES = {
      { LLM_ARCH_CLIP,             "clip"             }, // dummy, only used by llama-quantize
      { LLM_ARCH_LLAMA,            "llama"            },
      ...
  ```
  There are about 150 entries on master today.
- The builder keeps one git checkout at `<DataDir>/llama.cpp`
  (`internal/builder/builder.go:346`). Each `BuildResult` records the
  `GitSHA` it was built from (`builder.go:53-87`).

## Files touched
- `internal/builder/archs.go` (new): parse the name table.
- `internal/builder/builder.go`: record the list on each build; fill it in
  for builds made before this phase.
- `internal/builder/archs_test.go` (new).
- `internal/api/server.go`: a `supportedArchs()` helper for the active
  build.

## Steps
1. **Parse.** `ParseArchNames(src []byte) []string` finds the
   `LLM_ARCH_NAMES` map and collects every quoted name in a
   `{ LLM_ARCH_…, "name" }` entry.
   - Leave out `clip` (marked as a dummy) and `unknown`.
   - Return nil when the map is not found, so a layout change upstream
     means "unknown", not "nothing supported".
2. **Record on each build.** After a successful build, read
   `src/llama-arch.cpp` from the checkout and store the names on the
   result as `Archs []string` (`json:"archs,omitempty"`).
3. **Builds from before this phase.** When the list of builds is loaded,
   fill in `Archs` for any successful build that has a `GitSHA` but no
   list: run `git -C <DataDir>/llama.cpp show <sha>:src/llama-arch.cpp`
   and save the result. A failure leaves the list empty.
4. **Lookup.** `Server.supportedArchs() (set map[string]bool, known bool)`
   returns the active build's list. `known` is false when the list is
   empty. In that case the feed does not check architectures at all,
   because a missing list must never count against a model (the same rule
   as vllm-toolchest's `ArchsKnown`).

## Build gate
`go build ./... && go vet ./... && go test ./...`

## Test plan
- `ParseArchNames` on a trimmed copy of today's `llama-arch.cpp` in
  `testdata`: the count is right, `clip` is left out, and `llama`,
  `qwen3moe` and `gemma3` are in.
- A file without the map returns nil.
- Filling in old builds: use the `gitFixture` helper from
  `builder_refs_test.go` with a commit containing the file. Check that
  `Archs` is filled in and saved, and that an unknown SHA leaves it empty.

## Commit
`feat(builder): record which model architectures each build supports`

## Rollback
Revert the commit. The new field is optional and older code ignores it.
