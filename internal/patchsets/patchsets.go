// Package patchsets holds the curated patch sets: sets of patches on
// upstream llama.cpp that the builder applies at a known upstream commit.
// A patched build is a llama.cpp build in every way the router, jobs and
// launch options care about, because its upstream version is exact: the
// version of the commit the patches apply to. That is what sets a patch
// set apart from a curated fork (internal/forks), whose history does not
// follow upstream's build numbers.
//
// The builder and the API both import this package, so it imports
// nothing from the toolchest.
package patchsets

import (
	"regexp"
	"strconv"
)

// PatchSet is one curated set of patches on upstream llama.cpp.
type PatchSet struct {
	// ID is a lowercase slug ("rdna-boosts").
	ID   string
	Name string
	// ShortName starts the IDs of the set's builds ("rdna" gives
	// "rdna-b11887-r206-rocm").
	ShortName string
	// Repo is the git repo that publishes the set's releases, each as a
	// tag with a release.json and a patches/ folder.
	Repo string
	// Pinned is the release tested with this toolchest. A build of
	// another release is marked untested.
	Pinned Release
	// Backends lists the build profile backends the set is for.
	Backends []string
	// ROCmArchs are the AMD GPU architectures the set is tuned for.
	ROCmArchs []string
	License   string
	// Notes is one or two sentences on what the set is for, shown as a
	// tooltip.
	Notes string
}

// Release is one release of a patch set, as its release.json records it.
type Release struct {
	// Tag is the release's git tag in the set's repo.
	Tag string
	// Base is the upstream llama.cpp commit the patches apply to, as a
	// full SHA.
	Base string
	// Tree is the git tree of the patched source. A build whose tree
	// differs did not build this release.
	Tree string
	// NBlocks is the number of patch files.
	NBlocks int
	// Date is the day the release was pinned (YYYY-MM-DD); empty on a
	// release read from the repo.
	Date string
}

// Curated returns the curated patch sets, in the order the UI lists them.
func Curated() []PatchSet {
	return []PatchSet{rdnaBoosts()}
}

// Find returns the curated patch set with the given ID.
func Find(id string) (PatchSet, bool) {
	for _, p := range Curated() {
		if p.ID == id {
			return p, true
		}
	}
	return PatchSet{}, false
}

// releaseTagRE matches a release tag, v<blocks>-<base>-r<N>.
var releaseTagRE = regexp.MustCompile(`^v\d+-([0-9a-f]+)-r(\d+)$`)

// ParseTag returns the abbreviated upstream base and N from a release
// tag v<blocks>-<base>-r<N>. N alone does not order releases: a set may
// start again at r1 when it moves to a new base.
func ParseTag(tag string) (base string, n int, ok bool) {
	m := releaseTagRE.FindStringSubmatch(tag)
	if m == nil {
		return "", 0, false
	}
	n, err := strconv.Atoi(m[2])
	return m[1], n, err == nil
}

// ReleaseNumber returns N from a release tag v<blocks>-<base>-r<N>.
func ReleaseNumber(tag string) (int, bool) {
	_, n, ok := ParseTag(tag)
	return n, ok
}

// rdnaBoosts is stew675/llama-cpp-rdna-boosts: RDNA2–RDNA4 speed-ups for
// every model (MTP draft depth, chunked gated-delta-net prefill, BF16 KV
// and WMMA flash attention, quantized prefill GEMM, k-quant decode,
// host-resident MoE experts). Every feature is on by default, so it needs
// no launch options.
func rdnaBoosts() PatchSet {
	return PatchSet{
		ID:        "rdna-boosts",
		Name:      "RDNA boosts",
		ShortName: "rdna",
		Repo:      "https://github.com/stew675/llama-cpp-rdna-boosts",
		// From the release.json of the tag, read on 2026-11-02. The base
		// is upstream b11887.
		Pinned: Release{
			Tag:     "v16-6050b467c-r206",
			Base:    "6050b467c07a95c7b87debd4ddac2f8ba8d9b807",
			Tree:    "1bac1001f1780a5907fbce256ceeeaac8b8561ec",
			NBlocks: 16,
			Date:    "2026-11-02",
		},
		Backends:  []string{"rocm"},
		ROCmArchs: []string{"gfx1030", "gfx1100", "gfx1150", "gfx1151", "gfx1200", "gfx1201"},
		License:   "MIT",
		Notes:     "Upstream llama.cpp with the RDNA boosts patches. Faster on AMD RDNA GPUs. ROCm only.",
	}
}
