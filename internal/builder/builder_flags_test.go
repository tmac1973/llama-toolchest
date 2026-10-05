package builder

import (
	"os"
	"path/filepath"
	"regexp"
	"testing"
	"time"
)

// hashFlags names untagged builds whose flags differ from an existing one,
// so the same flag set must always give the same suffix. Go visits map keys
// in a random order, so a hash that followed that order would give one flag
// set a new name on each run.
func TestHashFlagsIsStableAcrossMapOrder(t *testing.T) {
	keys := []string{"GGML_CUDA", "GGML_NATIVE", "CMAKE_BUILD_TYPE", "GGML_CUDA_FA_ALL_QUANTS", "LLAMA_CURL"}
	vals := []string{"ON", "OFF", "Release", "ON", "OFF"}
	forward := map[string]string{}
	for i := range keys {
		forward[keys[i]] = vals[i]
	}
	backward := map[string]string{}
	for i := len(keys) - 1; i >= 0; i-- {
		backward[keys[i]] = vals[i]
	}

	want := hashFlags(forward)
	if !regexp.MustCompile(`^[0-9a-f]{6}$`).MatchString(want) {
		t.Fatalf("hashFlags = %q, want six hex characters", want)
	}
	for i := 0; i < 50; i++ {
		if got := hashFlags(forward); got != want {
			t.Fatalf("run %d: hashFlags = %q, want %q", i, got, want)
		}
		if got := hashFlags(backward); got != want {
			t.Fatalf("run %d: same flags inserted in another order hash to %q, want %q", i, got, want)
		}
	}
}

// A changed value, an added flag or a removed flag is a different build,
// so it must get a different suffix; otherwise two builds with different
// flags would share a name.
func TestHashFlagsDiffersWhenFlagsChange(t *testing.T) {
	base := map[string]string{"GGML_CUDA": "ON", "GGML_NATIVE": "OFF"}
	cases := []struct {
		name  string
		flags map[string]string
	}{
		{"value changed", map[string]string{"GGML_CUDA": "ON", "GGML_NATIVE": "ON"}},
		{"flag added", map[string]string{"GGML_CUDA": "ON", "GGML_NATIVE": "OFF", "LLAMA_CURL": "OFF"}},
		{"flag removed", map[string]string{"GGML_CUDA": "ON"}},
		{"value moved to another key", map[string]string{"GGML_CUDA": "OFF", "GGML_NATIVE": "ON"}},
		{"empty set", map[string]string{}},
	}
	want := hashFlags(base)
	for _, c := range cases {
		if got := hashFlags(c.flags); got == want {
			t.Errorf("%s: hash %q equals the base hash", c.name, got)
		}
	}
	// No flags at all, nil or empty, is one flag set.
	if hashFlags(nil) != hashFlags(map[string]string{}) {
		t.Error("nil and empty flag maps hash differently")
	}
}

// flagsEqual decides whether a new untagged build reuses an existing
// build's name or gets a hash suffix. It compares contents only, and a
// legacy build with no stored flags (nil) never matches a build that has
// some, so the two are never mixed up.
func TestFlagsEqual(t *testing.T) {
	cases := []struct {
		name string
		a, b map[string]string
		want bool
	}{
		{"both nil", nil, nil, true},
		{"nil and empty", nil, map[string]string{}, true},
		{"same contents", map[string]string{"A": "1", "B": "2"}, map[string]string{"B": "2", "A": "1"}, true},
		{"nil and non-empty", nil, map[string]string{"A": "1"}, false},
		{"different value", map[string]string{"A": "1"}, map[string]string{"A": "2"}, false},
		{"different key, same size", map[string]string{"A": "1"}, map[string]string{"B": "1"}, false},
		{"extra key", map[string]string{"A": "1"}, map[string]string{"A": "1", "B": "2"}, false},
		{"empty value is still a value", map[string]string{"A": ""}, map[string]string{"B": ""}, false},
	}
	for _, c := range cases {
		if got := flagsEqual(c.a, c.b); got != c.want {
			t.Errorf("%s: flagsEqual(a, b) = %v, want %v", c.name, got, c.want)
		}
		if got := flagsEqual(c.b, c.a); got != c.want {
			t.Errorf("%s: flagsEqual(b, a) = %v, want %v", c.name, got, c.want)
		}
	}
}

// refTagNumber reads N from llama.cpp's "bN" release tags so legacy builds
// can be ranked. Anything not shaped like that must be unrankable rather
// than given a made-up number.
func TestRefTagNumber(t *testing.T) {
	cases := []struct {
		ref  string
		want int
		ok   bool
	}{
		{"b1", 1, true},
		{"b10400", 10400, true},
		{"b007", 7, true},
		{"", 0, false},
		{"b", 0, false},
		{"B10400", 0, false},
		{"v0.1.0", 0, false},
		{"b10400-rc1", 0, false},
		{"b10.4", 0, false},
		{"master", 0, false},
		{"beta", 0, false},
		{"a1b2c3d", 0, false},
	}
	for _, c := range cases {
		got, ok := refTagNumber(c.ref)
		if got != c.want || ok != c.ok {
			t.Errorf("refTagNumber(%q) = (%d, %v), want (%d, %v)", c.ref, got, ok, c.want, c.ok)
		}
	}
}

// Rank is the exported form of buildRank, used outside the package to
// order builds; it must give the same answer.
func TestRankMatchesBuildRank(t *testing.T) {
	for _, r := range []BuildResult{
		{GitRef: "b10400"},
		{GitRef: "v0.2.0", CommitCount: 10500},
		{GitRef: "master"},
	} {
		n1, ok1 := r.Rank()
		n2, ok2 := buildRank(r)
		if n1 != n2 || ok1 != ok2 {
			t.Errorf("%s: Rank = (%d, %v), buildRank = (%d, %v)", r.GitRef, n1, ok1, n2, ok2)
		}
	}
}

// The router walks this list from the front and skips builds that cannot
// run in the current image, so the whole order matters, not just the
// first entry: ranked builds newest code first, a tie in rank broken by
// the newer build time, then every unrankable build by newest build time,
// and failed or still-building builds left out.
func TestSuccessfulBuildsRankedOrder(t *testing.T) {
	base := time.Date(2026, 9, 1, 0, 0, 0, 0, time.UTC)
	at := func(h int) time.Time { return base.Add(time.Duration(h) * time.Hour) }
	b := &Builder{builds: []BuildResult{
		{ID: "branch-old", GitRef: "master", Status: BuildStatusSuccess, StartedAt: at(1)},
		{ID: "b10400", GitRef: "b10400", Status: BuildStatusSuccess, StartedAt: at(2)},
		{ID: "failed", GitRef: "b10900", Status: BuildStatusFailed, StartedAt: at(9)},
		{ID: "release-10500-early", GitRef: "v0.2.0", CommitCount: 10500, Status: BuildStatusSuccess, StartedAt: at(3)},
		{ID: "branch-new", GitRef: "feature", Status: BuildStatusSuccess, StartedAt: at(8)},
		{ID: "b10500-late", GitRef: "b10500", Status: BuildStatusSuccess, StartedAt: at(4)},
		{ID: "b10300", GitRef: "b10300", Status: BuildStatusSuccess, StartedAt: at(7)},
		{ID: "building", GitRef: "b11000", Status: BuildStatusBuilding, StartedAt: at(10)},
	}}

	var got []string
	for _, r := range b.SuccessfulBuildsRanked() {
		got = append(got, r.ID)
	}
	want := []string{"b10500-late", "release-10500-early", "b10400", "b10300", "branch-new", "branch-old"}
	if len(got) != len(want) {
		t.Fatalf("order = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("order = %v, want %v", got, want)
		}
	}

	if got := (&Builder{}).SuccessfulBuildsRanked(); len(got) != 0 {
		t.Errorf("no builds: got %v, want none", got)
	}
	if got := (&Builder{}).LatestSuccessfulBuild(); got != nil {
		t.Errorf("no builds: LatestSuccessfulBuild = %+v, want nil", got)
	}
}

// nvcc found on PATH wins over the fixed install locations, and the
// directory returned is the one that holds it, which the build adds to
// PATH for cmake.
func TestFindNVCCUsesPATH(t *testing.T) {
	dir := t.TempDir()
	nvcc := filepath.Join(dir, "nvcc")
	if err := os.WriteFile(nvcc, []byte("#!/bin/sh\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", dir)

	path, binDir := findNVCC()
	if path != nvcc || binDir != dir {
		t.Errorf("findNVCC = (%q, %q), want (%q, %q)", path, binDir, nvcc, dir)
	}
}

// With no nvcc on PATH and none in the usual CUDA install locations, the
// result is empty so the caller can report that CUDA is missing. Skipped
// on a host that has CUDA installed in one of those locations.
func TestFindNVCCMissing(t *testing.T) {
	for _, p := range []string{"/usr/local/cuda/bin/nvcc", "/opt/cuda/bin/nvcc"} {
		if _, err := os.Stat(p); err == nil {
			t.Skipf("%s exists on this host", p)
		}
	}
	t.Setenv("PATH", t.TempDir())

	if path, binDir := findNVCC(); path != "" || binDir != "" {
		t.Errorf("findNVCC = (%q, %q), want empty", path, binDir)
	}
}
