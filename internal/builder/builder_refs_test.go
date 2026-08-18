package builder

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"
)

// TestBuildRank covers the ranking ladder: a recorded CommitCount wins
// outright, a bN tag recovers the same scale for legacy builds, and
// anything else is unrankable.
func TestBuildRank(t *testing.T) {
	cases := []struct {
		name   string
		build  BuildResult
		want   int
		ranked bool
	}{
		{"commit count recorded", BuildResult{GitRef: "v0.1.0", CommitCount: 10500}, 10500, true},
		{"count beats tag parse", BuildResult{GitRef: "b9000", CommitCount: 9001}, 9001, true},
		{"legacy b-tag", BuildResult{GitRef: "b10400"}, 10400, true},
		{"legacy branch", BuildResult{GitRef: "master"}, 0, false},
		{"legacy semver, no count", BuildResult{GitRef: "v0.1.0"}, 0, false},
		{"legacy sha", BuildResult{GitRef: "a1b2c3d"}, 0, false},
	}
	for _, c := range cases {
		got, ranked := buildRank(c.build)
		if got != c.want || ranked != c.ranked {
			t.Errorf("%s: buildRank = (%d, %v), want (%d, %v)", c.name, got, ranked, c.want, c.ranked)
		}
	}
}

// TestLatestSuccessfulBuildAcrossTagSchemes pins the property the semver
// transition needs: builds rank by upstream commit count wherever it is
// known, whatever the ref looks like — so a semver-tagged build of newer
// code beats a legacy b-tag build, and an older semver build does NOT
// beat a newer nightly.
func TestLatestSuccessfulBuildAcrossTagSchemes(t *testing.T) {
	base := time.Date(2026, 8, 1, 0, 0, 0, 0, time.UTC)
	b := &Builder{builds: []BuildResult{
		{ID: "old-release", GitRef: "v0.1.0", CommitCount: 10300, Status: BuildStatusSuccess, StartedAt: base.Add(3 * time.Hour)},
		{ID: "legacy-nightly", GitRef: "b10400", Status: BuildStatusSuccess, StartedAt: base},
		{ID: "new-release", GitRef: "v0.2.0", CommitCount: 10500, Status: BuildStatusSuccess, StartedAt: base.Add(1 * time.Hour)},
		{ID: "failed-newest", GitRef: "b10600", Status: BuildStatusFailed, StartedAt: base.Add(4 * time.Hour)},
		{ID: "legacy-branch", GitRef: "master", Status: BuildStatusSuccess, StartedAt: base.Add(5 * time.Hour)},
	}}

	if got := b.LatestSuccessfulBuild(); got == nil || got.ID != "new-release" {
		t.Fatalf("latest = %+v, want new-release (count 10500 beats legacy b10400 and older v0.1.0; failed and unrankable builds excluded)", got)
	}

	// With no rankable builds at all, newest StartedAt wins.
	b2 := &Builder{builds: []BuildResult{
		{ID: "older", GitRef: "master", Status: BuildStatusSuccess, StartedAt: base},
		{ID: "newer", GitRef: "feature", Status: BuildStatusSuccess, StartedAt: base.Add(time.Hour)},
	}}
	if got := b2.LatestSuccessfulBuild(); got == nil || got.ID != "newer" {
		t.Fatalf("latest unrankable = %+v, want newer (StartedAt fallback)", got)
	}
}

// gitFixture creates <dataDir>/llama.cpp as a real repo with three
// commits tagged b1, b2 and v0.1.0 (v0.1.0 on the second commit, the way
// upstream semver releases point at an existing nightly).
func gitFixture(t *testing.T) (dataDir, srcDir string) {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not installed")
	}
	dataDir = t.TempDir()
	srcDir = filepath.Join(dataDir, "llama.cpp")
	if err := os.MkdirAll(srcDir, 0o755); err != nil {
		t.Fatal(err)
	}
	run := func(args ...string) {
		cmd := exec.Command("git", append([]string{"-C", srcDir}, args...)...)
		cmd.Env = append(os.Environ(),
			"GIT_AUTHOR_NAME=t", "GIT_AUTHOR_EMAIL=t@t",
			"GIT_COMMITTER_NAME=t", "GIT_COMMITTER_EMAIL=t@t")
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
	}
	run("init", "-q")
	for i, tag := range []string{"b1", "b2", ""} {
		if err := os.WriteFile(filepath.Join(srcDir, "f"), []byte{byte(i)}, 0o644); err != nil {
			t.Fatal(err)
		}
		run("add", "f")
		run("commit", "-q", "-m", "c", "--no-gpg-sign")
		if tag != "" {
			run("tag", tag)
		}
	}
	run("tag", "v0.1.0", "b2")
	return dataDir, srcDir
}

// TestFetchRefsListsBothTagFamilies verifies releases come first, then
// nightlies, each newest-first. FetchRefs' remote fetch is best-effort,
// so a fixture without an origin still lists local tags.
func TestFetchRefsListsBothTagFamilies(t *testing.T) {
	dataDir, _ := gitFixture(t)
	b := &Builder{dataDir: dataDir}

	refs, err := b.FetchRefs()
	if err != nil {
		t.Fatalf("FetchRefs: %v", err)
	}
	want := []string{"v0.1.0", "b2", "b1"}
	if len(refs) != len(want) {
		t.Fatalf("refs = %v, want %v", refs, want)
	}
	for i := range want {
		if refs[i] != want[i] {
			t.Fatalf("refs = %v, want %v", refs, want)
		}
	}

	// The release anchors to the nightly it was cut from: v0.1.0 sits on
	// the same commit as b2, the second commit, so its b-number is 2.
	anchors := b.ReleaseAnchors()
	if anchors["v0.1.0"] != 2 {
		t.Errorf("anchors = %v, want v0.1.0 → 2", anchors)
	}
	if _, ok := anchors["b2"]; ok {
		t.Errorf("nightly tags must not be anchored: %v", anchors)
	}
}

// TestCheckoutRefLatestAndCount verifies "latest" still resolves to the
// newest NIGHTLY tag (not the semver release), and that the commit count
// of the checkout is recorded.
func TestCheckoutRefLatestAndCount(t *testing.T) {
	_, srcDir := gitFixture(t)
	b := &Builder{}
	logCh := make(chan string, 16)

	ref, sha, count, err := b.checkoutRef(context.Background(), srcDir, "latest", logCh)
	if err != nil {
		t.Fatalf("checkoutRef: %v", err)
	}
	if ref != "b2" {
		t.Errorf("latest resolved to %q, want b2 (newest nightly, not v0.1.0)", ref)
	}
	if sha == "" {
		t.Error("empty sha")
	}
	// b2 is the second of three commits.
	if count != 2 {
		t.Errorf("commit count = %d, want 2", count)
	}

	// The contingency this change exists for: if upstream stops cutting
	// nightly b-tags, "latest" falls back to the newest semver release.
	for _, tag := range []string{"b1", "b2"} {
		if out, err := exec.Command("git", "-C", srcDir, "tag", "-d", tag).CombinedOutput(); err != nil {
			t.Fatalf("deleting %s: %v\n%s", tag, err, out)
		}
	}
	ref, _, _, err = b.checkoutRef(context.Background(), srcDir, "latest", logCh)
	if err != nil {
		t.Fatalf("checkoutRef after b-tag removal: %v", err)
	}
	if ref != "v0.1.0" {
		t.Errorf("latest with no b-tags resolved to %q, want v0.1.0", ref)
	}
}
