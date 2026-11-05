package builder

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/patchsets"
)

// patchFixture is an upstream clone (gitFixture's, base at b2) and a patch
// set repo with releases of two patches on b2.
type patchFixture struct {
	dataDir, srcDir string
	repo            string // the patch set repo
	base            string // full SHA of b2
	tree            string // tree of b2 + the two patches
	set             patchsets.PatchSet
}

// fixtureGit runs git with no user or system config, as gitFixture does.
func fixtureGit(t *testing.T, dir string, args ...string) string {
	t.Helper()
	cmd := exec.Command("git", append([]string{"-C", dir}, args...)...)
	cmd.Env = append(os.Environ(),
		"GIT_CONFIG_GLOBAL=/dev/null", "GIT_CONFIG_NOSYSTEM=1",
		"GIT_AUTHOR_NAME=Patch Bot", "GIT_AUTHOR_EMAIL=bot@patch.invalid",
		"GIT_COMMITTER_NAME=t", "GIT_COMMITTER_EMAIL=t@t")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("git %v: %v\n%s", args, err, out)
	}
	return strings.TrimSpace(string(out))
}

func writeFile(t *testing.T, path, body string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

// newPatchFixture makes the fixture and points findPatchSet at it. The
// pinned release is r1.
func newPatchFixture(t *testing.T) *patchFixture {
	t.Helper()
	dataDir, srcDir, _ := covBuildSetup(t, "", "")
	f := &patchFixture{dataDir: dataDir, srcDir: srcDir}
	f.base = fixtureGit(t, srcDir, "rev-parse", "b2^{commit}")

	// The patches: two commits on b2, in a scratch clone.
	work := filepath.Join(t.TempDir(), "work")
	fixtureGit(t, filepath.Dir(work), "clone", "-q", srcDir, work)
	fixtureGit(t, work, "checkout", "-q", f.base)
	writeFile(t, filepath.Join(work, "boost.c"), "int boost;\n")
	fixtureGit(t, work, "add", "boost.c")
	fixtureGit(t, work, "commit", "-q", "--no-gpg-sign", "-m", "block 00: foundation")
	writeFile(t, filepath.Join(work, "f"), "patched\n")
	fixtureGit(t, work, "commit", "-q", "--no-gpg-sign", "-am", "block 01: kernels")
	f.tree = fixtureGit(t, work, "rev-parse", "HEAD^{tree}")
	patchDir := t.TempDir()
	fixtureGit(t, work, "format-patch", "-q", "-o", patchDir, f.base)
	patches, _ := filepath.Glob(filepath.Join(patchDir, "*.patch"))
	if len(patches) != 2 {
		t.Fatalf("format-patch made %d patches", len(patches))
	}

	f.repo = filepath.Join(t.TempDir(), "patchset")
	if err := os.MkdirAll(f.repo, 0o755); err != nil {
		t.Fatal(err)
	}
	fixtureGit(t, f.repo, "init", "-q")
	for i, p := range patches {
		body, _ := os.ReadFile(p)
		writeFile(t, filepath.Join(f.repo, "patches", fmt.Sprintf("%04d-block-%02d.patch", i, i)), string(body))
	}
	writeFile(t, filepath.Join(f.repo, "patches", "README.md"), "not a patch\n")
	f.release(t, "v2-"+f.base[:9]+"-r1", f.base[:9], f.tree, 2)

	f.set = patchsets.PatchSet{
		ID: "test-set", Name: "Test boosts", ShortName: "tst", Repo: f.repo,
		Pinned:   patchsets.Release{Tag: "v2-" + f.base[:9] + "-r1", Base: f.base, Tree: f.tree, NBlocks: 2},
		Backends: []string{"cpu"},
	}
	orig := findPatchSet
	findPatchSet = func(id string) (patchsets.PatchSet, bool) {
		if id == f.set.ID {
			return f.set, true
		}
		return patchsets.PatchSet{}, false
	}
	t.Cleanup(func() { findPatchSet = orig })
	return f
}

// release commits a release.json with these values and tags it.
func (f *patchFixture) release(t *testing.T, tag, base, tree string, n int) {
	t.Helper()
	writeFile(t, filepath.Join(f.repo, "release.json"),
		fmt.Sprintf(`{"release":%q,"base":%q,"tree":%q,"n_blocks":%d}`, tag, base, tree, n))
	fixtureGit(t, f.repo, "add", "-A")
	fixtureGit(t, f.repo, "commit", "-q", "--no-gpg-sign", "--allow-empty", "-m", tag)
	fixtureGit(t, f.repo, "tag", tag)
}

func (f *patchFixture) build(t *testing.T, release string, force bool) (*BuildResult, error) {
	t.Helper()
	b := NewBuilder(f.dataDir)
	res, err := b.BuildPatched(context.Background(), PatchSpec{SetID: f.set.ID, Release: release}, "cpu", "", force, nil, "", "")
	if err != nil {
		return nil, err
	}
	got := covWaitForBuild(t, b, res.ID)
	return &got, nil
}

func TestBuildPatchedPinnedRelease(t *testing.T) {
	f := newPatchFixture(t)
	got, err := f.build(t, "", false)
	if err != nil {
		t.Fatalf("BuildPatched: %v", err)
	}
	if got.Status != BuildStatusSuccess {
		t.Fatalf("status = %s (%s)", got.Status, got.Error)
	}
	if got.ID != "tst-b2-r1-cpu" {
		t.Errorf("ID = %q, want tst-b2-r1-cpu", got.ID)
	}
	// The base's count (b2), not base + 2 patches.
	if got.CommitCount != 2 {
		t.Errorf("CommitCount = %d, want 2", got.CommitCount)
	}
	if got.GitRef != f.set.Pinned.Tag || got.PatchSet != "test-set" || got.PatchRelease != f.set.Pinned.Tag || got.PatchBase != f.base {
		t.Errorf("patch fields = ref %q set %q release %q base %q", got.GitRef, got.PatchSet, got.PatchRelease, got.PatchBase)
	}
	if tree := fixtureGit(t, f.srcDir, "rev-parse", got.GitSHA+"^{tree}"); tree != f.tree {
		t.Errorf("built tree %s, want %s", tree, f.tree)
	}
	if author := fixtureGit(t, f.srcDir, "log", "-1", "--format=%cn <%ce>", got.GitSHA); author != "Patch Bot <bot@patch.invalid>" {
		t.Errorf("committer = %q, want the series author", author)
	}

	// After an unpatched build moves the clone, the patched commit can
	// still be read (arch backfill does).
	b := NewBuilder(f.dataDir)
	res, err := b.Build(context.Background(), "cpu", "b1", "", false, nil, "", "")
	if err != nil {
		t.Fatalf("plain build after a patched one: %v", err)
	}
	covWaitForBuild(t, b, res.ID)
	fixtureGit(t, f.srcDir, "gc", "-q", "--prune=now")
	fixtureGit(t, f.srcDir, "cat-file", "-e", got.GitSHA+":boost.c")

	// The same release and profile again is a duplicate.
	if _, err := f.build(t, "", false); err == nil {
		t.Error("second build of the same release was not refused")
	}
}

// The commit IDs do not depend on the environment of whoever builds.
func TestBuildPatchedIsReproducible(t *testing.T) {
	f := newPatchFixture(t)
	first, err := f.build(t, "", false)
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv("GIT_COMMITTER_NAME", "Someone Else")
	t.Setenv("GIT_COMMITTER_DATE", "2001-01-01T00:00:00Z")
	t.Setenv("HOME", t.TempDir())
	second, err := f.build(t, "", true)
	if err != nil {
		t.Fatal(err)
	}
	if first.GitSHA != second.GitSHA {
		t.Errorf("patched HEAD %s then %s", first.GitSHA, second.GitSHA)
	}
}

// "latest" takes the highest release number, not the newest tag.
func TestBuildPatchedLatestRelease(t *testing.T) {
	f := newPatchFixture(t)
	f.release(t, "v2-"+f.base[:9]+"-r10", f.base[:9], f.tree, 2)
	f.release(t, "v2-"+f.base[:9]+"-r9", f.base[:9], f.tree, 2)
	got, err := f.build(t, "latest", false)
	if err != nil {
		t.Fatal(err)
	}
	if got.PatchRelease != "v2-"+f.base[:9]+"-r10" || got.ID != "tst-b2-r10-cpu" {
		t.Errorf("latest built %q as %q", got.PatchRelease, got.ID)
	}
}

// A set that moves to a new base may start again at r1. "latest" takes
// the newest base first, so a higher number on an older base loses.
func TestBuildPatchedLatestPrefersNewerBase(t *testing.T) {
	f := newPatchFixture(t)
	b1 := fixtureGit(t, f.srcDir, "rev-parse", "b1^{commit}")
	f.release(t, "v2-"+b1[:9]+"-r50", b1[:9], f.tree, 2)
	// A base the clone does not have ranks below the ones it has.
	f.release(t, "v2-0123456789-r99", "0123456789", f.tree, 2)
	got, err := f.build(t, "latest", false)
	if err != nil {
		t.Fatal(err)
	}
	if got.PatchRelease != f.set.Pinned.Tag || got.ID != "tst-b2-r1-cpu" {
		t.Errorf("latest built %q as %q, want the r1 on the newer base", got.PatchRelease, got.ID)
	}
}

func TestPatchSetReleases(t *testing.T) {
	f := newPatchFixture(t)
	b := NewBuilder(f.dataDir)
	if _, err := b.PatchSetReleases(f.set.ID, false); err == nil {
		t.Error("listed releases before the set's repo was cloned")
	}
	if _, err := f.build(t, "", false); err != nil {
		t.Fatal(err)
	}
	b1 := fixtureGit(t, f.srcDir, "rev-parse", "b1^{commit}")
	f.release(t, "v2-"+f.base[:9]+"-r3", f.base[:9], f.tree, 2)
	f.release(t, "v2-"+b1[:9]+"-r50", b1[:9], f.tree, 2)
	rels, err := b.PatchSetReleases(f.set.ID, true)
	if err != nil {
		t.Fatal(err)
	}
	var got []string
	for _, r := range rels {
		got = append(got, fmt.Sprintf("r%d@%d", r.N, r.BaseCount))
	}
	if want := []string{"r3@2", "r1@2", "r50@1"}; !slices.Equal(got, want) {
		t.Errorf("releases = %v, want %v", got, want)
	}
}

func TestBuildPatchedRefusals(t *testing.T) {
	f := newPatchFixture(t)
	wantErr := func(name, release, want string) {
		t.Helper()
		_, err := f.build(t, release, false)
		if err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("%s: err = %v, want %q", name, err, want)
		}
	}

	f.release(t, "v2-"+f.base[:9]+"-r2", f.base[:9], strings.Repeat("0", 40), 2)
	wantErr("wrong tree", "v2-"+f.base[:9]+"-r2", "does not match release")

	f.release(t, "v2-"+f.base[:9]+"-r3", f.base[:9], f.tree, 3)
	wantErr("block count", "v2-"+f.base[:9]+"-r3", "has 2 patch files")

	wantErr("not a release", "b2", "is not a release")

	pinned := f.set
	f.set.Pinned.Tree = strings.Repeat("1", 40)
	wantErr("pinned values", "", "pinned release does not match")
	f.set = pinned

	f.set.Backends = []string{"rocm"}
	wantErr("backend", "", "can only be built with the rocm profiles")
	f.set = pinned

	b := NewBuilder(f.dataDir)
	if _, err := b.BuildPatched(context.Background(), PatchSpec{SetID: "nope"}, "cpu", "", false, nil, "", ""); err == nil {
		t.Error("unknown patch set was not refused")
	}
	if len(b.ListAll()) != 0 {
		t.Errorf("refused builds were recorded: %+v", b.ListAll())
	}
}

// A patch that does not apply names its block and leaves a clean clone.
func TestBuildPatchedPatchDoesNotApply(t *testing.T) {
	f := newPatchFixture(t)
	// Move the release's base to b1, where the second patch's change to
	// f does not apply.
	b1 := fixtureGit(t, f.srcDir, "rev-parse", "b1^{commit}")
	f.release(t, "v2-"+b1[:9]+"-r4", b1[:9], f.tree, 2)
	_, err := f.build(t, "v2-"+b1[:9]+"-r4", false)
	if err == nil || !strings.Contains(err.Error(), "block 01: kernels") {
		t.Fatalf("err = %v, want the failed block named", err)
	}
	gitDir := fixtureGit(t, f.srcDir, "rev-parse", "--absolute-git-dir")
	if _, err := os.Stat(filepath.Join(gitDir, "rebase-apply")); err == nil {
		t.Error("git am was left in progress")
	}
	if got, err := f.build(t, "", false); err != nil || got.Status != BuildStatusSuccess {
		t.Errorf("pinned build after a failed apply: %v", err)
	}
}

func TestSeriesAuthor(t *testing.T) {
	p := filepath.Join(t.TempDir(), "p.patch")
	writeFile(t, p, "From abc Mon Sep 17 00:00:00 2001\nFrom: Stew Forster <ci@rdna-boosts.invalid>\nSubject: x\n")
	if n, e := seriesAuthor(p); n != "Stew Forster" || e != "ci@rdna-boosts.invalid" {
		t.Errorf("seriesAuthor = %q %q", n, e)
	}
	writeFile(t, p, "Subject: none\n")
	if n, e := seriesAuthor(p); n != "rdna-boosts" || e != "ci@rdna-boosts.invalid" {
		t.Errorf("fallback = %q %q", n, e)
	}
}
