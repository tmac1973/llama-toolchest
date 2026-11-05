package builder

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"slices"
	"strings"

	"github.com/tmac1973/llama-toolchest/internal/patchsets"
)

// PatchSpec asks for a llama.cpp build with a curated patch set applied.
type PatchSpec struct {
	// SetID is the patch set's ID (internal/patchsets).
	SetID string
	// Release is the release tag to build: "" for the pinned release,
	// "latest" for the newest one in the set's repo.
	Release string
}

// findPatchSet looks up a curated patch set. A var so tests can point a
// set at a local fixture repo.
var findPatchSet = patchsets.Find

// patchRelease is a patch set release ready to apply: its release.json
// values, checked, and the patch files in apply order.
type patchRelease struct {
	set     patchsets.PatchSet
	rel     patchsets.Release
	repoDir string
	// files are the patch paths in the release's tree ("patches/0000-….patch").
	files []string
}

// patchFileRE matches a release's patch files; beta/ and other folders
// are left out by the path.
var patchFileRE = regexp.MustCompile(`^patches/\d{4}-[^/]*\.patch$`)

// releaseJSON is the part of a release's release.json the builder reads.
type releaseJSON struct {
	Release string `json:"release"`
	Base    string `json:"base"`
	Tree    string `json:"tree"`
	NBlocks int    `json:"n_blocks"`
}

// checkPatchSpec refuses a patch spec the project or profile cannot take.
func checkPatchSpec(spec PatchSpec, proj Project, prof BuildProfile) (patchsets.PatchSet, error) {
	set, ok := findPatchSet(spec.SetID)
	if !ok {
		return set, fmt.Errorf("unknown patch set: %s", spec.SetID)
	}
	if proj.ID != ProjectLlamaCpp {
		return set, fmt.Errorf("the %s patch set applies to llama.cpp only", set.Name)
	}
	if !slices.Contains(set.Backends, prof.Backend) {
		return set, fmt.Errorf("the %s patch set can only be built with the %s profiles", set.Name, strings.Join(set.Backends, ", "))
	}
	return set, nil
}

// preparePatchSet fetches the patch set's repo and reads the release to
// build. srcDir is the llama.cpp clone, already fetched, where an
// abbreviated base is resolved to a full SHA.
func (b *Builder) preparePatchSet(ctx context.Context, set patchsets.PatchSet, spec PatchSpec, srcDir string, logCh chan string) (*patchRelease, error) {
	sendLog(logCh, "==> Patch set: "+set.Name)
	repoDir := filepath.Join(b.dataDir, "patchsets", set.ID)
	if err := b.ensurePatchRepo(ctx, set, repoDir, logCh); err != nil {
		return nil, fmt.Errorf("patch set repo: %w", err)
	}

	tag := spec.Release
	switch tag {
	case "":
		tag = set.Pinned.Tag
	case "latest":
		latest, err := latestReleaseTag(ctx, repoDir)
		if err != nil {
			return nil, err
		}
		tag = latest
		sendLog(logCh, "==> Newest release: "+tag)
	}
	if _, ok := patchsets.ReleaseNumber(tag); !ok {
		return nil, fmt.Errorf("%q is not a release of %s", tag, set.Name)
	}

	out, err := gitOutput(ctx, repoDir, "show", tag+":release.json")
	if err != nil {
		return nil, fmt.Errorf("reading release.json of %s: %w", tag, err)
	}
	var rj releaseJSON
	if err := json.Unmarshal([]byte(out), &rj); err != nil {
		return nil, fmt.Errorf("parsing release.json of %s: %w", tag, err)
	}
	if rj.Base == "" || rj.Tree == "" || rj.NBlocks <= 0 {
		return nil, fmt.Errorf("release.json of %s has no base, tree or block count", tag)
	}
	if tag == set.Pinned.Tag {
		p := set.Pinned
		if !strings.HasPrefix(p.Base, rj.Base) || p.Tree != rj.Tree || p.NBlocks != rj.NBlocks {
			return nil, errors.New("the pinned release does not match its release.json")
		}
	}

	base, err := gitOutput(ctx, srcDir, "rev-parse", "--verify", "--quiet", rj.Base+"^{commit}")
	if err != nil {
		return nil, fmt.Errorf("upstream commit %s of release %s is not in the llama.cpp clone", rj.Base, tag)
	}

	list, err := gitOutput(ctx, repoDir, "ls-tree", "--name-only", tag, "patches/")
	if err != nil {
		return nil, fmt.Errorf("listing the patches of %s: %w", tag, err)
	}
	var files []string
	for _, f := range strings.Split(list, "\n") {
		if patchFileRE.MatchString(f) {
			files = append(files, f)
		}
	}
	slices.Sort(files)
	if len(files) != rj.NBlocks {
		return nil, fmt.Errorf("release %s has %d patch files, but its release.json says %d", tag, len(files), rj.NBlocks)
	}

	rel := patchsets.Release{Tag: tag, Base: base, Tree: rj.Tree, NBlocks: rj.NBlocks}
	if tag == set.Pinned.Tag {
		rel.Date = set.Pinned.Date
	}
	return &patchRelease{set: set, rel: rel, repoDir: repoDir, files: files}, nil
}

// ensurePatchRepo clones the patch set's repo, or fetches its tags when
// the clone exists. The clone is a cache we own, so one that git cannot
// read is thrown away and cloned again.
func (b *Builder) ensurePatchRepo(ctx context.Context, set patchsets.PatchSet, repoDir string, logCh chan string) error {
	if _, err := gitOutput(ctx, repoDir, "rev-parse", "--git-dir"); err == nil {
		sendLog(logCh, "==> Fetching the patch set...")
		return b.runCmd(ctx, repoDir, logCh, "", nil, "git", "fetch", "--tags", "--force", "origin")
	}
	if err := os.RemoveAll(repoDir); err != nil {
		return fmt.Errorf("removing unusable clone: %w", err)
	}
	if err := os.MkdirAll(filepath.Dir(repoDir), 0o755); err != nil {
		return fmt.Errorf("creating clone parent: %w", err)
	}
	sendLog(logCh, "==> Cloning the patch set...")
	// Only the release tags' files are read; the repo's history folders
	// are not downloaded until a release needs them.
	return b.runCmd(ctx, filepath.Dir(repoDir), logCh, "", nil, "git", "clone", "--filter=blob:none", "--no-checkout", set.Repo, filepath.Base(repoDir))
}

// latestReleaseTag returns the release tag with the highest release
// number. The number, not the tag date: a re-cut tag can be newer than a
// later release.
func latestReleaseTag(ctx context.Context, repoDir string) (string, error) {
	out, err := gitOutput(ctx, repoDir, "tag", "--list", "v*")
	if err != nil {
		return "", fmt.Errorf("listing releases: %w", err)
	}
	best, bestN := "", -1
	for _, tag := range strings.Split(out, "\n") {
		if n, ok := patchsets.ReleaseNumber(tag); ok && n > bestN {
			best, bestN = tag, n
		}
	}
	if best == "" {
		return "", errors.New("the patch set repo has no releases")
	}
	return best, nil
}

// clearStaleAM ends a git am left in the clone by an interrupted build.
// It must run before the checkout: aborting moves HEAD back to where the
// old am started.
func clearStaleAM(ctx context.Context, srcDir string) {
	gitDir, err := gitOutput(ctx, srcDir, "rev-parse", "--absolute-git-dir")
	if err != nil {
		return
	}
	if _, err := os.Stat(filepath.Join(gitDir, "rebase-apply")); err == nil {
		abortAM(ctx, srcDir)
	}
}

// abortAM ends a stopped git am. The abort writes a reflog entry, so it
// needs a committer: without one (a container with no git config) it
// fails and leaves the am in progress. It gets the set's fallback
// identity and the same cut-off config as the apply.
func abortAM(ctx context.Context, srcDir string) {
	name, email := seriesAuthor("")
	cmd := exec.CommandContext(ctx, "git", "-C", srcDir, "am", "--abort")
	cmd.Env = amEnv(os.TempDir(), name, email)
	_ = cmd.Run()
}

// amEnv is the environment git am runs in: the process environment
// without the variables that change what it commits, with home and
// config cut off and the given committer.
func amEnv(home, name, email string) []string {
	return append(cleanGitEnv(os.Environ()),
		"HOME="+home,
		"GIT_CONFIG_NOSYSTEM=1",
		"GIT_CONFIG_GLOBAL=/dev/null",
		"GIT_COMMITTER_NAME="+name,
		"GIT_COMMITTER_EMAIL="+email,
	)
}

// applyPatchSet applies the release's patches on the checked-out base
// with strict git am, checks the tree, and keeps the result on a branch.
// It returns the patched HEAD.
//
// The committer is the series author and each commit's date its author
// date, as the set's own apply script does, so the commit IDs are the
// release's and do not depend on who runs the build. HOME and the git
// config are cut off for the same reason.
func (b *Builder) applyPatchSet(ctx context.Context, pr *patchRelease, srcDir string, logCh chan string) (string, error) {
	tmp, err := os.MkdirTemp("", "patchset-")
	if err != nil {
		return "", err
	}
	defer os.RemoveAll(tmp)

	var paths []string
	for _, f := range pr.files {
		body, err := gitOutputRaw(ctx, pr.repoDir, "show", pr.rel.Tag+":"+f)
		if err != nil {
			return "", fmt.Errorf("reading %s: %w", f, err)
		}
		p := filepath.Join(tmp, filepath.Base(f))
		if err := os.WriteFile(p, body, 0o644); err != nil {
			return "", err
		}
		paths = append(paths, p)
	}
	name, email := seriesAuthor(paths[0])

	env := amEnv(tmp, name, email)
	for _, p := range paths {
		sendLog(logCh, "==> Applying "+filepath.Base(p))
	}
	args := append([]string{"am", "--committer-date-is-author-date", "--no-3way", "--no-gpg-sign", "--quiet"}, paths...)
	if err := b.runCmd(ctx, srcDir, logCh, "", env, "git", args...); err != nil {
		failed := failedPatch(ctx, srcDir)
		abortAM(ctx, srcDir)
		if failed != "" {
			return "", fmt.Errorf("patch %s does not apply: %w", failed, err)
		}
		return "", fmt.Errorf("applying the patches: %w", err)
	}

	tree, err := gitOutput(ctx, srcDir, "rev-parse", "HEAD^{tree}")
	if err != nil {
		return "", err
	}
	if tree != pr.rel.Tree {
		return "", fmt.Errorf("the patched source does not match release %s (tree %s, want %s)", pr.rel.Tag, shortSHA(tree), shortSHA(pr.rel.Tree))
	}
	sendLog(logCh, fmt.Sprintf("==> Tree matches release %s (%s)", pr.rel.Tag, shortSHA(tree)))

	head, err := gitOutput(ctx, srcDir, "rev-parse", "HEAD")
	if err != nil {
		return "", err
	}
	// The branch keeps the patched commits after the clone moves to
	// another ref, so the build's GitSHA can still be read later.
	if err := b.runCmd(ctx, srcDir, logCh, "", nil, "git", "branch", "--force", patchBranch(pr), head); err != nil {
		return "", err
	}
	return head, nil
}

// patchBranch is the clone branch that keeps a release's patched commits.
func patchBranch(pr *patchRelease) string {
	return "patchset/" + pr.set.ID + "/" + pr.rel.Tag
}

// patchBuildIDRef is the ref part of a patched build's ID ("rdna-r206").
func patchBuildIDRef(pr *patchRelease) string {
	n, _ := patchsets.ReleaseNumber(pr.rel.Tag)
	return fmt.Sprintf("%s-r%d", pr.set.ShortName, n)
}

// seriesAuthor reads the name and email of the first patch's From:
// header, with the set's own fallbacks.
func seriesAuthor(patch string) (name, email string) {
	name, email = "rdna-boosts", "ci@rdna-boosts.invalid"
	data, err := os.ReadFile(patch)
	if err != nil {
		return name, email
	}
	for _, line := range strings.Split(string(data), "\n") {
		from, ok := strings.CutPrefix(line, "From: ")
		if !ok {
			continue
		}
		if i := strings.LastIndex(from, " <"); i > 0 && strings.HasSuffix(from, ">") {
			name, email = from[:i], from[i+2:len(from)-1]
		}
		break
	}
	return name, email
}

// failedPatch names the patch a stopped git am could not apply.
func failedPatch(ctx context.Context, srcDir string) string {
	gitDir, err := gitOutput(ctx, srcDir, "rev-parse", "--absolute-git-dir")
	if err != nil {
		return ""
	}
	dir := filepath.Join(gitDir, "rebase-apply")
	// final-commit holds the message of the patch that stopped; its
	// subject names the block.
	if final, err := os.ReadFile(filepath.Join(dir, "final-commit")); err == nil {
		if subject, _, _ := strings.Cut(string(final), "\n"); subject != "" {
			return subject
		}
	}
	if next, err := os.ReadFile(filepath.Join(dir, "next")); err == nil {
		return "number " + strings.TrimSpace(string(next))
	}
	return ""
}

// cleanGitEnv drops the GIT_* variables that would change what git am
// commits, so an inherited one cannot change the commit IDs.
func cleanGitEnv(env []string) []string {
	out := env[:0:0]
	for _, kv := range env {
		k, _, _ := strings.Cut(kv, "=")
		switch k {
		case "HOME", "GIT_COMMITTER_NAME", "GIT_COMMITTER_EMAIL", "GIT_COMMITTER_DATE",
			"GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_AUTHOR_DATE",
			"GIT_CONFIG", "GIT_CONFIG_GLOBAL", "GIT_CONFIG_SYSTEM", "GIT_CONFIG_NOSYSTEM",
			"GIT_CONFIG_COUNT", "GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE":
			continue
		}
		out = append(out, kv)
	}
	return out
}

// gitOutput runs git in dir and returns its trimmed output.
func gitOutput(ctx context.Context, dir string, args ...string) (string, error) {
	out, err := gitOutputRaw(ctx, dir, args...)
	return strings.TrimSpace(string(out)), err
}

// gitOutputRaw runs git in dir and returns its output as it is.
func gitOutputRaw(ctx context.Context, dir string, args ...string) ([]byte, error) {
	out, err := exec.CommandContext(ctx, "git", append([]string{"-C", dir}, args...)...).Output()
	if err != nil {
		return nil, fmt.Errorf("git %s: %w%s", args[0], err, exitErrDetail(err))
	}
	return out, nil
}

// shortSHA is the first nine characters of a SHA, as the set's releases
// print them.
func shortSHA(sha string) string {
	if len(sha) > 9 {
		return sha[:9]
	}
	return sha
}
