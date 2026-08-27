package builder

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/broadcast"
)

const (
	llamaCppRepo = "https://github.com/ggml-org/llama.cpp"

	// Build statuses.
	BuildStatusBuilding = "building"
	BuildStatusSuccess  = "success"
	BuildStatusFailed   = "failed"
)

// BuildResult records the outcome of a build.
type BuildResult struct {
	ID         string            `json:"id"`
	Tag        string            `json:"tag,omitempty"`
	Profile    string            `json:"profile"`
	GitSHA     string            `json:"git_sha"`
	GitRef     string            `json:"git_ref"`
	Status     string            `json:"status"` // BuildStatusBuilding, BuildStatusSuccess, BuildStatusFailed
	BinaryPath string            `json:"binary_path"`
	CMakeFlags map[string]string `json:"cmake_flags,omitempty"`
	StartedAt  time.Time         `json:"started_at"`
	FinishedAt time.Time         `json:"finished_at,omitempty"`
	Error      string            `json:"error,omitempty"`
	// CommitCount is `git rev-list --count HEAD` of the built checkout.
	// llama.cpp's bN nightly tags ARE the master commit count, so this
	// number ranks builds of ANY ref — semver release tags (v0.x.y,
	// introduced upstream Aug 2026), branches, bare SHAs — on the same
	// scale as b-tags. Zero on builds from before the field existed;
	// those fall back to parsing the b-number out of GitRef.
	CommitCount int `json:"commit_count,omitempty"`
}

const buildLogHistorySize = 2000

// Builder orchestrates llama.cpp builds.
type Builder struct {
	dataDir string

	mu     sync.Mutex
	builds []BuildResult
	logChs map[string]chan string

	// Log history and broadcasting per build
	logMu       sync.Mutex
	logBcasts   map[string]*broadcast.Broadcaster[string] // build ID → log stream
	lastBuildID string                                    // most recent build ID

	refsMu        sync.Mutex
	cachedRefs    []string
	cachedAnchors map[string]int // release tag → its nightly b-number

	// Saved build-flag presets (see flag_presets.go), loaded lazily.
	fpMu        sync.Mutex
	fpLoaded    bool
	flagPresets []FlagPreset
}

// NewBuilder creates a Builder and loads persisted build state.
func NewBuilder(dataDir string) *Builder {
	b := &Builder{
		dataDir:   dataDir,
		logChs:    make(map[string]chan string),
		logBcasts: make(map[string]*broadcast.Broadcaster[string]),
	}
	b.loadBuilds()
	return b
}

// List returns all builds.
func (b *Builder) List() []BuildResult {
	b.mu.Lock()
	defer b.mu.Unlock()
	out := make([]BuildResult, len(b.builds))
	copy(out, b.builds)
	return out
}

// Find returns a copy of the build with the given ID, or false if unknown.
func (b *Builder) Find(id string) (*BuildResult, bool) {
	b.mu.Lock()
	defer b.mu.Unlock()
	for _, br := range b.builds {
		if br.ID == id {
			out := br
			return &out, true
		}
	}
	return nil, false
}

// LatestSuccessfulBuild returns the successful build of the newest
// upstream code: highest buildRank (upstream commit count) first, with
// unrankable builds below ranked ones and ordered by newest StartedAt.
// Returns nil if no successful build exists.
func (b *Builder) LatestSuccessfulBuild() *BuildResult {
	b.mu.Lock()
	defer b.mu.Unlock()

	var ok []BuildResult
	for _, br := range b.builds {
		if br.Status == BuildStatusSuccess {
			ok = append(ok, br)
		}
	}
	if len(ok) == 0 {
		return nil
	}
	sort.SliceStable(ok, func(i, j int) bool {
		ni, oki := buildRank(ok[i])
		nj, okj := buildRank(ok[j])
		switch {
		case oki && okj:
			if ni != nj {
				return ni > nj
			}
		case oki && !okj:
			return true
		case !oki && okj:
			return false
		}
		return ok[i].StartedAt.After(ok[j].StartedAt)
	})
	res := ok[0]
	return &res
}

// buildRank places a build on llama.cpp's own version scale: the master
// commit count. Recorded directly on newer builds; recovered from the
// bN tag (whose N is that same count) on builds from before CommitCount
// existed. A legacy build of a branch or bare SHA has neither and is
// unrankable — (0, false).
func buildRank(r BuildResult) (int, bool) {
	if r.CommitCount > 0 {
		return r.CommitCount, true
	}
	return refTagNumber(r.GitRef)
}

// refTagNumber extracts N from refs shaped like "bN" (llama.cpp's release
// tag format). Returns (0, false) for non-matching refs.
func refTagNumber(ref string) (int, bool) {
	if len(ref) < 2 || ref[0] != 'b' {
		return 0, false
	}
	n, err := strconv.Atoi(ref[1:])
	if err != nil {
		return 0, false
	}
	return n, true
}

// LogChannel returns the log channel for a build in progress.
func (b *Builder) LogChannel(buildID string) (<-chan string, bool) {
	b.mu.Lock()
	defer b.mu.Unlock()
	ch, ok := b.logChs[buildID]
	return ch, ok
}

// logBcast returns the log broadcaster for a build, or nil if the build has
// no logs this process run.
func (b *Builder) logBcast(buildID string) *broadcast.Broadcaster[string] {
	b.logMu.Lock()
	defer b.logMu.Unlock()
	return b.logBcasts[buildID]
}

// SubscribeLogs returns a channel that receives log lines for a build,
// starting with any existing history. Returns nil if the build has no logs.
func (b *Builder) SubscribeLogs(buildID string) chan string {
	bc := b.logBcast(buildID)
	if bc == nil {
		return nil
	}
	return bc.Subscribe()
}

// UnsubscribeLogs removes a log subscriber.
func (b *Builder) UnsubscribeLogs(buildID string, ch chan string) {
	if bc := b.logBcast(buildID); bc != nil {
		bc.Unsubscribe(ch)
	}
}

// broadcastLog stores a log line and sends it to all subscribers.
func (b *Builder) broadcastLog(buildID, line string) {
	if bc := b.logBcast(buildID); bc != nil {
		bc.Broadcast(line)
	}
}

// LastBuildID returns the most recently started build ID.
func (b *Builder) LastBuildID() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.lastBuildID
}

// BuildStatus returns the status of a build by ID.
func (b *Builder) BuildStatus(id string) string {
	b.mu.Lock()
	defer b.mu.Unlock()
	for _, build := range b.builds {
		if build.ID == id {
			return build.Status
		}
	}
	return ""
}

// DuplicateBuildError is returned when a build with the same ref+profile already exists.
type DuplicateBuildError struct {
	ID string
}

func (e *DuplicateBuildError) Error() string {
	return fmt.Sprintf("build %s already exists", e.ID)
}

// Build runs the full build pipeline asynchronously.
// It returns the initial BuildResult immediately; logs stream via LogChannel.
// If force is true, an existing build with the same ID will be replaced.
// tag is an optional user-supplied label; when non-empty it becomes part of
// the build ID so multiple builds of the same ref+profile can coexist.
// optionOverrides allows toggling profile-specific cmake flags.
// extraCMake allows passing additional raw cmake flags.
func (b *Builder) Build(ctx context.Context, profile string, gitRef string, tag string, force bool, optionOverrides map[string]bool, extraCMake string) (*BuildResult, error) {
	slog.Info("build requested", "profile", profile, "git_ref", gitRef, "tag", tag, "force", force)

	prof, ok := FindProfile(profile)
	if !ok {
		return nil, fmt.Errorf("unknown profile: %s", profile)
	}

	if prof.Backend == "vulkan" && RunningInContainer() {
		return nil, fmt.Errorf("vulkan builds are only supported in host mode, not inside containers")
	}

	tag = strings.ToLower(strings.TrimSpace(tag))
	if tag != "" && !validTagRE.MatchString(tag) {
		return nil, fmt.Errorf("invalid tag %q: only lowercase letters, digits, and hyphens are allowed", tag)
	}

	// Apply option overrides (or defaults) to the profile's cmake flags,
	// using the same resolution as the API's flag preview.
	ApplyOptionOverrides(prof.CMakeFlags, ProfileOptions(profile), optionOverrides)

	// Parse extra cmake flags (e.g. "-DFOO=BAR -DBAZ=ON")
	if extraCMake != "" {
		for _, flag := range strings.Fields(extraCMake) {
			flag = strings.TrimPrefix(flag, "-D")
			if parts := strings.SplitN(flag, "=", 2); len(parts) == 2 {
				prof.CMakeFlags[parts[0]] = parts[1]
			}
		}
	}

	if gitRef == "" {
		gitRef = "latest"
	}

	result := &BuildResult{
		Profile:    prof.Name,
		GitRef:     gitRef,
		Tag:        tag,
		Status:     BuildStatusBuilding,
		StartedAt:  time.Now(),
		CMakeFlags: copyFlags(prof.CMakeFlags),
	}

	logCh := make(chan string, 256)

	// Clone/fetch and resolve ref synchronously to get the ID before returning
	srcDir := filepath.Join(b.dataDir, "llama.cpp")
	if err := b.ensureRepo(ctx, srcDir, logCh); err != nil {
		close(logCh)
		return nil, fmt.Errorf("repo setup: %w", err)
	}

	resolvedRef, sha, commitCount, err := b.checkoutRef(ctx, srcDir, gitRef, logCh)
	if err != nil {
		close(logCh)
		return nil, fmt.Errorf("checkout: %w", err)
	}

	result.GitRef = resolvedRef
	result.GitSHA = sha
	result.CommitCount = commitCount

	// Compute ID. Tag wins. If untagged and the bare ID is already taken
	// by a build with different flags, auto-suffix a short hash of this
	// build's flags so it can coexist. If flags are identical, fall through
	// to DuplicateBuildError so the user sees the rebuild prompt.
	baseID := fmt.Sprintf("%s-%s", resolvedRef, prof.Name)
	if tag != "" {
		result.ID = baseID + "-" + tag
	} else {
		result.ID = baseID
		b.mu.Lock()
		for _, br := range b.builds {
			if br.ID == baseID && !flagsEqual(br.CMakeFlags, result.CMakeFlags) {
				result.ID = baseID + "-" + hashFlags(result.CMakeFlags)
				break
			}
		}
		b.mu.Unlock()
	}

	// Check for duplicate build
	b.mu.Lock()
	for i, br := range b.builds {
		if br.ID == result.ID {
			if !force {
				b.mu.Unlock()
				close(logCh)
				return nil, &DuplicateBuildError{ID: result.ID}
			}
			// Replace existing build
			buildDir := filepath.Join(b.dataDir, "builds", br.ID)
			os.RemoveAll(buildDir)
			b.builds = append(b.builds[:i], b.builds[i+1:]...)
			break
		}
	}
	b.logChs[result.ID] = logCh
	b.builds = append(b.builds, *result)
	b.lastBuildID = result.ID
	b.mu.Unlock()

	// (Re)create the log stream for this build, discarding any history
	// from a previous build with the same ID.
	b.logMu.Lock()
	b.logBcasts[result.ID] = broadcast.New[string](buildLogHistorySize, buildLogHistorySize)
	b.logMu.Unlock()

	// The git clone/fetch/checkout lines were buffered before the build ID
	// existed; replay them into the broadcaster so they show up in the
	// streamed log rather than being silently discarded.
drain:
	for {
		select {
		case line := <-logCh:
			b.broadcastLog(result.ID, line)
		default:
			break drain
		}
	}

	slog.Info("build started", "id", result.ID, "git_ref", resolvedRef, "git_sha", sha)

	// Run the actual build asynchronously
	go b.runBuild(ctx, prof, srcDir, result, logCh)

	return result, nil
}

// Delete removes a build and its files.
func (b *Builder) Delete(id string) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	idx := -1
	for i, br := range b.builds {
		if br.ID == id {
			idx = i
			break
		}
	}
	if idx == -1 {
		return fmt.Errorf("build not found: %s", id)
	}

	// Remove build directory
	buildDir := filepath.Join(b.dataDir, "builds", id)
	os.RemoveAll(buildDir)

	b.builds = append(b.builds[:idx], b.builds[idx+1:]...)
	b.saveBuilds()
	return nil
}

func (b *Builder) runBuild(ctx context.Context, prof BuildProfile, srcDir string, result *BuildResult, logCh chan string) {
	defer func() {
		close(logCh)
		// Close all subscriber channels for this build; history is kept so
		// later subscribers still get a replay.
		if bc := b.logBcast(result.ID); bc != nil {
			bc.CloseSubscribers()
		}
	}()

	sendLog := func(msg string) {
		select {
		case logCh <- msg:
		default:
		}
		b.broadcastLog(result.ID, msg)
	}

	buildDir := filepath.Join(srcDir, "build-"+prof.Name)
	os.RemoveAll(buildDir) // clean stale cmake state from previous builds
	os.MkdirAll(buildDir, 0o755)

	// cmake — only build server and required libs, skip tests and examples
	slog.Info("running cmake", "id", result.ID)
	sendLog("==> Running cmake...")
	cmakeArgs := []string{"..", "-G", "Ninja",
		"-DLLAMA_BUILD_TESTS=OFF",
		"-DLLAMA_BUILD_EXAMPLES=OFF",
		"-DLLAMA_BUILD_SERVER=ON",
	}
	for k, v := range prof.CMakeFlags {
		cmakeArgs = append(cmakeArgs, fmt.Sprintf("-D%s=%s", k, v))
	}

	// NVIDIA's official RPM/DEB installs nvcc at /usr/local/cuda/bin without
	// adding it to PATH. The systemd-managed service inherits a minimal PATH,
	// so cmake's enable_language(CUDA) probe fails with "No CMAKE_CUDA_COMPILER
	// could be found" even when the toolkit is installed. Locate nvcc and pin
	// it explicitly when building the cuda profile.
	//
	// Separately, modern distros now ship a default g++ that nvcc rejects
	// (e.g. Fedora 44 ships gcc 16, but CUDA 12.x's host_config.h refuses
	// anything past gcc 13/14/15 depending on toolkit version). Probe for
	// a compatible side-by-side g++ and hand it to cmake as the host
	// compiler — without this, build fails on "unsupported GNU version".
	buildEnv := os.Environ()
	// Arch-family distros install ROCm under /opt/rocm without adding it
	// to PATH (Fedora/Debian symlink the tools into /usr/bin), so cmake's
	// HIP compiler probe and rocm_agent_enumerator fail even with ROCm
	// fully installed. Put the detected tool directory on PATH for the
	// build; harmless when it's already there.
	//
	// ROCM_PATH/HIP_PATH matter for the same layout: Arch's HIP clang
	// lives at <root>/lib/llvm/bin, and its automatic ROCm detection
	// walks up to <root>/lib instead of <root>. The HIP compiler-id link
	// then fails with "unable to find -lamdhip64", cmake reports "The
	// HIP compiler identification is unknown", and the HIP sources get
	// compiled in CUDA mode — surfacing as "unsupported CUDA gpu
	// architecture: gfxNNNN". Point clang at the real root (mirroring
	// the ENV block in Dockerfile.rocm); values already exported around
	// the service win, matching applyExtraEnv's precedence rule.
	if prof.Backend == "rocm" {
		if tool := FindROCmTool("hipconfig"); tool != "" {
			binDir := filepath.Dir(tool)
			buildEnv = prependPath(buildEnv, binDir)
			root := filepath.Dir(binDir)

			// Two ROCm layouts, and what helps one breaks the other.
			//
			// Arch-family (and AMD's own) install a self-contained ROCm
			// under /opt/rocm whose clang lives at <root>/lib/llvm/bin;
			// that clang can't locate its own ROCm and needs ROCM_PATH to
			// find it. Debian/Ubuntu and Fedora instead spread ROCm across
			// /usr, and there ROCM_PATH=/usr is actively harmful: clang
			// takes it as the ROCm root and looks for the device bitcode
			// at /usr/amdgcn/bitcode, while the distro ships it beside the
			// compiler in /usr/lib/llvm-N/lib/clang/N/amdgcn/bitcode. It
			// then fails with "cannot find ROCm device library", which
			// surfaces as "The HIP compiler identification is unknown" —
			// and since cmake derives CMAKE_HIP_LIBRARY_ARCHITECTURE from
			// that identification, it also stops searching the multiarch
			// directory holding hip-lang-config.cmake and dies on "does
			// not contain the HIP runtime CMake package" with the config
			// sitting right there.
			//
			// So: point cmake at the root with a cache variable, which
			// costs clang nothing, and only export ROCM_PATH for the
			// layout that actually needs it.
			if root == "/usr" {
				if !hasCMakeArg(cmakeArgs, "CMAKE_HIP_COMPILER_ROCM_ROOT") {
					cmakeArgs = append(cmakeArgs, "-DCMAKE_HIP_COMPILER_ROCM_ROOT="+root)
				}
			} else {
				buildEnv = setEnvDefault(buildEnv, "ROCM_PATH", root)
				buildEnv = setEnvDefault(buildEnv, "HIP_PATH", root)
				if devlib := filepath.Join(root, "amdgcn", "bitcode"); dirExists(devlib) {
					buildEnv = setEnvDefault(buildEnv, "HIP_DEVICE_LIB_PATH", devlib)
				}
			}
		}
		// cmake asks `hipconfig --hipclangpath` for the HIP clang and
		// otherwise falls back to a bare "clang++" on PATH. On Debian/Ubuntu
		// neither answer lands: hipconfig names a directory with no clang++
		// in it, and the clang HIP needs is installed as the versioned
		// /usr/lib/llvm-N/bin/clang++ with no unversioned symlink. Pin it
		// ourselves, exactly as the cuda profile pins nvcc below.
		if !hasCMakeArg(cmakeArgs, "CMAKE_HIP_COMPILER") {
			if hipcc := findHIPCompiler(buildEnv); hipcc != "" {
				cmakeArgs = append(cmakeArgs, "-DCMAKE_HIP_COMPILER="+hipcc)
				sendLog(fmt.Sprintf("==> Using HIP compiler at %s", hipcc))
			}
		}
	}
	if prof.Backend == "cuda" {
		if nvcc, dir := findNVCC(); nvcc != "" {
			alreadySet := false
			for _, a := range cmakeArgs {
				if strings.HasPrefix(a, "-DCMAKE_CUDA_COMPILER=") {
					alreadySet = true
					break
				}
			}
			if !alreadySet {
				cmakeArgs = append(cmakeArgs, "-DCMAKE_CUDA_COMPILER="+nvcc)
				sendLog(fmt.Sprintf("==> Using nvcc at %s", nvcc))
			}
			buildEnv = prependPath(buildEnv, dir)
		}
		hostCC := findCUDAHostCompiler()
		if hostCC != "" {
			hostCCAlreadySet := false
			for _, a := range cmakeArgs {
				if strings.HasPrefix(a, "-DCMAKE_CUDA_HOST_COMPILER=") {
					hostCCAlreadySet = true
					break
				}
			}
			if !hostCCAlreadySet {
				cmakeArgs = append(cmakeArgs, "-DCMAKE_CUDA_HOST_COMPILER="+hostCC)
				sendLog(fmt.Sprintf("==> Using CUDA host compiler %s", hostCC))
			}
		}
	}

	if err := b.runCmd(ctx, buildDir, logCh, result.ID, buildEnv, "cmake", cmakeArgs...); err != nil {
		b.finishBuild(result, BuildStatusFailed, fmt.Sprintf("cmake failed: %v", err))
		sendLog(fmt.Sprintf("==> cmake FAILED: %v", err))
		return
	}

	// ninja — build all targets (target names vary across llama.cpp versions)
	slog.Info("running ninja", "id", result.ID)
	sendLog("==> Running ninja...")
	if err := b.runCmd(ctx, buildDir, logCh, result.ID, buildEnv, "ninja", "-j", fmt.Sprintf("%d", runtime.NumCPU())); err != nil {
		b.finishBuild(result, BuildStatusFailed, fmt.Sprintf("ninja failed: %v", err))
		sendLog(fmt.Sprintf("==> ninja FAILED: %v", err))
		return
	}

	// Install binary — check common locations across llama.cpp versions
	outDir := filepath.Join(b.dataDir, "builds", result.ID)
	os.MkdirAll(outDir, 0o755)

	srcBin := ""
	for _, candidate := range []string{
		filepath.Join(buildDir, "bin", "llama-server"),
		filepath.Join(buildDir, "bin", "server"),
	} {
		if _, err := os.Stat(candidate); err == nil {
			srcBin = candidate
			break
		}
	}
	if srcBin == "" {
		b.finishBuild(result, BuildStatusFailed, "llama-server binary not found in build output")
		sendLog(fmt.Sprintf("==> Install FAILED: llama-server binary not found"))
		return
	}
	dstBin := filepath.Join(outDir, "llama-server")

	if err := copyFile(srcBin, dstBin); err != nil {
		b.finishBuild(result, BuildStatusFailed, fmt.Sprintf("install failed: %v", err))
		sendLog(fmt.Sprintf("==> Install FAILED: %v", err))
		return
	}
	os.Chmod(dstBin, 0o755)

	// Copy shared libraries the binary depends on
	libDir := filepath.Join(buildDir, "lib")
	if entries, err := os.ReadDir(libDir); err == nil {
		for _, e := range entries {
			name := e.Name()
			if strings.HasSuffix(name, ".so") || strings.Contains(name, ".so.") {
				src := filepath.Join(libDir, name)
				dst := filepath.Join(outDir, name)
				if err := copyFile(src, dst); err == nil {
					sendLog(fmt.Sprintf("    Installed lib: %s", name))
				}
			}
		}
	}
	// Also check bin/ for .so files (some versions put them there)
	binDir := filepath.Dir(srcBin)
	if entries, err := os.ReadDir(binDir); err == nil {
		for _, e := range entries {
			name := e.Name()
			if strings.HasSuffix(name, ".so") || strings.Contains(name, ".so.") {
				src := filepath.Join(binDir, name)
				dst := filepath.Join(outDir, name)
				if err := copyFile(src, dst); err == nil {
					sendLog(fmt.Sprintf("    Installed lib: %s", name))
				}
			}
		}
	}

	// Copy llama-bench if it exists (used for benchmarking)
	for _, benchCandidate := range []string{
		filepath.Join(buildDir, "bin", "llama-bench"),
		filepath.Join(buildDir, "bin", "bench"),
	} {
		if _, err := os.Stat(benchCandidate); err == nil {
			dstBench := filepath.Join(outDir, "llama-bench")
			if err := copyFile(benchCandidate, dstBench); err == nil {
				os.Chmod(dstBench, 0o755)
				sendLog("    Installed: llama-bench")
			}
			break
		}
	}

	// Copy llama-perplexity if it exists (used for capability
	// evaluations, which run it directly without llama-server). A
	// missing binary is NOT a build failure: builds made before this
	// step existed (old refs, exotic layouts) simply lack it, and
	// capability cells detect the absence at run time with a rebuild
	// hint.
	for _, perplexityCandidate := range llamaPerplexityCandidates(buildDir) {
		if _, err := os.Stat(perplexityCandidate); err == nil {
			dstPerplexity := filepath.Join(outDir, "llama-perplexity")
			if err := copyFile(perplexityCandidate, dstPerplexity); err == nil {
				os.Chmod(dstPerplexity, 0o755)
				sendLog("    Installed: llama-perplexity")
			}
			break
		}
	}

	// Cleanup temp build dir
	os.RemoveAll(buildDir)

	result.BinaryPath = dstBin
	b.finishBuild(result, BuildStatusSuccess, "")
	sendLog(fmt.Sprintf("==> Build complete: %s", dstBin))
}

func (b *Builder) finishBuild(result *BuildResult, status, errMsg string) {
	b.mu.Lock()
	defer b.mu.Unlock()

	result.Status = status
	result.Error = errMsg
	result.FinishedAt = time.Now()

	if status == BuildStatusFailed {
		slog.Error("build failed", "id", result.ID, "error", errMsg)
	} else {
		slog.Info("build succeeded", "id", result.ID, "binary", result.BinaryPath)
	}

	for i, br := range b.builds {
		if br.ID == result.ID {
			b.builds[i] = *result
			break
		}
	}
	b.saveBuilds()
}

func (b *Builder) ensureRepo(ctx context.Context, srcDir string, logCh chan string) error {
	if _, err := os.Stat(filepath.Join(srcDir, ".git")); err == nil {
		slog.Info("fetching llama.cpp", "dir", srcDir)
		sendLog(logCh, "==> Fetching latest from llama.cpp...")
		return b.runCmd(ctx, srcDir, logCh, "", nil, "git", "fetch", "--all", "--tags")
	}

	slog.Info("cloning llama.cpp", "repo", llamaCppRepo, "dir", srcDir)
	sendLog(logCh, "==> Cloning llama.cpp...")
	return b.runCmd(ctx, filepath.Dir(srcDir), logCh, "", nil, "git", "clone", llamaCppRepo, filepath.Base(srcDir))
}

// checkoutRef checks out the given ref and returns (resolvedRef, sha,
// commitCount, error). commitCount is `git rev-list --count HEAD` of the
// checkout — llama.cpp's own version scale (its bN tags are exactly that
// count) — and is 0 if counting fails; ranking then falls back to the
// ref's tag number.
//
// If ref is "latest", it resolves to the newest bN nightly tag — the
// meaning "latest" has always had here. Upstream added semver release
// tags (v0.x.y) in Aug 2026 alongside the nightlies; those are offered
// in the ref picker but never chosen implicitly. Should upstream ever
// stop cutting b-tags, the newest v-tag is the fallback, then HEAD.
func (b *Builder) checkoutRef(ctx context.Context, srcDir string, ref string, logCh chan string) (string, string, int, error) {
	if ref == "latest" {
		// --sort=-v:refname gives proper version ordering within a family.
		ref = ""
		for _, family := range []string{"b*", "v*"} {
			out, err := exec.CommandContext(ctx, "git", "-C", srcDir, "tag", "--sort=-v:refname", "-l", family).Output()
			if err != nil {
				return "", "", 0, fmt.Errorf("listing tags: %w%s", err, exitErrDetail(err))
			}
			tags := strings.Split(strings.TrimSpace(string(out)), "\n")
			if len(tags) > 0 && tags[0] != "" {
				ref = tags[0]
				break
			}
		}
		if ref == "" {
			ref = "HEAD"
		}
		sendLog(logCh, fmt.Sprintf("==> Latest tag: %s", ref))
	}

	slog.Info("checking out ref", "ref", ref)
	sendLog(logCh, fmt.Sprintf("==> Checking out %s...", ref))
	// The clone is a build cache managed entirely by us — nothing in it is
	// ever worth preserving. Force the checkout so leftover local changes
	// (e.g. a checkout interrupted by a restart, which leaves the tree
	// half-updated) can't block future builds with "Please commit your
	// changes or stash them".
	if err := b.runCmd(ctx, srcDir, logCh, "", nil, "git", "checkout", "--force", ref); err != nil {
		return "", "", 0, err
	}

	out, err := exec.CommandContext(ctx, "git", "-C", srcDir, "rev-parse", "HEAD").Output()
	if err != nil {
		return "", "", 0, fmt.Errorf("rev-parse: %w%s", err, exitErrDetail(err))
	}
	sha := strings.TrimSpace(string(out))

	// Best-effort: a build is still usable without its count, it just
	// ranks by its ref's tag number (or not at all) instead.
	count := 0
	if cout, err := exec.CommandContext(ctx, "git", "-C", srcDir, "rev-list", "--count", "HEAD").Output(); err == nil {
		count, _ = strconv.Atoi(strings.TrimSpace(string(cout)))
	} else {
		slog.Warn("counting commits failed; build will rank by ref only", "ref", ref, "error", err)
	}
	return ref, sha, count, nil
}

// FetchRefs pulls the latest tags from the llama.cpp remote and returns
// the available tags: v* semver release tags first (few, stable), then
// the b* nightly tags — each family newest-first. Results are cached;
// call this to refresh.
//
// The remote fetch is best-effort: if it fails (no network, transient error,
// timeout), we still return whatever tags are already in the local clone so
// the user can pick from cached tags rather than seeing an error.
func (b *Builder) FetchRefs() ([]string, error) {
	srcDir := filepath.Join(b.dataDir, "llama.cpp")
	if _, err := os.Stat(filepath.Join(srcDir, ".git")); err != nil {
		return nil, fmt.Errorf("llama.cpp repo not cloned yet — run a build first")
	}

	// Fetch from origin so newly-pushed upstream tags become visible to
	// the subsequent `git tag` listing. Bounded timeout so a slow or
	// unreachable remote doesn't hang the UI request.
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if err := exec.CommandContext(ctx, "git", "-C", srcDir, "fetch", "--tags", "--prune", "origin").Run(); err != nil {
		slog.Warn("git fetch failed; returning cached tags", "error", err)
	}

	// The two families are listed separately: mixing them in one
	// -v:refname sort would interleave v0.x.y between b-numbers
	// meaninglessly. Prefix keeps them distinguishable downstream.
	var refs []string
	var releases []string
	for _, family := range []string{"v*", "b*"} {
		out, err := exec.Command("git", "-C", srcDir, "tag", "--sort=-v:refname", "-l", family).Output()
		if err != nil {
			return nil, fmt.Errorf("listing tags: %w", err)
		}
		for _, t := range strings.Split(strings.TrimSpace(string(out)), "\n") {
			if t = strings.TrimSpace(t); t != "" {
				refs = append(refs, t)
				if family == "v*" {
					releases = append(releases, t)
				}
			}
		}
	}

	// Anchor each release to the nightly scale: its commit count IS the
	// b-number of the nightly it was cut from (upstream tags releases on
	// nightlies, and bN = commit count N). Releases are few, so one
	// rev-list per tag is cheap; failures just leave that tag unanchored.
	anchors := make(map[string]int, len(releases))
	for _, tag := range releases {
		out, err := exec.Command("git", "-C", srcDir, "rev-list", "--count", tag).Output()
		if err != nil {
			continue
		}
		if n, err := strconv.Atoi(strings.TrimSpace(string(out))); err == nil && n > 0 {
			anchors[tag] = n
		}
	}

	b.refsMu.Lock()
	b.cachedRefs = refs
	b.cachedAnchors = anchors
	b.refsMu.Unlock()

	return refs, nil
}

// CachedRefs returns the last fetched refs without hitting git.
func (b *Builder) CachedRefs() []string {
	b.refsMu.Lock()
	defer b.refsMu.Unlock()
	out := make([]string, len(b.cachedRefs))
	copy(out, b.cachedRefs)
	return out
}

// ReleaseAnchors returns, for each release tag from the last FetchRefs,
// the nightly b-number that release corresponds to (its commit count).
// The UI uses it to label releases like "v0.2.0 (b10500)".
func (b *Builder) ReleaseAnchors() map[string]int {
	b.refsMu.Lock()
	defer b.refsMu.Unlock()
	out := make(map[string]int, len(b.cachedAnchors))
	for k, v := range b.cachedAnchors {
		out[k] = v
	}
	return out
}

// HasBuild checks if a build with the given ID already exists.
func (b *Builder) HasBuild(id string) bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	for _, br := range b.builds {
		if br.ID == id {
			return true
		}
	}
	return false
}

// runCmd runs a command, streaming stdout+stderr line-by-line to the log channel.
// If env is non-nil it overrides the inherited environment; pass nil to inherit
// os.Environ() unchanged (the common case for git operations).
func (b *Builder) runCmd(ctx context.Context, dir string, logCh chan string, buildID string, env []string, name string, args ...string) error {
	cmd := exec.CommandContext(ctx, name, args...)
	cmd.Dir = dir
	if env != nil {
		cmd.Env = env
	}

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}
	cmd.Stderr = cmd.Stdout // merge stderr into stdout

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("starting %s: %w", name, err)
	}

	// Keep the last few output lines so a failure's error message carries
	// the actual tool output (e.g. git's "fatal: ..."), not just an exit code.
	const tailSize = 10
	var tail []string

	scanner := bufio.NewScanner(stdout)
	for scanner.Scan() {
		line := scanner.Text()
		select {
		case logCh <- line:
		default:
			// drop if channel full
		}
		if buildID != "" {
			b.broadcastLog(buildID, line)
		}
		tail = append(tail, line)
		if len(tail) > tailSize {
			tail = tail[1:]
		}
	}

	if err := cmd.Wait(); err != nil {
		if detail := strings.TrimSpace(strings.Join(tail, "\n")); detail != "" {
			return fmt.Errorf("%s: %w\n%s", name, err, detail)
		}
		return fmt.Errorf("%s: %w", name, err)
	}
	return nil
}

// exitErrDetail returns the captured stderr of an exec.ExitError (as produced
// by cmd.Output()), formatted for appending to an error message.
func exitErrDetail(err error) string {
	var ee *exec.ExitError
	if errors.As(err, &ee) && len(ee.Stderr) > 0 {
		return ": " + strings.TrimSpace(string(ee.Stderr))
	}
	return ""
}

func sendLog(ch chan string, msg string) {
	select {
	case ch <- msg:
	default:
	}
}

func (b *Builder) buildsPath() string {
	return filepath.Join(b.dataDir, "config", "builds.json")
}

func (b *Builder) loadBuilds() {
	data, err := os.ReadFile(b.buildsPath())
	if err != nil {
		return
	}
	json.Unmarshal(data, &b.builds)

	// Clean up stale "building" entries — they can't recover after restart.
	cleaned := b.builds[:0]
	for _, br := range b.builds {
		if br.Status != BuildStatusBuilding {
			cleaned = append(cleaned, br)
		} else {
			slog.Warn("discarding build interrupted by restart", "id", br.ID)
		}
	}
	if len(cleaned) != len(b.builds) {
		b.builds = cleaned
		b.saveBuilds()
	}
}

func (b *Builder) saveBuilds() {
	os.MkdirAll(filepath.Dir(b.buildsPath()), 0o755)
	data, err := json.MarshalIndent(b.builds, "", "  ")
	if err != nil {
		slog.Error("failed to marshal builds", "error", err)
		return
	}
	os.WriteFile(b.buildsPath(), data, 0o644)
}

var validTagRE = regexp.MustCompile(`^[a-z0-9][a-z0-9-]*$`)

// hashFlags returns a short stable hash of a cmake flag set, used to
// disambiguate untagged builds whose flags differ from an existing one.
func hashFlags(flags map[string]string) string {
	keys := make([]string, 0, len(flags))
	for k := range flags {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	h := sha256.New()
	for _, k := range keys {
		h.Write([]byte(k))
		h.Write([]byte{'='})
		h.Write([]byte(flags[k]))
		h.Write([]byte{'\n'})
	}
	return hex.EncodeToString(h.Sum(nil))[:6]
}

// flagsEqual reports whether two flag maps have identical contents.
// A nil map (legacy builds predating flag persistence) compares unequal
// to any non-empty map, which is the conservative choice — we'd rather
// hash-suffix an unknown legacy build than collide with it.
func flagsEqual(a, b map[string]string) bool {
	if len(a) != len(b) {
		return false
	}
	for k, va := range a {
		if vb, ok := b[k]; !ok || vb != va {
			return false
		}
	}
	return true
}

func copyFlags(in map[string]string) map[string]string {
	out := make(map[string]string, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()

	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer out.Close()

	_, err = io.Copy(out, in)
	return err
}

// llamaPerplexityCandidates returns the build-output locations the
// install step searches for the llama-perplexity binary, mirroring the
// llama-server search (bin/ across llama.cpp versions). The list is a
// function so tests can pin it; a build whose layout differs simply
// ships without the binary, which capability cells handle at run time.
func llamaPerplexityCandidates(buildDir string) []string {
	return []string{
		filepath.Join(buildDir, "bin", "llama-perplexity"),
	}
}

// findCUDAHostCompiler probes for a side-by-side g++ that nvcc will
// accept as host compiler. Modern distros (Fedora 44 with gcc 16,
// Ubuntu 24.04 with gcc 13/14) ship a default g++ that exceeds CUDA's
// supported range; the side-by-side packages (gcc15-c++, g++-13, ...)
// land at /usr/bin/g++-N. Returns the absolute path of the highest
// available version, or "" if none is installed.
func findCUDAHostCompiler() string {
	for _, c := range []string{
		"/usr/bin/g++-15",
		"/usr/bin/g++-14",
		"/usr/bin/g++-13",
		"/usr/bin/g++-12",
	} {
		if st, err := os.Stat(c); err == nil && !st.IsDir() {
			return c
		}
	}
	return ""
}

// findNVCC locates the CUDA compiler. NVIDIA's official RPM/DEB installs
// nvcc under /usr/local/cuda/bin without adding it to PATH, so PATH alone
// isn't enough. Returns (full path, containing dir) or ("", "") if absent.
func findNVCC() (string, string) {
	if p, err := exec.LookPath("nvcc"); err == nil {
		return p, filepath.Dir(p)
	}
	for _, c := range []string{"/usr/local/cuda/bin/nvcc", "/opt/cuda/bin/nvcc"} {
		if st, err := os.Stat(c); err == nil && !st.IsDir() {
			return c, filepath.Dir(c)
		}
	}
	return "", ""
}

// hasCMakeArg reports whether -D<name>=... was already supplied, so a
// user-provided value in the build's extra cmake flags wins over ours.
func hasCMakeArg(args []string, name string) bool {
	prefix := "-D" + name + "="
	for _, a := range args {
		if strings.HasPrefix(a, prefix) {
			return true
		}
	}
	return false
}

// findHIPCompiler locates a clang++ that can compile HIP device code, for
// cmake's CMAKE_HIP_COMPILER. Returns "" when nothing suitable is found,
// leaving cmake to its own search.
//
// Order matters. A ROCm install that ships its own llvm knows best, so ask
// hipconfig first and then look in the usual ROCm roots. Only then fall
// back to a distro-packaged clang, and only one that actually has the
// AMD device bitcode beside it: on Debian/Ubuntu the ROCm device libs are
// packaged per-LLVM-version (rocm-device-libs-17 installs into
// /usr/lib/llvm-17/lib/clang/17/amdgcn/bitcode), so the newest clang on
// the system is often the wrong one. Pairing on the bitcode picks the
// clang the distro actually built its HIP stack against.
func findHIPCompiler(env []string) string {
	if p := hipconfigClangPath(env); p != "" {
		return p
	}
	var roots []string
	if rp := envValue(env, "ROCM_PATH"); rp != "" {
		roots = append(roots, rp)
	}
	roots = append(roots, "/opt/rocm", "/usr/lib/rocm", "/usr/lib64/rocm")
	if c := hipClangInRoots(roots); c != "" {
		return c
	}
	return newestDistroHIPClang("/usr/lib")
}

// hipClangInRoots returns the first <root>/llvm/bin/clang++ that exists.
func hipClangInRoots(roots []string) string {
	for _, r := range roots {
		if c := filepath.Join(r, "llvm", "bin", "clang++"); isExecutable(c) {
			return c
		}
	}
	return ""
}

// hipconfigClangPath asks hipconfig where the HIP clang is and returns
// <path>/clang++ if that file exists. Empty when hipconfig is missing,
// fails, or names a directory that isn't there — which is the common case
// on Debian/Ubuntu and the reason this whole function exists.
func hipconfigClangPath(env []string) string {
	tool := FindROCmTool("hipconfig")
	if tool == "" {
		return ""
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, tool, "--hipclangpath")
	cmd.Env = env
	out, err := cmd.Output()
	if err != nil {
		return ""
	}
	dir := strings.TrimSpace(string(out))
	if dir == "" {
		return ""
	}
	if c := filepath.Join(dir, "clang++"); isExecutable(c) {
		return c
	}
	return ""
}

// newestDistroHIPClang returns the highest-versioned
// <libDir>/llvm-N/bin/clang++ that has AMD device bitcode installed
// alongside it, or "" if none does.
func newestDistroHIPClang(libDir string) string {
	matches, err := filepath.Glob(filepath.Join(libDir, "llvm-*", "bin", "clang++"))
	if err != nil {
		return ""
	}
	best, bestVer := "", -1
	for _, m := range matches {
		if !isExecutable(m) {
			continue
		}
		llvmRoot := filepath.Dir(filepath.Dir(m))
		bitcode, err := filepath.Glob(filepath.Join(llvmRoot, "lib", "clang", "*", "amdgcn", "bitcode"))
		if err != nil || len(bitcode) == 0 {
			continue
		}
		ver, err := strconv.Atoi(strings.TrimPrefix(filepath.Base(llvmRoot), "llvm-"))
		if err != nil {
			continue
		}
		if ver > bestVer {
			best, bestVer = m, ver
		}
	}
	return best
}

// isExecutable reports whether path is a regular file with an execute bit.
func isExecutable(path string) bool {
	st, err := os.Stat(path)
	return err == nil && !st.IsDir() && st.Mode()&0o111 != 0
}

// envValue returns the value of KEY in a KEY=value environment slice.
func envValue(env []string, key string) string {
	prefix := key + "="
	for _, kv := range env {
		if strings.HasPrefix(kv, prefix) {
			return strings.TrimPrefix(kv, prefix)
		}
	}
	return ""
}

// setEnvDefault returns env with KEY=value appended unless KEY is
// already present — an inherited value stays authoritative.
func setEnvDefault(env []string, key, value string) []string {
	for _, kv := range env {
		if strings.HasPrefix(kv, key+"=") {
			return env
		}
	}
	return append(env, key+"="+value)
}

// dirExists reports whether path exists and is a directory.
func dirExists(path string) bool {
	st, err := os.Stat(path)
	return err == nil && st.IsDir()
}

// prependPath returns env with dir prepended to PATH (no-op if dir is
// already first). Operates on a copy so the caller's slice is unchanged.
func prependPath(env []string, dir string) []string {
	out := make([]string, 0, len(env)+1)
	found := false
	for _, kv := range env {
		if strings.HasPrefix(kv, "PATH=") {
			cur := strings.TrimPrefix(kv, "PATH=")
			if cur == dir || strings.HasPrefix(cur, dir+":") {
				out = append(out, kv)
			} else {
				out = append(out, "PATH="+dir+":"+cur)
			}
			found = true
		} else {
			out = append(out, kv)
		}
	}
	if !found {
		out = append(out, "PATH="+dir)
	}
	return out
}
