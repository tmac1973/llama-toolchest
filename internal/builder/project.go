package builder

import (
	"context"
	"os/exec"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
)

// Project IDs.
const (
	ProjectLlamaCpp = "llama.cpp"
	ProjectSDCpp    = "stable-diffusion.cpp"
	ProjectAceStep  = "acestep.cpp"
)

// BinarySpec names a program a build installs and where to find it in the
// cmake build directory.
type BinarySpec struct {
	Name       string
	Candidates []string // relative to the cmake build directory
}

// Project describes a ggml-based source tree the builder can build: where
// it comes from, how a build profile maps to its cmake options, which
// programs a build must produce, and how its release tags look.
type Project struct {
	ID string
	// Kind is the generation backend kind ("sd", "ace") the project's
	// builds run as; empty for llama.cpp. A plain string, so this package
	// does not import internal/generation.
	Kind string
	Repo string
	// CloneDir is the clone's directory under the data dir.
	CloneDir   string
	Submodules bool
	// IDPrefix starts the build IDs of this project ("" for llama.cpp, so
	// existing IDs keep their form).
	IDPrefix string
	// BaseCMakeArgs are passed to every build of the project.
	BaseCMakeArgs []string
	// Backends lists the profile backends the project can be built for;
	// nil means all of them.
	Backends []string
	Required BinarySpec
	Optional []BinarySpec
	// ReleaseNotesURL is where the UI links for a ref; "{ref}" is
	// replaced by the ref, and LatestNotesURL is used for "latest".
	ReleaseNotesURL string
	LatestNotesURL  string
	// mapFlags turns profile flags (GGML_*) into the project's own. nil
	// means the project reads the GGML_* flags directly.
	mapFlags func(map[string]string)
	// latestRef picks the ref "latest" builds; listRefs lists the release
	// tags offered in the ref picker, newest first.
	latestRef func(ctx context.Context, srcDir string) (string, error)
	listRefs  func(ctx context.Context, srcDir string) ([]string, error)
}

// projects returns the projects the builder knows. A function rather than
// a package var so the llama.cpp repo follows llamaCppRepo, which tests
// point at a local fixture.
func projects() []Project {
	llama := Project{
		ID:            ProjectLlamaCpp,
		Repo:          llamaCppRepo,
		CloneDir:      "llama.cpp",
		BaseCMakeArgs: []string{"-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF", "-DLLAMA_BUILD_SERVER=ON"},
		Required: BinarySpec{"llama-server", []string{
			filepath.Join("bin", "llama-server"), filepath.Join("bin", "server"),
		}},
		Optional: []BinarySpec{
			{"llama-bench", []string{filepath.Join("bin", "llama-bench"), filepath.Join("bin", "bench")}},
			{"llama-perplexity", llamaPerplexityCandidates("")},
		},
		ReleaseNotesURL: "https://github.com/ggml-org/llama.cpp/releases/tag/{ref}",
		LatestNotesURL:  "https://github.com/ggml-org/llama.cpp/releases/latest",
		latestRef:       llamaLatestRef,
		listRefs:        llamaListRefs,
	}
	gpuOnly := []string{"rocm", "cuda", "vulkan"}
	sd := Project{
		ID:         ProjectSDCpp,
		Kind:       "sd",
		Repo:       "https://github.com/leejet/stable-diffusion.cpp",
		CloneDir:   "stable-diffusion.cpp",
		Submodules: true,
		IDPrefix:   "sd-",
		// The flags that built sd-server on gfx1201 on 2026-10-24, minus
		// CMAKE_BUILD_WITH_INSTALL_RPATH: it left the binary without the
		// path to the GPU runtime (/opt/rocm-7.2.4/lib), which the Fedora
		// ROCm image does not give the system loader, so sd-server could not
		// start in a container. llama.cpp builds keep that path the same way.
		BaseCMakeArgs: []string{"-DSD_BUILD_EXAMPLES=ON", "-DCMAKE_POSITION_INDEPENDENT_CODE=ON"},
		Backends:      gpuOnly,
		Required:      BinarySpec{"sd-server", []string{filepath.Join("bin", "sd-server")}},
		Optional:      []BinarySpec{{"sd-cli", []string{filepath.Join("bin", "sd-cli")}}},
		// sd.cpp tags each master build; there is no "latest" release page.
		ReleaseNotesURL: "https://github.com/leejet/stable-diffusion.cpp/releases/tag/{ref}",
		LatestNotesURL:  "https://github.com/leejet/stable-diffusion.cpp/releases",
		mapFlags:        sdMapFlags,
		latestRef:       sdLatestRef,
		listRefs:        sdListRefs,
	}
	ace := Project{
		ID:         ProjectAceStep,
		Kind:       "ace",
		Repo:       "https://github.com/ServeurpersoCom/acestep.cpp",
		CloneDir:   "acestep.cpp",
		Submodules: true,
		IDPrefix:   "ace-",
		Backends:   gpuOnly,
		mapFlags:   aceMapFlags,
		Required:   BinarySpec{"ace-server", []string{"ace-server", filepath.Join("bin", "ace-server")}},
		Optional: []BinarySpec{
			{"ace-lm", []string{"ace-lm", filepath.Join("bin", "ace-lm")}},
			{"ace-synth", []string{"ace-synth", filepath.Join("bin", "ace-synth")}},
		},
		// acestep.cpp has no release tags; builds follow its default branch.
		ReleaseNotesURL: "https://github.com/ServeurpersoCom/acestep.cpp/commits/{ref}",
		LatestNotesURL:  "https://github.com/ServeurpersoCom/acestep.cpp/commits",
		latestRef:       defaultBranchRef,
		listRefs:        func(context.Context, string) ([]string, error) { return nil, nil },
	}
	return []Project{llama, sd, ace}
}

// Projects returns every project the builder knows, llama.cpp first.
func Projects() []Project { return projects() }

// ProjectForKind returns the project that builds a generation backend
// kind. Video sets ("video") run stable-diffusion.cpp's sd-server, so they
// use its project.
func ProjectForKind(kind string) (Project, bool) {
	if kind == "video" {
		kind = "sd"
	}
	for _, p := range projects() {
		if p.Kind != "" && p.Kind == kind {
			return p, true
		}
	}
	return Project{}, false
}

// FindProject returns the project with the given ID. "" means llama.cpp,
// which is what builds recorded before projects existed were.
func FindProject(id string) (Project, bool) {
	if id == "" {
		id = ProjectLlamaCpp
	}
	for _, p := range projects() {
		if p.ID == id {
			return p, true
		}
	}
	return Project{}, false
}

// SupportsBackend reports whether the project can be built for a profile
// backend.
func (p Project) SupportsBackend(backend string) bool {
	return p.Backends == nil || slices.Contains(p.Backends, backend)
}

// NotesURL returns the release-notes link for a ref.
func (p Project) NotesURL(ref string) string {
	if ref == "" || ref == "latest" {
		return p.LatestNotesURL
	}
	return strings.ReplaceAll(p.ReleaseNotesURL, "{ref}", ref)
}

// ProjectID returns the build's project, treating builds from before
// projects existed as llama.cpp.
func (r BuildResult) ProjectID() string {
	if r.Project == "" {
		return ProjectLlamaCpp
	}
	return r.Project
}

// buildIDRef is the ref's part of a build ID. A branch build (a ref like
// "origin/master", which is what "latest" means for a project without
// tags) gets the commit's short SHA, so each new commit on the branch gets
// its own ID instead of a rebuild prompt. Slashes are not valid in an ID.
func buildIDRef(ref, sha string) string {
	if branch, ok := strings.CutPrefix(ref, "origin/"); ok {
		if len(sha) > 7 {
			sha = sha[:7]
		}
		ref = branch + "-" + sha
	}
	return strings.ReplaceAll(ref, "/", "-")
}

// sdMapFlags adds stable-diffusion.cpp's own backend switches for the
// GGML_* ones a profile sets. The GGML_* flags stay too: sd.cpp passes
// them on to its bundled ggml.
func sdMapFlags(flags map[string]string) {
	for ggml, sd := range map[string]string{
		"GGML_HIP":    "SD_HIPBLAS",
		"GGML_CUDA":   "SD_CUDA",
		"GGML_VULKAN": "SD_VULKAN",
	} {
		if flags[ggml] == "ON" {
			flags[sd] = "ON"
		}
	}
	// llama.cpp's own option; sd.cpp has no such switch.
	delete(flags, "LLAMA_OPENSSL")
}

// aceSafeCUDAArchs is acestep.cpp's own CUDA architecture list without
// sm_121a. Upstream picks sm_121a whenever CUDA is 12.8 or newer, but
// nvcc only accepts it from CUDA 12.9, so the CUDA 12.8 container image
// failed at cmake with "Unsupported gpu architecture 'compute_121a'".
const aceSafeCUDAArchs = "75-virtual;80-virtual;86-real;89-real;120a-real"

// aceMapFlags chooses the CUDA architectures for acestep.cpp, which does
// not follow GGML_NATIVE for them: the GPUs present when native CPU
// optimizations are on, as llama.cpp and sd.cpp do, otherwise
// aceSafeCUDAArchs. A value from the build's extra cmake flags is applied
// afterwards and wins.
func aceMapFlags(flags map[string]string) {
	if flags["GGML_CUDA"] != "ON" {
		return
	}
	if _, set := flags["CMAKE_CUDA_ARCHITECTURES"]; set {
		return
	}
	if flags["GGML_NATIVE"] == "ON" {
		flags["CMAKE_CUDA_ARCHITECTURES"] = "native"
	} else {
		flags["CMAKE_CUDA_ARCHITECTURES"] = aceSafeCUDAArchs
	}
}

// llamaLatestRef is the newest bN nightly tag, the meaning "latest" has
// always had here. Upstream added semver release tags (v0.x.y) in Aug 2026
// alongside the nightlies; those are offered in the ref picker but never
// chosen implicitly. Should upstream ever stop cutting b-tags, the newest
// v-tag is the fallback, then HEAD.
func llamaLatestRef(ctx context.Context, srcDir string) (string, error) {
	for _, family := range []string{"b", "v"} {
		tags, err := listTags(ctx, srcDir, family)
		if err != nil {
			return "", err
		}
		if len(tags) > 0 {
			return tags[0], nil
		}
	}
	return "HEAD", nil
}

// llamaListRefs lists v* release tags first (few, stable), then the b*
// nightlies, each family newest first. The two families are listed
// separately: mixing them in one -v:refname sort would interleave v0.x.y
// between b-numbers meaninglessly.
func llamaListRefs(ctx context.Context, srcDir string) ([]string, error) {
	var refs []string
	for _, family := range []string{"v", "b"} {
		tags, err := listTags(ctx, srcDir, family)
		if err != nil {
			return nil, err
		}
		refs = append(refs, tags...)
	}
	return refs, nil
}

// sdTagRE matches stable-diffusion.cpp's numbered build tags, e.g.
// "master-1000-95ff762". Older tags carry only a hash ("master-ff4fdbb")
// and cannot be ordered, so they are left out.
var sdTagRE = regexp.MustCompile(`^master-([0-9]+)-[0-9a-f]+$`)

func sdListRefs(ctx context.Context, srcDir string) ([]string, error) {
	out, err := exec.CommandContext(ctx, "git", "-C", srcDir, "tag", "-l", "master-*").Output()
	if err != nil {
		return nil, err
	}
	return sortSDTags(strings.Split(string(out), "\n")), nil
}

// sortSDTags keeps the numbered sd.cpp tags and orders them newest first.
func sortSDTags(lines []string) []string {
	type tag struct {
		name string
		n    int
	}
	var tags []tag
	for _, l := range lines {
		l = strings.TrimSpace(l)
		if m := sdTagRE.FindStringSubmatch(l); m != nil {
			n, _ := strconv.Atoi(m[1])
			tags = append(tags, tag{l, n})
		}
	}
	slices.SortStableFunc(tags, func(a, b tag) int { return b.n - a.n })
	names := make([]string, len(tags))
	for i, t := range tags {
		names[i] = t.name
	}
	return names
}

func sdLatestRef(ctx context.Context, srcDir string) (string, error) {
	tags, err := sdListRefs(ctx, srcDir)
	if err != nil {
		return "", err
	}
	if len(tags) > 0 {
		return tags[0], nil
	}
	return defaultBranchRef(ctx, srcDir)
}

// defaultBranchRef is the tip of the remote's default branch, for projects
// without release tags.
func defaultBranchRef(ctx context.Context, srcDir string) (string, error) {
	out, err := exec.CommandContext(ctx, "git", "-C", srcDir, "rev-parse", "--abbrev-ref", "origin/HEAD").Output()
	if err == nil {
		if ref := strings.TrimSpace(string(out)); ref != "" && ref != "origin/HEAD" {
			return ref, nil
		}
	}
	return "origin/master", nil
}

// llamaOnlyOptions are profile options that do nothing for
// stable-diffusion.cpp and acestep.cpp, although their bundled ggml may
// define them. Checked against sd.cpp master-1006 and acestep.cpp de858d8
// on 2026-10-25:
//   - GGML_CUDA_FA_ALL_QUANTS: kernels for quantized LLM KV caches, which
//     neither backend has.
//   - GGML_HIP_RCCL, GGML_CUDA_NO_PEER_COPY: for one model split across
//     several GPUs; a generation instance runs on one GPU.
//   - GGML_HIP_ROCWMMA_FATTN: not defined by either project's ggml.
var llamaOnlyOptions = []string{
	"GGML_CUDA_FA_ALL_QUANTS",
	"GGML_HIP_RCCL",
	"GGML_CUDA_NO_PEER_COPY",
	"GGML_HIP_ROCWMMA_FATTN",
}

// ProjectOptions returns the toggleable build options for a project and
// profile: for llama.cpp the profile's options; for the other projects the
// ones their bundled ggml uses (no LLAMA_* options and none of
// llamaOnlyOptions), plus each project's extras.
func ProjectOptions(project, profile string) []BuildOption {
	opts := ProfileOptions(profile)
	if project == "" || project == ProjectLlamaCpp {
		return opts
	}
	opts = slices.DeleteFunc(opts, func(o BuildOption) bool {
		return strings.HasPrefix(o.Flag, "LLAMA_") || slices.Contains(llamaOnlyOptions, o.Flag)
	})
	// stable-diffusion.cpp leaves CUDA graphs off (its ggml labels them
	// "llama.cpp only"); llama.cpp and acestep.cpp turn them on.
	if project == ProjectSDCpp {
		for i := range opts {
			if opts[i].Flag == "GGML_CUDA_GRAPHS" {
				opts[i].Default = false
			}
		}
	}
	if project == ProjectSDCpp {
		_, err := exec.LookPath("pnpm")
		opts = append(opts, BuildOption{
			Flag:  sdFrontendFlag,
			Label: "Built-in web UI (needs pnpm)",
			Description: "Build sd-server's own web page for generating images in a browser. Building it needs pnpm on this machine; " +
				"without the web UI, sd-server still serves the image API.",
			Default:  err == nil,
			OffValue: "OFF",
		})
	}
	return opts
}

// sdFrontendFlag is stable-diffusion.cpp's switch for building sd-server's
// web UI with pnpm. It defaults to ON upstream and is skipped silently
// when pnpm is missing, so the build turns it OFF explicitly then.
const sdFrontendFlag = "SD_SERVER_BUILD_FRONTEND"

// MapFlags applies the project's own cmake switches for the profile flags
// in place, exactly as a build does. Exposed for the flag preview.
func (p Project) MapFlags(flags map[string]string) {
	if p.mapFlags != nil {
		p.mapFlags(flags)
	}
}
