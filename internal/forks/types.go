// Package forks holds the curated llama.cpp forks: what each one is, how
// it is built and launched, and the model files it is made for. A fork
// runs as a fork instance, a llama-server of its own on one whole GPU,
// next to the router; it is never the router's build, because the router
// writes launch options by upstream build number, which a fork's history
// does not follow.
//
// The builder imports this package for the fork projects, so it must not
// import the builder, nor internal/generation (which does).
package forks

import "strconv"

// Fork is one curated fork.
type Fork struct {
	// ID is a lowercase slug ("avalon"). It is also the builder project
	// ID and the build ID prefix ("avalon-").
	ID   string
	Name string
	// Repo is built from its default branch: the forks move too fast
	// to pin a commit.
	Repo string
	// Backends lists the build profile backends the fork supports.
	Backends []string
	// MaxGPUs is how many GPUs one instance can use; 0 means 1. The fork
	// itself decides how to use them: it sees only the instance's GPUs.
	MaxGPUs int
	Recipe  Recipe
	// ModelRepo is the Hugging Face repo of the paired model, ModelName
	// its display name, and ModelFiles every file a fork instance needs.
	// ModelRevision is the repo commit the files are downloaded from: the
	// sizes in ModelFiles are that commit's, so a later upload cannot
	// change the set. Move it with ModelFiles, never alone.
	ModelRepo     string
	ModelRevision string
	ModelName     string
	ModelFiles    []ModelFile
	License       string
	// Notes is one or two sentences on what the fork is for, shown as a
	// tooltip.
	Notes string
	Needs Needs
	// SamplingDefaults are added to a chat request that does not set
	// them, as the router's sampling defaults are.
	SamplingDefaults map[string]any
}

// GPULimit is how many GPUs one instance of f can use.
func (f Fork) GPULimit() int {
	return max(1, f.MaxGPUs)
}

// Recipe is how a fork instance is launched.
type Recipe struct {
	// Args are the fixed llama-server options.
	Args []string
	// ModelFileArg is the model file passed as -m, and DraftFileArg the
	// draft model passed as -md ("" for none); both are paths in
	// ModelFiles.
	ModelFileArg string
	DraftFileArg string
	// Env is NAME=value lines set for every instance of the fork.
	Env []string
}

// Needs is what the host should have for the fork to run well. Falling
// short gives a warning, not a refusal.
type Needs struct {
	MinRAMGiB int
	AVX2      bool
	// ROCmArchs are the AMD GPU architectures the fork is tuned for; CUDA
	// reports whether NVIDIA GPUs are supported.
	ROCmArchs  []string
	CUDA       bool
	MinDiskGiB int
}

// ModelFile is one file of a fork's model set. Path may name a
// subfolder ("MTP/draft.gguf").
type ModelFile struct {
	Path  string
	Bytes int64
}

// ModelSet is a fork's downloaded model, one per fork.
type ModelSet struct {
	ForkID string `json:"fork_id"`
	// Dir holds the set's files, laid out as in ModelFiles.
	Dir      string `json:"dir"`
	Complete bool   `json:"complete"`
}

// Instance is one fork instance: a fork's llama-server on one whole GPU,
// or on more when the fork allows it.
type Instance struct {
	// ID is the instance's name and the model name clients send.
	ID     string `json:"id"`
	ForkID string `json:"fork_id"`
	// BuildID is the fork build to run; "" means the newest successful
	// build of the fork.
	BuildID string `json:"build_id,omitempty"`
	// GPU is the instance's first GPU, and ExtraGPUs the others, up to
	// the fork's GPULimit. The instance holds each one whole.
	GPU       int      `json:"gpu"`
	ExtraGPUs []int    `json:"extra_gpus,omitempty"`
	Port      int      `json:"port"`
	ExtraArgs []string `json:"extra_args,omitempty"`
	// ExtraEnv is NAME=value lines for this instance only.
	ExtraEnv  []string `json:"extra_env,omitempty"`
	AutoStart bool     `json:"auto_start,omitempty"`
}

// GPUs returns every GPU of the instance, the first one first.
func (inst Instance) GPUs() []int {
	return append([]int{inst.GPU}, inst.ExtraGPUs...)
}

// BaseURL is the address the toolchest reaches the instance at, on
// loopback whatever host it listens on.
func (inst Instance) BaseURL() string {
	return "http://127.0.0.1:" + strconv.Itoa(inst.Port)
}
