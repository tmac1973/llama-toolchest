// Package generation holds the image, video and music generation backends that
// run next to llama-server: their model sets, backend instances and the
// store that persists them.
package generation

import "time"

// BackendKind names a generation backend program.
type BackendKind string

const (
	// KindSD is stable-diffusion.cpp's sd-server (images).
	KindSD BackendKind = "sd"
	// KindVideo also runs sd-server, with a video model. It is a kind of
	// its own so image jobs never pick a video set and video jobs never
	// pick an image set.
	KindVideo BackendKind = "video"
	// KindAce is acestep.cpp's ace-server (music).
	KindAce BackendKind = "ace"
)

// Component is one file of a model set.
type Component struct {
	// Slot is the role of the file. For sd it is an sd-server recipe slot
	// such as "diffusion_model", "llm", "tokenizer" or "vae"; for ace it
	// is "lm", "dit", "embedding" or "vae".
	Slot string `json:"slot"`
	// Repo and File locate the file on Hugging Face. Both are empty for a
	// custom set made from files already on disk.
	Repo string `json:"repo,omitempty"`
	File string `json:"file,omitempty"`
	// Path is the absolute local path once the file is present.
	Path  string `json:"path,omitempty"`
	Bytes int64  `json:"bytes,omitempty"`
}

// ModelSet is a named group of component files that a backend loads
// together, e.g. a diffusion model, its text encoder and its VAE.
type ModelSet struct {
	// ID is the catalog ID for a catalog set, or the slug of Name for a
	// custom set. It is unique.
	ID   string      `json:"id"`
	Name string      `json:"name"`
	Kind BackendKind `json:"kind"`
	// CatalogID is empty for a custom set.
	CatalogID string `json:"catalog_id,omitempty"`
	// Recipe is the sd-server recipe ID, e.g. "ming-image". Empty for ace.
	Recipe     string      `json:"recipe,omitempty"`
	Components []Component `json:"components"`
	// LaunchArgs are extra backend flags, e.g. "--sigmas qwen-image-2.1-turbo".
	LaunchArgs []string `json:"launch_args,omitempty"`
	// Capabilities lists extra abilities, e.g. "alpha" or "edit".
	Capabilities []string `json:"capabilities,omitempty"`
	License      string   `json:"license,omitempty"`
	// DefaultWidth and DefaultHeight are the image or video size the
	// measured test run uses. Zero for ace.
	DefaultWidth  int `json:"default_width,omitempty"`
	DefaultHeight int `json:"default_height,omitempty"`
	// DefaultFrames and DefaultFPS are the test video's length and frame
	// rate, and what a video request that gives none gets. Video sets only.
	DefaultFrames int `json:"default_frames,omitempty"`
	DefaultFPS    int `json:"default_fps,omitempty"`
	// MaxFrames, MaxWidth and MaxHeight are the largest video a request
	// may ask for; zero means no limit. Video sets only.
	MaxFrames int `json:"max_frames,omitempty"`
	MaxWidth  int `json:"max_width,omitempty"`
	MaxHeight int `json:"max_height,omitempty"`
	// Complete is true once every component file is present.
	Complete bool      `json:"complete"`
	Created  time.Time `json:"created"`
}

// Instance is one backend process: a model set, run by one build, on one
// GPU and one port.
type Instance struct {
	// ID is the slug of Name and is unique. API clients use it as the
	// model name.
	ID         string `json:"id"`
	Name       string `json:"name"`
	ModelSetID string `json:"model_set_id"`
	// BuildID is empty for "the newest successful build of the backend".
	BuildID string `json:"build_id,omitempty"`
	// GPU is the monitor's GPU index.
	GPU int `json:"gpu"`
	// Port is 0 until one is assigned from the generation port range.
	Port int `json:"port,omitempty"`
	// ReservedMiB is the VRAM held for this instance on its GPU; 0 means
	// it has not been measured yet.
	ReservedMiB     int       `json:"reserved_mib,omitempty"`
	MeasuredPeakMiB int       `json:"measured_peak_mib,omitempty"`
	MeasuredAt      time.Time `json:"measured_at,omitempty"`
	// MeasureEstimated is true when the peak came from whole-GPU use,
	// because the GPU tools gave no per-process figure (e.g. Vulkan).
	MeasureEstimated bool `json:"measure_estimated,omitempty"`
	// ExtraArgs are backend flags for this instance, added after the
	// set's LaunchArgs.
	ExtraArgs []string `json:"extra_args,omitempty"`
	// Enabled instances hold their reservation even while stopped, so
	// starting one later never fails for lack of VRAM. A disabled
	// instance cannot run and holds nothing.
	Enabled bool `json:"enabled"`
	// AutoStart starts the instance with toolchest. It is separate from
	// the router's own auto-start setting.
	AutoStart bool `json:"auto_start,omitempty"`
	// KeepRunning exempts the instance from eviction: LLM loads and other
	// backends' jobs never stop it to make room, and its reservation
	// counts even while it is stopped, as it does with eviction off.
	KeepRunning bool `json:"keep_running,omitempty"`
}
