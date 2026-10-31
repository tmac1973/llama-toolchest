// Package generation holds the image, video and music generation backends that
// run next to llama-server: their model sets, backend instances and the
// store that persists them.
package generation

import (
	"fmt"
	"time"
)

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
	// KindUpscale also runs sd-server, with no diffusion model: only an
	// ESRGAN upscaler, for /v1/images/upscale. It needs about 1 GiB of
	// VRAM, so it is a set of its own rather than part of an image set.
	KindUpscale BackendKind = "upscale"
)

// Component is one file of a model set.
type Component struct {
	// Slot is the role of the file. For sd it is an sd-server recipe slot
	// such as "diffusion_model", "llm", "tokenizer" or "vae", or a LoRA
	// slot ("lora", "high_noise_lora"; see LoraSlots); for ace it is
	// "lm", "dit", "embedding" or "vae"; for upscale it is "upscaler".
	Slot string `json:"slot"`
	// Repo and File locate the file on Hugging Face. Both are empty for a
	// custom set made from files already on disk.
	Repo string `json:"repo,omitempty"`
	File string `json:"file,omitempty"`
	// Path is the absolute local path once the file is present.
	Path  string `json:"path,omitempty"`
	Bytes int64  `json:"bytes,omitempty"`
}

// VideoSize is a video's length and size: a Measure preset, or the size a
// video instance was measured for.
type VideoSize struct {
	Label  string `json:"label,omitempty"`
	Frames int    `json:"frames"`
	Width  int    `json:"width"`
	Height int    `json:"height"`
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
	// VideoSizes are the sizes Measure offers a video set, smallest (the
	// test size) first. See VideoPresets.
	VideoSizes []VideoSize `json:"video_sizes,omitempty"`
	// StemTasks says whether a custom music set has a base DiT, which can
	// run lego, extract and complete; nil for a catalog set and for custom
	// sets saved before it existed. See HasStemTasks.
	StemTasks *bool `json:"stem_tasks,omitempty"`
	// Complete is true once every component file is present.
	Complete bool      `json:"complete"`
	Created  time.Time `json:"created"`
}

// MeasurePoint is one video measurement: the size, the peak, and the
// budget it ran inside (0 for the whole room). Over says the test was
// stopped once its peak passed the room, so PeakMiB is only a lower bound.
type MeasurePoint struct {
	Frames    int  `json:"frames"`
	Width     int  `json:"width"`
	Height    int  `json:"height"`
	PeakMiB   int  `json:"peak_mib"`
	BudgetMiB int  `json:"budget_mib,omitempty"`
	Over      bool `json:"over,omitempty"`
	// Fallback says the decode only fitted by being done in smaller
	// pieces, which changes how memory grows with the size.
	Fallback bool `json:"fallback,omitempty"`
}

// LastVideo is one finished video job, for the instance row.
type LastVideo struct {
	Frames  int       `json:"frames"`
	Width   int       `json:"width"`
	Height  int       `json:"height"`
	Inputs  string    `json:"inputs"` // "text", "a start image" or "start and end images"
	Seconds int       `json:"seconds"`
	At      time.Time `json:"at"`
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
	// MeasuredSuggestedMiB is the reservation the last measurement
	// suggested, after capping it at what the card could give (sd-server
	// instances only; zero for music). It is stored because that cap
	// depends on what else was on the card at the time.
	MeasuredSuggestedMiB int `json:"measured_suggested_mib,omitempty"`
	// MeasuredFrames, MeasuredWidth and MeasuredHeight are the video size
	// the last measurement made (video instances only). See VideoLimit.
	MeasuredFrames int `json:"measured_frames,omitempty"`
	MeasuredWidth  int `json:"measured_width,omitempty"`
	MeasuredHeight int `json:"measured_height,omitempty"`
	// MeasuredDecodeFallback is true when the measured decode did not fit
	// at once and sd.cpp retried it with temporal tiling: the size fits,
	// but its decode is slower than with more room.
	MeasuredDecodeFallback bool `json:"measured_decode_fallback,omitempty"`
	// MeasuredArgs are the backend flags the last measurement ran with:
	// the set's run arguments, then the instance's extra ones. When they
	// differ from today's (a catalog change), the measurement is out of
	// date. MeasuredRAMMiB is the most system RAM the backend held during
	// it: a text encoder on the CPU, and any weights sd.cpp kept in RAM.
	MeasuredArgs   []string `json:"measured_args,omitempty"`
	MeasuredRAMMiB int      `json:"measured_ram_mib,omitempty"`
	// MeasuredEditImages is how many reference images the test edit of
	// the last measurement had (edit sets only); 0 for measurements made
	// before it was recorded, which all had one.
	MeasuredEditImages int `json:"measured_edit_images,omitempty"`
	// BudgetMiB is the GPU budget (sd-server's --max-vram) a video
	// instance was measured inside, when the user chose one smaller than
	// the room on the card; 0 means the whole room. sd-server runs with it,
	// so it keeps to what was measured: inside a smaller budget sd.cpp
	// keeps weights in system RAM and decodes in smaller pieces.
	BudgetMiB int `json:"budget_mib,omitempty"`
	// MeasurePoints are a video instance's recent measurements, for
	// predicting what other sizes need (see PredictPeak).
	MeasurePoints []MeasurePoint `json:"measure_points,omitempty"`
	// LastVideo is the instance's last finished video job: its size, its
	// inputs and how long sd-server took to make it (video instances
	// only). A measurement cannot say this: the steps of a job differ too
	// much in time (LTX-2.3's first two steps with start and end images
	// took 120-137 s each and the other six 18-22 s; R9700, 2026-10-28).
	LastVideo LastVideo `json:"last_video,omitzero"`
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

// BaseURL is the address the toolchest reaches the instance at, on
// loopback whatever host it listens on.
func (inst Instance) BaseURL() string {
	return fmt.Sprintf("http://127.0.0.1:%d", inst.Port)
}
