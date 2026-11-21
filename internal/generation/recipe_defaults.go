package generation

import (
	"fmt"
	"log/slog"
	"slices"
	"strconv"
	"strings"
	"sync"
)

// sd-server applies a recipe's defaults and context only when it loads
// the recipe itself. Toolchest passes the files by flag, so it turns them
// into sd-server flags for the custom set form to fill in.

// recipeFlag maps one recipe key to the sd-server flag that sets it.
type recipeFlag struct {
	key, flag string
	// context is true for a key of the recipe's context, false for one of
	// its defaults.
	context bool
}

// recipeFlags are the recipe keys that become launch flags, in the order
// of the flags in LaunchDefaults.
var recipeFlags = []recipeFlag{
	{key: "steps", flag: "--steps"},
	{key: "cfg_scale", flag: "--cfg-scale"},
	{key: "img_cfg_scale", flag: "--img-cfg-scale"},
	{key: "distilled_guidance", flag: "--guidance"},
	{key: "sampling_method", flag: "--sampling-method"},
	{key: "scheduler", flag: "--scheduler"},
	{key: "flow_shift", flag: "--flow-shift"},
	{key: "shifted_timestep", flag: "--timestep-shift"},
	{key: "custom_sigmas", flag: "--sigmas"},
	{key: "strength", flag: "--strength"},
	{key: "high_noise_steps", flag: "--high-noise-steps"},
	{key: "high_noise_cfg_scale", flag: "--high-noise-cfg-scale"},
	{key: "high_noise_sampling_method", flag: "--high-noise-sampling-method"},
	{key: "moe_boundary", flag: "--moe-boundary"},
	{key: "vae_format", flag: "--vae-format", context: true},
	{key: "model_args", flag: "--model-args", context: true},
}

// recipeSizeKeys are the defaults that fill the form's test-size fields,
// not launch flags.
var recipeSizeKeys = []string{"width", "height", "video_frames", "fps"}

// skippedDefaults are recipe defaults toolchest leaves out on purpose,
// with the reason.
var skippedDefaults = map[string]string{
	// The Wan2.2 A14B recipes give the high-noise expert its own shift.
	// sd-server has no flag for it; --flow-shift sets both experts.
	"high_noise_flow_shift": "sd-server has no flag for it; --flow-shift covers both experts",
}

// loggedRecipeKeys are the unknown recipe keys already logged, so each is
// logged once.
var loggedRecipeKeys sync.Map

// logUnknownRecipeKeys logs, once per key, a defaults or context key that
// toolchest neither turns into a flag nor skips on purpose. It is left
// out of the form; TestRecipeKeysAreKnown fails on it at the next
// snapshot refresh.
func logUnknownRecipeKeys(r *Recipe) {
	for _, k := range unknownRecipeKeys(r) {
		if _, seen := loggedRecipeKeys.LoadOrStore(k, true); !seen {
			slog.Debug("recipe setting not used", "recipe", r.ID, "key", k)
		}
	}
}

// unknownRecipeKeys are the recipe's defaults and context keys that are
// not in recipeFlags, recipeSizeKeys or skippedDefaults, A–Z.
func unknownRecipeKeys(r *Recipe) []string {
	known := func(key string, context bool) bool {
		for _, f := range recipeFlags {
			if f.key == key && f.context == context {
				return true
			}
		}
		if context {
			return false
		}
		return slices.Contains(recipeSizeKeys, key) || skippedDefaults[key] != ""
	}
	var out []string
	for k := range r.rawDefaults {
		if !known(k, false) {
			out = append(out, k)
		}
	}
	for k := range r.rawContext {
		if !known(k, true) {
			out = append(out, "context."+k)
		}
	}
	slices.Sort(out)
	return out
}

// formatRecipeValue writes a recipe value as one flag value: a number
// without a needless decimal point, a string as it is, a list
// comma-separated.
func formatRecipeValue(v any) string {
	switch v := v.(type) {
	case float64:
		return strconv.FormatFloat(v, 'f', -1, 64)
	case string:
		return v
	case []any:
		parts := make([]string, len(v))
		for i, e := range v {
			parts[i] = formatRecipeValue(e)
		}
		return strings.Join(parts, ",")
	default:
		return fmt.Sprint(v)
	}
}

// LaunchDefaults are the recipe's defaults and context as sd-server
// flags, for example ["--steps", "20", "--cfg-scale", "1"].
func (r *Recipe) LaunchDefaults() []string {
	var args []string
	for _, f := range recipeFlags {
		src := r.rawDefaults
		if f.context {
			src = r.rawContext
		}
		if v, ok := src[f.key]; ok && v != nil {
			args = append(args, f.flag, formatRecipeValue(v))
		}
	}
	return args
}

// SizeDefaults are the recipe's default size, and for a video recipe its
// frames and frame rate; each is empty when the recipe gives none.
func (r *Recipe) SizeDefaults() (width, height, frames, fps string) {
	get := func(key string) string {
		if v, ok := r.rawDefaults[key]; ok && v != nil {
			return formatRecipeValue(v)
		}
		return ""
	}
	width, height = get("width"), get("height")
	if r.Kind == KindVideo {
		frames, fps = get("video_frames"), get("fps")
	}
	return width, height, frames, fps
}
