package models

// Target is the llama-server build that launch options are written for.
//
// Most options mean the same thing in every build, but a few have been
// renamed or removed upstream, and llama-server does not skip an option it
// does not know. The router reads every model's section of the preset
// file when it starts and stops at the first unknown option, so one model
// with an option its build lacks leaves every model unable to load. The
// options that depend on the build are written only in the spelling that
// build accepts.
type Target struct {
	// Backend is the build's backend ("cuda", "rocm", ...). It selects
	// the device names in a GPU restriction; "" writes the padded
	// tensor split instead.
	Backend string
	// Version is the build's place in llama.cpp's history: the master
	// commit count, which is the N of its bN release tag. 0 means
	// unknown, and is written for as the newest llama.cpp, since a build
	// made today is the likeliest one to be running.
	Version int
	// GPUMiB is each GPU's total memory, by index. It lets a config that
	// keeps experts in system memory be given a split balanced by size
	// (MoESplitConfig); without it llama.cpp's own split by count is used.
	GPUMiB []int
}

// Where the options below changed upstream, as the first release tag
// with each change. Checked against llama.cpp's own history.
const (
	// versionNoDraftContext is b9109 (#22838, parallel drafting), which
	// removed --ctx-size-draft. A draft context has had the size of the
	// model's own context since.
	versionNoDraftContext = 9109
	// versionLazyRead is b10653 (#27794), which added
	// --tensor-read-lazy for on-demand reading of per-layer embeddings.
	versionLazyRead = 10653
	// versionLazyMode is b10700 (#27969), which renamed
	// --tensor-read-lazy to --lazy-mode and kept no alias.
	versionLazyMode = 10700
)

// newest reports whether the target is to be written for as current
// llama.cpp: a build at or past v, or one whose version is unknown.
func (t Target) newest(v int) bool { return t.Version == 0 || t.Version >= v }

// draftContextOption reports whether the build takes --ctx-size-draft.
// Only builds known to predate its removal do; an unknown build is
// taken to be current.
func (t Target) draftContextOption() bool {
	return t.Version > 0 && t.Version < versionNoDraftContext
}

// lazyReadOption returns the name of the option that sets on-demand
// reading of per-layer embeddings, or "" when the build has none.
func (t Target) lazyReadOption() string {
	switch {
	case t.newest(versionLazyMode):
		return "lazy-mode"
	case t.Version >= versionLazyRead:
		return "tensor-read-lazy"
	default:
		return ""
	}
}
