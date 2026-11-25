package generation

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/builder"
	"github.com/tmac1973/llama-toolchest/internal/process"
)

// highNoiseFlags are the sd-server flags for the high-noise expert of a
// Wan2.2 A14B pair that stable-diffusion.cpp removed in its Diffusion
// Graph Language rewrite (master-1116, commit 6cd0ef1), each with the
// flag whose value the high-noise expert uses from then on. Older builds
// need them: there the high-noise expert has its own defaults (CFG 7.0),
// not those of the low-noise one. Newer builds stop at start with
// "unknown argument", so LaunchSpec leaves them out for a build whose
// --help does not list them. Each takes one value.
// --high-noise-steps and --high-noise-diffusion-model stayed.
var highNoiseFlags = map[string]string{
	"--high-noise-cfg-scale":        "--cfg-scale",
	"--high-noise-img-cfg-scale":    "--img-cfg-scale",
	"--high-noise-guidance":         "--guidance",
	"--high-noise-slg-scale":        "--slg-scale",
	"--high-noise-skip-layer-start": "--skip-layer-start",
	"--high-noise-skip-layer-end":   "--skip-layer-end",
	"--high-noise-skip-layers":      "--skip-layers",
	"--high-noise-eta":              "--eta",
	"--high-noise-sampling-method":  "--sampling-method",
}

// droppedFlag is a flag adaptArgs left out, with its value.
type droppedFlag struct{ flag, value string }

// adaptArgs returns args without the highNoiseFlags (and their values)
// that help, the build's --help text, does not list. With an empty help
// (the build could not be asked) args come back as they are.
func adaptArgs(args []string, help string) ([]string, []droppedFlag) {
	if help == "" {
		return args, nil
	}
	var out []string
	var dropped []droppedFlag
	for i := 0; i < len(args); i++ {
		flag, value, inline := strings.Cut(args[i], "=")
		if _, ok := highNoiseFlags[flag]; !ok || helpLists(help, flag) {
			out = append(out, args[i])
			continue
		}
		if !inline && i+1 < len(args) {
			i++
			value = args[i]
		}
		dropped = append(dropped, droppedFlag{flag, value})
	}
	return out, dropped
}

// helpLists says whether a --help text lists flag as an option.
func helpLists(help, flag string) bool {
	return regexp.MustCompile(`(^|[\s,])` + regexp.QuoteMeta(flag) + `([\s,=]|$)`).MatchString(help)
}

// argValue returns the value of the last flag in args, and whether it is
// there.
func argValue(args []string, flag string) (string, bool) {
	value, found := "", false
	for i, a := range args {
		if f, v, inline := strings.Cut(a, "="); f == flag {
			found = true
			switch {
			case inline:
				value = v
			case i+1 < len(args):
				value = args[i+1]
			}
		}
	}
	return value, found
}

// adaptArgsForBuild leaves out the high-noise flags the build does not
// have, and logs each one. It warns when the dropped value differs from
// the value the high-noise expert now uses.
func adaptArgsForBuild(setID string, build builder.BuildResult, args []string) []string {
	out, dropped := adaptArgs(args, buildHelp(build))
	for _, d := range dropped {
		main := highNoiseFlags[d.flag]
		mainValue, ok := argValue(out, main)
		if ok && mainValue == d.value {
			slog.Info("leaving out a flag this sd-server build does not have; the high-noise expert uses the same value from "+main,
				"set", setID, "build", build.ID, "flag", d.flag, "value", d.value)
			continue
		}
		if !ok {
			mainValue = "sd-server's default"
		}
		slog.Warn("leaving out a flag this sd-server build does not have; the high-noise expert uses the "+main+" value instead",
			"set", setID, "build", build.ID, "flag", d.flag, "value", d.value, "used", mainValue)
	}
	return out
}

// BuildHasDGL reports whether the build's sd-server has DGL (it lists
// --dgl-dir), which video upscaler sets need. When --help cannot be read,
// it answers true and logs a warning: the instance then tries to start,
// and sd.cpp's own error shows if DGL is missing, as adaptArgsForBuild
// does for its flags.
func BuildHasDGL(build builder.BuildResult) bool {
	help := buildHelp(build)
	if help == "" {
		slog.Warn("could not read the sd-server build's --help; assuming it has DGL", "build", build.ID)
		return true
	}
	return helpLists(help, "--dgl-dir")
}

// helpCache keeps each sd-server binary's --help text, by path, size and
// modification time, so a rebuild in place is asked again.
var helpCache sync.Map // helpKey -> string

type helpKey struct {
	path    string
	size    int64
	modTime time.Time
}

// helpTimeout limits one --help run. sd-server prints its help before it
// loads a GPU backend; it took 0.15 s on a ROCm build.
const helpTimeout = 10 * time.Second

// buildHelp returns the --help text of the build's binary, or "" when it
// cannot be read. A failed run is not kept, so the next launch tries
// again.
func buildHelp(build builder.BuildResult) string {
	st, err := os.Stat(build.BinaryPath)
	if err != nil {
		return ""
	}
	key := helpKey{build.BinaryPath, st.Size(), st.ModTime()}
	if h, ok := helpCache.Load(key); ok {
		return h.(string)
	}
	help, err := runHelp(build.BinaryPath)
	if err != nil {
		slog.Warn("could not read the sd-server build's --help; starting with the set's flags as they are", "build", build.ID, "error", err)
		return ""
	}
	helpCache.Store(key, help)
	return help
}

// runHelp runs "binary --help" and returns its output. It is "" unless
// the text looks like sd-server's options list (it has --steps).
func runHelp(binary string) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), helpTimeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, binary, "--help")
	cmd.Env = process.BuildEnv(filepath.Dir(binary), nil)
	out, err := cmd.CombinedOutput()
	help := string(out)
	if !helpLists(help, "--steps") {
		if err == nil {
			err = fmt.Errorf("the output has no --steps option")
		}
		return "", err
	}
	// sd-server may exit non-zero after it prints the help; the text is
	// what counts.
	return help, nil
}
