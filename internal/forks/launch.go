package forks

import (
	"path/filepath"
	"strconv"
	"strings"
)

// deviceEnvVar is the variable that limits a fork instance to its GPUs,
// by build profile backend.
var deviceEnvVar = map[string]string{
	"rocm": "HIP_VISIBLE_DEVICES",
	"cuda": "CUDA_VISIBLE_DEVICES",
}

// Args returns llama-server's arguments for an instance: where to listen,
// the model files from the set's folder, the fork's own options, then the
// instance's extra arguments, so a later copy of an option wins.
func Args(f Fork, set ModelSet, inst Instance) []string {
	args := []string{
		"--host", "127.0.0.1",
		"--port", strconv.Itoa(inst.Port),
		"-m", filepath.Join(set.Dir, filepath.FromSlash(f.Recipe.ModelFileArg)),
	}
	if f.Recipe.DraftFileArg != "" {
		args = append(args, "-md", filepath.Join(set.Dir, filepath.FromSlash(f.Recipe.DraftFileArg)))
	}
	args = append(args, f.Recipe.Args...)
	return append(args, inst.ExtraArgs...)
}

// Env returns the NAME=value lines for an instance: the GPU selector for
// the backend, set to devices (the instance's GPUs as the backend numbers
// them, the first one first), then the fork's lines, then the instance's.
// A later line for the same name replaces an earlier one.
func Env(f Fork, inst Instance, backend string, devices []int) []string {
	var lines []string
	if v := deviceEnvVar[backend]; v != "" && len(devices) > 0 {
		ids := make([]string, len(devices))
		for i, d := range devices {
			ids[i] = strconv.Itoa(d)
		}
		lines = append(lines, v+"="+strings.Join(ids, ","))
	}
	lines = append(lines, f.Recipe.Env...)
	lines = append(lines, inst.ExtraEnv...)

	var out []string
	at := map[string]int{}
	for _, line := range lines {
		name, _, _ := strings.Cut(line, "=")
		if i, ok := at[name]; ok {
			out[i] = line
			continue
		}
		at[name] = len(out)
		out = append(out, line)
	}
	return out
}
