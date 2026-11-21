package builder

import (
	"fmt"
	"os/exec"
	"runtime"
	"strings"
)

// buildTools are the programs every build runs. A missing one otherwise
// shows up late and unclear: without ninja, cmake stops with "unable to
// find a build program corresponding to Ninja".
var buildTools = []string{"git", "cmake", "ninja"}

// lookPath finds a program on PATH; a var so tests can replace it.
var lookPath = exec.LookPath

// CheckBuildTools reports the build tools missing from PATH, with how to
// install them on this system, or nil when all are there.
func CheckBuildTools() error {
	return missingToolsError(missingBuildTools(), runtime.GOOS)
}

func missingBuildTools() []string {
	var missing []string
	for _, t := range buildTools {
		if _, err := lookPath(t); err != nil {
			missing = append(missing, t)
		}
	}
	return missing
}

func missingToolsError(missing []string, goos string) error {
	if len(missing) == 0 {
		return nil
	}
	msg := fmt.Sprintf("cannot build: %s not found on PATH", strings.Join(missing, ", "))
	switch goos {
	case "darwin":
		var brew []string
		for _, t := range missing {
			if t != "git" {
				brew = append(brew, t)
			}
		}
		msg += ". Install the Xcode command-line tools (xcode-select --install)"
		if len(brew) > 0 {
			msg += " and run: brew install " + strings.Join(brew, " ")
		}
		msg += ". Homebrew puts them in /opt/homebrew/bin, which must be on the PATH llama-toolchest starts with"
	default:
		msg += ". Install them with your package manager (the ninja package is often called ninja-build) and restart llama-toolchest"
	}
	return fmt.Errorf("%s", msg)
}
