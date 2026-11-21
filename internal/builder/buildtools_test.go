package builder

import (
	"errors"
	"slices"
	"strings"
	"testing"
)

func TestMissingBuildTools(t *testing.T) {
	old := lookPath
	defer func() { lookPath = old }()
	lookPath = func(name string) (string, error) {
		if name == "ninja" {
			return "", errors.New("not found")
		}
		return "/usr/bin/" + name, nil
	}
	if got := missingBuildTools(); !slices.Equal(got, []string{"ninja"}) {
		t.Errorf("missing = %v, want [ninja]", got)
	}
}

func TestMissingToolsError(t *testing.T) {
	if err := missingToolsError(nil, "darwin"); err != nil {
		t.Errorf("nothing missing gave %v", err)
	}
	err := missingToolsError([]string{"git", "cmake", "ninja"}, "darwin")
	if err == nil || !strings.Contains(err.Error(), "brew install cmake ninja") || !strings.Contains(err.Error(), "xcode-select --install") {
		t.Errorf("darwin message: %v", err)
	}
	err = missingToolsError([]string{"ninja"}, "linux")
	if err == nil || !strings.Contains(err.Error(), "ninja not found") || strings.Contains(err.Error(), "brew") {
		t.Errorf("linux message: %v", err)
	}
}
