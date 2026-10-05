package libpath

import (
	"os"
	"slices"
	"testing"
)

// The router used to get a second LD_LIBRARY_PATH appended, which replaced
// the container's value (on compute2 it lost /usr/local/cuda/lib64). The
// build's directory must go first and the existing value must stay.
func TestPrependKeepsTheExistingValue(t *testing.T) {
	env := []string{"HOME=/root", "LD_LIBRARY_PATH=/usr/local/cuda/lib64"}
	got := prependVar(env, "LD_LIBRARY_PATH", "/data/builds/b1")
	want := []string{"HOME=/root", "LD_LIBRARY_PATH=/data/builds/b1" + string(os.PathListSeparator) + "/usr/local/cuda/lib64"}
	if !slices.Equal(got, want) {
		t.Errorf("got %q, want %q", got, want)
	}
}

func TestPrependAddsTheVariableWhenMissing(t *testing.T) {
	got := prependVar([]string{"HOME=/root"}, "LD_LIBRARY_PATH", "/data/builds/b1")
	if !slices.Equal(got, []string{"HOME=/root", "LD_LIBRARY_PATH=/data/builds/b1"}) {
		t.Errorf("got %q", got)
	}
}

func TestPrependEmptyValueHasNoTrailingSeparator(t *testing.T) {
	got := prependVar([]string{"LD_LIBRARY_PATH="}, "LD_LIBRARY_PATH", "/b")
	if got[0] != "LD_LIBRARY_PATH=/b" {
		t.Errorf("got %q", got[0])
	}
}

func TestPrependMatchesWindowsPathCaseInsensitively(t *testing.T) {
	got := prependVar([]string{`Path=C:\Windows`}, "PATH", `C:\b`)
	if len(got) != 1 || got[0] != `Path=C:\b`+string(os.PathListSeparator)+`C:\Windows` {
		t.Errorf("got %q", got)
	}
}
