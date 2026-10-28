package builder

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/testutil"
)

// covNinjaInstalls is a fake ninja that lays out a llama.cpp build: the
// server and llama-bench in bin/ (the server without its execute bit, so
// the install must set it), shared libraries in lib/ and beside the
// binary, and a file in lib/ that is not a library.
const covNinjaInstalls = `#!/bin/sh
echo "[1/2] Building CXX object"
mkdir -p bin lib
printf 'server' > bin/llama-server
chmod 644 bin/llama-server
printf 'bench' > bin/llama-bench
chmod 755 bin/llama-bench
printf 'lib' > lib/libggml-base.so
printf 'lib' > lib/libllama.so.0
printf 'lib' > bin/libmtmd.so
printf 'text' > lib/notes.txt
echo "[2/2] Linking CXX executable bin/llama-server"
`

// covBuildSetup prepares a data directory with a llama.cpp git fixture
// (see gitFixture), puts fake cmake and ninja first on PATH, and returns
// the data directory, the clone and the file the fake cmake writes its
// arguments to. cmakeScript and ninjaScript are the bodies of the two
// tools; "" uses a cmake that records its arguments and covNinjaInstalls.
func covBuildSetup(t *testing.T, cmakeScript, ninjaScript string) (dataDir, srcDir, argsFile string) {
	t.Helper()
	dataDir, srcDir = gitFixture(t)
	bin := t.TempDir()
	argsFile = filepath.Join(t.TempDir(), "cmake-args")
	if cmakeScript == "" {
		cmakeScript = "#!/bin/sh\nprintf '%s\\n' \"$@\" > '" + argsFile + "'\necho '-- Configuring done'\n"
	}
	if ninjaScript == "" {
		ninjaScript = covNinjaInstalls
	}
	for name, body := range map[string]string{"cmake": cmakeScript, "ninja": ninjaScript} {
		if err := os.WriteFile(filepath.Join(bin, name), []byte(body), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	t.Setenv("PATH", bin+string(os.PathListSeparator)+os.Getenv("PATH"))
	return dataDir, srcDir, argsFile
}

// covWaitForBuild waits until the build has finished and its last log
// line is out: the final line is sent after the status changes, so
// waiting on the status alone would race with it.
func covWaitForBuild(t *testing.T, b *Builder, id string) BuildResult {
	t.Helper()
	testutil.EventuallyMsg(t, func() bool {
		if b.BuildStatus(id) == BuildStatusBuilding {
			return false
		}
		for _, line := range covBuildLog(b, id) {
			if strings.Contains(line, "==> Build complete") || strings.Contains(line, "FAILED") {
				return true
			}
		}
		return false
	}, func() string {
		return "build " + id + " did not finish; log: " + strings.Join(covBuildLog(b, id), " | ")
	})
	r, ok := b.Find(id)
	if !ok {
		t.Fatalf("build %s not found after it finished", id)
	}
	return *r
}

// covBuildLog is the log history of a build.
func covBuildLog(b *Builder, id string) []string {
	bc := b.logBcast(id)
	if bc == nil {
		return nil
	}
	return bc.History()
}

// A successful build installs the server with its execute bit, the
// shared libraries from lib/ and from the binary's directory, and the
// optional programs that exist, then removes the build directory and
// saves the record. A missing optional program (llama-perplexity here)
// must not fail the build. The cmake call gets the project's base
// arguments, the profile flags and the user's extra flags.
func TestBuildRunsCMakeAndNinjaAndInstalls(t *testing.T) {
	dataDir, srcDir, argsFile := covBuildSetup(t, "", "")
	arch, err := os.ReadFile("testdata/llama-arch.cpp")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(srcDir, "src"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(srcDir, archSourceFile), arch, 0o644); err != nil {
		t.Fatal(err)
	}

	b := NewBuilder(dataDir)
	res, err := b.Build(context.Background(), "cpu", "b2", "", false, nil, "-DGGML_FOO=BAR NOT_A_FLAG", "")
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	if res.ID != "b2-cpu" {
		t.Fatalf("build ID = %q, want b2-cpu", res.ID)
	}
	got := covWaitForBuild(t, b, res.ID)

	if got.Status != BuildStatusSuccess || got.Error != "" {
		t.Fatalf("build = %+v; log: %v", got, covBuildLog(b, res.ID))
	}
	outDir := filepath.Join(dataDir, "builds", res.ID)
	if got.BinaryPath != filepath.Join(outDir, "llama-server") {
		t.Errorf("BinaryPath = %q", got.BinaryPath)
	}
	if st, err := os.Stat(got.BinaryPath); err != nil {
		t.Errorf("installed server: %v", err)
	} else if st.Mode().Perm() != 0o755 {
		t.Errorf("installed server mode = %v, want 0755", st.Mode().Perm())
	}
	for _, name := range []string{"llama-bench", "libggml-base.so", "libllama.so.0", "libmtmd.so"} {
		if _, err := os.Stat(filepath.Join(outDir, name)); err != nil {
			t.Errorf("%s not installed: %v", name, err)
		}
	}
	for _, name := range []string{"notes.txt", "llama-perplexity"} {
		if _, err := os.Stat(filepath.Join(outDir, name)); err == nil {
			t.Errorf("%s was installed, want it left out", name)
		}
	}
	if _, err := os.Stat(filepath.Join(srcDir, "build-cpu")); !os.IsNotExist(err) {
		t.Errorf("build directory still exists: %v", err)
	}
	if got.GitRef != "b2" || got.GitSHA == "" || got.CommitCount != 2 {
		t.Errorf("ref/sha/count = %q/%q/%d, want b2, a SHA and 2", got.GitRef, got.GitSHA, got.CommitCount)
	}
	if got.FinishedAt.IsZero() || got.FinishedAt.Before(got.StartedAt) {
		t.Errorf("StartedAt %v, FinishedAt %v", got.StartedAt, got.FinishedAt)
	}
	if !slices.Contains(got.Archs, "llama") {
		t.Errorf("Archs = %v, want the names read from the checkout", got.Archs)
	}
	if got.CMakeFlags["GGML_FOO"] != "BAR" || got.CMakeFlags["CMAKE_BUILD_TYPE"] != "Release" {
		t.Errorf("CMakeFlags = %v", got.CMakeFlags)
	}

	raw, err := os.ReadFile(argsFile)
	if err != nil {
		t.Fatalf("cmake did not run: %v", err)
	}
	args := strings.Split(strings.TrimSpace(string(raw)), "\n")
	if !slices.Equal(args[:3], []string{"..", "-G", "Ninja"}) {
		t.Errorf("cmake args start %v", args[:3])
	}
	for _, want := range []string{"-DLLAMA_BUILD_SERVER=ON", "-DLLAMA_BUILD_TESTS=OFF", "-DCMAKE_BUILD_TYPE=Release", "-DGGML_FOO=BAR"} {
		if !slices.Contains(args, want) {
			t.Errorf("cmake args %v lack %s", args, want)
		}
	}
	for _, a := range args {
		if strings.Contains(a, "NOT_A_FLAG") {
			t.Errorf("an extra word without '=' reached cmake: %v", args)
		}
	}

	log := strings.Join(covBuildLog(b, res.ID), "\n")
	for _, want := range []string{"==> Checking out b2", "==> Running cmake", "-- Configuring done", "[2/2] Linking", "Installed: llama-bench", "Installed lib: libmtmd.so", "==> Build complete"} {
		if !strings.Contains(log, want) {
			t.Errorf("log lacks %q:\n%s", want, log)
		}
	}

	// The record survives a restart.
	if r, ok := NewBuilder(dataDir).Find(res.ID); !ok || r.Status != BuildStatusSuccess || r.BinaryPath != got.BinaryPath {
		t.Errorf("reloaded record = %+v, %v", r, ok)
	}
}

// Building the same ref and flags again is refused unless forced; a
// forced build replaces the old record and deletes the old files. The
// same ref with other flags gets its own ID instead of a refusal.
func TestBuildDuplicateForceAndFlagSuffix(t *testing.T) {
	dataDir, _, _ := covBuildSetup(t, "", "")
	b := NewBuilder(dataDir)
	first, err := b.Build(context.Background(), "cpu", "b2", "", false, nil, "", "")
	if err != nil {
		t.Fatal(err)
	}
	covWaitForBuild(t, b, first.ID)

	_, err = b.Build(context.Background(), "cpu", "b2", "", false, nil, "", "")
	var dup *DuplicateBuildError
	if !errors.As(err, &dup) || dup.ID != first.ID {
		t.Fatalf("second build: %v, want a DuplicateBuildError for %s", err, first.ID)
	}

	marker := filepath.Join(dataDir, "builds", first.ID, "old-file")
	if err := os.WriteFile(marker, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	forced, err := b.Build(context.Background(), "cpu", "b2", "", true, nil, "", "")
	if err != nil {
		t.Fatalf("forced build: %v", err)
	}
	if forced.ID != first.ID {
		t.Fatalf("forced build ID = %q, want %q", forced.ID, first.ID)
	}
	if r := covWaitForBuild(t, b, forced.ID); r.Status != BuildStatusSuccess {
		t.Fatalf("forced build = %+v", r)
	}
	if _, err := os.Stat(marker); !os.IsNotExist(err) {
		t.Errorf("old build files survived a forced rebuild: %v", err)
	}
	n := 0
	for _, r := range b.ListAll() {
		if r.ID == first.ID {
			n++
		}
	}
	if n != 1 {
		t.Errorf("%d records with ID %s, want 1", n, first.ID)
	}

	other, err := b.Build(context.Background(), "cpu", "b2", "", false, nil, "-DGGML_FOO=BAZ", "")
	if err != nil {
		t.Fatalf("build with other flags: %v", err)
	}
	if !strings.HasPrefix(other.ID, first.ID+"-") || len(other.ID) != len(first.ID)+7 {
		t.Errorf("build with other flags has ID %q, want %s-<6 hex>", other.ID, first.ID)
	}
	covWaitForBuild(t, b, other.ID)

	tagged, err := b.Build(context.Background(), "cpu", "b2", " Fast-1 ", false, nil, "", "")
	if err != nil {
		t.Fatalf("tagged build: %v", err)
	}
	if tagged.ID != "b2-cpu-fast-1" {
		t.Errorf("tagged build ID = %q, want b2-cpu-fast-1", tagged.ID)
	}
	covWaitForBuild(t, b, tagged.ID)
}

// A bad tag or profile is refused before anything is checked out, so no
// record appears.
func TestBuildRefusesBadTagAndProfile(t *testing.T) {
	b := NewBuilder(t.TempDir())
	if _, err := b.Build(context.Background(), "cpu", "b2", "bad_tag", false, nil, "", ""); err == nil || !strings.Contains(err.Error(), "invalid tag") {
		t.Errorf("tag with underscore: %v, want an invalid tag error", err)
	}
	if _, err := b.Build(context.Background(), "-cpu", "b2", "", false, nil, "", ""); err == nil || !strings.Contains(err.Error(), "unknown profile") {
		t.Errorf("unknown profile: %v", err)
	}
	if n := len(b.ListAll()); n != 0 {
		t.Errorf("%d records after refused builds, want 0", n)
	}
}

// Each failing step marks the build failed with a message that names the
// step and, for a failing tool, carries the tool's last output lines, so
// the Builds page shows why it failed.
func TestBuildFailures(t *testing.T) {
	cases := []struct {
		name      string
		cmake     string
		ninja     string
		wantError []string
	}{
		{"cmake fails", "#!/bin/sh\necho 'CMake Error: no CXX compiler'\nexit 1\n", "",
			[]string{"cmake failed", "CMake Error: no CXX compiler"}},
		{"ninja fails", "", "#!/bin/sh\necho 'error: undefined reference'\nexit 2\n",
			[]string{"ninja failed", "error: undefined reference"}},
		{"server not built", "", "#!/bin/sh\nmkdir -p bin\nprintf x > bin/llama-bench\n",
			[]string{"llama-server binary not found in build output"}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			dataDir, _, _ := covBuildSetup(t, c.cmake, c.ninja)
			b := NewBuilder(dataDir)
			res, err := b.Build(context.Background(), "cpu", "b2", "", false, nil, "", "")
			if err != nil {
				t.Fatalf("Build: %v", err)
			}
			got := covWaitForBuild(t, b, res.ID)
			if got.Status != BuildStatusFailed || got.FinishedAt.IsZero() || got.BinaryPath != "" {
				t.Fatalf("build = %+v", got)
			}
			for _, want := range c.wantError {
				if !strings.Contains(got.Error, want) {
					t.Errorf("Error = %q, want it to contain %q", got.Error, want)
				}
			}
			if _, err := os.Stat(filepath.Join(dataDir, "builds", res.ID, "llama-server")); err == nil {
				t.Error("a failed build installed a server")
			}
			if r, ok := NewBuilder(dataDir).Find(res.ID); !ok || r.Status != BuildStatusFailed {
				t.Errorf("reloaded failed record = %+v, %v", r, ok)
			}
		})
	}
}

// finishBuild updates the record with the same ID in place and saves it;
// a result whose ID is not in the list must not be added.
func TestFinishBuildUpdatesAndSaves(t *testing.T) {
	dir := t.TempDir()
	b := NewBuilder(dir)
	b.builds = []BuildResult{
		{ID: "b1-cpu", Status: BuildStatusBuilding},
		{ID: "b2-cpu", Status: BuildStatusSuccess},
	}
	r := BuildResult{ID: "b1-cpu", Status: BuildStatusBuilding}
	b.finishBuild(&r, BuildStatusFailed, "cmake failed: exit status 1")

	if r.Status != BuildStatusFailed || r.Error != "cmake failed: exit status 1" || r.FinishedAt.IsZero() {
		t.Errorf("result after finishBuild = %+v", r)
	}
	reloaded := NewBuilder(dir)
	got, ok := reloaded.Find("b1-cpu")
	if !ok || got.Status != BuildStatusFailed || got.Error != r.Error {
		t.Errorf("saved record = %+v, %v", got, ok)
	}
	if other, _ := reloaded.Find("b2-cpu"); other == nil || other.Status != BuildStatusSuccess {
		t.Errorf("other record changed: %+v", other)
	}

	stray := BuildResult{ID: "not-listed"}
	b.finishBuild(&stray, BuildStatusSuccess, "")
	if b.HasBuild("not-listed") || len(NewBuilder(dir).ListAll()) != 2 {
		t.Errorf("finishBuild added a build that was not in the list: %v", b.ListAll())
	}
}

// Delete removes the record and the build's files and saves the list;
// an unknown ID is an error and changes nothing.
func TestDeleteRemovesRecordAndFiles(t *testing.T) {
	dir := t.TempDir()
	b := NewBuilder(dir)
	b.builds = []BuildResult{
		{ID: "b1-cpu", Status: BuildStatusSuccess},
		{ID: "b2-cpu", Status: BuildStatusSuccess},
	}
	for _, id := range []string{"b1-cpu", "b2-cpu"} {
		if err := os.MkdirAll(filepath.Join(dir, "builds", id), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, "builds", id, "llama-server"), []byte("x"), 0o755); err != nil {
			t.Fatal(err)
		}
	}

	if err := b.Delete("b1-cpu"); err != nil {
		t.Fatalf("Delete: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, "builds", "b1-cpu")); !os.IsNotExist(err) {
		t.Errorf("build directory still exists: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, "builds", "b2-cpu", "llama-server")); err != nil {
		t.Errorf("the other build's files were touched: %v", err)
	}
	if b.HasBuild("b1-cpu") || !b.HasBuild("b2-cpu") {
		t.Errorf("records after Delete = %v", b.ListAll())
	}
	if got := NewBuilder(dir).ListAll(); len(got) != 1 || got[0].ID != "b2-cpu" {
		t.Errorf("saved records = %v, want only b2-cpu", got)
	}

	err := b.Delete("no-such-build")
	if err == nil || !strings.Contains(err.Error(), "build not found: no-such-build") {
		t.Errorf("Delete of an unknown ID: %v", err)
	}
	if len(b.ListAll()) != 1 {
		t.Errorf("a failed Delete changed the list: %v", b.ListAll())
	}
}
