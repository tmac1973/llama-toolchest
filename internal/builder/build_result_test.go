package builder

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

// The result Build returns is the caller's own copy: the Builds page
// renders it while the build goes on in the background, and the build
// writing to the same struct was a data race (seen with -race when cmake
// fails at once). The copy keeps the state the build started in.
func TestBuildReturnsACopyOfTheResult(t *testing.T) {
	dataDir, _, _ := covBuildSetup(t, "#!/bin/sh\nexit 1\n", "")
	b := NewBuilder(dataDir)
	res, err := b.Build(context.Background(), "cpu", "b2", "", false, nil, "", "")
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	got := covWaitForBuild(t, b, res.ID)
	// Read the returned copy after the build has finished writing: with
	// -race this fails if the two are the same struct.
	if res.Status != BuildStatusBuilding || !res.FinishedAt.IsZero() {
		t.Errorf("returned result changed with the build: %+v", *res)
	}
	if got.Status != BuildStatusFailed {
		t.Errorf("build status = %q, want failed", got.Status)
	}
}

// A patched build's patch set fields survive a save and load of
// builds.json, and a builds.json from before them loads with them empty.
func TestPatchFieldsRoundTrip(t *testing.T) {
	dir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dir, "config"), 0o755); err != nil {
		t.Fatal(err)
	}
	b := NewBuilder(dir)
	b.builds = []BuildResult{{
		ID: "rdna-r206-rocm", Status: BuildStatusSuccess, CommitCount: 11887,
		PatchSet: "rdna-boosts", PatchRelease: "v16-6050b467c-r206",
		PatchBase: "6050b467c07a95c7b87debd4ddac2f8ba8d9b807",
	}}
	b.saveBuilds()
	got, ok := NewBuilder(dir).Find("rdna-r206-rocm")
	if !ok || got.PatchSet != "rdna-boosts" || got.PatchRelease != "v16-6050b467c-r206" ||
		got.PatchBase != "6050b467c07a95c7b87debd4ddac2f8ba8d9b807" {
		t.Fatalf("patch fields lost: %+v", got)
	}

	old := `[{"id":"b2-cpu","status":"success","git_ref":"b2"}]`
	if err := os.WriteFile(filepath.Join(dir, "config", "builds.json"), []byte(old), 0o644); err != nil {
		t.Fatal(err)
	}
	got, ok = NewBuilder(dir).Find("b2-cpu")
	if !ok || got.PatchSet != "" || got.PatchRelease != "" || got.PatchBase != "" {
		t.Fatalf("old build = %+v, %v; want no patch fields", got, ok)
	}
}
