package builder

import (
	"context"
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
