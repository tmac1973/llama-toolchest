package builder

import (
	"context"
	"strings"
	"testing"
	"time"
)

// readAll reads a log channel until it closes, or fails after a while.
func readAll(t *testing.T, ch chan string) []string {
	t.Helper()
	var lines []string
	timeout := time.After(10 * time.Second)
	for {
		select {
		case line, ok := <-ch:
			if !ok {
				return lines
			}
			lines = append(lines, line)
		case <-timeout:
			t.Fatalf("the stream did not end; got %q", lines)
		}
	}
}

// The fetch and checkout lines of a start reach the stream of its token
// while they happen, the stream ends with the build ID, and the same lines
// are replayed into the build's own log. A reader that comes late gets
// the lines and the end.
func TestStartLogStreamsTheFetchAndCheckout(t *testing.T) {
	dataDir, _, _ := covBuildSetup(t, "#!/bin/sh\nexit 1\n", "")
	b := NewBuilder(dataDir)
	const token = "abcdef0123456789"
	early, done, ok := b.SubscribeStart(token)
	if !ok {
		t.Fatal("a valid token was refused")
	}
	defer done()

	res, err := b.Build(WithStartToken(context.Background(), token), "cpu", "b2", "", false, nil, "", "")
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	lines := readAll(t, early)
	joined := strings.Join(lines, "\n")
	if !strings.Contains(joined, "==> Checking out b2") || !strings.HasSuffix(joined, "==> Started build "+res.ID+".") {
		t.Errorf("start stream:\n%s", joined)
	}

	late, _, _ := b.SubscribeStart(token)
	if got := readAll(t, late); strings.Join(got, "\n") != joined {
		t.Errorf("a late reader got %q, want the same lines", got)
	}

	covWaitForBuild(t, b, res.ID)
	if log := strings.Join(covBuildLog(b, res.ID), "\n"); !strings.Contains(log, "==> Checking out b2") {
		t.Errorf("the checkout lines are not in the build log:\n%s", log)
	}
}

// A start that fails ends its stream with the reason, so the page's
// Starting panel does not wait for ever.
func TestStartLogEndsWhenTheStartFails(t *testing.T) {
	dataDir, _, _ := covBuildSetup(t, "#!/bin/sh\nexit 1\n", "")
	b := NewBuilder(dataDir)
	const token = "failedstart0001"
	ch, done, _ := b.SubscribeStart(token)
	defer done()
	if _, err := b.Build(WithStartToken(context.Background(), token), "cpu", "no-such-ref", "", false, nil, "", ""); err == nil {
		t.Fatal("a missing ref started a build")
	}
	lines := readAll(t, ch)
	if len(lines) == 0 || !strings.HasPrefix(lines[len(lines)-1], "==> The build did not start: ") {
		t.Errorf("start stream: %q", lines)
	}
}

func TestStartTokens(t *testing.T) {
	for token, want := range map[string]bool{"abcdef01": true, "ABCDEF01": false, "abc": false, "../../etc": false, strings.Repeat("a", 65): false} {
		if ValidStartToken(token) != want {
			t.Errorf("ValidStartToken(%q) = %v", token, !want)
		}
	}
	b := NewBuilder(t.TempDir())
	if _, _, ok := b.SubscribeStart("../x"); ok {
		t.Error("a malformed token got a stream")
	}
}

// RunningBuilds lists only running builds, newest first.
func TestRunningBuilds(t *testing.T) {
	now := time.Now()
	b := &Builder{builds: []BuildResult{
		{ID: "old", Status: BuildStatusBuilding, StartedAt: now.Add(-time.Hour)},
		{ID: "done", Status: BuildStatusSuccess, StartedAt: now},
		{ID: "new", Status: BuildStatusBuilding, StartedAt: now.Add(-time.Minute)},
	}}
	var ids []string
	for _, br := range b.RunningBuilds() {
		ids = append(ids, br.ID)
	}
	if strings.Join(ids, ",") != "new,old" {
		t.Errorf("running = %v, want new,old", ids)
	}
}
